#!/bin/bash
# glm5 first-tokens pod: 2xH200. Bootstraps rust, clones tang @ frog/glm5-forward,
# downloads GLM-5.3-Flash UD-Q4_K_XL, runs the W1 run-1 sequence, publishes results
# on :8001, then TERMINATES ITSELF (idle 15 min after ready, hard cap 240 min).
set -u
R=/workspace/results; mkdir -p "$R"; W=/workspace; T0=$(date +%s)
IDLE_MIN=${IDLE_MIN:-15}; HARD_MIN=${HARD_MIN:-240}
log(){ echo "$(date -u +%FT%TZ) [$(( ($(date +%s)-T0)/60 ))m] $*" | tee -a "$R/pod.log"; }
gql(){ curl -s -m 20 -H 'Content-Type: application/json' -H "Authorization: Bearer ${RUNPOD_API_KEY:-}" \
  https://api.runpod.io/graphql -d "{\"query\":\"$1\"}"; }
die_pod(){ log "TERMINATE: $1"
  for i in 1 2 3; do gql "mutation { podTerminate(input: {podId: \\\"$RUNPOD_POD_ID\\\"}) }" >>"$R/pod.log" 2>&1; sleep 30; done
  sleep 600; exit 0; }
# Phantom-create guard (pod 14: the API accepted the create, no worker claimed it).
# Poll our own log file's existence + mtime instead of GraphQL (no key juggling):
# the script itself writes pod.log every few seconds; if it was never claimed, no log.
if [ ! -s "$R/pod.log" ]; then
  log "PHANTOM CREATE GUARD: empty log at start (should never happen)"; sleep 300; exit 0
fi
( sleep $((HARD_MIN*60)); die_pod "hard deadline ${HARD_MIN}m" ) &
log "pod $RUNPOD_POD_ID; self-terminate auth: $(gql "query { pod(input: {podId: \\\"$RUNPOD_POD_ID\\\"}) { id } }" | head -c 120)"

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq --no-install-recommends curl ca-certificates git build-essential pkg-config python3 python3-venv rsync >/dev/null 2>&1 || die_pod "apt failed"
log "apt done"
nohup python3 -m http.server 8001 --directory "$R" >/dev/null 2>&1 &
sleep 2; log "results server: $(curl -s -m 5 localhost:8001/ | head -c 80)"
( python3 -m venv /opt/hf && /opt/hf/bin/pip install -q 'huggingface_hub[cli]==0.35.3' \
  && HF_HOME=$W/hf /opt/hf/bin/hf download unsloth/GLM-5.3-Flash-GGUF --include 'UD-Q4_K_XL/*' --local-dir $W/glm53 > $W/model.dir 2>>"$R/dl.log" \
  && touch $W/MODEL_OK ) &
curl -sSf https://sh.rustup.rs | sh -s -- -y -q --profile minimal --default-toolchain 1.94.0 >/dev/null 2>&1 || die_pod "rustup failed"
export PATH=/root/.cargo/bin:$PATH
git clone -q --filter=blob:none https://github.com/ecto/tang.git $W/tang && cd $W/tang && git checkout -q "${TANG_SHA:-3e9a487e}" || die_pod "checkout failed"
log "building tang-llm (cuda)"
cargo build -q --release -p tang-llm --no-default-features --features cuda >"$R/build.log" 2>&1 || { tail -40 "$R/build.log"; die_pod "build failed"; }
BIN=$W/tang/target/release/tang-llm; log "build done"

while [ ! -f $W/MODEL_OK ]; do sleep 10; [ $(( ($(date +%s)-T0)/60 )) -ge 75 ] && die_pod "model download >75m: $(tail -3 $R/dl.log)"; done
G=$(find $W/glm53 -name "*UD-Q4_K_XL-00001-of-*.gguf" | head -1); log "model $G ($(du -sh $W/glm53 | cut -f1))"
[ -n "$G" ] || { ls -R $W/glm53 | head -20; die_pod "no model shard found"; }
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader >>"$R/pod.log"

# 3. GPU kernel tests on the tiny GGUF (fixtures are committed in-repo)
mkdir -p /private/tmp/glm5-tiny
cp -r $W/tang/crates/tang-llm/tests/fixtures/glm5/* /private/tmp/glm5-tiny/
log "fixtures: $(ls /private/tmp/glm5-tiny | wc -l) dirs"
log "kernel tests (tiny, cuda)"
TANG_REQUIRE_CUDA=1 cargo test -q --release -p tang-llm --no-default-features --features cuda --lib glm5 -- --nocapture >"$R/kernels-tiny.log" 2>&1
log "kernel tests rc=$? | $(grep -E 'test result' $R/kernels-tiny.log | tail -1)"
export RUST_BACKTRACE=1; grep -A4 panicked "$R/kernels-tiny.log" | head -60
TANG_REQUIRE_CUDA=1 TANG_GPUS=2 TANG_SPLIT_AT=2 cargo test -q --release -p tang-llm --no-default-features --features cuda --lib glm5::engine -- --nocapture >"$R/engine-split.log" 2>&1
log "engine split tests rc=$? | $(grep -E 'test result' $R/engine-split.log | tail -1)"

# 4. FIRST TOKENS (the headline)
export TANG_GPUS=2 TANG_GLM5_PROF=1
# EARLY EVIDENCE: one 1-token tiny-fixture window so a fast death still leaves prof lines
TINY=$(find $W/tang/crates/tang-llm/tests/fixtures/glm5 -name '*.gguf' | head -1)
if [ -n "$TINY" ]; then
  log "smoke: tiny 1-token window"
  TANG_GPUS=2 TANG_GLM5_PROF=1 timeout 120 $BIN flash-generate "$TINY" --prompt-ids 1000 1001 -n 2 --temp 0 -v >"$R/smoke.log" 2>&1
  echo "SMOKE rc=$?"; grep -E "prof|window|tok/s" "$R/smoke.log" | head -8
fi
log "flash-generate (haiku)"
( timeout 900 $BIN flash-generate "$G" --prompt "Write a haiku about the sea." -n 32 --temp 0 -v --out /root/g2.ids >"$R/haiku.log" 2>&1; echo "haiku rc=$?"; echo "=== HAIKU IDS: $(head -c 120 /root/g2.ids 2>/dev/null)"; tail -8 "$R/haiku.log" ) > >(tee -a "$R/pod.log" >&2) 2>&1
log "haiku rc=$?"; echo "=== HAIKU LOG TAIL"; tail -25 "$R/haiku.log"; echo "=== HAIKU IDS: $(cat /root/g2.ids 2>/dev/null)"
( timeout 900 $BIN flash-generate "$G" --prompt "Write a Rust function that reverses a string." -n 48 --temp 0 -v --out /root/code.ids >"$R/code.log" 2>&1; echo "code rc=$?"; echo "=== CODE IDS: $(head -c 120 /root/code.ids 2>/dev/null)"; tail -8 "$R/code.log" ) > >(tee -a "$R/pod.log" >&2) 2>&1
log "code rc=$?"; echo "=== CODE LOG TAIL"; tail -25 "$R/code.log"

# 5. split-vs-CPU-ref greedy check
export TANG_GLM5_GPU=0
timeout 1800 $BIN flash-generate "$G" --prompt "Write a haiku about the sea." -n 8 --temp 0 --out /root/cpu.ids >"$R/cpuref.log" 2>&1
log "cpu-ref rc=$?"
unset TANG_GLM5_GPU
A=$(tr -s " \n" " " </root/cpu.ids 2>/dev/null | cut -d" " -f1-8); B=$(tr -s " \n" " " </root/g2.ids 2>/dev/null | cut -d" " -f1-8)
if [ -n "$A" ] && [ "$A" = "$B" ]; then log "GREEDY MATCH gpu==cpu-ref (8 ids): yes [$A]"; else log "GREEDY MATCH: NO gpu=[$B] cpu=[$A]"; fi
echo "=== CPUREF LOG TAIL"; tail -12 "$R/cpuref.log"

# 6. throughput, batch 1
export TANG_GPUS=2
timeout 1200 $BIN flash-bench "$G" --prompt-ids "1000 1001 1002" -n 64 >"$R/bench.log" 2>&1
log "bench rc=$?"; echo "=== BENCH TAIL"; tail -15 "$R/bench.log"; echo "=== CPUREF TAIL"; tail -8 "$R/cpuref.log"
grep -iE 'tok/s|tokens/s|prefill|ttft' "$R"/bench.log "$R"/haiku.log 2>/dev/null | head -20 >>"$R/pod.log"
touch "$R/DONE"; log "ALL DONE; holding ${IDLE_MIN}m"
sleep $((IDLE_MIN*60)); die_pod "done + idle"
