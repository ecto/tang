#!/bin/bash
# dart fair-share prefill bench pod. Builds tang at BENCH_REF (default frog/dart-prefill-bench = eaffdc84 fair-share), serves
# Qwen3-Coder-Next UD-Q4_K_XL at production flags, runs correctness then TTFT for
# TANG_PREFILL_POLICY=fair and =srf, publishes results on :8001, then TERMINATES ITSELF.
set -u
R=/workspace/results; mkdir -p "$R"; W=/workspace; T0=$(date +%s)
HARD_MIN=${HARD_MIN:-115}; HOLD_MIN=${HOLD_MIN:-15}
log(){ echo "$(date -u +%FT%TZ) [$(( ($(date +%s)-T0)/60 ))m] $*" | tee -a "$R/pod.log"; }
gql(){ curl -s -m 20 -H 'Content-Type: application/json' -H "Authorization: Bearer ${RUNPOD_API_KEY:-}" \
  https://api.runpod.io/graphql -d "{\"query\":\"$1\"}"; }
die_pod(){ log "TERMINATE: $1"
  for i in 1 2 3; do gql "mutation { podTerminate(input: {podId: \\\"$RUNPOD_POD_ID\\\"}) }" >>"$R/pod.log" 2>&1; sleep 30; done
  sleep 600; exit 0; }
( sleep $((HARD_MIN*60)); die_pod "hard deadline ${HARD_MIN}m" ) &
log "pod $RUNPOD_POD_ID; self-terminate auth: $(gql "query { pod(input: {podId: \\\"$RUNPOD_POD_ID\\\"}) { id } }" | head -c 120)"
(cd "$R" && nohup python3 -m http.server 8001 >/dev/null 2>&1 &) 2>/dev/null

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq --no-install-recommends curl ca-certificates git build-essential pkg-config python3 python3-venv >/dev/null 2>&1 || die_pod "apt failed"
(cd "$R" && nohup python3 -m http.server 8001 >/dev/null 2>&1 &)
log "apt done"

# model download in the background (production: unsloth/Qwen3-Coder-Next-GGUF *UD-Q4_K_XL*)
( python3 -m venv /opt/hf && /opt/hf/bin/pip install -q 'huggingface_hub[cli]==0.35.3' \
  && HF_HOME=$W/hf /opt/hf/bin/hf download unsloth/Qwen3-Coder-Next-GGUF --include '*UD-Q4_K_XL*' --quiet > $W/model.dir 2>>"$R/dl.log" \
  && touch $W/MODEL_OK ) &

curl -sSf https://sh.rustup.rs | sh -s -- -y -q --profile minimal --default-toolchain 1.94.0 >/dev/null 2>&1 || die_pod "rustup failed"
export PATH=/root/.cargo/bin:$PATH
git clone -q --filter=blob:none https://github.com/ecto/tang.git $W/tang && cd $W/tang || die_pod "clone failed"
git checkout -q "${BENCH_REF:-frog/dart-prefill-bench}" || die_pod "checkout failed"
cp tools/bench/dart_prefill_bench.py $W/prefill_bench.py
log "building tang-llm (cuda)"
cargo build -q --release -p tang-llm --no-default-features --features cuda >"$R/build.log" 2>&1 || die_pod "build failed: $(tail -5 $R/build.log | tr '\n' ' ')"
BIN=$W/tang/target/release/tang-llm; log "build done"
python3 $W/prefill_bench.py prompts $W/tang $R >>"$R/pod.log" 2>&1

while [ ! -f $W/MODEL_OK ]; do sleep 10; [ $(( ($(date +%s)-T0)/60 )) -ge 60 ] && die_pod "model download >60m: $(tail -3 $R/dl.log)"; done
GGUF=$(find "$(cat $W/model.dir)" -name '*.gguf' | sort | head -1); log "model $GGUF"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader >>"$R/pod.log"

serve(){ # $1 policy
  pkill -f "tang-llm serve" 2>/dev/null; sleep 3
  TANG_PREFILL_POLICY=$1 TANG_MAX_BATCH=8 nohup $BIN serve "$GGUF" --host 127.0.0.1 --port 8912 --ctx 131072 \
    --kv-ttl 3600 --batch 8 >"$R/serve-$1-$(date +%s).log" 2>&1 &
  for i in $(seq 1 120); do curl -sf -m 3 localhost:8912/health >/dev/null && { log "serve[$1] up"; break; }; sleep 5; done
  curl -sf -m 3 localhost:8912/health >/dev/null || die_pod "serve[$1] never came up"
  curl -s -m 120 localhost:8912/v1/chat/completions -H 'content-type: application/json' \
    -d '{"model":"dart","messages":[{"role":"user","content":"hi"}],"max_tokens":8,"temperature":0}' >/dev/null
}
for pol in fair srf; do
  serve $pol; log "solo[$pol]"; python3 $W/prefill_bench.py solo $R $pol >>"$R/pod.log" 2>&1
  serve $pol; log "conc[$pol]"; python3 $W/prefill_bench.py conc $R $pol >>"$R/pod.log" 2>&1
done
serve fair; log "long[fair]"; python3 $W/prefill_bench.py long $R fair >>"$R/pod.log" 2>&1
pkill -f "tang-llm serve"
touch "$R/DONE"; log "ALL DONE; holding ${HOLD_MIN}m for result pickup"
sleep $((HOLD_MIN*60)); die_pod "done + hold"
