#!/usr/bin/env bash
# GLM-5 KDA staging verify run: one command for the pod.
#
#   scripts/glm5-verify-run.sh root@HOST -p PORT [source-dir]
#
# 1. rsync the local worktree into ~/tang-verify/src on the pod (no local cargo
#    needed; the pod builds with --features cuda).
# 2. md5 the source tree on both sides — REFUSE to build or run on a mismatch
#    (this is the stale-binary guard; there have been four).
# 3. cargo build --release -p tang-llm --bin tang-llm --features cuda.
# 4. Run flash-generate on the real GLM-5.3 weights with TANG_GLM5_KDA_DUMP=1
#    (same prompt / -n 8 / --temp 0 as the CPU oracle), capturing the kda gpu
#    dump lines AND the ids in one log. The verdict is the ids line: it must
#    equal the CPU oracle's 785 1196 6801 264 6386 38038 911 279; the first
#    divergent kda gpu vs kda cpu line (layer-by-layer) names the staging bug.
#
# The pod keeps its disk, so the weights stay at /root/glm53/UD-Q4_K_XL
# (see the pod's earlier dump run); override with GLM5_GGUF if it moved.
set -euo pipefail
cd "$(dirname "$0")/.."
DEST=${1:?usage: glm5-verify-run.sh root@HOST [-p PORT]}; shift
PORT=
if [ "${1:-}" = -p ]; then PORT=$2; shift 2; fi
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new ${PORT:+-p $PORT} $DEST"
R=~/tang-verify
SRC=${1:-.}

# 1. source up
rsync -a --delete --exclude target --exclude .git --exclude .frog --exclude node_modules \
  ${PORT:+-e "ssh -p $PORT"} "$SRC/" "$DEST:$R/src/"

# 2. md5 the tree, both sides
L=$(cd "$SRC" && find crates -name '*.rs' -o -name Cargo.toml | LC_ALL=C sort | xargs md5 -r | md5 -q)
Rp=$($SSH "cd $R/src && find crates -name '*.rs' -o -name Cargo.toml | LC_ALL=C sort | xargs md5sum | awk '{print \$1\" \"\$2}' | md5sum")
R=$(echo "$Rp" | awk '{print $1}')
echo "src md5: local=$L pod=$R"
[ "$L" = "$R" ] || { echo "REFUSING: source md5 mismatch"; exit 1; }

# 3. build (cuda feature; NVRTC sm_103 fallback is already on main, ec3890aa)
$SSH "cd $R/src && cargo build --release -p tang-llm --bin tang-llm --features cuda \
  > $R/build.log 2>&1 && echo BUILD_RC=0 >> $R/build.log || { echo BUILD_RC=\$? >> $R/build.log; tail -30 $R/build.log; exit 1; }"
$SSH "grep -E 'BUILD_RC|error' $R/build.log | tail -5"

# 4. run with the dump on
G5=${GLM5_GGUF:-/root/glm53/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf}
$SSH "ls -la $G5 || { echo 'weights not found, set GLM5_GGUF'; exit 1; }
cd $R/src && TANG_GLM5_KDA_DUMP=1 timeout 3600 target/release/tang-llm flash-generate '$G5' \
  --prompt 'Write a haiku about the sea.' -n 8 --temp 0 --out $R/verify.ids \
  > $R/verify-run.log 2>&1; echo RUN_RC=\$? | tee -a $R/verify-run.log"
echo "---- ids ----"; $SSH "cat $R/verify.ids; echo; echo expected: 785 1196 6801 264 6386 38038 911 279"
echo "---- log tail ----"; $SSH "tail -8 $R/verify-run.log"
echo "next: pull $R/verify-run.log and diff its 'kda gpu' lines against the CPU oracle's 'kda cpu' lines layer by layer"
