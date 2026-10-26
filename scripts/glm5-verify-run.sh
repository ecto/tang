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
SSHOPTS="-o BatchMode=yes -o StrictHostKeyChecking=accept-new"
SSH="ssh $SSHOPTS ${PORT:+-p $PORT} $DEST"
# literal path: a ~ here would expand to the LOCAL home (openrsync quirk)
R=/root/tang-verify
SRC=${1:-.}

# 1. source up
$SSH "mkdir -p $R/src"
rsync -a --delete --exclude target --exclude .git --exclude .frog --exclude node_modules \
  ${PORT:+-e "ssh $SSHOPTS -p $PORT"} "$SRC/" "$DEST:$R/src/"

# 2. md5 the tree, both sides
L=$(cd "$SRC" && find crates -name '*.rs' -o -name Cargo.toml | LC_ALL=C sort | xargs md5 -r | md5 -q)
Rp=$($SSH 'cd '"$R"'/src && find crates -name "*.rs" -o -name Cargo.toml | LC_ALL=C sort | xargs md5sum | awk "{print \$1\" \"\$2}" | md5sum')
RM=$(echo "$Rp" | awk '{print $1}')
echo "src md5: local=$L pod=$RM"
[ "$L" = "$RM" ] || { echo "REFUSING: source md5 mismatch"; exit 1; }
# md5 output is a single 32-char token; if the ssh banner or an error leaked in,
# $RM would be multi-word — refuse rather than proceed on garbage
case "$RM" in *[!0-9a-f]*) echo "REFUSING: bad pod md5 output: $Rp"; exit 1;; esac

# 3. build (cuda feature; NVRTC sm_103 fallback is already on main, ec3890aa)
$SSH "[ -f /root/.cargo/bin/cargo ] || { curl -sSf https://sh.rustup.rs -o /tmp/rustup.sh && sh /tmp/rustup.sh -y --profile minimal --default-toolchain stable; }
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:\$PATH
[ -x /usr/local/cuda/bin/nvcc ] || { echo 'REFUSING: no nvcc on the pod'; exit 1; }
cd $R/src && CUDA_PATH=/usr/local/cuda cargo build --release -p tang-llm --bin tang-llm --features cuda \
  > $R/build.log 2>&1 && echo BUILD_RC=0 >> $R/build.log || { echo BUILD_RC=\$? >> $R/build.log; tail -30 $R/build.log; exit 1; }"
$SSH "export PATH=/root/.cargo/bin:\$PATH; grep -E 'BUILD_RC|error' $R/build.log | tail -5"

# 4. run with the dump on. The pod was created with no volume, so treat every
#    start as COLD: if the shards are missing, fetch them first
#    (unsloth/GLM-5.3-Flash-GGUF UD-Q4_K_XL, six shards, ~186 GB — on the B300
#    pod this took ~30 min; budget ~45 min to verify-ready on a cold disk).
G5=${GLM5_GGUF:-/workspace/glm53/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf}
$SSH "if [ ! -f $G5 ]; then
  echo 'weights missing — cold start, downloading unsloth/GLM-5.3-Flash-GGUF UD-Q4_K_XL';
  pip install -q -U huggingface_hub 2>/dev/null | tail -1 || pip3 install -q -U huggingface_hub;
  python3 -m huggingface_hub.commands.huggingface_cli download unsloth/GLM-5.3-Flash-GGUF \
    --include '*UD-Q4_K_XL*' --local-dir /workspace/glm53 >> $R/download.log 2>&1 \\
    || huggingface-cli download unsloth/GLM-5.3-Flash-GGUF --include '*UD-Q4_K_XL*' \\
    --local-dir /workspace/glm53 >> $R/download.log 2>&1;
fi
ls -la $G5 || { echo 'weights still missing after download attempt; set GLM5_GGUF'; exit 1; }
cd $R/src && TANG_GLM5_KDA_DUMP=1 timeout 3600 target/release/tang-llm flash-generate '$G5' \
  --prompt 'Write a haiku about the sea.' -n 8 --temp 0 --out $R/verify.ids \
  > $R/verify-run.log 2>&1; echo RUN_RC=\$? | tee -a $R/verify-run.log"
echo "---- ids ----"; $SSH "cat $R/verify.ids; echo; echo expected: 785 1196 6801 264 6386 38038 911 279"
echo "---- log tail ----"; $SSH "tail -8 $R/verify-run.log"
echo "next: pull $R/verify-run.log and diff its 'kda gpu' lines against the CPU oracle's 'kda cpu' lines layer by layer"
