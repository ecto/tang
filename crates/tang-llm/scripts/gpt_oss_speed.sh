#!/usr/bin/env bash
# gpt-oss single-stream speed, Artificial-Analysis style: a ~1k-token chat prompt, then output
# tok/s over the completion (and time to first token), plus an nsys kernel summary of decode.
#
#   gpt_oss_speed.sh <model-dir> [out-tokens]
#
# Run from the repo root on a CUDA box with nsys on PATH (or NSYS=...). Prints markdown rows for
# docs/speed-log.md, tagged with the commit. Variants compared on the same pod: the MXFP4 GEMV
# with its input read through L1 (default) and staged in shared memory (TANG_MOE_GEMV=smem).
set -euo pipefail
model=${1:?model dir}
out=${2:-1000}
commit=$(git rev-parse --short HEAD)
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
nsys=${NSYS:-nsys}

cargo clean -q -p tang-compute -p tang-llm --release
cargo build -q --release -p tang-llm --no-default-features --features cuda --bin tang-llm
bin=${CARGO_TARGET_DIR:-target}/release/tang-llm

# A ~1k-token prompt through the model's own chat template (harmony).
ids=$(python3 - "$model" <<'EOF'
import sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1])
para = ("The history of the printing press spans movable type in East Asia, Gutenberg's press in "
        "Mainz, and the spread of print shops across Europe within fifty years. ")
body = para * 1
while len(tok(body)["input_ids"]) < 950:
    body += para
msgs = [{"role": "user", "content": body + "\nSummarise the above in detail."}]
text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
print(" ".join(map(str, tok(text, add_special_tokens=False)["input_ids"])))
EOF
)
n_prompt=$(wc -w <<<"$ids")
echo "commit $commit · $gpu · prompt $n_prompt tokens · $out out" >&2

run() { # label, env...
  local label=$1; shift
  "$bin" generate "$model" $ids -n 16 >/dev/null 2>&1 # warm: kernel compile, pool
  for rep in 1 2 3; do
    line=$(env "$@" "$bin" generate "$model" $ids -n "$out" 2>&1 | grep -o "prefill.*")
    pre_ms=$(sed -E 's/.* in ([0-9]+)ms.*/\1/' <<<"$line")
    tps=$(sed -E 's/.*decode [0-9]+ tok at ([0-9.]+) tok\/s.*/\1/' <<<"$line")
    echo "| $(date -u +%F) | $commit | gpt-oss $label | $gpu | $(basename "$model") | 1 | $tps | ttft ${pre_ms} ms |"
  done
}
run "MXFP4 GEMV, x in smem" TANG_MOE_GEMV=smem
run "MXFP4 GEMV, x via L1" TANG_MOE_GEMV=l1

# Where decode time goes: kernels (time, count, avg) and the API calls that block.
$nsys profile -q -t cuda -o /tmp/gpt-oss-speed -f true "$bin" generate "$model" $ids -n 256 >/dev/null 2>&1
$nsys stats -q --report cuda_gpu_kern_sum --format csv /tmp/gpt-oss-speed.nsys-rep | head -16 | cut -d, -f1,2,3,4,9
$nsys stats -q --report cuda_api_sum --format csv /tmp/gpt-oss-speed.nsys-rep | head -8 | cut -d, -f1,2,3,9
