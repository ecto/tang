#!/bin/sh
# Qwen3 prefill/decode bench for the Booster K1 (Adreno 740, OpenCL), after
# `qwen_decode tune` has written ~/.config/tang/tune.json. One command:
#
#   scripts/k1-prefill-bench.sh [model dir, default ~/models/qwen3-1.7b-4bit]
#
# Prints, per variant, the device time of a 20/49/512-token prefill, a
# 20-token turn after 512 cached (as a Session splits it), and 128 decode
# steps; then the accuracy check of the packed-weight prefills (f32, fp16)
# against the f32-map one. Variants:
#   maps   TANG_CL_PREFILL_Q4=0   dequantized f32 weights per pass (before)
#   q4     default                dequantized in the register-block kernel
#   fp16   TANG_CL_PREFILL_F16=1  q4 with half operands, f32 accumulation
set -e
dir=${1:-$HOME/models/qwen3-1.7b-4bit}
cd "$(dirname "$0")/.."
cargo build --release -p tang-bench --features opencl --example qwen_decode
bin=target/release/examples/qwen_decode
export TANG_BACKEND=opencl
echo "== maps"; TANG_CL_PREFILL_Q4=0 "$bin" bench "$dir" --gen 0 2>/dev/null
echo "== q4";   "$bin" bench "$dir" --gen 128 2>/dev/null
echo "== fp16"; TANG_CL_PREFILL_F16=1 "$bin" bench "$dir" --gen 0 2>/dev/null
echo "== check"; "$bin" bench "$dir" --sizes 20,49 --gen 0 --check 2>/dev/null | grep -E "^(bench|check)"
