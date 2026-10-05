#!/usr/bin/env bash
# Measure an exported MTP checkpoint using the same four cells as flash-handoff.md.
# Usage: flash-mtp-evaluate.sh <tang-llm> <main.gguf> <mtp.gguf> <new-log-dir>
set -euo pipefail
[[ $# == 4 ]] || { echo "usage: $0 <tang-llm> <main.gguf> <mtp.gguf> <new-log-dir>" >&2; exit 2; }
X=$1; MAIN=$2; MTP=$3; OUT=$4
LOCK=${TANG_GPU_LOCK:-$HOME/tang-gpu.lock}
PROMPTS=${TANG_FLASH_PROMPT_DIR:-$HOME/flash-engine}
TRUTH=${TANG_FLASH_TRUTH_DIR:-$HOME/flash-truth}
[[ ! -e "$OUT" ]] || { echo "output already exists: $OUT" >&2; exit 2; }
mkdir -p "$OUT"
trap 'status=$?; printf "%s\n" "$status" > "$OUT/exit-code"; touch "$OUT/done"' EXIT
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-/usr/local/cuda/lib64}
flock "$LOCK" "$X" flash-spec-test "$MAIN" --ids-file "$TRUTH/code.ids" -n 96 --mtp "$MTP" > "$OUT/spec.log" 2>&1
grep -q '^spec test: PASS$' "$OUT/spec.log"
for prompt in code chat; do
    for thinking in off on; do
        flags=(); [[ "$thinking" == off ]] && flags=(--no-think)
        draft=mtp; steps=3
        [[ "$prompt/$thinking" == code/off ]] && { draft=hybrid; steps=5; }
        TANG_FLASH_MTP_STEPS=$steps flock "$LOCK" "$X" flash-bench "$MAIN" \
            --prompt-file "$PROMPTS/${prompt}4k.txt" "${flags[@]}" -n 512 \
            --mtp "$MTP" --draft "$draft" > "$OUT/$prompt-$thinking.log" 2>&1
    done
done
