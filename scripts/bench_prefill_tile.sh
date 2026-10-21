#!/usr/bin/env bash
# A/B tiled vs split-K wide prefill on an H200, including greedy and KL gates.
# Usage: scripts/bench_prefill_tile.sh <model.gguf> [prompt-ids.txt]
set -euo pipefail
[[ $# -ge 1 && $# -le 2 ]] || { echo "usage: $0 <model.gguf> [prompt-ids.txt]" >&2; exit 2; }
MODEL=$1
PROMPT=${2:-${TANG_PREFILL_TILE_IDS:-${HOME}/.cache/tang/code.ids}}
OUT=${TANG_PREFILL_TILE_OUT:-prefill-tile-$(date +%Y%m%d-%H%M%S).md}
TMP=$(mktemp -d "${TMPDIR:-/tmp}/prefill-tile.XXXXXX")
cleanup() {
    rm -f "$TMP"/*.log
    rmdir "$TMP"
}
trap cleanup EXIT
[[ -s "$PROMPT" ]] || { echo "set TANG_PREFILL_TILE_IDS to a prompt ids file (missing $PROMPT)" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi is required; run this on the H200 pod" >&2; exit 2; }
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
[[ "$GPU" == *H200* ]] || { echo "expected an H200, found: $GPU" >&2; exit 2; }

cargo build --release -p tang-llm --no-default-features --features cuda --bin tang-llm
X=${TANG_PREFILL_TILE_BIN:-target/release/tang-llm}
[[ -x "$X" ]] || { echo "binary not executable: $X" >&2; exit 2; }

printf '# Tiled prefill A/B\n\nGPU: %s  \nModel: `%s`  \nCorrectness: greedy identical for 64 steps; teacher-forced max KL <= 1e-3 vs narrow.\n\n| tokens | tiled cold (s) | split-K cold (s) | speedup | baseline wide (s) | target | tiled gate | split-K gate |\n|---:|---:|---:|---:|---:|---:|:---:|:---:|\n' "$GPU" "$MODEL" > "$OUT"
for n in 8192 30000 54000; do
    gate=()
    for mode in 1 0; do
        log="$TMP/$n-$mode.log"
        TANG_WIDE_CTXS=$n TANG_CODER_ATTN_TILE=$mode "$X" flash-wide-check "$MODEL" --ids-file "$PROMPT" -n 64 >"$log" 2>&1
        grep -Eq "wide-check $n: PASS \\(.*64 decoded tokens equal" "$log" || { cat "$log" >&2; echo "greedy identity gate against narrow failed for n=$n tile=$mode" >&2; exit 1; }
        kl=$(sed -nE 's/.*teacher-forced max KL\(narrow \|\| wide\) over 64 steps: ([0-9.eE+-]+).*/\1/p' "$log" | tail -1)
        [[ -n "$kl" ]] || { cat "$log" >&2; echo "missing teacher-forced KL for n=$n tile=$mode" >&2; exit 1; }
        awk -v kl="$kl" 'BEGIN { exit !(kl <= 1e-3) }' || { cat "$log" >&2; echo "KL gate failed for n=$n tile=$mode: $kl > 1e-3" >&2; exit 1; }
        gate+=(PASS)
    done
    tiled=$(sed -nE "s/.*prefill $n tokens: narrow [0-9.]+ s \\([^)]*\\), wide ([0-9.]+) s cold.*/\\1/p" "$TMP/$n-1.log" | tail -1)
    split=$(sed -nE "s/.*prefill $n tokens: narrow [0-9.]+ s \\([^)]*\\), wide ([0-9.]+) s cold.*/\\1/p" "$TMP/$n-0.log" | tail -1)
    [[ -n "$tiled" && -n "$split" ]] || { echo "could not parse cold wide timings at $n tokens" >&2; exit 1; }
    speed=$(awk -v a="$tiled" -v b="$split" 'BEGIN { printf "%.2fx", b/a }')
    baseline=$(awk -v n="$n" 'BEGIN { if(n==8192) print "4.7"; else if(n==30000) print "28.5"; else print "72" }')
    target=$(awk -v n="$n" 'BEGIN { if(n==30000) print "~10 s"; else print "—" }')
    printf '| %s | %s | %s | %s | %s | %s | %s | %s |\n' "$n" "$tiled" "$split" "$speed" "$baseline" "$target" "${gate[0]}" "${gate[1]}" >> "$OUT"
done
cat "$OUT"
echo "gates passed: greedy identity over 64 steps and teacher-forced max KL <= 1e-3 for both modes at all lengths" >&2
