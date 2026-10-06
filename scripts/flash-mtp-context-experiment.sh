#!/usr/bin/env bash
# Usage: script <frozen tang-llm> <main.gguf> <original mtp.gguf> <corpus root> <prompt dir> <new out dir>
# The corpus root must have data/, prefix/, and their completed validation manifests.
set -euo pipefail
[[ $# == 6 ]] || { echo "usage: $0 binary main mtp corpus prompt-dir new-out" >&2; exit 2; }
X=$1; MAIN=$2; MTP=$3; CORPUS=$4; PROMPTS=$5; OUT=$6
LOCK=${TANG_GPU_LOCK:-$HOME/tang-gpu.lock}
[[ ! -e "$OUT" ]] || { echo "output exists: $OUT" >&2; exit 2; }
[[ -f "$CORPUS/data/gen.done" && -f "$CORPUS/prefix/done" && -f "$CORPUS/finite-validation.json" && -f "$CORPUS/sha256-manifest.json" ]]
mkdir -p "$OUT"
trap 'task_exit=$?; printf "%s\n" "$task_exit" > "$OUT/exit-code"; touch "$OUT/done"' EXIT
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-/usr/local/cuda/lib64}
# Do not collide with a GPU workload that does not use the lock. Check before and after
# taking it; waiting outside it allows other users to finish their tests and clean up.
gpu_run() {
    while true; do
        free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1)
        if (( free >= 22000 )); then
            if flock "$LOCK" bash -c '
                free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1)
                (( free >= 22000 )) || exit 75
                exec "$@"
            ' _ "$@"; then return 0; else
                task_exit=$?
                (( task_exit == 75 )) || return "$task_exit"
            fi
        fi
        echo "waiting for exclusive GPU headroom ($free MiB free)" >&2
        sleep 30
    done
}
sha256sum "$X" "$MTP" > "$OUT/binary-and-mtp.sha256"
common=("$MAIN" "$MTP" --seq 16 --burn 4 --lr 1e-5 --beta .8 --clip 1 --vocab 0 --eval-every 25 --eval-windows 3)
mkdir "$OUT/smoke-data"
ln -s "$CORPUS/data/p000" "$OUT/smoke-data/p000"
ln -s "$CORPUS/data/p157" "$OUT/smoke-data/p157"
# Seed 47 samples generated offset 11195 of the maximum-length p157 sequence.
gpu_run /usr/bin/time -v "$X" flash-mtp-train "${common[@]}" --data "$OUT/smoke-data" --prefix-data "$CORPUS/prefix" --out "$OUT/smoke" --steps 1 --seed 47 > "$OUT/smoke.log" 2>&1
gpu_run "$X" flash-spec-test "$MAIN" --mtp "$MTP" --ids-file "$HOME/flash-truth/code.ids" -n 96 > "$OUT/original-spec.log" 2>&1
grep -q '^spec test: PASS$' "$OUT/original-spec.log"
gpu_run "$X" flash-bench-panel "$MAIN" --mtp "$MTP" --prompt-dir "$PROMPTS" --heldout-dir "$CORPUS/data" --slots 11000 --reserve-mb 1024 --repeats 3 -n 512 > "$OUT/original-panel.jsonl" 2> "$OUT/original-panel.log"
for arm in control context; do
    flags=(); [[ "$arm" == context ]] && flags=(--prefix-data "$CORPUS/prefix")
    gpu_run /usr/bin/time -v "$X" flash-mtp-train "${common[@]}" --data "$CORPUS/data" --out "$OUT/$arm" --steps 50 --seed 42 "${flags[@]}" > "$OUT/$arm-train.log" 2>&1
    for step in 000025 000050; do
        checkpoint="$OUT/$arm/step-$step/mtp.gguf"
        gpu_run "$X" flash-mtp-train "$MAIN" "$checkpoint" --data "$CORPUS/data" --out "$OUT/$arm-reload-$step" --steps 0 --seq 16 --burn 4 --eval-windows 3 "${flags[@]}" > "$OUT/$arm-reload-$step.log" 2>&1
        gpu_run "$X" flash-spec-test "$MAIN" --mtp "$checkpoint" --ids-file "$HOME/flash-truth/code.ids" -n 96 > "$OUT/$arm-spec-$step.log" 2>&1
        grep -q '^spec test: PASS$' "$OUT/$arm-spec-$step.log"
        gpu_run "$X" flash-bench-panel "$MAIN" --mtp "$checkpoint" --prompt-dir "$PROMPTS" --heldout-dir "$CORPUS/data" --slots 11000 --reserve-mb 1024 --repeats 3 -n 512 > "$OUT/$arm-panel-$step.jsonl" 2> "$OUT/$arm-panel-$step.log"
    done
done
python3 - "$OUT" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); baseline={(r['kind'],r['case'],r['repeat']):r['generated'] for r in map(json.loads,(root/'original-panel.jsonl').read_text().splitlines())}
assert len(baseline)==32, f'expected 12 serving and 20 held-out cases, got {len(baseline)}'
for arm in ['control','context']:
 for step in ['000025','000050']:
  rows=list(map(json.loads,(root/f'{arm}-panel-{step}.jsonl').read_text().splitlines()))
  assert len(rows)==len(baseline)
  for r in rows:
   key=(r['kind'],r['case'],r['repeat'])
   assert r['generated']==baseline[key], f'target output differs: {arm}/{step}/{key}'
(root/'panel-token-exactness.json').write_text(json.dumps({'all_checkpoint_panel_tokens_match_original':True,'cases_per_checkpoint':len(baseline)}))
print('All benchmark and held-out generated tokens match the original')
PY
