#!/usr/bin/env bash
# Decode speed: Cool-chic reference CPU decode vs our CPU and GPU decoders,
# plus GPU contention snapshots. Usage: speed.sh ROOT IMGCODEC OUT [IMG-LAMBDA...]
set -u
ROOT=$1; BIN=$2; OUT=$3; shift 3
EXP=$(cd "$(dirname "$0")" && pwd)/export_cc.py
echo "# $(date -u)  $(nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader)" >> $OUT
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | sed 's/^/# gpu-app /' >> $OUT
for run in "$@"; do
  d=$ROOT/runs/n10000/$run; img=${run%-*}
  cc=$(cd $ROOT/Cool-Chic && for i in 1 2 3; do CUDA_VISIBLE_DEVICES= $ROOT/venv/bin/python cc_decode.py -i $d/$img.cool -o /tmp/ccdec.png 2>&1 | grep -o "time = *[0-9.]*" | grep -o "[0-9.]*$"; done | sort -n | head -1)
  (cd $ROOT/Cool-Chic && PYTHONPATH=. CUDA_VISIBLE_DEVICES= $ROOT/venv/bin/python $EXP $d/$img.cool /tmp/sp.ccx > /dev/null 2>&1)
  for c in "64:0" "64:0x128" "128:128x128"; do
    L=${c%%:*}; T=${c#*:}
    o=$($BIN /tmp/sp.ccx $L $T 2>&1)
    g=$(echo "$o" | grep GPU | sed -E 's/.*decode ([0-9.]+) ms.*identical to CPU: ([a-z]+).*/\1 \2/')
    cpu=$(echo "$o" | grep "CPU decode" | sed -E 's/.*CPU decode ([0-9.]+)s.*/\1/')
    bpp=$(echo "$o" | head -1 | sed -E 's/.*: [0-9]+ B = ([0-9.]+) bpp.*/\1/')
    echo "$run L=$L T=$T bpp=$bpp coolchic_cpu_s=$cc ours_cpu_s=$cpu gpu_ms,identical=$g" >> $OUT
  done
  rm -f /tmp/sp.ccx
done
