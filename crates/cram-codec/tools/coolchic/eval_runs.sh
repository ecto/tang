#!/usr/bin/env bash
# For every Cool-chic encode under ROOT/runs/nNITR/IMG-LAMBDA: export, code as
# TCI1 in several lane/tile layouts, and append "label lambda image bpp psnr"
# lines to OUT. Usage: eval_runs.sh ROOT NITR IMGCODEC OUT
set -u
ROOT=$1; NITR=$2; BIN=$3; OUT=$4
EXP=$(cd "$(dirname "$0")" && pwd)/export_cc.py
touch $OUT
for d in $ROOT/runs/n$NITR/*-*; do
  [ -f $d/0000-results_decoder.tsv ] || continue
  b=$(basename $d); img=${b%-*}; l=${b##*-}
  grep -q " $l $img " $OUT && continue
  (cd $ROOT/Cool-Chic && PYTHONPATH=. CUDA_VISIBLE_DEVICES= $ROOT/venv/bin/python $EXP $d/$img.cool $d/$img.ccx $ROOT/kodak/$img.png > /dev/null 2>&1) || { echo "export failed $d"; continue; }
  for c in "serial:1:0" "L64:64:0" "L64-b128:64:0x128" "L128-t128:128:128x128"; do
    lab=${c%%:*}; rest=${c#*:}; L=${rest%%:*}; T=${rest#*:}
    o=$($BIN $d/$img.ccx $L $T 2>&1)
    bpp=$(echo "$o" | head -1 | sed -E 's/.*: [0-9]+ B = ([0-9.]+) bpp.*/\1/')
    ps=$(echo "$o" | grep PSNR | sed -E 's/PSNR ours ([0-9.]+) dB.*/\1/')
    [ $lab = serial ] && {
      cb=$(echo "$o" | head -1 | sed -E 's/.*coolchic [0-9]+ B = ([0-9.]+) bpp.*/\1/')
      cp=$(echo "$o" | grep PSNR | sed -E 's/.*coolchic ref ([0-9.]+) dB.*/\1/')
      echo "cc-n$NITR $l $img $cb $cp" >> $OUT
    }
    echo "tci-$lab-n$NITR $l $img $bpp $ps" >> $OUT
  done
  rm -f $d/$img.ccx $d/0000-frame_encoder.pt
done
