#!/usr/bin/env bash
# Cool-chic 5 (HOP) encodes of Kodak at the published lambdas.
# Usage: run_kodak.sh ROOT N_ITR [IMAGES...]   (ROOT has Cool-Chic/, kodak/, venv/)
set -u
ROOT=$1; NITR=$2; shift 2
IMGS=${@:-$(cd $ROOT/kodak && ls kodim*.png | sed 's/.png//')}
cd $ROOT/Cool-Chic
for img in $IMGS; do
  for l in 0.0001 0.0004 0.001 0.004 0.02; do
    out=$ROOT/runs/n$NITR/$img-$l
    [ -f $out/$img.cool ] && continue
    mkdir -p $out
    $ROOT/venv/bin/python cc_encode.py -i $ROOT/kodak/$img.png -o $out/$img.cool --workdir $out \
      --dec_cfg_residue cfg/dec/intra/hop.cfg --lmbda $l --n_itr $NITR --start_lr 1e-2 > $out/log.txt 2>&1
    tail -1 $out/0000-results_decoder.tsv | sed "s|^|$img $l |"
  done
done
