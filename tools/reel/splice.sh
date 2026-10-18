#!/usr/bin/env bash
# Build the reel's soundtrack: the intro from 0, then two drops, each cut on a
# downbeat. Times come from beats.py / drop.py for the track you supply.
#   ./splice.sh track.wav
# Defaults are for the track the 47 s cut was made to (135 BPM, first downbeat
# at 0.862 s, drops on the downbeats at 39.974 s and 95.085 s).
set -euo pipefail
IN=${1:?usage: splice.sh track.wav}
P=$(python3 -c 'print(60/135)')
INTRO_END=${INTRO_END:-7.9736}   # first downbeat + 16 beats
DROP1=${DROP1:-39.974}           # 48 beats
DROP2=${DROP2:-95.085}           # 40 beats
D1=$(python3 -c "print(48*$P)"); D2=$(python3 -c "print(40*$P)")
ffmpeg -v error -y -i "$IN" -filter_complex \
  "[0:a]atrim=0:$INTRO_END,asetpts=PTS-STARTPTS,afade=t=out:st=$(python3 -c "print($INTRO_END-0.015)"):d=0.015[a];\
   [0:a]atrim=$DROP1:$(python3 -c "print($DROP1+$D1)"),asetpts=PTS-STARTPTS,afade=t=in:d=0.01,afade=t=out:st=$(python3 -c "print($D1-0.015)"):d=0.015[b];\
   [0:a]atrim=$DROP2:$(python3 -c "print($DROP2+$D2)"),asetpts=PTS-STARTPTS,afade=t=in:d=0.01,afade=t=out:st=$(python3 -c "print($D2-1.2)"):d=1.17[c];\
   [a][b][c]concat=n=3:v=0:a=1[o]" -map "[o]" audio/spliced.wav
echo "audio/spliced.wav"
