#!/bin/zsh
# GPU-only timing + byte-identity on the saved learned TCG3 streams.
cd "$(dirname $0)/.."
TC_GPU_PROFILE=${PROFILE:-} ./target/release/examples/gpubench gpu3 bench/out/gpu3_streams/*.tcg3 --reps ${REPS:-3} pretrained:bench/models/pre3_32x16.tcm pretrained:bench/models/pre3_16.tcm 2>&1 | grep -o '"stream":"[^"]*"\|"identical":[a-z]*\|"kernels_gbs":[0-9.]*\|"total_gbs":[0-9.]*\|decode3l", [0-9.]*\|panicked.*' | paste -s -d' ' - | sed 's/"stream"/\n"stream"/g'
echo; uptime
