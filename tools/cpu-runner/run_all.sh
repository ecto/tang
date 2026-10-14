#!/bin/sh
# Build and run every exported case (cpu_export export ROOT) on one core.
#   ./run_all.sh ROOT [CORE] [STEPS]
# Each run is ≤ ~4 s, niced, single-threaded. Waits while the hottest
# thermal zone is ≥ 82 °C until it is ≤ 78 °C. Logs load and temperature.
ROOT=$(cd "$1" && pwd)
CORE=${2:-7}
STEPS=${3:-20000}
CFLAGS=${CFLAGS:-"-O3 -march=armv8.6-a+fp16+fp16fml -mtune=cortex-x1 -ffp-contract=fast -fno-strict-aliasing"}
HERE=$(cd "$(dirname "$0")" && pwd)
temp() { cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -n | tail -1; }
cool() {
    t=$(temp)
    if [ "$t" -ge 82000 ]; then
        echo "hot ($t), waiting"
        while [ "$(temp)" -gt 78000 ]; do sleep 5; done
    fi
}
echo "start: load $(cut -d' ' -f1-3 /proc/loadavg) temp $(temp)"
for d in "$ROOT"/*/; do
    d=$(cd "$d" && pwd)
    [ -f "$d/model.c" ] || continue
    cool
    gcc $CFLAGS -DMODEL="\"$d/model.c\"" -o "$d/bench" "$HERE/bench.c" -lm || exit 1
    nice -n 10 "$d/bench" "$d" "$STEPS" "$CORE"
    echo "  load $(cut -d' ' -f1-3 /proc/loadavg) temp $(temp)"
done
if [ -d "$ROOT/gru_b1" ]; then
    cool
    gcc $CFLAGS -o "$ROOT/naive_gru" "$HERE/naive_gru.c" -lm &&
        nice -n 10 "$ROOT/naive_gru" "$ROOT/gru_b1" "$STEPS" "$CORE"
fi
echo "end: load $(cut -d' ' -f1-3 /proc/loadavg) temp $(temp)"
