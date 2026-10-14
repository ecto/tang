#!/bin/sh
# Build and run every exported case (cpu_export export ROOT) on one core.
#   ./run_all.sh ROOT [CORE] [STEPS] [CASE...]
# Each run is ≤ ~4.5 s, niced, single-threaded. Before every run, waits
# until the hottest thermal zone is ≤ 78 °C (runs heat the core past
# 82 °C within seconds). Logs load and temperature.
ROOT=$(cd "$1" && pwd)
CORE=${2:-7}
STEPS=${3:-20000}
shift 3 2>/dev/null
CASES="$*"
CFLAGS=${CFLAGS:-"-O3 -march=armv8.6-a+fp16+fp16fml -mtune=cortex-x1 -ffp-contract=fast -fno-strict-aliasing"}
HERE=$(cd "$(dirname "$0")" && pwd)
temp() { cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -n | tail -1; }
cool() {
    t=$(temp)
    if [ "$t" -gt 78000 ]; then
        echo "  hot ($t), waiting"
        while [ "$(temp)" -gt 78000 ]; do sleep 3; done
    fi
}
echo "start: load $(cut -d' ' -f1-3 /proc/loadavg) temp $(temp)"
for d in "$ROOT"/*/; do
    d=$(cd "$d" && pwd)
    [ -f "$d/model.c" ] || continue
    if [ -n "$CASES" ]; then
        case " $CASES " in *" $(basename "$d") "*) ;; *) continue ;; esac
    fi
    cool
    gcc $CFLAGS -DMODEL="\"$d/model.c\"" -o "$d/bench" "$HERE/bench.c" -lm || exit 1
    nice -n 10 "$d/bench" "$d" "$STEPS" "$CORE"
    echo "  load $(cut -d' ' -f1-3 /proc/loadavg) temp $(temp)"
done
if [ -d "$ROOT/gru_b1" ] && { [ -z "$CASES" ] || case " $CASES " in *" naive "*) true ;; *) false ;; esac; }; then
    cool
    gcc $CFLAGS -o "$ROOT/naive_gru" "$HERE/naive_gru.c" -lm &&
        nice -n 10 "$ROOT/naive_gru" "$ROOT/gru_b1" "$STEPS" "$CORE"
fi
echo "end: load $(cut -d' ' -f1-3 /proc/loadavg) temp $(temp)"
