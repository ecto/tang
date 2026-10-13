#!/bin/sh
# Run every exported case under ROOT with tang_cl, pausing while the SoC
# runs hot: before each case, at >= 82 C wait until it is back to <= 78 C.
# Usage: run_all.sh ROOT [budget_s] [max_reps] [build opts...]
root=$1
shift
hot() { cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -n | tail -1; }
echo "before: $(uptime) max temp $(hot)"
for d in "$root"/*/; do
    t=$(hot)
    if [ "$t" -ge 82000 ]; then
        echo "hot: $t mC, waiting"
        while [ "$(hot)" -gt 78000 ]; do sleep 3; done
    fi
    echo "== $(basename "$d") ($t mC)"
    ./tang_cl "$d" "$@" 2>&1 | grep -E "reps|failed|error" | head -5
    sleep 1
done
echo "after: $(uptime) max temp $(hot)"
