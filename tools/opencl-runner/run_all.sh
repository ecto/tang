#!/bin/sh
# Run every exported case under ROOT with tang_cl, stopping if the SoC runs
# hot. Usage: run_all.sh ROOT [budget_s] [max_reps] [build opts...]
root=$1
shift
hot() { cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -n | tail -1; }
echo "before: $(uptime) max temp $(hot)"
for d in "$root"/*/; do
    t=$(hot)
    if [ "$t" -gt 85000 ]; then
        echo "stopping: $t mC"
        break
    fi
    echo "== $(basename "$d")"
    ./tang_cl "$d" "$@" 2>&1 | grep -E "reps|failed|error" | head -5
    sleep 1
done
echo "after: $(uptime) max temp $(hot)"
