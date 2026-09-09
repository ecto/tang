#!/usr/bin/env bash
# Publish the workspace to crates.io in dependency order.
#
# The order is the topological order of the publishable crates
# (cargo metadata over the workspace); each crate's dependencies are already
# on the index by the time it goes up. The sleep is index propagation — a
# crate is not resolvable by the next `cargo publish` the instant it lands.
#
# Stops on the first failure. Re-run with the already-published crates removed
# from the list if you need to resume.
set -euo pipefail

CRATES=(
    tang
    tang-expr
    tang-la
    tang-ad
    tang-gpu
    tang-tensor
    tang-3dgs
    tang-optim
    tang-train
)

for crate in "${CRATES[@]}"; do
    echo "==> cargo publish -p $crate"
    cargo publish -p "$crate"
    if [ "$crate" != "${CRATES[${#CRATES[@]} - 1]}" ]; then
        echo "    waiting 30s for the index"
        sleep 30
    fi
done

echo "all crates published"
