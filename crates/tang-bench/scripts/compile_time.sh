#!/usr/bin/env bash
# Measures clean and incremental compile times for tang vs nalgebra vs glam.
set -euo pipefail

RUNS=3
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

echo "=== Compile Time Benchmark (Release, median of $RUNS) ==="
echo ""

setup_crate() {
    local name=$1
    local dep=$2
    local code=$3
    local dir="$TMPDIR/$name"
    mkdir -p "$dir/src"
    cat > "$dir/Cargo.toml" <<TOML
[package]
name = "bench-$name"
version = "0.1.0"
edition = "2021"

[dependencies]
$dep
TOML
    cat > "$dir/src/main.rs" <<RUST
$code
RUST
}

setup_crate "tang" \
    'tang = { path = "'$(cd "$(dirname "$0")/../../tang" && pwd)'" }' \
    'use tang::{Vec3, Mat4};
fn main() {
    let v = Vec3::new(1.0f64, 2.0, 3.0);
    let m = Mat4::rotation_z(0.5) * Mat4::translation(1.0, 2.0, 3.0);
    println!("{:?} {:?}", v.normalize(), m.determinant());
}'

setup_crate "nalgebra" \
    'nalgebra = "0.33"' \
    'use nalgebra::{Vector3, Matrix4};
fn main() {
    let v = Vector3::new(1.0f64, 2.0, 3.0);
    let m = Matrix4::new_translation(&Vector3::new(1.0, 2.0, 3.0));
    println!("{:?} {:?}", v.normalize(), m.determinant());
}'

setup_crate "glam" \
    'glam = "0.29"' \
    'use glam::{DVec3, DMat4};
fn main() {
    let v = DVec3::new(1.0, 2.0, 3.0);
    let m = DMat4::from_translation(DVec3::new(1.0, 2.0, 3.0));
    println!("{:?} {:?}", v.normalize(), m.determinant());
}'

median() {
    local -a arr=("$@")
    IFS=$'\n' sorted=($(sort -n <<<"${arr[*]}")); unset IFS
    echo "${sorted[$((${#sorted[@]} / 2))]}"
}

measure_build() {
    local name=$1
    local dir="$TMPDIR/$name"

    # Clean build times
    local -a clean_times=()
    for i in $(seq 1 $RUNS); do
        (cd "$dir" && cargo clean 2>/dev/null)
        local start=$(date +%s%3N 2>/dev/null || python3 -c 'import time; print(int(time.time()*1000))')
        (cd "$dir" && cargo build --release 2>&1 >/dev/null)
        local end=$(date +%s%3N 2>/dev/null || python3 -c 'import time; print(int(time.time()*1000))')
        clean_times+=($((end - start)))
    done

    # Incremental build times
    local -a incr_times=()
    for i in $(seq 1 $RUNS); do
        touch "$dir/src/main.rs"
        local start=$(date +%s%3N 2>/dev/null || python3 -c 'import time; print(int(time.time()*1000))')
        (cd "$dir" && cargo build --release 2>&1 >/dev/null)
        local end=$(date +%s%3N 2>/dev/null || python3 -c 'import time; print(int(time.time()*1000))')
        incr_times+=($((end - start)))
    done

    local clean_med=$(median "${clean_times[@]}")
    local incr_med=$(median "${incr_times[@]}")
    printf "%-16s %6s ms       %6s ms\n" "$name" "$clean_med" "$incr_med"
}

printf "%-16s %9s       %9s\n" "" "clean" "incremental"
printf "%-16s %9s       %9s\n" "" "-----" "-----------"

measure_build "tang"
measure_build "glam"
measure_build "nalgebra"
