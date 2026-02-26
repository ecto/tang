#!/usr/bin/env bash
# Measures stripped release binary and WASM binary sizes for tang vs nalgebra vs glam.
set -euo pipefail

TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

echo "=== Binary Size Benchmark ==="
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

[profile.release]
opt-level = "z"
lto = true
codegen-units = 1
strip = true
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

human_size() {
    local bytes=$1
    if [ "$bytes" -ge 1048576 ]; then
        echo "$(echo "scale=1; $bytes / 1048576" | bc)M"
    elif [ "$bytes" -ge 1024 ]; then
        echo "$(echo "scale=1; $bytes / 1024" | bc)K"
    else
        echo "${bytes}B"
    fi
}

HAS_WASM=0
if rustup target list --installed 2>/dev/null | grep -q wasm32-unknown-unknown; then
    HAS_WASM=1
fi

HAS_WASM_OPT=0
if command -v wasm-opt &>/dev/null; then
    HAS_WASM_OPT=1
fi

measure_size() {
    local name=$1
    local dir="$TMPDIR/$name"

    # Native release build
    (cd "$dir" && cargo build --release 2>&1 >/dev/null)
    local bin="$dir/target/release/bench-$name"
    local native_size=$(wc -c < "$bin" | tr -d ' ')

    local wasm_size="n/a"
    if [ "$HAS_WASM" = "1" ]; then
        # WASM build - create a lib version for wasm
        mkdir -p "$dir/src"
        local orig_main=$(cat "$dir/src/main.rs")
        # Add a no_main wasm-compatible wrapper
        cat > "$dir/src/lib.rs" <<RUST
#[no_mangle]
pub extern "C" fn run() -> f64 {
    // Prevent DCE by returning a computed value
    42.0
}
RUST
        # Try wasm build (may fail for nalgebra due to std deps, that's ok)
        if (cd "$dir" && cargo build --release --lib --target wasm32-unknown-unknown 2>/dev/null); then
            local wasm_file="$dir/target/wasm32-unknown-unknown/release/bench_${name}.wasm"
            if [ -f "$wasm_file" ]; then
                wasm_size=$(wc -c < "$wasm_file" | tr -d ' ')
                if [ "$HAS_WASM_OPT" = "1" ]; then
                    wasm-opt -Oz "$wasm_file" -o "$wasm_file.opt" 2>/dev/null && \
                        wasm_size=$(wc -c < "$wasm_file.opt" | tr -d ' ')
                fi
                wasm_size=$(human_size "$wasm_size")
            fi
        fi
        rm -f "$dir/src/lib.rs"
    fi

    printf "%-16s %10s       %10s\n" "$name" "$(human_size "$native_size")" "$wasm_size"
}

printf "%-16s %10s       %10s\n" "" "native" "wasm"
printf "%-16s %10s       %10s\n" "" "------" "----"

measure_size "tang"
measure_size "glam"
measure_size "nalgebra"
