#!/bin/zsh
# Assemble the cram.sh static site (Cloudflare Pages) into a dist directory:
#   scripts/site.sh <fields-dir> <model.tcm> <dist-dir>
# <fields-dir> holds the five 64x128x128 demo crops (miranda-density, ...).
# Layout: /           landing page (site/index.html)
#         /demo/      WebGPU decode demo + wasm pkg + corpus
#         /docs       302 → docs.rs/cram-codec (_redirects)
set -e
F=${1:?fields dir}; M=${2:?model.tcm}; OUT=${3:?dist dir}
C=$(cd "$(dirname $0)/.." && pwd)
cd "$C"
wasm-pack build --target web --release --out-dir demo/pkg --no-typescript
for n in miranda-density miranda-velocityx hurricane-TC hurricane-U nyx-temperature; do
  echo "$F/$n-64x128x128.f32 64x128x128"
done > demo/fields.txt
cargo build --release -p cram-codec-wasm --example corpus
mkdir -p demo/corpus
cp "$M" demo/corpus/model.tcm
"$(cargo metadata --format-version 1 --no-deps | python3 -c 'import json,sys;print(json.load(sys.stdin)["target_directory"])')/release/examples/corpus" \
  demo/fields.txt demo/corpus demo/corpus/model.tcm --threads ${THREADS:-6}
rm -rf "$OUT" && mkdir -p "$OUT/demo"
cp site/index.html "$OUT/"
mkdir -p "$OUT/examples" && cp site/examples/*.webp site/examples/*.json "$OUT/examples/"
cp demo/index.html "$OUT/demo/"
cp -R demo/pkg demo/corpus "$OUT/demo/"
rm -f "$OUT/demo/pkg/.gitignore" "$OUT/demo/pkg/package.json"
cat > "$OUT/_redirects" <<'EOF'
/docs https://docs.rs/cram-codec 302
/docs/ https://docs.rs/cram-codec 302
EOF
# WebGPU needs no COOP/COEP (no SharedArrayBuffer or wasm threads in the demo).
cat > "$OUT/_headers" <<'EOF'
/demo/pkg/*
  Cache-Control: public, max-age=3600
/demo/corpus/*
  Cache-Control: public, max-age=86400
EOF
find "$OUT" -type f -size +24M -print | grep . && { echo "file over the 25 MiB Pages limit" >&2; exit 1; }
du -sh "$OUT"
