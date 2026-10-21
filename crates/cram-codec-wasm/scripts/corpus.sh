#!/bin/zsh
# Build the browser gate corpus from a cram-codec bench checkout:
#   scripts/corpus.sh /path/to/cram-codec/bench
# Fields: the 5 original 64x128x128 crops (bench/out) + the 14 held-out fields
# (bench/data/heldout); model: bench/models/pre3_32x16.tcm.
set -e
B=${1:?bench dir}
D=$(cd "$(dirname $0)/../demo" && pwd)
python3 - "$B" > "$D/fields.txt" <<'PY'
import json, sys
b = sys.argv[1]
for n in ["miranda-density", "miranda-velocityx", "hurricane-TC", "hurricane-U", "nyx-temperature"]:
    print(f"{b}/out/{n}-64x128x128.f32 64x128x128")
for m in json.load(open(f"{b}/data/heldout/manifest.json")):
    print(f"{b}/data/heldout/{m['name']}.f32 " + "x".join(map(str, m["shape"])))
PY
cargo build --release -p cram-codec-wasm --example corpus
mkdir -p "$D/corpus"
cp "$B/models/pre3_32x16.tcm" "$D/corpus/model.tcm"
"$(cargo metadata --format-version 1 --no-deps | python3 -c 'import json,sys;print(json.load(sys.stdin)["target_directory"])')/release/examples/corpus" \
  "$D/fields.txt" "$D/corpus" "$D/corpus/model.tcm" --threads ${THREADS:-6}
