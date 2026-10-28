"""The retarget fixture for tang-chip: the put/get/list trace's tokens as IR features.

    python -I scripts/chip/export_fixture.py IR.json CKPT_DIR TRACES_DIR OUT.json

For every token the trace feeds, its embedding row in the qwenputer Filesystem checkpoint
(chip dims only: Qwen's own dims are ignored by the norms next to the ballast) is mapped back to
IR token dims through the python allocation, so tang-chip can lay the row out for any shape. Also
the output decoder table (prep/decode.json), the store heads and the dumped store states.
"""
import json, sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

IR, CK, TR, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
ir = json.loads(IR.read_text())
meta = json.loads((CK / "chipformer.json").read_text())
off, m, B = meta["chip_dims_offset"], meta["ballast_dims"], 2.0**40
slot_dim = {}
for d in ir["token_dims"]:
    s = ir["python_alloc"]["slots"][str(d)]
    assert s not in slot_dim
    slot_dim[s] = d
tr = safe_open(TR / "trace_put_get_list.safetensors", "np")
ops = json.loads((TR / "trace_put_get_list.json").read_text())
lay = json.loads((TR / "layout.json").read_text())
toks = set()
fx_ops, states = [], {}
for op in ops:
    t = tr.get_tensor(f"{op['op']}/tokens").tolist()
    toks |= set(t)
    fx_ops.append({"op": op["op"], "forced": op["forced_tokens"], "tokens": t})
    st = {}
    for name, h in lay["store_heads"].items():
        st[name] = [{"rows": tr.get_tensor(f"{op['op']}/{name}/v{x}/rows").tolist(),
                     "vals": tr.get_tensor(f"{op['op']}/{name}/v{x}/vals").tolist()}
                    for x in range(len(h["value_heads"]))]
    states[op["op"]] = st
emb = safe_open(CK / "chip.safetensors", "pt")
E = emb.get_slice("model.embed_tokens.weight")
rows = {}
for t in sorted(toks):
    r = E[t:t + 1].float()[0].numpy().astype(np.float64)
    assert (r[off:off + m] == B).all(), t
    assert r[off + m] == 0, t
    feats = []
    for i in np.nonzero(r[off + m + 1:])[0]:
        feats.append([slot_dim[int(i)], float(r[off + m + 1 + i])])
    rows[str(t)] = feats
dec = json.loads((CK / "prep" / "decode.json").read_text())
fx = {"source": "qwenputer Filesystem checkpoint + reproduction trace put/get/list (MIT, (c) 2026 Divyansh Agrawal)",
      "v0": meta["chip_token_offset"], "decoder": {k: dec[k] for k in ("nfam", "families", "printable")},
      "rows": rows, "ops": fx_ops, "store_heads": lay["store_heads"], "states": states}
OUT.write_text(json.dumps(fx, separators=(",", ":")))
print("tokens", len(rows), "ops", len(fx_ops))
