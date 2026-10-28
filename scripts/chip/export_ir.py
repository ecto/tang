"""Export qwenputer's store circuit IR (core.build, CPU, no weights) for tang-chip.

    python -I scripts/chip/export_ir.py QWENPUTER_EXPERIMENTAL_RISCV CKPT_DIR OUT.json

qwenputer is MIT, Copyright (c) 2026 Divyansh Agrawal. Writes the scheduled q35 Graph (dims,
FFN and Head nodes, unit counts, big dims, keep) plus the python allocation, for parity checks of
tang-chip's port of `q35/emit.py`.
"""
import json, sys
from fractions import Fraction
from pathlib import Path

REPO, CK, OUT = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, REPO)
from chipformer.q35.emit import allocate
from chipformer.q35.ir import FFN, Head
from chipformer.rv import bridge as BR, core
from chipformer.rv.store import Config

meta = json.loads((CK / "chipformer.json").read_text())
T = BR.Text(CK / "tokenizer.json")
c = core.build(core.T4B, T, Config(**meta["store"]))


def num(v):
    f = float(v)
    assert Fraction(f) == Fraction(v), v  # every coefficient is a dyadic rational
    return f


def ex(e):
    return [[d, num(v)] for d, v in e.terms.items()]


def chans(ch):
    return [[i, ex(c.expr), [num(t) for t in c.taps]] for i, c in ch.items()]


def vhead(vh):
    return {"v": chans(vh.v), "a": ex(vh.a), "z": [[i, ex(e)] for i, e in vh.z.items()],
            "out": [[col, d, num(w)] for col, d, w in vh.out],
            "beta": None if vh.beta is None else ex(vh.beta), "into": vh.into}


nodes = []
for n in c.nodes:
    if isinstance(n, FFN):
        nodes.append({"ffn": n.name, "level": n.level, "out": n.out, "into": n.into,
                      "terms": [[num(k), ex(a), ex(b)] for k, a, b in n.terms]})
    else:
        assert isinstance(n, Head) and n.n_cols == 128
        nodes.append({"head": n.name, "level": n.level, "tiny": n.tiny,
                      "q": chans(n.q), "k": chans(n.k), "vheads": [vhead(v) for v in n.vheads()]})
keep = list(c.keep)
alloc = allocate(c, None, set(keep), c.target.intermediate)
ir = {
    "source": "qwenputer rv-store experimental/riscv core.build(T4B, text, store) (MIT, (c) 2026 Divyansh Agrawal)",
    "target": {"key_heads": c.target.key_heads, "value_heads": c.target.value_heads,
               "intermediate": c.target.intermediate},
    "dims": [[d.name, d.kind, d.level] for d in c.dims],
    "token_dims": sorted(c.token_fns), "units": {str(k): v for k, v in c.units.items()},
    "big": sorted(getattr(c, "big", set())), "keep": keep, "nodes": nodes,
    "code": [next(iter(e.terms)) for e in c.code], "tau": next(iter(c.tau.terms)),
    "mu": next(iter(c.mu.terms)), "id_code": [next(iter(e.terms)) for e in c.store.yc],
    "python_alloc": {"slots": {str(k): v for k, v in alloc.slots.items()}, "n_slots": alloc.n_slots,
                     "subtract": {str(k): v for k, v in alloc.subtract.items()},
                     "clears": {str(k): v for k, v in alloc.clears.items()}, "final": alloc.final},
}
OUT.write_text(json.dumps(ir, separators=(",", ":")))
print("dims", len(c.dims), "nodes", len(nodes), "slots", alloc.n_slots, "layers", c.n_layers())
