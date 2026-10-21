"""Export a Cool-chic 5 (.cool) bitstream into a CCX model for cram-codec `img`.

Runs the reference decoder (Cool-chic repo on PYTHONPATH) once, capturing the
integer latents and every decoder network as it is used, plus the reference
float decode for comparison. Usage:

    python export_cc.py IN.cool OUT.json [ORIG.png]
"""

import struct
import sys

import numpy as np
import torch

import coolchic.bitstream.component.coolchic as bcc
from coolchic.bitstream.component.constants import WEIGHT_SHIFT
from coolchic.bitstream.decode import decode_video
from coolchic.component.core.arm import MAX_ARM_MASK_SIZE, _get_non_zero_pixel_ctx_index

cap = {"latents": [], "modules": None, "header": None}

_orig_decode_network = bcc.decode_network
_orig_ecla = bcc.entropy_coding_latent_arm
_orig_ecc = bcc.encode_decode_coolchic


def decode_network(*a, **k):
    out = _orig_decode_network(*a, **k)
    cap["modules"] = out[0]
    return out


def ecla(*a, **k):
    y = _orig_ecla(*a, **k)
    cap["latents"].insert(0, y.to(torch.int64))
    return y


def ecc(header, bytes_nn, mode, dec_bytes_latent=None, **k):
    cap["header"] = header
    cap["n_bytes_nn"] = len(bytes_nn)
    cap["n_bytes_latent"] = len(dec_bytes_latent)
    return _orig_ecc(header, bytes_nn, mode, dec_bytes_latent=dec_bytes_latent, **k)


bcc.decode_network = decode_network
bcc.entropy_coding_latent_arm = ecla
bcc.encode_decode_coolchic = ecc
import coolchic.bitstream.decode as bdec  # noqa: E402

bdec.encode_decode_coolchic = ecc


def t(x):
    return x.detach().cpu().double().flatten().tolist()


def arm_float(arm):
    """ARM / IFCE MLP as float (dequantized) layers: [(w[out][in], b[out])], stab."""
    lays = []
    for lay in arm.mlp.children():
        if hasattr(lay, "weight"):
            lays.append({"w": t(lay.weight), "b": t(lay.bias), "out": lay.weight.shape[0], "in": lay.weight.shape[1]})
    stab = None
    if getattr(arm, "flag_linear_stabiliser", False):
        s = arm.stabiliser_branch
        stab = {"w": t(s.weight), "b": t(s.bias), "out": s.weight.shape[0], "in": s.weight.shape[1]}
    return {"layers": lays, "stab": stab}


def main():
    inp, outp = sys.argv[1], sys.argv[2]
    total_bytes = len(open(inp, "rb").read())
    frames = decode_video(inp, verbosity=0)
    rec = frames["0"].data
    h = cap["header"]
    p = h.get_coolchic_parameter()
    mods = cap["modules"]
    arm = mods["arm"]
    # NB: arm.mlp layers are quantized float params (integer before scaling by q-step)
    arm_q = h.get_value("nn_q_step").arm
    # ArmLinear stores integer params scaled by q_step at decode; dequantize here.
    def deq(lay_param, q):
        return t(lay_param.double() * q)

    def arm_deq(a, q):
        lays = []
        for lay in a.mlp.children():
            if hasattr(lay, "weight"):
                w = lay.weight
                lays.append(
                    {
                        "w": deq(w, q.weight) if not torch.is_floating_point(w) else t(w),
                        "b": deq(lay.bias, q.bias) if not torch.is_floating_point(lay.bias) else t(lay.bias),
                        "out": w.shape[0],
                        "in": w.shape[1],
                    }
                )
        stab = None
        if getattr(a, "flag_linear_stabiliser", False):
            s = a.stabiliser_branch
            stab = {
                "w": t(s.weight) if torch.is_floating_point(s.weight) else deq(s.weight, q.weight),
                "b": t(s.bias) if torch.is_floating_point(s.bias) else deq(s.bias, q.bias),
                "out": s.weight.shape[0],
                "in": s.weight.shape[1],
            }
        return {"layers": lays, "stab": stab}

    ups = mods["upsampling"]
    syn = mods["synthesis"]

    rec_list = []

    def put(name, arr, shape=None):
        a = np.asarray(arr, dtype=np.float64)
        if shape is not None:
            a = a.reshape(shape)
        rec_list.append((name, a))

    put("img_size", list(p.img_size))
    put("sizes", [list(s[-2:]) for s in p.size_per_latent])
    put("is_hyper", [float(x) for x in p.flag_is_hyperlatent])
    put("ctx_index", [int(i) for i in _get_non_zero_pixel_ctx_index(p.spatial_context_arm)])
    put("mask_size", [MAX_ARM_MASK_SIZE])
    put("n_ifce_out", [int(p.output_feature_ifce) if p.flag_ifce else 0])
    assert p.final_upsampling_type is None or list(p.size_per_latent[0][-2:]) == list(p.img_size), "final upsampling unsupported"

    def put_arm(prefix, a):
        for i, l in enumerate(a["layers"]):
            put(f"{prefix}.L{i}.w", l["w"], (l["out"], l["in"]))
            put(f"{prefix}.L{i}.b", l["b"])
        if a["stab"] is not None:
            put(f"{prefix}.stab.w", a["stab"]["w"], (a["stab"]["out"], a["stab"]["in"]))
            put(f"{prefix}.stab.b", a["stab"]["b"])

    put_arm("arm", arm_deq(arm, arm_q))
    if p.flag_ifce:
        im = mods["ifce"]
        ifce_q = h.get_value("nn_q_step").ifce
        n = len(p.size_per_latent)
        put("ifce.input_features", [int(x) for x in p.input_features_ifce])
        put("ifce.index_to_arm", [int(im.index_to_arm[k]) if k in im.index_to_arm else -1 for k in range(n)])
        for j, a in enumerate(im.arms):
            put_arm(f"ifce.A{j}", arm_deq(a, ifce_q))
    for i, c in enumerate(ups.conv_transpose2ds):
        put(f"ups.tconv.{i}", t(c.weight))
    for i, c in enumerate(ups.conv2ds):
        put(f"ups.conv.{i}", t(c.weight))

    def put_conv(prefix, c, relu=False):
        put(prefix + ".w", t(c.weight), tuple(c.weight.shape))
        put(prefix + ".b", t(c.bias))
        put(prefix + ".meta", [float(bool(getattr(c, "residual", False))), float(relu)])

    lst = list(syn.main_branch.children())
    for i in range(0, len(lst), 2):
        put_conv(f"syn.L{i // 2}", lst[i], isinstance(lst[i + 1], torch.nn.ReLU))
    if syn.flag_linear_stabiliser:
        put_conv("syn.stab", syn.stabiliser_branch)
    put_conv("syn.out", syn.output_transform)
    for i, x in enumerate(cap["latents"]):
        put(f"latent.{i}", x.flatten().tolist(), tuple(x.shape[-2:]))
    put("bytes", [total_bytes, cap["n_bytes_nn"], cap["n_bytes_latent"]])
    r = (rec.squeeze(0) * 255).round().clamp(0, 255)
    put("ref_rgb", r.permute(1, 2, 0).flatten().tolist(), tuple(r.permute(1, 2, 0).shape))
    if len(sys.argv) > 3:
        from PIL import Image

        o = np.asarray(Image.open(sys.argv[3]).convert("RGB"), dtype=np.float64)
        put("orig_rgb", o.flatten(), o.shape)
    with open(outp, "wb") as f:
        f.write(b"CCX1")
        f.write(struct.pack("<I", len(rec_list)))
        for name, a in rec_list:
            nb = name.encode()
            f.write(struct.pack("<H", len(nb)) + nb)
            f.write(struct.pack("<B", a.ndim) + struct.pack(f"<{a.ndim}I", *a.shape))
            f.write(a.astype("<f8").tobytes())
    print(f"exported {outp}: {total_bytes} B (nn {cap['n_bytes_nn']}, latent {cap['n_bytes_latent']})")


if __name__ == "__main__":
    main()
