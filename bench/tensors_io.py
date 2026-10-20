"""numpy-only safetensors reader shared by the ML-tensor bench scripts."""
import json
import struct
from pathlib import Path

import numpy as np

T = Path(__file__).resolve().parent / "tensors"
SMOL = T / "hf/SmolLM2-135M/model.safetensors"
RESNET = T / "hf/resnet18.a1_in1k/model.safetensors"
VIT = T / "hf/vit_tiny_patch16_224.augreg_in21k_ft_in1k/model.safetensors"

_DT = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16, "I64": np.int64}


def bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even f32 -> bf16 bit pattern (uint16)."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def load_safetensors(path: Path) -> dict[str, tuple[np.ndarray, str]]:
    """name -> (f32 array, native dtype string)."""
    raw = path.read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    hdr = json.loads(raw[8 : 8 + n])
    hdr.pop("__metadata__", None)
    out = {}
    for k, v in hdr.items():
        a, b = v["data_offsets"]
        x = np.frombuffer(raw[8 + n + a : 8 + n + b], dtype=_DT[v["dtype"]]).reshape(v["shape"])
        if v["dtype"] == "BF16":
            x = bf16_to_f32(x)
        out[k] = (np.asarray(x, dtype=np.float32) if v["dtype"] != "I64" else x, v["dtype"])
    return out
