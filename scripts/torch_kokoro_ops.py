# PyTorch MPS reference timings for crates/tang-bench/examples/kokoro_ops.rs.
# Same shapes, same method: 5 warmups, 50 launches, one torch.mps.synchronize.
# fp32 by default; `f16` as the first argument runs everything in half
# (weights and activations; MPS accumulates matmul/conv in f32).
#   ~/.cache/kv/bin/python scripts/torch_kokoro_ops.py [f16]
import sys, time, math
import torch, torch.nn.functional as F

dev = "mps"
ITERS = 50
DT = torch.float16 if "f16" in sys.argv[1:] else torch.float32
torch.set_default_dtype(DT)

def bench(fn, iters=ITERS, warm=5):
    for _ in range(warm): fn()
    torch.mps.synchronize()
    t = time.perf_counter()
    for _ in range(iters): fn()
    torch.mps.synchronize()
    return (time.perf_counter() - t) * 1e3 / iters

def r(*shape, scale=1.0):
    return ((torch.rand(*shape, device=dev) * 2 - 1) * scale).to(DT)

torch.manual_seed(0)
print("torch", torch.__version__, "MPS", DT, "kernel-only,", ITERS, "launches per sample\n")
rows = []
def row(name, ms, note=""):
    rows.append((name, ms, note)); print(f"{name}: {ms:.3f} ms {note}")

C, T, K, DIL = 512, 2000, 7, 3
# (a) conv1d 512->512 k7 d3 same padding
x = r(1, C, T); w = r(C, C, K, scale=0.05)
row("a conv1d 512→512 k7 d3 T2000", bench(lambda: F.conv1d(x, w, padding=DIL * (K - 1) // 2, dilation=DIL)))

# (b) upsample: conv_transpose1d k20 s10 p5 vs the polyphase form as conv1d
CI, CO, S = 512, 256, 10
xu = r(1, CI, T); wt = r(CI, CO, 2 * S, scale=0.05)
row("b conv_transpose1d 512→256 k20 s10 p5", bench(lambda: F.conv_transpose1d(xu, wt, stride=S, padding=5)))
# polyphase: conv1d with 256*10 output channels, 2 taps, then interleave
wpoly = r(CO * S, CI, 2, scale=0.05)
def poly():
    z = F.conv1d(F.pad(xu, (1, 0)), wpoly)            # [1, CO*S, T]
    return z.view(1, CO, S, T).transpose(2, 3).reshape(1, CO, T * S)
row("b polyphase as conv1d(k2, 2560 ch) + interleave", bench(poly))
row("b polyphase conv1d only (no interleave)", bench(lambda: F.conv1d(F.pad(xu, (1, 0)), wpoly)))

# (c) iSTFT head: [22, frames] -> exp/sin -> istft(n_fft 20, hop 5, hann, center)
FR = 4000
spec = r(1, 22, FR).float(); win = torch.hann_window(20, device=dev, dtype=torch.float32)  # MPS FFT: f32 only
def istft():
    mag = torch.exp(spec[:, :11]); ph = torch.sin(spec[:, 11:])
    z = torch.polar(mag, ph)
    return torch.istft(z, n_fft=20, hop_length=5, win_length=20, window=win, center=True)
try:
    row("c iSTFT n_fft20 hop5 T4000", bench(istft))
except Exception as e:
    print("c istft on mps failed:", e)
    rows.append(("c iSTFT n_fft20 hop5 T4000", float("nan"), "failed"))
# the same algebra written out (irfft + fold), in case istft falls back
def istft_manual():
    mag = torch.exp(spec[0, :11]); ph = torch.sin(spec[0, 11:])
    fr = torch.fft.irfft(torch.polar(mag, ph), n=20, dim=0) * win[:, None]  # [20, FR]
    y = F.fold(fr[None], output_size=(1, 5 * (FR - 1) + 20), kernel_size=(1, 20), stride=(1, 5))
    env = F.fold((win ** 2)[None, :, None].expand(1, 20, FR), output_size=(1, 5 * (FR - 1) + 20), kernel_size=(1, 20), stride=(1, 5))
    return (y / env).flatten()[10:-10]
try:
    row("c iSTFT manual (irfft + fold)", bench(istft_manual))
except Exception as e:
    print("c manual failed:", e)

# (d) conv + bias + snake
b = r(C, scale=0.1); alpha = r(1, C, 1, scale=0.5) + 1
def conv_snake():
    z = F.conv1d(x, w, b, padding=9, dilation=DIL)
    return z + torch.sin(alpha * z) ** 2 / alpha
row("d conv + bias + snake", bench(conv_snake))

# (e) AdaIN
SD = 128
xa = r(1, C, T); st = r(1, SD); fc = torch.nn.Linear(SD, 2 * C).to(dev)
def adain():
    h = fc(st)[..., None]
    g, bb = h.chunk(2, dim=1)
    return (1 + g) * F.instance_norm(xa, eps=1e-5) + bb
row("e AdaIN C512 T2000 style128", bench(adain))

# (f) ConvNeXt block, C=512, T=200, F=1536
CF, TF, FF = 512, 200, 1536
class ConvNeXt(torch.nn.Module):
    def __init__(s):
        super().__init__()
        s.dw = torch.nn.Conv1d(CF, CF, 7, padding=3, groups=CF)
        s.norm = torch.nn.LayerNorm(CF, eps=1e-6)
        s.pw1 = torch.nn.Linear(CF, FF); s.pw2 = torch.nn.Linear(FF, CF)
    def forward(s, x):
        h = s.dw(x).transpose(1, 2)
        h = s.pw2(F.gelu(s.pw1(s.norm(h)), approximate="tanh"))
        return x + h.transpose(1, 2)
blk = ConvNeXt().to(dev)
xf = r(1, CF, TF)
with torch.no_grad():
    row("f ConvNeXt block C512 T200 F1536", bench(lambda: blk(xf)))

# forward + backward
xg = x.clone().requires_grad_(True); wg = w.clone().requires_grad_(True)
def conv_train():
    y = F.conv1d(xg, wg, padding=9, dilation=DIL)
    y.backward(torch.ones_like(y))
row("a fwd+bwd (dx, dw)", bench(conv_train))
xfg = xf.clone().requires_grad_(True)
def cn_train():
    y = blk(xfg)
    y.backward(torch.ones_like(y))
row("f fwd+bwd (all params)", bench(cn_train))

# kernel_only shapes: matmul (TFLOP/s) and causal attention, d = 64
for (m, k, n) in [(1024, 1024, 1024), (128, 576, 1536), (512, 576, 1536)]:
    a = r(m, k); b = r(k, n)
    ms = bench(lambda: a @ b)
    row(f"matmul {m}x{k}x{n}", ms, f"{2 * m * k * n / ms / 1e9:.2f} TFLOP/s")
for n in (512, 1024):
    q = r(1, 1, n, 64); kk = r(1, 1, n, 64); v = r(1, 1, n, 64)
    row(f"attention n={n} sdpa causal", bench(lambda: F.scaled_dot_product_attention(q, kk, v, is_causal=True)))

print("\n| op | MPS (ms) |\n|---|---:|")
for n, ms, note in rows:
    print(f"| {n} | {ms:.3f} | {note}")
