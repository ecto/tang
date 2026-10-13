# PyTorch reference for examples/whole_model.rs: N SmolLM2-135M-like blocks
# (no biases), MSE loss, backward, AdamW on every projection weight.
# Usage: python3 torch_model_bench.py [layers] [seq]
import sys, torch, torch.nn.functional as F
dev = "cuda"
L = int(sys.argv[1]) if len(sys.argv) > 1 else 4
T = int(sys.argv[2]) if len(sys.argv) > 2 else 128
D, H, KV, HD, FF = 576, 9, 3, 64, 1536

def bench(fn, iters=10, warm=3, batches=5):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(batches):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters): fn()
        e.record(); torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) / iters)
    return best

class Block(torch.nn.Module):
    def __init__(s):
        super().__init__()
        lin = lambda i, o: torch.nn.Linear(i, o, bias=False)
        s.wq, s.wk, s.wv, s.wo = lin(D, D), lin(D, KV*HD), lin(D, KV*HD), lin(D, D)
        s.gate, s.up, s.down = lin(D, FF), lin(D, FF), lin(FF, D)
    def rms(s, x): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    def rope(s, x, cos, sin):
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1*cos - x2*sin, x1*sin + x2*cos], -1).flatten(-2)
    def forward(s, x, cos, sin):
        h = s.rms(x)
        q = s.rope(s.wq(h).view(T,H,HD), cos, sin); k = s.rope(s.wk(h).view(T,KV,HD), cos, sin); v = s.wv(h).view(T,KV,HD)
        q, k, v = (t.transpose(0,1).unsqueeze(0) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        x = x + s.wo(a.squeeze(0).transpose(0,1).reshape(T, D))
        h = s.rms(x)
        return x + s.down(F.silu(s.gate(h)) * s.up(h))

model = torch.nn.Sequential(*[Block() for _ in range(L)]).to(dev)
x = torch.randn(T, D, device=dev); target = torch.randn(T, D, device=dev)
pos = torch.arange(T, device=dev).float().unsqueeze(1); i = torch.arange(HD//2, device=dev).float()
theta = pos / (10000 ** (2*i/HD)); cos, sin = theta.cos().unsqueeze(1), theta.sin().unsqueeze(1)

def run(opt, amp=False):
    def step():
        opt.zero_grad(set_to_none=True)
        # bf16 autocast: fp32 master weights and AdamW, bf16 matmuls and
        # attention (bf16 needs no loss scaling)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            h = x
            for b in model: h = b(h, cos, sin)
        F.mse_loss(h.float(), target).backward()
        opt.step()
    return step

print("torch", torch.__version__, torch.cuda.get_device_name(0), f"layers {L} seq {T}")
for tf32 in (False, True):
    torch.backends.cuda.matmul.allow_tf32 = tf32
    for kind in ("foreach", "fused"):
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01, **{kind: True})
        torch.cuda.reset_peak_memory_stats()
        ms = bench(run(opt))
        peak = torch.cuda.max_memory_allocated() / 2**20
        print(f"step {'tf32' if tf32 else 'fp32'} adamw-{kind}: {ms:.2f} ms, peak allocated {peak:.0f} MB")
torch.backends.cuda.matmul.allow_tf32 = False
for kind in ("foreach", "fused"):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01, **{kind: True})
    torch.cuda.reset_peak_memory_stats()
    ms = bench(run(opt, amp=True))
    peak = torch.cuda.max_memory_allocated() / 2**20
    print(f"step bf16-autocast adamw-{kind}: {ms:.2f} ms, peak allocated {peak:.0f} MB")
