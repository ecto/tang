# PyTorch reference timings for the kernels_vs_ir shapes. Kernel-only via
# CUDA events, fp32 (matmul precision "highest" = true fp32, no TF32).
import torch, torch.nn.functional as F, time
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
dev = "cuda"

def bench(fn, iters=50, warm=10):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters

print("torch", torch.__version__, torch.cuda.get_device_name(0), "fp32 (tf32 off)")
# matmul 1024^2
a = torch.randn(1024, 1024, device=dev); b = torch.randn(1024, 1024, device=dev)
print(f"matmul 1024^2 fp32: {bench(lambda: a @ b):.3f} ms")
torch.backends.cuda.matmul.allow_tf32 = True
print(f"matmul 1024^2 tf32: {bench(lambda: a @ b):.3f} ms")
torch.backends.cuda.matmul.allow_tf32 = False
for (m,k,n) in [(128,576,1536),(512,576,1536)]:
    x = torch.randn(m,k,device=dev); w = torch.randn(k,n,device=dev)
    print(f"matmul {m}x{k}x{n} fp32: {bench(lambda: x @ w):.3f} ms")
# causal attention, single head, d=64
for n in (512, 1024):
    q = torch.randn(1,1,n,64,device=dev); k = torch.randn(1,1,n,64,device=dev); v = torch.randn(1,1,n,64,device=dev)
    print(f"attention n={n} d=64 sdpa(flash/mem-eff) fp32: {bench(lambda: F.scaled_dot_product_attention(q,k,v,is_causal=True)):.3f} ms")
    def naive():
        s = (q @ k.transpose(-1,-2)) / 8.0
        mask = torch.triu(torch.ones(n,n,device=dev,dtype=torch.bool),1)
        s = s.masked_fill(mask, float("-inf"))
        return torch.softmax(s,-1) @ v
    print(f"attention n={n} naive (matmul+softmax+matmul) fp32: {bench(naive):.3f} ms")
# transformer block: rmsnorm -> qkv(+bias) -> rope -> GQA causal attn -> wo -> res -> rmsnorm -> swiglu -> res
D, H, KV, HD, FF = 576, 9, 3, 64, 1536
class Block(torch.nn.Module):
    def __init__(s):
        super().__init__()
        s.wq = torch.nn.Linear(D, D); s.wk = torch.nn.Linear(D, KV*HD); s.wv = torch.nn.Linear(D, KV*HD); s.wo = torch.nn.Linear(D, D)
        s.gate = torch.nn.Linear(D, FF); s.up = torch.nn.Linear(D, FF); s.down = torch.nn.Linear(FF, D)
    def rms(s, x): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    def rope(s, x, cos, sin):
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1*cos - x2*sin, x1*sin + x2*cos], -1).flatten(-2)
    def forward(s, x, cos, sin):
        T = x.shape[0]
        h = s.rms(x)
        q = s.rope(s.wq(h).view(T,H,HD), cos, sin); k = s.rope(s.wk(h).view(T,KV,HD), cos, sin); v = s.wv(h).view(T,KV,HD)
        q = q.transpose(0,1).unsqueeze(0); k = k.transpose(0,1).unsqueeze(0); v = v.transpose(0,1).unsqueeze(0)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        a = a.squeeze(0).transpose(0,1).reshape(T, D)
        x = x + s.wo(a)
        h = s.rms(x)
        return x + s.down(F.silu(s.gate(h)) * s.up(h))
blk = Block().to(dev)
for T in (128, 512):
    x = torch.randn(T, D, device=dev)
    pos = torch.arange(T, device=dev).float().unsqueeze(1); i = torch.arange(HD//2, device=dev).float()
    theta = pos / (10000 ** (2*i/HD)); cos, sin = theta.cos().unsqueeze(1), theta.sin().unsqueeze(1)
    with torch.no_grad():
        print(f"block fwd seq={T} fp32: {bench(lambda: blk(x,cos,sin)):.3f} ms")
        # tang's CUDA path runs its matmuls and attention on TF32 tensor
        # cores; this is the like-for-like PyTorch setting
        torch.backends.cuda.matmul.allow_tf32 = True
        print(f"block fwd seq={T} tf32: {bench(lambda: blk(x,cos,sin)):.3f} ms")
        torch.backends.cuda.matmul.allow_tf32 = False
    xg = x.clone().requires_grad_(True)
    def step():
        blk.zero_grad(set_to_none=True)
        out = blk(xg,cos,sin); out.sum().backward()
    print(f"block fwd+bwd seq={T} fp32: {bench(step):.3f} ms")
# adam 4M
p = torch.randn(4_194_304, device=dev); g = torch.randn_like(p)*0.1
opt = torch.optim.Adam([torch.nn.Parameter(p)], lr=1e-3, foreach=True)
opt.param_groups[0]["params"][0].grad = g
print(f"adam step 4M (foreach) fp32: {bench(lambda: opt.step()):.3f} ms")
