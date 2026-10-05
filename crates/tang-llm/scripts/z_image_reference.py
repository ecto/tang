"""Create deterministic fp32 diffusers tensors for tang's Z-Image parity runner.
This requires torch/diffusers/safetensors, and downloads no model weights.
"""
import json
import sys
from pathlib import Path
import torch
from diffusers import ZImageTransformer2DModel

root = Path(sys.argv[1]); root.mkdir(parents=True, exist_ok=True)
torch.manual_seed(20261005)
model = ZImageTransformer2DModel(dim=24, n_heads=2, n_kv_heads=2, n_layers=2,
    n_refiner_layers=2, in_channels=4, cap_feat_dim=16, axes_dims=[4,4,4],
    axes_lens=[1024,64,64]).float().eval()
model.save_pretrained(root)
for height,width,tokens in [(8,8,5),(32,32,35)]:
    traces = {}; hooks = []
    def capture(name):
        def hook(module,args,out):traces[name] = out.detach().float().reshape(-1).tolist()
        return hook
    def capture_input(name):
        def hook(module,args):traces[name] = args[0].detach().float().reshape(-1).tolist()
        return hook
    hooks.append(model.noise_refiner[0].register_forward_pre_hook(capture_input('image_embed')))
    hooks.append(model.context_refiner[0].register_forward_pre_hook(capture_input('caption_embed')))
    for group in ['noise_refiner','context_refiner','layers']:
        for i,layer in enumerate(getattr(model,group)): hooks.append(layer.register_forward_hook(capture(f'{group}.{i}')))
    hooks.append(model.all_final_layer['2-1'].register_forward_hook(capture('final')))
    latent = torch.randn(4,1,height,width);caption = torch.randn(tokens,16);t=0.625
    with torch.no_grad():out = model([latent],torch.tensor([t]),[caption],return_dict=False)[0][0]
    for hook in hooks:hook.remove()
    case={'height':height,'width':width,'caption_tokens':tokens,'t':t,'latent':latent.flatten().tolist(),
        'caption':caption.flatten().tolist(),'traces':traces,'output':out.flatten().tolist()}
    (root/f'case-{height}.json').write_text(json.dumps(case))
    print(f'reference {height}x{width}: {len(traces)} tensors',flush=True)
