"""Create deterministic fp32 diffusers tensors for tang's Z-Image parity runner.
This requires torch/diffusers/safetensors, and downloads no model weights.
"""
import json
import argparse
import re
import subprocess
import sys
from pathlib import Path
import torch
from diffusers import ZImageTransformer2DModel

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('root',type=Path)
parser.add_argument('--head128',action='store_true')
parser.add_argument('--checkpoint',type=Path,help='Existing local trained transformer directory; no downloads or weight copies')
parser.add_argument('--small-only',action='store_true',help='Export only the 8x8 latent case')
args=parser.parse_args()
root = args.root; root.mkdir(parents=True, exist_ok=True)
torch.manual_seed(20261005)
if args.checkpoint:
    checkpoint=args.checkpoint.resolve()
    required=0
    for path in checkpoint.glob('*.safetensors'):
        with path.open('rb') as f:
            header=json.loads(f.read(int.from_bytes(f.read(8),'little')))
        for name,value in header.items():
            if name=='__metadata__':continue
            count=1
            for dim in value['shape']:count*=dim
            required+=count*4
    if sys.platform=='darwin':
        stat=subprocess.check_output(['vm_stat'],text=True)
        page=int(re.search(r'page size of (\d+)',stat)[1])
        available=sum(int(re.search(rf'Pages {kind}:\s+(\d+)',stat)[1]) for kind in ['free','inactive'])*page
        assert available>=required+4*1024**3,f'need {required+4*1024**3} available bytes, have {available}; release only test-owned pipelines first'
    torch.set_num_threads(8)
    print(f'Loading trained fp32 reference ({required/1e9:.2f} GB weights)',flush=True)
    model=ZImageTransformer2DModel.from_pretrained(checkpoint,torch_dtype=torch.float32,local_files_only=True,low_cpu_mem_usage=True).eval()
    for path in checkpoint.iterdir():
        if path.suffix in ['.json','.safetensors']:
            dest=root/path.name
            assert not dest.exists(),f'output already exists: {dest}'
            dest.symlink_to(path)
else:
    model = ZImageTransformer2DModel(dim=256 if args.head128 else 24, n_heads=2, n_kv_heads=2, n_layers=2,
        n_refiner_layers=2, in_channels=4, cap_feat_dim=16, axes_dims=[32,48,48] if args.head128 else [4,4,4],
        axes_lens=[1024,64,64]).float().eval()
    model.save_pretrained(root)
cases=[(8,8,5)] if args.small_only else [(8,8,5),(32,32,35)]
for height,width,tokens in cases:
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
    latent = torch.randn(model.config.in_channels,1,height,width);caption = torch.randn(tokens,model.config.cap_feat_dim);t=0.625
    with torch.no_grad():out = model([latent],torch.tensor([t]),[caption],return_dict=False)[0][0]
    for hook in hooks:hook.remove()
    case={'height':height,'width':width,'caption_tokens':tokens,'t':t,'latent':latent.flatten().tolist(),
        'caption':caption.flatten().tolist(),'traces':traces,'output':out.flatten().tolist()}
    (root/f'case-{height}.json').write_text(json.dumps(case))
    print(f'reference {height}x{width}: {len(traces)} tensors',flush=True)
