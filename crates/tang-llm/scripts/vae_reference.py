"""Export fp32 diffusers VAE decoder boundary tensors without model downloads."""
import json
import sys
from pathlib import Path
import torch
from diffusers import AutoencoderKL
root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=True)
torch.manual_seed(20261005)
model=AutoencoderKL(latent_channels=4,block_out_channels=(8,8,16,16),norm_num_groups=4,
    down_block_types=('DownEncoderBlock2D',)*4,up_block_types=('UpDecoderBlock2D',)*4,
    layers_per_block=2,scaling_factor=0.3611,shift_factor=0.1159,
    use_quant_conv=False,use_post_quant_conv=False).float().eval()
model.save_pretrained(root)
(root/"vae-fixture").touch()
for size in [4,32]:
    hooks=[];traces={}
    def capture(name):
        def hook(module,args,out):traces[name]=out.detach().float().flatten().tolist()
        return hook
    hooks.append(model.decoder.conv_in.register_forward_hook(capture('conv_in')))
    for i,block in enumerate(model.decoder.mid_block.resnets):hooks.append(block.register_forward_hook(capture(f'mid.resnets.{i}')))
    hooks.append(model.decoder.mid_block.attentions[0].register_forward_hook(capture('mid.attention')))
    for i,block in enumerate(model.decoder.up_blocks):
        for j,res in enumerate(block.resnets):hooks.append(res.register_forward_hook(capture(f'up.{i}.resnets.{j}')))
        if block.upsamplers:hooks.append(block.upsamplers[0].register_forward_hook(capture(f'up.{i}.upsample')))
    hooks.append(model.decoder.conv_out.register_forward_hook(capture('output')))
    latent=torch.randn(1,4,size,size)
    with torch.no_grad():out=model.decode(latent).sample
    for hook in hooks:hook.remove()
    (root/f'case-{size}.json').write_text(json.dumps({'height':size,'width':size,'latent':latent.flatten().tolist(),'traces':traces,'output':out.flatten().tolist()}))
    print(f'VAE {size}x{size} -> {size*8}x{size*8}: {len(traces)} boundaries',flush=True)
