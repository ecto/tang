"""Audit fp32 attention-kernel variation against an existing diffusers reference case.
No downloads or weight copies. This measures reference variability, not tang accuracy.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from diffusers import ZImageTransformer2DModel

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--reference-case', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--atol', type=float, default=1e-3)
parser.add_argument('--rtol', type=float, default=1e-6)
args = parser.parse_args()
assert not args.output.exists(), 'choose a new metrics output'
required = 0
for path in args.checkpoint.glob('*.safetensors'):
    with path.open('rb') as stream:
        header = json.loads(stream.read(int.from_bytes(stream.read(8), 'little')))
    for name, tensor in header.items():
        if name == '__metadata__':
            continue
        count = 1
        for dimension in tensor['shape']:
            count *= dimension
        required += count * 4
if sys.platform == 'darwin':
    stat = subprocess.check_output(['vm_stat'], text=True)
    page = int(re.search(r'page size of (\d+)', stat)[1])
    available = sum(int(re.search(rf'Pages {kind}:\s+(\d+)', stat)[1]) for kind in ['free', 'inactive']) * page
    assert available >= required + 4 * 1024**3, 'release only test-owned pipelines before loading the reference'
case = json.loads(args.reference_case.read_text())
torch.set_num_threads(8)
print(f'Loading local fp32 reference ({required/1e9:.2f} GB)', flush=True)
model = ZImageTransformer2DModel.from_pretrained(args.checkpoint, torch_dtype=torch.float32,
    local_files_only=True, low_cpu_mem_usage=True).eval()
latent = torch.tensor(case['latent']).reshape(model.config.in_channels, 1, case['height'], case['width'])
caption = torch.tensor(case['caption']).reshape(case['caption_tokens'], model.config.cap_feat_dim)
records = {}

def metrics(actual, values):
    actual = actual.detach().float().flatten()
    expected = torch.tensor(values)
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    error = (actual - expected).abs()
    ratio = error / (args.atol + args.rtol * expected.abs())
    return {'max_abs': error.max().item(), 'relative_l2':
        (torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected).clamp_min(1e-30)).item(),
        'max_tolerance_ratio': ratio.max().item()}

def capture(name):
    def hook(module, inputs, output):
        records[name] = metrics(output, case['traces'][name])
        print(name, records[name], flush=True)
    return hook

for group in ['noise_refiner', 'context_refiner', 'layers']:
    for index, layer in enumerate(getattr(model, group)):
        layer.register_forward_hook(capture(f'{group}.{index}'))
model.all_final_layer['2-1'].register_forward_hook(capture('final'))
with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
    output = model([latent], torch.tensor([case['t']]), [caption], return_dict=False)[0][0]
records['output'] = metrics(output, case['output'])
summary = {'torch_version': torch.__version__, 'attention_backend': 'MATH',
    'checkpoint': str(args.checkpoint.resolve()), 'reference_case': str(args.reference_case.resolve()),
    'atol': args.atol, 'rtol': args.rtol, 'metrics': records,
    'failed_pointwise': [name for name, value in records.items() if value['max_tolerance_ratio'] > 1]}
with args.output.open('x') as stream:
    json.dump(summary, stream, indent=2)
print(f'Independent reference variation: {len(summary["failed_pointwise"])} boundaries exceed the supplied bound', flush=True)
