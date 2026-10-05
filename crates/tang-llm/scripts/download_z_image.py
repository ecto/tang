"""Download pinned public Z-Image weights, converting F32 matrices to BF16 in flight.
Retains 4 GiB disk headroom, verifies original LFS hashes, never replaces existing files.
Only the new output directory is written; no production weights are downloaded twice.
"""
import hashlib
import json
import os
import shutil
import struct
import sys
from pathlib import Path
import numpy as np
import requests
from huggingface_hub import model_info

root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=True)
info=model_info('Tongyi-MAI/Z-Image-Turbo',files_metadata=True)
revision=info.sha
base=f'https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/resolve/{revision}/'
reserve=4*1024**3
manifest={'repo':info.id,'revision':revision,'conversion':'F32 matrices to BF16 round-to-nearest-even; vectors unchanged','files':{}}

def read_exact(stream,n):
    chunks=[];left=n
    while left:
        part=stream.read(left)
        if not part:raise EOFError(f'expected {left} more bytes')
        chunks.append(part);left-=len(part)
    return b''.join(chunks)

files=[s for s in info.siblings if s.rfilename.startswith(('transformer/','text_encoder/','vae/','tokenizer/','scheduler/')) and s.rfilename.endswith(('.json','.safetensors','.txt','.model','.jinja'))]
# All text/VAE weights already use BF16; transformer storage is halved.
expected=sum((s.size or 0)//2 if s.rfilename.startswith('transformer/') and s.rfilename.endswith('.safetensors') else (s.size or 0) for s in files)
assert shutil.disk_usage(root).free >= expected+reserve, f'insufficient disk: need {expected+reserve} bytes'
print(f'Pinned {revision}; approximately {expected/1e9:.2f} GB output',flush=True)
for entry in files:
    name=entry.rfilename;dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():raise FileExistsError(dest)
    tmp=dest.with_suffix(dest.suffix+'.part')
    original=hashlib.sha256();converted=hashlib.sha256()
    print(f'Downloading {name}',flush=True)
    with requests.get(base+name+'?download=true',stream=True,timeout=(30,120)) as response:
        response.raise_for_status();stream=response.raw
        with tmp.open('xb') as out:
            def write(data):
                if shutil.disk_usage(root).free < len(data)+reserve:raise RuntimeError('disk headroom exhausted')
                out.write(data);converted.update(data)
            if name.endswith('.safetensors'):
                prefix=read_exact(stream,8);original.update(prefix);header_size=int.from_bytes(prefix,'little');assert header_size<8*1024**2
                raw=read_exact(stream,header_size);original.update(raw);header=json.loads(raw)
                tensors=sorted(((n,v) for n,v in header.items() if n!='__metadata__'),key=lambda pair:pair[1]['data_offsets'][0])
                new_header={'__metadata__':header.get('__metadata__',{})};offset=0;position=0
                for key,value in tensors:
                    value=dict(value);size=value['data_offsets'][1]-value['data_offsets'][0]
                    if value['dtype']=='F32' and len(value['shape'])>=2:size//=2;value['dtype']='BF16'
                    value['data_offsets']=[offset,offset+size];offset+=size;new_header[key]=value
                encoded=json.dumps(new_header,separators=(',',':')).encode();encoded+=b' '*((-len(encoded))%8)
                write(struct.pack('<Q',len(encoded)));write(encoded)
                done=0;total=sum(v['data_offsets'][1]-v['data_offsets'][0] for _,v in tensors);next_report=0
                for key,value in tensors:
                    start,end=value['data_offsets'];assert start==position,'non-contiguous tensor offsets'
                    left=end-start
                    while left:
                        n=min(left,4*1024**2);data=read_exact(stream,n);original.update(data)
                        if value['dtype']=='F32' and len(value['shape'])>=2:
                            bits=np.frombuffer(data,dtype='<u4');rounded=bits+np.uint32(0x7fff)+((bits>>16)&1)
                            data=(rounded>>16).astype('<u2').tobytes()
                        write(data);left-=n;done+=n
                        if done>=next_report:print(f'  {done/total:.0%}',flush=True);next_report=done+1024**3
                    position=end
                assert not stream.read(1),'unexpected trailing safetensors data'
            else:
                while True:
                    data=stream.read(4*1024**2)
                    if not data:break
                    original.update(data);write(data)
            out.flush();os.fsync(out.fileno())
    if entry.lfs:assert original.hexdigest()==entry.lfs.sha256,f'upstream hash mismatch: {name}'
    tmp.rename(dest)
    manifest['files'][name]={'upstream_sha256':original.hexdigest(),'output_sha256':converted.hexdigest(),'bytes':dest.stat().st_size}
    (root/'conversion.json').write_text(json.dumps(manifest,indent=2))
print('Download and integrity verification complete',flush=True)
