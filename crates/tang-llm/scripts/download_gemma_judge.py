"""Explicitly download the pinned public Gemma 3 4-bit vision judge; no inference downloads.
Verifies completed files and LFS hashes, decodes HTTP compression and reserves 4 GiB disk.
"""
import hashlib
import argparse
import json
import os
import shutil
from pathlib import Path
import requests
from huggingface_hub import model_info
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('output',type=Path)
parser.add_argument('--model',choices=('4b','12b','27b'),default='4b')
args=parser.parse_args()
repo=f'mlx-community/gemma-3-{args.model}-it-4bit'
revision={'4b':'93724907d4ed1745d2fe50baadf3b0b01a65abf2','12b':'86cc6a8dedbc456dd0e4af01a9d09f396f77e558','27b':'83acee3d10064661a7a39ead3732dddb16fe15bb'}[args.model]
root=args.output;root.mkdir(parents=True,exist_ok=True)
record=root/'download.json'
manifest=json.loads(record.read_text()) if record.exists() else {'repo':repo,'revision':revision,'files':{}}
assert manifest['repo']==repo and manifest['revision']==revision
info=model_info(repo,revision=revision,files_metadata=True)
files=[s for s in info.siblings if s.rfilename.endswith(('.json','.safetensors','.txt','.model','.jinja'))]
reserve=4*1024**3
remaining=sum(s.size or 0 for s in files if s.rfilename not in manifest['files'])
assert shutil.disk_usage(root).free>=remaining+reserve,'insufficient disk headroom'
session=requests.Session()
session.mount('https://',HTTPAdapter(max_retries=Retry(total=3,read=0,backoff_factor=.5,status_forcelist=[429,502,503,504])))
for entry in files:
    dest=root/entry.rfilename;dest.parent.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256()
    if dest.exists():
        previous=manifest['files'].get(entry.rfilename)
        assert previous,'untracked existing file'
        with dest.open('rb') as f:
            while data:=f.read(16*1024**2):digest.update(data)
        assert digest.hexdigest()==previous['sha256'] and dest.stat().st_size==previous['bytes']
        assert dest.stat().st_size==entry.size,'upstream size mismatch'
        if entry.lfs:assert digest.hexdigest()==entry.lfs.sha256,'upstream SHA256 mismatch'
        print('Verified completed',entry.rfilename,flush=True)
        continue
    assert entry.rfilename not in manifest['files'],'completed file missing'
    tmp=dest.with_suffix(dest.suffix+'.part');created=False
    print('Downloading',entry.rfilename,entry.size,flush=True)
    try:
        with session.get(f'https://huggingface.co/{repo}/resolve/{revision}/{entry.rfilename}',headers={'Accept-Encoding':'identity'},stream=True,timeout=(30,120)) as response:
            response.raise_for_status();response.raw.decode_content=True
            with tmp.open('xb') as out:
                created=True;total=0;report=0
                while data:=response.raw.read(4*1024**2):
                    assert shutil.disk_usage(root).free>=len(data)+reserve,'disk headroom exhausted'
                    out.write(data);digest.update(data);total+=len(data)
                    if total>=report:
                        print(f'  {total} bytes',flush=True);report=total+512*1024**2
                out.flush();os.fsync(out.fileno())
        assert total==entry.size,'size mismatch'
        if entry.lfs:assert digest.hexdigest()==entry.lfs.sha256,'upstream SHA256 mismatch'
        if dest.suffix=='.json':json.loads(tmp.read_bytes())
        tmp.rename(dest)
    except BaseException:
        if created and tmp.exists():tmp.unlink()
        raise
    manifest['files'][entry.rfilename]={'sha256':digest.hexdigest(),'bytes':total}
    record.write_text(json.dumps(manifest,indent=2))
manifest['complete']=True
record.write_text(json.dumps(manifest,indent=2))
print('Judge download and integrity verification complete',flush=True)
