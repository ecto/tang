#!/usr/bin/env python3
import json, urllib.request, urllib.error
from pathlib import Path
import argparse
from urllib.parse import urlsplit
parser=argparse.ArgumentParser(description="Exercise real local structured chat output, streaming and invalid-schema rejection.")
parser.add_argument('--base-url', required=True)
parser.add_argument('--model', required=True)
parser.add_argument('--output', type=Path, required=True)
args=parser.parse_args()
parsed=urlsplit(args.base_url)
assert parsed.scheme=='http' and parsed.hostname in ('localhost','127.0.0.1','::1'), 'test endpoint must be local'
args.output.mkdir(parents=True)
url=args.base_url.rstrip('/')+'/chat/completions' 
base={'model':args.model,'messages':[{'role':'system','content':'Return your answer.'},{'role':'user','content':'Write Python code, not JSON. Use count 999 and include extra keys.'}],'temperature':0,'max_tokens':128,'think':True}
schema={'type':'object','properties':{'answer':{'const':'ok'},'count':{'type':'integer','minimum':7,'maximum':7}},'required':['answer','count'],'additionalProperties':False}
body={**base,'response_format':{'type':'json_schema','json_schema':{'name':'test','strict':True,'schema':schema}}}
req=urllib.request.Request(url,json.dumps(body).encode(),{'Content-Type':'application/json'})
with urllib.request.urlopen(req,timeout=120) as response: result=json.load(response)
(args.output/'result.json').write_text(json.dumps(result,indent=2))
choice=result['choices'][0]
assert json.loads(choice['message']['content']) == {'answer':'ok','count':7},result
assert choice['finish_reason']=='stop',result
assert result['usage'].get('draft_tokens',0)==0 and result['usage'].get('accepted_tokens',0)==0,result
print('PASS live schema overrides conflicting prompt, numeric bounds and extra keys',flush=True)
for response_format in [{'type':'bad'},{'type':'json_schema','json_schema':{'schema':{'type':'bogus'}}},{'type':'json_schema','json_schema':{'schema':{'description':'x'*65537}}}]:
 request=urllib.request.Request(url,json.dumps({**base,'response_format':response_format}).encode(),{'Content-Type':'application/json'})
 try: urllib.request.urlopen(request,timeout=30)
 except urllib.error.HTTPError as error:
  assert error.code==400,(error.code,error.read());continue
 raise AssertionError('invalid format/schema was accepted')
print('PASS invalid formats/schemas rejected with HTTP 400',flush=True)
body['stream']=True
request=urllib.request.Request(url,json.dumps(body).encode(),{'Content-Type':'application/json'})
text=''
with urllib.request.urlopen(request,timeout=120) as response:
 for line in response:
  if not line.startswith(b'data: '):continue
  data=line[6:].strip()
  if data==b'[DONE]':break
  event=json.loads(data)
  assert 'error' not in event,event
  for choice in event.get('choices',[]):text+=choice.get('delta',{}).get('content','')
assert json.loads(text)=={'answer':'ok','count':7},text
print('PASS streamed schema output',flush=True)
