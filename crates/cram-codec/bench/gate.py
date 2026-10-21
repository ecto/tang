import csv,sys
rows=list(csv.DictReader(open(sys.argv[1])))
files=[];by={}
for r in rows:
    f=r['file']
    if f not in by: files.append(f); by[f]={}
    by[f][r['arm']]=r
def hdr(n):
    v=1
    while n>=128: n>>=7; v+=1
    return 9+v
print("| file | in bytes | best arm | best arm bytes | cram bytes | cram pick | Δ vs best | header | gate | cram c MB/s | cram d MB/s |")
print("|---|---:|---|---:|---:|---|---:|---:|---|---:|---:|")
allok=True
for f in files:
    d=by[f]; c=d['cram']; arms={k:v for k,v in d.items() if k!='cram'}
    b=min(arms.values(),key=lambda r:int(r['out_bytes']))
    n=int(c['in_bytes']); h=hdr(n); delta=int(c['out_bytes'])-int(b['out_bytes'])
    ok=delta<=h; allok&=ok
    tie="tie (= best + header)" if delta==h else ("PASS" if ok else "FAIL")
    print(f"| {f} | {n} | {b['arm']} | {b['out_bytes']} | {c['out_bytes']} | {c['pick']} | {delta:+d} | {h} | {tie} | {c['c_MBps']} | {c['d_MBps']} |")
print("\nALL PASS" if allok else "\nSOME FAIL")
