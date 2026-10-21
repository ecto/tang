import sqlite3,sys,collections,re
db=sqlite3.connect(sys.argv[1]); steps_hint=None
q=lambda s:db.execute(s).fetchall()
names=dict(q("select id,value from StringIds"))
K=q("select start,end,shortName,gridX*gridY*gridZ,blockX*blockY*blockZ from CUPTI_ACTIVITY_KIND_KERNEL order by start")
M=q("select start,end,copyKind,bytes from CUPTI_ACTIVITY_KIND_MEMCPY order by start") if q("select name from sqlite_master where name='CUPTI_ACTIVITY_KIND_MEMCPY'") else []
S=q("select start,end from CUPTI_ACTIVITY_KIND_MEMSET") if q("select name from sqlite_master where name='CUPTI_ACTIVITY_KIND_MEMSET'") else []
R=q("select start,end,nameId,globalTid from CUPTI_ACTIVITY_KIND_RUNTIME order by start")
# decode window: densest busy stretch at the end; find the last idle gap >200ms before the end
ends=[k[1] for k in K]; starts=[k[0] for k in K]
last=ends[-1]
# walk back to find start of final burst (gap >300ms)
i=len(K)-1
while i>0 and K[i][0]-K[i-1][1] < 300e6: i-=1
w0=K[i][0]; 
# drop first 15% (prefill) of the burst
w0=w0+(last-w0)*0.15; w1=last-(last-w0)*0.02
win=[k for k in K if w0<=k[0]<w1]
T=w1-w0
busy=0; cur_s=cur_e=None
for s,e,*_ in win:
    if cur_e is None or s>cur_e:
        if cur_e is not None: busy+=cur_e-cur_s
        cur_s,cur_e=s,e
    else: cur_e=max(cur_e,e)
busy+=cur_e-cur_s
# step detection: lm_head-ish? use memcpy DtoH as step boundary
dtoh=[m for m in M if w0<=m[0]<w1 and m[2]==2]
htod=[m for m in M if w0<=m[0]<w1 and m[2]==1]
dtod=[m for m in M if w0<=m[0]<w1 and m[2] not in (1,2)]
steps=len(dtoh) or 1
print(f"window {T/1e6:.1f} ms, kernels {len(win)}, DtoH copies {len(dtoh)} (steps), HtoD {len(htod)}, other memcpy {len(dtod)}, memsets {len([x for x in S if w0<=x[0]<w1])}")
print(f"per step: {T/steps/1e6:.3f} ms, kernels/step {len(win)/steps:.1f}, GPU busy {busy/T*100:.1f}% ({busy/steps/1e6:.3f} ms/step), idle {(T-busy)/steps/1e6:.3f} ms/step")
print(f"DtoH bytes/copy avg {sum(m[3] for m in dtoh)/max(1,len(dtoh)):.0f}, DtoH time/step {sum(m[1]-m[0] for m in dtoh)/steps/1e3:.1f} us; HtoD/step {len(htod)/steps:.1f} copies {sum(m[1]-m[0] for m in htod)/steps/1e3:.1f} us")
fam=collections.defaultdict(lambda:[0,0])
def famname(n):
    n=names.get(n,str(n)); return re.sub(r'_?\d+$','',n)
for s,e,n,g,b in win:
    f=fam[names.get(n,str(n))]; f[0]+=1; f[1]+=e-s
print("\nkernel families (per step): count, us, share of busy, avg us")
for n,(c,t) in sorted(fam.items(),key=lambda x:-x[1][1])[:25]:
    print(f"  {n[:60]:60s} {c/steps:6.1f} {t/steps/1e3:8.1f}us {t/busy*100:5.1f}% {t/c/1e3:7.2f}")
# gaps between consecutive kernels
gaps=[win[j+1][0]-win[j][1] for j in range(len(win)-1)]
gaps=[g for g in gaps if g>0]
gaps.sort()
import statistics
if gaps:
  big=[g for g in gaps if g>20e3]
  print(f"\ngaps: n={len(gaps)} median {statistics.median(gaps)/1e3:.2f}us p90 {gaps[int(.9*len(gaps))]/1e3:.2f}us; gaps>20us: {len(big)/steps:.1f}/step totalling {sum(big)/steps/1e6:.3f} ms/step; small gaps total {(sum(gaps)-sum(big))/steps/1e6:.3f} ms/step")
# runtime API summary in window
api=collections.defaultdict(lambda:[0,0])
for s,e,n,t in R:
    if w0<=s<w1: a=api[names.get(n,str(n))]; a[0]+=1; a[1]+=e-s
print("\nCUDA API (per step): count, us")
for n,(c,t) in sorted(api.items(),key=lambda x:-x[1][1])[:15]:
    print(f"  {n[:50]:50s} {c/steps:7.1f} {t/steps/1e3:8.1f}us")
