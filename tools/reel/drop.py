import array, math, json, sys
sr=11025; hop=128
a=array.array('h'); a.frombytes(open(sys.argv[1],'rb').read())
d=json.load(open(sys.argv[2])); beats=d['beats']; P=60/d['bpm']
def rms(t0,t1):
    i0,i1=int(t0*sr),int(t1*sr); s=0
    for v in a[i0:i1:4]: s+=v*v
    return math.sqrt(s/max(1,(i1-i0)//4))
# low-ish band proxy: energy in first 60ms after each beat (kick)
hit=[rms(b,b+0.06) for b in beats]
phase=max(range(4),key=lambda p: sum(hit[p::4])/len(hit[p::4]))
print('downbeat phase', phase, [round(sum(hit[p::4])/len(hit[p::4])) for p in range(4)])
# bar energy, find biggest jump on a downbeat after 30 s
bars=[(k,beats[k]) for k in range(phase,len(beats)-16,4)]
def e(k,n): return sum(hit[k:k+n])/n
cands=sorted(((e(k,16)-e(k-8,8)) if k>=8 else 0, k, round(t,3)) for k,t in bars)[-6:]
for c in cands: print('jump', round(c[0]), 'beat', c[1], 't', c[2], 'energy after', round(e(c[1],16)))
