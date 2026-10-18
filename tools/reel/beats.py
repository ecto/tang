import array, math, json, sys
sr=11025; hop=256
a=array.array('h'); a.frombytes(open(sys.argv[1],'rb').read())
n=len(a)//hop
env=[]
for i in range(n):
    s=0
    for v in a[i*hop:(i+1)*hop]: s+=v*v
    env.append(math.log(1e-9+s/hop))
fps=sr/hop
on=[max(0,env[i]-env[i-1]) for i in range(1,n)]; on=[0]+on
m=sum(on)/len(on); on=[x-m for x in on]
best=None
for bpm10 in range(700,1800,2):
    bpm=bpm10/10; lag=60/bpm*fps
    sc=0; k=1
    while k*lag < min(len(on)-1, 8*fps*4):
        L=int(round(k*lag)); sc+=sum(on[i]*on[i+L] for i in range(0,len(on)-L,3)); k+=1
    sc/=k
    if not best or sc>best[0]: best=(sc,bpm)
bpm=best[1]; period=60/bpm*fps
# phase
bp=max(range(int(period)), key=lambda p: sum(on[int(p+k*period)] for k in range(int((len(on)-p)/period))))
beats=[ (bp+k*period)/fps for k in range(int((len(on)-bp)/period)) ]
# energy per second
sec=[sum(math.exp(e) for e in env[int(t*fps):int((t+1)*fps)])/fps for t in range(int(n/fps))]
mx=max(sec); sec=[round(x/mx,2) for x in sec]
json.dump({"bpm":bpm,"beats":[round(b,3) for b in beats],"energy":sec},open(sys.argv[2],'w'))
print(bpm, len(beats), beats[:4]); print(sec)
