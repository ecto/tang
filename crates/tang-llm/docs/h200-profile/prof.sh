# usage: prof.sh <name> <streams> <spec 0|1>
set -u
name=$1; n=$2; spec=$3
model=$(cat /root/model_path)
mkdir -p /root/out
export TANG_SPECULATE=$spec TANG_MAX_BATCH=16
nsys profile -t cuda,nvtx,osrt --cuda-memory-usage=false --sample=cpu --cpuctxsw=none \
  -o /root/out/$name -f true --wait=primary \
  /src/tang/target/release/tang-llm serve "$model" --device cuda --host 127.0.0.1 --port 8912 \
  --ctx 8192 --kv-slots 16 > /root/out/$name.serve.log 2>&1 &
for _ in $(seq 1 600); do curl -sf -m 2 http://127.0.0.1:8912/health >/dev/null && break; sleep 1; done
req() { curl -s -m 600 -H 'content-type: application/json' -d "{\"max_tokens\":$1,\"temperature\":0.7,\"prompt_cache_key\":\"s$2\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a long, detailed essay on the history of $3. /no_think\"}]}" http://127.0.0.1:8912/v1/chat/completions; }
req 16 0 tea >/dev/null   # warm-up
topics=(tea bridges glass rivers clocks paper salt maps bells ships trains coins ink silk iron wheat)
t0=$(date +%s.%N)
for i in $(seq 0 $((n-1))); do req 512 $i "${topics[$i]}" > /root/out/$name.r$i.json & done
wait $(jobs -p | tail -n $n)
t1=$(date +%s.%N)
python3 - "$name" "$n" "$t0" "$t1" <<'PY'
import json,sys,glob
name,n,t0,t1=sys.argv[1],int(sys.argv[2]),float(sys.argv[3]),float(sys.argv[4])
tok=0
for i in range(n):
    try: tok+=json.load(open(f"/root/out/{name}.r{i}.json"))["usage"]["completion_tokens"]
    except Exception as e: print("bad",i,e)
print(f"{name}: streams={n} tokens={tok} wall={t1-t0:.2f}s total_tok/s={tok/(t1-t0):.1f}")
PY
pkill -INT -f 'tang-llm serve'; wait
