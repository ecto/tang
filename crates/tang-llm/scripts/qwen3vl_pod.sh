#!/usr/bin/env bash
# Qwen3-VL GPU checks on a fresh RunPod pytorch pod, end to end, results to stdout (the pod log).
# Pod entrypoint (a fresh clone, not raw.githubusercontent.com, whose CDN can serve a stale copy):
#   bash -c "git clone -q --depth 1 -b $BRANCH https://github.com/ecto/tang /root/boot && BRANCH=$BRANCH bash /root/boot/crates/tang-llm/scripts/qwen3vl_pod.sh"
# STEPS (default "build tiny spec real") picks what runs.
set -uo pipefail
BRANCH=${BRANCH:-main}
STEPS=${STEPS:-build tiny spec real}
export HF_HOME=/root/hf CARGO_TARGET_DIR=/root/target
say() { echo "=== $* ($(date -u +%H:%M:%S))"; }

say setup
# The image's own start script runs sshd with PUBLIC_KEY (for direct SSH while this runs).
[ -x /start.sh ] && (/start.sh >/dev/null 2>&1 &)
command -v cargo >/dev/null || { curl -sSf https://sh.rustup.rs | sh -s -- -y -q --profile minimal >/dev/null; }
source "$HOME/.cargo/env"
rm -rf /root/tang && git clone -q --depth 1 -b "$BRANCH" https://github.com/ecto/tang /root/tang
cd /root/tang && git log --oneline -1
pip install -q "transformers==5.20.2" safetensors pillow numpy huggingface_hub accelerate 2>&1 | tail -1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
BIN=$CARGO_TARGET_DIR/release/tang-llm

if [[ $STEPS == *build* ]]; then
  say build
  cargo build -q --release -p tang-llm --no-default-features --features cuda --bin tang-llm 2>&1 | grep -E "^error" -A8 | head -40
fi
if [[ $STEPS == *tiny* ]]; then
  # The fp32 bar: full-FP32 cuBLAS (tang-llm turns TF32 on for CUDA inference by default).
  for seed in 0 1; do
    say tiny seed $seed
    GAIA_TF32=0 python crates/tang-llm/scripts/qwen3vl_tiny.py --tang $BIN --out /root/tiny$seed --seed $seed --device cuda 2>&1 \
      | grep --line-buffered -E "^(images|preprocess|prefill|decode|\(HF|PASS|FAIL)|embed|block|merger|deepstack|Error|error|failed"
  done
fi
if [[ $STEPS == *spec* ]]; then
  say speculate
  python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-0.6B')" >/dev/null 2>&1
  TANG_LLM_TEST_VL_MODEL=/root/tiny0/hf TANG_LLM_TEST_MODEL=Qwen/Qwen3-0.6B \
    cargo test -q --release -p tang-llm --no-default-features --features cuda --test speculate -- --nocapture 2>&1 \
    | grep --line-buffered -E "identical=|test result|panicked|differ" | head -60
fi
if [[ $STEPS == *real* ]]; then
  say download
  python -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen3-VL-8B-Instruct'))" > /root/model_path 2>/dev/null
  say real
  python crates/tang-llm/scripts/qwen3vl_real.py --tang $BIN --model "$(cat /root/model_path)" --out /root/real --device cuda 2>&1 \
    | grep -vE "Warning|warn\(|it/s\]|Loading"
fi
if [[ $STEPS == *llama* ]]; then
  say llama.cpp build
  M="$(python -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen3-VL-8B-Instruct'))" 2>/dev/null)"
  [ -d /root/llama.cpp ] || git clone -q --depth 1 https://github.com/ggml-org/llama.cpp /root/llama.cpp
  (cd /root/llama.cpp && cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=OFF >/dev/null && cmake --build build -j --target llama-server 2>&1 | grep -E "error" | head)
  say llama.cpp convert
  pip install -q gguf sentencepiece protobuf 2>&1 | tail -1
  python /root/llama.cpp/convert_hf_to_gguf.py "$M" --outtype bf16 --outfile /root/vl.gguf 2>&1 | tail -1
  python /root/llama.cpp/convert_hf_to_gguf.py "$M" --outtype bf16 --mmproj --outfile /root/mmproj.gguf 2>&1 | tail -1
  /root/llama.cpp/build/bin/llama-server -m /root/vl.gguf --mmproj /root/mmproj.gguf -ngl 99 -c 8192 --port 8080 >/root/llama.log 2>&1 &
  for i in $(seq 120); do curl -sf localhost:8080/health >/dev/null && break; sleep 5; done
  say llama.cpp compare
  python crates/tang-llm/scripts/qwen3vl_llamacpp.py --tang $BIN --model "$M" --image /root/real/view.png --device cuda 2>&1 \
    | grep -vE "Warning|warn\("
  tail -3 /root/llama.log
  pkill -f llama-server; sleep 3
fi
if [[ $STEPS == *eval* ]]; then
  say eval
  M="$(python -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen3-VL-8B-Instruct'))" 2>/dev/null)"
  pip install -q matplotlib 2>&1 | tail -1
  $BIN serve "$M" --device cuda --port 8090 --ctx 8192 --kv-slots 2 >/root/serve.log 2>&1 &
  SERVE=$!
  for i in $(seq 120); do curl -sf localhost:8090/v1/models >/dev/null && break; sleep 5; done
  python crates/tang-llm/scripts/qwen3vl_eval.py --phase tang --model "$M" --server http://127.0.0.1:8090 --out /root/eval 2>&1 | tail -3
  kill $SERVE; sleep 5; tail -3 /root/serve.log
  python crates/tang-llm/scripts/qwen3vl_eval.py --phase hf --model "$M" --out /root/eval 2>&1 \
    | grep --line-buffered -vE "Warning|warn\(|it/s\]|Loading"
fi
say DONE
sleep infinity
