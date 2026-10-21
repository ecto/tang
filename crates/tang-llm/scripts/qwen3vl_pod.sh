#!/usr/bin/env bash
# Qwen3-VL GPU checks on a fresh RunPod pytorch pod, end to end, results to stdout (the pod log).
# Pod entrypoint: bash -c "curl -fsSL https://raw.githubusercontent.com/ecto/tang/$BRANCH/crates/tang-llm/scripts/qwen3vl_pod.sh | BRANCH=$BRANCH bash"
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
  for seed in 0 1; do
    say tiny seed $seed
    python crates/tang-llm/scripts/qwen3vl_tiny.py --tang $BIN --out /root/tiny$seed --seed $seed --device cuda 2>&1 \
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
say DONE
sleep infinity
