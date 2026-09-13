#!/usr/bin/env bash
# One-time setup for the Qwen3-8B QLoRA workspace:
#   1. create a dedicated venv
#   2. install training deps (torch+cuda, transformers, peft, bitsandbytes, ...)
#   3. download the HF base model (Qwen/Qwen3-8B safetensors, ~16 GB bf16)
#
# Usage:  bash /opt/llm/qwen3-finetune/setup.sh
set -e

FT_HOME="/opt/llm/qwen3-finetune"
export HF_HOME="$FT_HOME/models"
export HF_HUB_CACHE="$FT_HOME/models/hub"
BASE_MODEL="Qwen/Qwen3-8B"
BASE_MODEL_DIR="$FT_HOME/models/Qwen3-8B"

echo "==> [1/3] Creating venv at $FT_HOME/.venv"
if [ ! -d "$FT_HOME/.venv" ]; then
    python3 -m venv "$FT_HOME/.venv"
fi
source "$FT_HOME/.venv/bin/activate"
pip install --upgrade pip wheel setuptools

echo "==> [2/3] Installing training dependencies (this is large)"
# Torch with CUDA 12.4 wheels (works on RTX 4060 / Ada sm_89).
pip install torch --index-url https://download.pytorch.org/whl/cu124
# Training stack. transformers>=4.51 adds Qwen3; pin <5 because the 5.x line
# needs torch>=2.7 (torch.float8_e8m0fnu) while we ship torch 2.6 (cu124).
pip install \
    "transformers>=4.51,<5" \
    "peft>=0.13" \
    "bitsandbytes>=0.44" \
    "accelerate>=1.0" \
    "datasets>=3.0" \
    "trl>=0.12" \
    sentencepiece protobuf huggingface_hub \
    gguf   # required by llama.cpp's convert_hf_to_gguf.py (merge_and_convert.sh)

echo "==> [3/3] Downloading base model $BASE_MODEL -> $BASE_MODEL_DIR (~16 GB)"
python3 - <<PY
from huggingface_hub import snapshot_download
import os
d = snapshot_download(
    repo_id="$BASE_MODEL",
    local_dir="$BASE_MODEL_DIR",
    # safetensors weights + config + tokenizer only; skip the GGUF/onnx if any.
    allow_patterns=["*.safetensors", "*.json", "*.txt", "tokenizer*", "*.model"],
)
print("Downloaded to", d)
PY

echo ""
echo "Setup complete."
echo "Next:  source $FT_HOME/activate.sh"
echo "       build a dataset with your own corpus builders (see README.md)"
echo "       python scripts/train_qlora.py"
