#!/bin/bash
# SPEAR — cloud QLoRA run for Qwen3.6-35B-A3B.
#
# Run this ON THE POD (RunPod/Vast.ai, PyTorch image, GPU >= 48 GB VRAM,
# >= 150 GB disk for the BF16 base download). Typical cost: 2-5 h on an
# L40S/A6000 (48 GB) or A100-80G at ~0.8-1.8 $/h.
#
# Confirm the HF base repo id first (the local model came as GGUF):
#   export BASE_MODEL=<exact-hf-id>     # default: Qwen/Qwen3.6-35B-A3B
#
# Usage on the pod:
#   1. scp/rsync this cloud/ dir AND the data/ dir to the pod, e.g.:
#        rsync -av cloud/ data/ root@<pod>:/workspace/ft/
#   2. ssh to the pod, then:
#        cd /workspace/ft && bash runpod_train.sh
#   3. Fetch the result from your laptop:
#        scp root@<pod>:/workspace/ft/$GGUF_NAME \
#            /opt/llm/models/
set -e

cd "$(dirname "$0")"
export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.6-35B-A3B}"
export FT_DATA_DIR="${FT_DATA_DIR:-$PWD/data}"
export FT_OUT_DIR="${FT_OUT_DIR:-$PWD/checkpoints}"
export FT_RUN_NAME="${FT_RUN_NAME:-qlora-spear-35b-v1}"
# GGUF adapter name derives from the run name so v1/v2/... stay distinct.
GGUF_NAME="${FT_RUN_NAME#qlora-}-lora.gguf"
export HF_HUB_ENABLE_HF_TRANSFER=1
# IMPORTANT: pod root overlay is tiny (~30 GB). Put the HF cache (~70 GB
# base) and tmp on the big mounted volume, or the download dies with
# "No space left on device". Override WORKDIR_VOL if your volume differs.
WORKDIR_VOL="${WORKDIR_VOL:-/workspace}"
export HF_HOME="${HF_HOME:-$WORKDIR_VOL/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$WORKDIR_VOL/hf/hub}"
export TMPDIR="${TMPDIR:-$WORKDIR_VOL/tmp}"
export HF_HUB_DISABLE_XET=1          # xet has hung before; use hf_transfer
mkdir -p "$HF_HOME" "$TMPDIR"
# avoid CUDA allocator fragmentation while quantize-loading the 35B
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "== base model: $BASE_MODEL =="

echo "== deps =="
# some pod images ship a PEP 668 "externally managed" system python
PIP="pip install -q"
$PIP --help >/dev/null 2>&1
pip install -q hf_transfer 2>/dev/null || PIP="pip install -q --break-system-packages"
# Qwen3.6 is model_type qwen3_5_moe — NOT in any stable transformers release
# (post-cutoff arch). Stable <5 fails with KeyError('qwen3_5_moe'); install
# transformers from source. peft/bnb/etc. from PyPI are fine.
$PIP peft datasets bitsandbytes accelerate hf_transfer sentencepiece
$PIP --upgrade "git+https://github.com/huggingface/transformers.git"
# Blackwell (sm_120): torch 2.8's _grouped_mm is Hopper-only and crashes the
# MoE forward. torch >= 2.9 supports SM80+; upgrade to the cu128 build (bnb
# 0.49.2 stays compatible — verified). Skip if already on >= 2.9.
python3 - <<'PYV' || $PIP --upgrade torch --index-url https://download.pytorch.org/whl/cu128
import torch, sys
maj, mnr = (int(x) for x in torch.__version__.split('+')[0].split('.')[:2])
sys.exit(0 if (maj, mnr) >= (2, 9) else 1)
PYV
# torchvision/torchaudio from the base image pin torch==2.8 and break the
# import chain after the torch upgrade (operator torchvision::nms missing).
# Text QLoRA needs neither — remove them so transformers skips vision cleanly.
pip uninstall -y -q --break-system-packages torchvision torchaudio 2>/dev/null || true
python3 -c "from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES as M; \
    assert 'qwen3_5_moe' in M, 'transformers too old for qwen3_5_moe'; \
    print('transformers OK: qwen3_5_moe supported')"

echo "== smoke test (2 steps) =="
FT_MAX_STEPS=2 python train_qlora_35b.py

echo "== full training =="
python train_qlora_35b.py

echo "== convert adapter to GGUF (no merge needed: llama-server --lora) =="
if [ ! -d llama.cpp ]; then
    git clone --depth 1 https://github.com/ggml-org/llama.cpp
fi
$PIP gguf
python llama.cpp/convert_lora_to_gguf.py \
    "$FT_OUT_DIR/$FT_RUN_NAME" \
    --base-model-id "$BASE_MODEL" \
    --outfile "$GGUF_NAME" \
    --outtype f16

echo "== done =="
ls -lh $GGUF_NAME
echo "Fetch it with:"
echo "  scp root@<pod>:$PWD/$GGUF_NAME /opt/llm/models/"
