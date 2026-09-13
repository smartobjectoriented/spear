#!/bin/bash
# SPEAR — QLoRA run for the DENSE Qwen2.5-Coder-32B-Instruct.
#
# Much simpler than the A3B launcher: Coder-32B is a qwen2 arch (in stable
# transformers — NO source build), dense (no MoE _grouped_mm, torch 2.8 cu128
# is fine), and vLLM serves the LoRA adapter directly (NO GGUF conversion).
#
# Run ON the pod (the same RTX PRO 6000 that serves vLLM). STOP vLLM first to
# free VRAM:  tmux kill-session -t vllm. 4-bit NF4 base ~18 GB + LoRA fits
# easily. Bring back only the adapter dir (~150-400 MB).
#
# Objective: EDIT DISCIPLINE — see scripts/build_edit_discipline.py.
set -e
cd "$(dirname "$0")"

export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-Coder-32B-Instruct}"
export FT_DATA_DIR="${FT_DATA_DIR:-$PWD/data}"
export FT_OUT_DIR="${FT_OUT_DIR:-$PWD/checkpoints}"
export FT_RUN_NAME="${FT_RUN_NAME:-qlora-coder32-edit-v1}"
export FT_TARGETS="${FT_TARGETS:-attn+mlp}"   # dense: q,k,v,o + gate,up,down
export FT_RANK="${FT_RANK:-16}"
export FT_EPOCHS="${FT_EPOCHS:-4}"
# full-file context in user turn + edits in assistant turn → long samples
export FT_MAX_LEN="${FT_MAX_LEN:-4096}"
export FT_BATCH="${FT_BATCH:-1}"
export FT_ACCUM="${FT_ACCUM:-8}"
# train on the edit-discipline set (override FT_TRAIN/FT_VAL via the py env)
export FT_TRAIN_FILE="${FT_TRAIN_FILE:-edit_discipline_train.jsonl}"
export FT_VAL_FILE="${FT_VAL_FILE:-edit_discipline_val.jsonl}"

WORKDIR_VOL="${WORKDIR_VOL:-/workspace}"
export HF_HOME="${HF_HOME:-$WORKDIR_VOL/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$WORKDIR_VOL/hf/hub}"
export TMPDIR="${TMPDIR:-$WORKDIR_VOL/tmp}"
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$HF_HOME" "$TMPDIR"

echo "== base: $BASE_MODEL  targets: $FT_TARGETS  rank: $FT_RANK =="

echo "== deps =="
PIP="pip install -q --break-system-packages"
# Coder-32B (qwen2) is supported by the transformers already on the pod
# (4.57.x from the vLLM serve). Just add the trainer stack. bitsandbytes
# >= 0.45 supports Blackwell sm_120.
$PIP peft "datasets" accelerate sentencepiece
$PIP "bitsandbytes>=0.46"
python3 -c "import torch,bitsandbytes,peft,datasets,transformers as t; \
  print('torch',torch.__version__,'bnb ok','transformers',t.__version__)"

echo "== smoke test (2 steps) =="
FT_MAX_STEPS=2 python3 train_qlora_coder32.py

echo "== full training =="
python3 train_qlora_coder32.py

echo "== done — adapter at: $FT_OUT_DIR/$FT_RUN_NAME =="
ls -lh "$FT_OUT_DIR/$FT_RUN_NAME" 2>/dev/null
echo "Serve it with vLLM:  vllm serve $BASE_MODEL --enable-lora \\"
echo "   --lora-modules edit=$FT_OUT_DIR/$FT_RUN_NAME ..."
