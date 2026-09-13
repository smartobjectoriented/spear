#!/usr/bin/env bash
# Source this to enter the Qwen3-8B fine-tuning workspace:
#   source /opt/llm/qwen3-finetune/activate.sh
export FT_HOME="/opt/llm/qwen3-finetune"
# Keep every HuggingFace download inside this workspace (not ~/.cache).
export HF_HOME="$FT_HOME/models"
export HF_HUB_CACHE="$FT_HOME/models/hub"
# Base model (HF safetensors — required for training, NOT the GGUF).
export BASE_MODEL="Qwen/Qwen3-8B"
export BASE_MODEL_DIR="$FT_HOME/models/Qwen3-8B"
# Source corpus used to build the training data.
# The build-system checkout these builders harvest. No default: it is a
# tree of yours, and a default naming somebody else's is wrong everywhere.
export VERDIN_REPO="${VERDIN_REPO:?set VERDIN_REPO to your build-system checkout}"

if [ -f "$FT_HOME/.venv/bin/activate" ]; then
    source "$FT_HOME/.venv/bin/activate"
fi
cd "$FT_HOME"
echo "Qwen3-8B finetune workspace: $FT_HOME"
echo "  base model dir : $BASE_MODEL_DIR  (present: $([ -d "$BASE_MODEL_DIR" ] && echo yes || echo NO — run setup.sh))"
echo "  HF cache       : $HF_HOME"
