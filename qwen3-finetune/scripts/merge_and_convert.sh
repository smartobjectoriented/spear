#!/usr/bin/env bash
# Merge the trained LoRA adapter into the base model, then convert + quantize
# to GGUF so llama.cpp can serve it.
#
# Usage:  source /opt/llm/qwen3-finetune/activate.sh
#         bash scripts/merge_and_convert.sh [adapter_dir] [quant]
#   adapter_dir : default checkpoints/qlora-spear-v1
#   quant       : default Q5_K_M (matches the current inference setup)
set -e

FT_HOME="/opt/llm/qwen3-finetune"
LLAMA="/opt/llm/llama.cpp"
ADAPTER="${1:-$FT_HOME/checkpoints/qlora-spear-v1}"
QUANT="${2:-Q5_K_M}"
BASE="${BASE_MODEL_DIR:-$FT_HOME/models/Qwen3-8B}"
MERGED="$FT_HOME/checkpoints/merged-qwen3-8b"
# Version the output from the adapter dir name (…-v1 → v1) so a new train
# never overwrites the previous servable GGUF before it's been validated.
VER="$(basename "$ADAPTER")"; VER="${VER##*-}"
OUT_GGUF="/opt/llm/models/Qwen3-8B-${VER}-${QUANT}.gguf"
F16_GGUF="$FT_HOME/checkpoints/merged-qwen3-8b-f16.gguf"

echo "==> [1/3] Merging LoRA adapter into base ($ADAPTER)"
python3 - <<PY
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
base = AutoModelForCausalLM.from_pretrained("$BASE", torch_dtype=torch.float16,
                                            trust_remote_code=True)
model = PeftModel.from_pretrained(base, "$ADAPTER")
model = model.merge_and_unload()
model.save_pretrained("$MERGED", safe_serialization=True)
AutoTokenizer.from_pretrained("$BASE", trust_remote_code=True).save_pretrained("$MERGED")
print("merged ->", "$MERGED")
PY

echo "==> [2/3] Converting merged model to f16 GGUF"
python3 "$LLAMA/convert_hf_to_gguf.py" "$MERGED" \
    --outfile "$F16_GGUF" --outtype f16

echo "==> [3/3] Quantizing to $QUANT -> $OUT_GGUF"
"$LLAMA/build/bin/llama-quantize" "$F16_GGUF" "$OUT_GGUF" "$QUANT"

echo ""
echo "Done: $OUT_GGUF"
echo "To serve it, point spear-chat MODEL= at this file."
