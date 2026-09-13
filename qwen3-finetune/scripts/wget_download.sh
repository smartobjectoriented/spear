#!/usr/bin/env bash
# Resilient HTTP download of Qwen3-8B safetensors shards via wget (bypasses the
# hf-xet library, which hangs here). wget -c resumes partials; --tries=0 retries
# forever; wrapped in a systemd service with Restart=on-failure as a backstop.
set -u
DIR="/opt/llm/qwen3-finetune/models/Qwen3-8B"
BASE="https://huggingface.co/Qwen/Qwen3-8B/resolve/main"
SHARDS="00001 00002 00003 00004 00005"

for i in $SHARDS; do
    f="model-${i}-of-00005.safetensors"
    echo "[$(date '+%F %T')] fetching $f"
    wget -c --tries=0 --retry-connrefused --waitretry=10 --timeout=60 \
         -q --show-progress \
         "$BASE/$f" -O "$DIR/$f"
done
echo "[$(date '+%F %T')] all shards fetched"
