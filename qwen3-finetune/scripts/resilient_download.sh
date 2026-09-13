#!/usr/bin/env bash
# Detached, self-restarting download of the Qwen3-8B base weights.
# Survives suspend / network change / session teardown: keeps retrying
# snapshot_download (which resumes from .incomplete) until all 5 shards land.
# Launch detached:
#   setsid nohup bash scripts/resilient_download.sh >/dev/null 2>&1 &
FT_HOME="/opt/llm/qwen3-finetune"
MODELDIR="$FT_HOME/models/Qwen3-8B"
LOG="$FT_HOME/download.log"
export HF_HOME="$FT_HOME/models"
export HF_HUB_CACHE="$FT_HOME/models/hub"

source "$FT_HOME/.venv/bin/activate"

echo "[$(date '+%F %T')] === resilient download started (pid $$) ===" >> "$LOG"
attempt=0
while true; do
    attempt=$((attempt+1))
    done_count=$(ls "$MODELDIR"/*.safetensors 2>/dev/null | wc -l)
    echo "[$(date '+%F %T')] attempt $attempt — shards complete: $done_count/5" >> "$LOG"
    if [ "$done_count" -ge 5 ]; then
        echo "[$(date '+%F %T')] ALL-SHARDS-PRESENT — done" >> "$LOG"
        du -sh "$MODELDIR" >> "$LOG" 2>&1
        break
    fi
    python3 - >> "$LOG" 2>&1 <<'PY' || echo "[retry] snapshot interrupted" >> "$FT_HOME/download.log"
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen3-8B",
    local_dir="/opt/llm/qwen3-finetune/models/Qwen3-8B",
    allow_patterns=["*.safetensors", "*.json", "*.txt", "tokenizer*", "*.model"],
    max_workers=2,
)
print("SNAPSHOT-RETURNED-OK")
PY
    [ "$attempt" -ge 200 ] && { echo "[$(date '+%F %T')] GIVE-UP after $attempt" >> "$LOG"; break; }
    sleep 5
done
