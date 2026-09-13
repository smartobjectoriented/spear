#!/bin/bash
# Does this card actually hold the model, loaded the way training will load it?
#
# Answers ONE question, empirically, and trains nothing:
#
#     stop inference -> load with the exact quantization config -> measure the
#     peak -> unload -> restart inference
#
# It exists because the answer has been guessed twice and been wrong twice. A
# documentation figure said one thing, a reading of the installed transformers
# said another, and the run that settled it cost a rented B200 -- for a job
# that, on the stack it actually pinned, would have fitted a far smaller card.
# The rule this file enforces is the one the rest of the harness already
# applies to sandboxes and budgets: measure, do not deduce.
#
#   bash cloud/load_preflight.sh              # 4-bit with MoE experts quantized
#   MOE_QUANT=0 bash cloud/load_preflight.sh  # 4-bit WITHOUT -- the failing case
#   LOAD_DTYPE=bf16 bash cloud/load_preflight.sh
#
# Run it ON the training host. On a shared host that means the reserved card, which
# is also the card serving the model: this script stops the server, and puts it
# back, because 5 GiB free out of 97 answers nothing.
#
# The point is to make renting a card unnecessary. If this says FITS, the
# local card does the fine-tune and no pod is booked; if it says
# OOM, it says so in the ten minutes a load takes rather than in the third hour
# of a paid run.
set -e
cd "$(dirname "$0")/.."

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-Coder-Next}"
BASE_GB="${BASE_GB:-159}"
MOE_QUANT="${MOE_QUANT:-1}"
LOAD_DTYPE="${LOAD_DTYPE:-4bit}"
SERVE_SCRIPT="${SERVE_SCRIPT:-$HOME/spear/serve-model.sh}"
export HF_HOME="${HF_HOME:-$HOME/spear/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# ── disk, before anything else ───────────────────────────────────────────
# The binding constraint may not be the card, it is the filesystem.
# Measured 2026-08-25: /home had 97 GB free of 1399, most of the rest being a
# colleagues' Hugging Face cache that is not ours to purge. The bf16 shards are
# 159 GB, and 4-bit does not change that: bitsandbytes quantizes AT LOAD, from
# shards that must land on disk first. Disk is the cheap constraint -- it is
# bought, not rented by the hour -- so this check exists to name the exact
# number of GB to add, in two seconds, rather than to have it discovered at
# 90 % of a download.
CACHED_DIR="$HF_HUB_CACHE/models--${BASE_MODEL//\//--}"
HAVE_GB=$(du -sBG --apparent-size "$CACHED_DIR" 2>/dev/null | awk '{print $1+0}')
HAVE_GB="${HAVE_GB:-0}"
NEED_GB=$((BASE_GB - HAVE_GB)); [ "$NEED_GB" -lt 0 ] && NEED_GB=0
FREE_GB=$(df -BG --output=avail "$HF_HOME" 2>/dev/null | tail -1 | tr -dc '0-9')
FREE_GB="${FREE_GB:-0}"
echo "== disk =="
echo "   $HF_HOME: ${FREE_GB} GB free, ${HAVE_GB} GB cached, ${NEED_GB} GB to fetch"
if [ "$FREE_GB" -lt "$NEED_GB" ]; then
    SHORT_GB=$((NEED_GB - FREE_GB))
    cat >&2 <<EOF

Not enough disk: ${FREE_GB} GB free where HF_HOME lives, ${NEED_GB} GB still to
fetch for $BASE_MODEL.

   ADD AT LEAST ${SHORT_GB} GB  (allow ${SHORT_GB} + 40 GB: adapters, checkpoints
   and the merged output land on the same filesystem).

4-bit does not reduce this -- the quantization happens at load, from shards
that must be on disk first.

Cheaper than that, if the space is not there yet:
  * HF_HOME=/some/roomier/path bash cloud/load_preflight.sh -- any filesystem
    on this host will do, the loader does not care where the cache is;
  * BASE_MODEL=<a pre-quantized NF4 checkpoint> (~45 GB). Valid for the
    VRAM question, but note it measures THAT checkpoint, not the bf16 one the
    bundle YAML names.

What is NOT an option: purging the caches under other users' home on this
shared machine.
EOF
    exit 1
fi

# ── the card ─────────────────────────────────────────────────────────────
GPU_CONF="${GPU_CONF:-$HOME/spear/spear/active-gpu.conf}"
if [ -z "$FT_GPU" ] && [ -f "$GPU_CONF" ]; then
    FT_GPU=$(tr -d '[:space:]' < "$GPU_CONF")
fi
[ -n "$FT_GPU" ] || { echo "No GPU pinned: set FT_GPU or fill $GPU_CONF" >&2; exit 2; }
export CUDA_VISIBLE_DEVICES="$FT_GPU"

STOPPED=0
if pgrep -f "build/bin/llama-server --model" >/dev/null 2>&1; then
    echo "== stopping inference (it holds the card) =="
    if command -v systemctl >/dev/null 2>&1 && systemctl is-active --quiet spear-llm; then
        sudo -n systemctl stop spear-llm && STOPPED=unit
    else
        SERVER_PID=$(pgrep -f "build/bin/llama-server --model" | head -1)
        kill "$SERVER_PID" 2>/dev/null && STOPPED=pid
    fi
    for _ in $(seq 1 60); do
        pgrep -f "build/bin/llama-server --model" >/dev/null 2>&1 || break
        sleep 2
    done
fi

restore_inference() {
    [ "$STOPPED" = 0 ] && return 0
    echo "== restarting inference =="
    if [ "$STOPPED" = unit ]; then
        sudo -n systemd-run --unit=spear-llm --collect "$SERVE_SCRIPT" >/dev/null 2>&1 || \
            setsid "$SERVE_SCRIPT" >>"$HOME/spear/llama-server.log" 2>&1 < /dev/null &
    else
        setsid "$SERVE_SCRIPT" >>"$HOME/spear/llama-server.log" 2>&1 < /dev/null &
    fi
}
# Put the server back whatever happens below -- including the OOM this script
# exists to provoke. A preflight that leaves the machine without its model is
# worse than no preflight.
trap restore_inference EXIT

nvidia-smi -i "$FT_GPU" --query-gpu=name,memory.total,memory.free \
    --format=csv,noheader | sed 's/^/   GPU /'

# ── the measurement ──────────────────────────────────────────────────────
PY="${FT_VENV:-$PWD/.venv}/bin/python"
[ -x "$PY" ] || PY=python3
BASE_MODEL="$BASE_MODEL" MOE_QUANT="$MOE_QUANT" LOAD_DTYPE="$LOAD_DTYPE" \
    "$PY" cloud/load_preflight.py
