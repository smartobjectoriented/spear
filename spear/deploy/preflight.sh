#!/bin/bash
# Survey a target machine BEFORE deploying the RAG stack. Changes nothing,
# installs nothing. Run it on the machine, paste the output back.
#
#   ssh <host> 'bash -s' < deploy/preflight.sh
#
# Every line answers a question the install procedure depends on.
set -u
ok()   { printf '  \033[32mOK\033[0m   %s\n' "$1"; }
warn() { printf '  \033[33mSEE\033[0m  %s\n' "$1"; }
bad()  { printf '  \033[31mNO\033[0m   %s\n' "$1"; }

echo "== machine =="
echo "  $(hostname) — $(uname -sr) — $(nproc) cores"
echo "  RAM: $(free -g 2>/dev/null | awk '/^Mem:/{print $2" GB"}')"

echo "== GPU (decides where the embedder and inference run) =="
if command -v nvidia-smi >/dev/null; then
    echo "  all visible GPUs:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,uuid \
               --format=csv,noheader | sed 's/^/    /'
    echo "  driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
    # List ALL toolkits: the nvcc on PATH is often a stale distro package (on
    # reds-ml it reports 12.0 while /usr/local/cuda is 13.0), and sm_120 needs
    # >= 12.8.
    echo "  CUDA compilers (sm_120 needs >= 12.8):"
    for C in /usr/bin/nvcc /usr/local/cuda/bin/nvcc /usr/local/cuda-1*/bin/nvcc; do
        [ -x "$C" ] && echo "    $C -> $("$C" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')"
    done | sort -u
    command -v cmake >/dev/null && echo "    cmake $(cmake --version | head -1 | awk '{print $3}')" \
        || bad "cmake missing — llama.cpp cannot be built"
    # Shared machine: check the allocated GPU REALLY exists, and see what is
    # already running on it. A missing UUID means wrong machine, or a GPU
    # pulled out — better known before installing 8 GB of torch.
    if [ -n "${SPEAR_GPU_UUID:-}" ]; then
        if nvidia-smi --query-gpu=uuid --format=csv,noheader | grep -qF "$SPEAR_GPU_UUID"; then
            ok "allocated GPU present: $SPEAR_GPU_UUID"
            nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu \
                       --format=csv,noheader -i "$SPEAR_GPU_UUID" | sed 's/^/    /'
            echo "    processes already on it:"
            nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader \
                       -i "$SPEAR_GPU_UUID" 2>/dev/null | sed 's/^/      /' \
                || echo "      (none)"
        else
            bad "allocated GPU NOT FOUND: $SPEAR_GPU_UUID"
        fi
    else
        warn "SPEAR_GPU_UUID not supplied — shared machine, a GPU must be pinned"
    fi
else
    bad "nvidia-smi missing — bge-m3 will have to index on CPU (slow but working)"
fi

echo "== python =="
for P in python3.12 python3.13 python3; do
    command -v $P >/dev/null && echo "  $P -> $($P -V 2>&1)"
done
python3 -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3,12) else 1)' \
    && ok "python3 >= 3.12" || bad "python3 < 3.12 — chromadb 1.5.9 requires 3.12+"

echo "== tools =="
for T in git curl rsync bwrap; do
    if command -v $T >/dev/null; then ok "$T present"
    elif [ "$T" = bwrap ]; then
        bad "bwrap missing — the chat's bash tool cannot be sandboxed (blocking in --bypass mode)"
    else bad "$T missing"; fi
done

echo "== disk =="
for D in "$HOME" /opt /tmp; do
    [ -d "$D" ] && echo "  $D: $(df -h "$D" 2>/dev/null | awk 'NR==2{print $4" free"}')"
done
echo "  (~15 GB needed: 2.2 GB of index + ~2.5 GB of bge-m3 + ~8 GB of venv with torch)"

echo "== network =="
for U in https://huggingface.co https://pypi.org ${PREFLIGHT_EXTRA_URLS:-}; do
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 8 "$U" 2>/dev/null)
    [ "$code" = 200 ] || [ "$code" = 302 ] && ok "$U ($code)" || bad "$U (${code:-timeout}) — no direct download"
done

echo "== source trees (decides: rebuild the indexes, or copy the index) =="
# Whatever THIS machine has registered, not a list of somebody's checkouts.
REG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/projects.json"
if [ -r "$REG" ]; then
    python3 -c 'import json,sys; print("\n".join(s["path"] for s in json.load(open(sys.argv[1])).values() if "path" in s))' "$REG" \
    | while read -r D; do [ -d "$D" ] && ok "$D" || warn "$D missing"; done
else
    warn "no projects.json yet — nothing registered to check"
fi
echo
echo "== already deployed? =="
APP_NAME="$(basename "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)")"
for D in "$HOME/spear" /opt/llm/spear; do
    [ -d "$D/$APP_NAME" ] && warn "$D already exists" || true
done
