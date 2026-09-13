#!/bin/bash
# Install the RAG stack on a target machine. Idempotent: safe to re-run.
# Run it ON the machine, from anywhere:
#     bash <checkout>/deploy/install.sh
#
# The venv is created INSIDE the checkout (bin/, lib/), as on the original
# workstation: spear-chat.sh calls "$SCRIPT_DIR/bin/python3" by absolute path,
# and a venv cannot be copied — it bakes in its own.
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_BOOT="${PY_BOOT:-python3.12}"
# Blackwell (RTX PRO 6000, sm_120) is not covered by the default PyPI wheels.
# cu130 is, and it is also what runs on the original workstation.
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
TORCH_VERSION="${TORCH_VERSION:-2.12.0}"
export HF_HUB_DISABLE_XET=1     # xet hangs on these networks

step() { printf '\n\033[1m== %s\033[0m\n' "$1"; }

step "interpreter"
command -v "$PY_BOOT" >/dev/null || { echo "$PY_BOOT not found"; exit 1; }
"$PY_BOOT" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3,12) else 1)' \
    || { echo "$PY_BOOT < 3.12 — chromadb 1.5.9 requires it"; exit 1; }
echo "  $PY_BOOT -> $("$PY_BOOT" -V 2>&1)"

step "venv in $APP_DIR"
if [ -x "$APP_DIR/bin/python3" ]; then
    echo "  already present -> $("$APP_DIR/bin/python3" -V 2>&1)"
else
    "$PY_BOOT" -m venv "$APP_DIR"
    echo "  created"
fi
PY="$APP_DIR/bin/python3"
"$PY" -m pip install -q --upgrade pip

step "torch ($TORCH_VERSION from $(basename "$TORCH_INDEX"))"
if "$PY" -c 'import torch' 2>/dev/null; then
    echo "  already installed -> $("$PY" -c 'import torch; print(torch.__version__)')"
else
    "$PY" -m pip install -q "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX"
fi
"$PY" - <<'EOF'
import torch
print(f"  torch {torch.__version__} | cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        cap = torch.cuda.get_device_capability(i)
        print(f"    [{i}] {torch.cuda.get_device_name(i)}  sm_{cap[0]}{cap[1]}")
EOF

step "dependencies"
"$PY" -m pip install -q -r "$APP_DIR/deploy/requirements.txt"
echo "  $("$PY" -m pip list 2>/dev/null | wc -l) packages"

step "embedding model"
EMB="$(cat "$APP_DIR/active-embedder.conf" 2>/dev/null || echo chroma-default)"
echo "  active: $EMB"
if [ "$EMB" != "chroma-default" ]; then
    "$PY" - "$EMB" <<'EOF'
import sys, time
from sentence_transformers import SentenceTransformer
t0 = time.time()
SentenceTransformer(sys.argv[1], device="cpu")   # downloads if absent
print(f"  cached ({time.time()-t0:.0f} s)")
EOF
fi

step "verification"
cd "$APP_DIR"
"$PY" -c "
import sys; sys.path.insert(0, '.')
import embedding, chromadb
print('  active embedder:', embedding.active_model())
c = chromadb.PersistentClient(path='chromadb')
cols = [x for x in c.list_collections() if not x.name.startswith('bench_')]
print(f'  collections    : {len(cols)}')
for col in sorted(cols, key=lambda x: x.name):
    print(f'    {col.name:<24}{col.count():>7}  {(col.metadata or {}).get(\"embed_model\", \"MiniLM\")}')
" 2>/dev/null || echo "  (no index yet — copy them over or rebuild)"
echo
echo "Install complete. Run: $APP_DIR/spear-chat.sh"
