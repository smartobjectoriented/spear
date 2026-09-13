#!/bin/bash
# Build llama.cpp with CUDA on this machine, and optionally fetch the weights.
#
#     install-llamacpp.sh [--no-model]
#
# Where things go is not this script's decision. The build location is derived
# from SPEAR_SERVER_LLAMA_BIN -- the same value serve.sh runs -- so the two
# cannot disagree about where the binary is, and there is no second variable
# to keep in step. The weights go wherever fetch-model.sh is told to put them.
#
# Nothing is built inside the Git checkout. A repository that also holds 80 GB
# of weights and a CMake build tree is a repository nobody can clone, and the
# server tree is meant to be public.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${SPEAR_SERVER_LLAMA_BIN:-}"

[ -n "$BIN" ] || {
    {
        echo "install-llamacpp.sh: SPEAR_SERVER_LLAMA_BIN is not set."
        echo "  It names the llama-server binary to produce, and the build"
        echo "  location is derived from it:"
        echo "      <src>/build/bin/llama-server  ->  <src>"
        echo "  see server/config/server.conf.example."
    } >&2
    exit 78
}

case "$BIN" in
    */build/bin/llama-server) SRC="${BIN%/build/bin/llama-server}" ;;
    *)  echo "install-llamacpp.sh: expected .../build/bin/llama-server, got: $BIN" >&2
        echo "  the build location is derived from this path." >&2
        exit 78 ;;
esac

step() { printf '\n\033[1m== %s\033[0m\n' "$1"; }

step "CUDA compiler"
# Blackwell (sm_120) needs CUDA >= 12.8, and /usr/bin/nvcc is often a stale
# distro package reporting an older release than the toolkit beside it. Pick
# explicitly rather than trusting whatever is first on PATH.
NVCC=""
for C in /usr/local/cuda/bin/nvcc /usr/local/cuda-13*/bin/nvcc /usr/local/cuda-12.8/bin/nvcc; do
    [ -x "$C" ] || continue
    V=$("$C" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')
    MAJ=${V%%.*}; MIN=${V#*.}
    if [ "$MAJ" -gt 12 ] || { [ "$MAJ" -eq 12 ] && [ "${MIN%%.*}" -ge 8 ]; }; then
        NVCC="$C"; echo "  $C -> CUDA $V"; break
    fi
done
[ -n "$NVCC" ] || { echo "  no nvcc >= 12.8 (sm_120 impossible)" >&2; exit 1; }
export CUDACXX="$NVCC"
PATH="$(dirname "$NVCC"):$PATH"; export PATH

step "llama.cpp -> $SRC"
if [ -x "$BIN" ]; then
    echo "  already built"
else
    mkdir -p "$(dirname "$SRC")"
    [ -d "$SRC/.git" ] || git clone -q https://github.com/ggml-org/llama.cpp "$SRC"
    cmake -S "$SRC" -B "$SRC/build" \
        -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="${SPEAR_SERVER_CUDA_ARCH:-120}" \
        -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release >/dev/null
    cmake --build "$SRC/build" -j"$(nproc)" --target llama-server >/dev/null
    echo "  built -> $BIN"
fi
LD_LIBRARY_PATH="$(dirname "$BIN")" "$BIN" --version 2>&1 | head -2 | sed 's/^/  /'

if [ "${1:-}" = "--no-model" ]; then
    echo; echo "Model fetch skipped."
else
    step "model"
    # One implementation of "download a sharded GGUF", not a second one here.
    "$HERE/../scripts/fetch-model.sh"
fi

echo
echo "Serve with:  server/inference/serve.sh"
echo "  configure it first -- see server/config/server.conf.example"
