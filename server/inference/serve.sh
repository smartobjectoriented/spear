#!/bin/bash
# Run a llama.cpp inference server from deployment configuration.
#
# One implementation. There were three: a launcher in the client checkout, a
# copy captured by hand from a running process on the GPU host, and the
# process itself. They disagreed -- the running server held a 98304-token
# context while the restart script would have brought it back at 65536 -- and
# nothing reconciled them, because none of the three was the place where the
# number was decided. Here, every such number comes from configuration and
# nothing is hidden in the script.
#
#     serve.sh [extra llama-server arguments...]
#
# Configuration, for every value:
#
#     SPEAR_SERVER_<NAME> in the environment  >  server.conf  >  nothing
#
# and where "nothing" will not do, this says which value is missing and stops.
# It guesses no path, no port and no context size: a default that happens to
# match one deployment is how the three implementations drifted apart.
#
# It execs llama-server in the foreground and writes no log of its own.
# Whatever supervises it -- a systemd unit, a terminal, the client's --local
# mode -- owns the output, and that is also who owns where the output goes.
set -euo pipefail

KEYS=(LLAMA_BIN MODEL CTX HOST PORT PARALLEL NGL THREADS NCPUMOE LORA GPU_UUID GPU_WAIT)

# The environment must survive the config file, so capture it first and put it
# back afterwards. Sourcing is what lets an operator keep a readable file with
# comments; it is also what would otherwise silently overwrite a deliberate
# one-off override on the command line.
for _k in "${KEYS[@]}"; do
    eval "_env_$_k=\${SPEAR_SERVER_$_k-}"
done

SPEAR_SERVER_ROOT="${SPEAR_SERVER_ROOT:-$HOME/spear-runtime}"
CONF="${SPEAR_SERVER_CONF:-$SPEAR_SERVER_ROOT/config/server.conf}"
GPU_CONF="${SPEAR_SERVER_GPU_CONF:-$SPEAR_SERVER_ROOT/config/gpu.conf}"

# shellcheck source=/dev/null
[ -r "$CONF" ] && . "$CONF"
# shellcheck source=/dev/null
[ -r "$GPU_CONF" ] && . "$GPU_CONF"

for _k in "${KEYS[@]}"; do
    eval "_v=\$_env_$_k"
    [ -n "$_v" ] && eval "SPEAR_SERVER_$_k=\$_v"
done
unset _k _v

# ── what this deployment has not said ────────────────────────────────
missing=()
[ -n "${SPEAR_SERVER_LLAMA_BIN:-}" ] || missing+=("SPEAR_SERVER_LLAMA_BIN   the llama-server binary")
[ -n "${SPEAR_SERVER_MODEL:-}" ]     || missing+=("SPEAR_SERVER_MODEL       absolute path to the GGUF (first shard)")
[ -n "${SPEAR_SERVER_CTX:-}" ]       || missing+=("SPEAR_SERVER_CTX         context window, in tokens")

if [ "${#missing[@]}" -gt 0 ]; then
    {
        echo "serve.sh: this deployment is not configured."
        printf '  missing: %s\n' "${missing[@]}"
        echo "  set them in $CONF, or in the environment."
        echo "  see server/config/server.conf.example for what each one means."
    } >&2
    exit 78                      # EX_CONFIG
fi

[ -x "$SPEAR_SERVER_LLAMA_BIN" ] || {
    echo "serve.sh: not an executable: $SPEAR_SERVER_LLAMA_BIN" >&2
    echo "  build it with server/inference/install-llamacpp.sh" >&2
    exit 78
}
[ -r "$SPEAR_SERVER_MODEL" ] || {
    echo "serve.sh: no readable model: $SPEAR_SERVER_MODEL" >&2
    echo "  fetch it with server/scripts/fetch-model.sh" >&2
    exit 78
}
case "$SPEAR_SERVER_CTX" in
    ''|*[!0-9]*) echo "serve.sh: SPEAR_SERVER_CTX is not a number: $SPEAR_SERVER_CTX" >&2; exit 78 ;;
esac
case "${SPEAR_SERVER_NATIVE_CTX:-0}" in
    *[!0-9]*) echo "serve.sh: SPEAR_SERVER_NATIVE_CTX is not a number: $SPEAR_SERVER_NATIVE_CTX" >&2; exit 78 ;;
esac

# ── the card ─────────────────────────────────────────────────────────
# Pinned before llama.cpp enumerates anything: CUDA reads this at
# initialisation. With --n-gpu-layers 99 and several visible cards, llama.cpp
# spreads the layers over ALL of them, which on a shared host means landing on
# somebody else's GPU. An already-set CUDA_VISIBLE_DEVICES is the caller's.
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && [ -n "${SPEAR_SERVER_GPU_UUID:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$SPEAR_SERVER_GPU_UUID"
fi
echo "serving on GPU ${CUDA_VISIBLE_DEVICES:-<unpinned>}" >&2

# ── the card is there ────────────────────────────────────────────────
# llama.cpp does not stop when CUDA finds no device: it serves the whole model
# from the CPU, two orders of magnitude slower, behind a health check that
# answers ok. At boot the driver can come up after this service starts, so
# wait for the card and refuse to start without it.
#
# Not checked when no layer goes to a GPU, when the caller hid every card
# (-1), or on a host without nvidia-smi, which this cannot speak for.

gpu_ready() {
    local listed wanted w line found
    listed="$(nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null)" || return 1
    [ -n "$listed" ] || return 1
    IFS=, read -ra wanted <<<"${CUDA_VISIBLE_DEVICES:-}"

    for w in "${wanted[@]}"; do
        case "$w" in
            GPU-*)
                found=""
                while read -r line; do
                    case "$line" in "$w"*) found=1 ;; esac
                done <<<"$listed"
                [ -n "$found" ] || return 1 ;;
            ''|*[!0-9]*) ;;
            *) [ "$w" -lt "$(wc -l <<<"$listed")" ] || return 1 ;;
        esac
    done
}

if [ "${SPEAR_SERVER_NGL:-99}" != 0 ] && [ "${CUDA_VISIBLE_DEVICES:-}" != -1 ] \
        && command -v nvidia-smi >/dev/null; then
    WAIT="${SPEAR_SERVER_GPU_WAIT:-300}"
    case "$WAIT" in
        ''|*[!0-9]*) echo "serve.sh: SPEAR_SERVER_GPU_WAIT is not a number: $WAIT" >&2; exit 78 ;;
    esac
    deadline=$((SECONDS + WAIT))

    until gpu_ready; do
        if [ "$SECONDS" -ge "$deadline" ]; then
            echo "serve.sh: GPU ${CUDA_VISIBLE_DEVICES:-<any>} not available after ${WAIT}s;" \
                 "not serving from the CPU" >&2
            echo "  check the driver with nvidia-smi, or set SPEAR_SERVER_NGL=0 to serve" \
                 "from the CPU on purpose" >&2
            exit 69              # EX_UNAVAILABLE
        fi
        sleep 5
    done
fi

# The build links shared libraries beside the binary.
export LD_LIBRARY_PATH="$(dirname "$SPEAR_SERVER_LLAMA_BIN")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ── mixture-of-experts offload ───────────────────────────────────────
# A MoE model can keep its experts in CPU RAM so a small GPU holds only
# attention and KV; a dense model cannot, and passing --n-cpu-moe to one is
# meaningless. Guessed from the filename, because that is the only thing we
# have before the weights are opened -- and overridable for when the guess is
# wrong or the GPU is large enough not to care.
if [ -n "${SPEAR_SERVER_NCPUMOE:-}" ]; then
    NCPUMOE="$SPEAR_SERVER_NCPUMOE"
else
    case "$(basename "$SPEAR_SERVER_MODEL")" in
        *A3B*|*a3b*|*MoE*|*moe*|*Next*|*next*) NCPUMOE=99 ;;
        *)                                     NCPUMOE=0  ;;
    esac
fi

LORA="${SPEAR_SERVER_LORA:-}"
[ "$LORA" = "none" ] && LORA=""
if [ -n "$LORA" ] && [ ! -r "$LORA" ]; then
    echo "serve.sh: adapter not readable, serving the base model: $LORA" >&2
    LORA=""
fi

ARGS=(--model "$SPEAR_SERVER_MODEL"
      --ctx-size "$SPEAR_SERVER_CTX"
      --parallel "${SPEAR_SERVER_PARALLEL:-1}"
      --n-gpu-layers "${SPEAR_SERVER_NGL:-99}"
      --flash-attn on
      --cache-type-k q8_0
      --cache-type-v q8_0
      --host "${SPEAR_SERVER_HOST:-127.0.0.1}"
      --port "${SPEAR_SERVER_PORT:-8080}"
      --jinja)

# Unset means "let llama.cpp decide for this host", which is usually better
# than a number chosen on a different one.
[ -n "${SPEAR_SERVER_THREADS:-}" ] && ARGS+=(--threads "$SPEAR_SERVER_THREADS")
[ "$NCPUMOE" -gt 0 ] 2>/dev/null && ARGS+=(--n-cpu-moe "$NCPUMOE")
[ -n "$LORA" ] && ARGS+=(--lora "$LORA")

# ── context beyond the trained window ────────────────────────────────
# Past the length the model was trained at, RoPE positions it has never seen
# degrade attention; YaRN rescales them. It is static: the scaling applies to
# every request, short ones included, so it is switched on only when the
# window asked for actually exceeds the trained one.
if [ "${SPEAR_SERVER_NATIVE_CTX:-0}" -gt 0 ] \
        && [ "$SPEAR_SERVER_CTX" -gt "$SPEAR_SERVER_NATIVE_CTX" ]; then
    SCALE=$(awk -v c="$SPEAR_SERVER_CTX" -v n="$SPEAR_SERVER_NATIVE_CTX" \
                'BEGIN { printf "%g", c / n }')
    # llama-server caps each slot at the trained length it reads from the
    # GGUF, YaRN or not: measured, --ctx-size 524288 served 262144. Raising
    # that metadata key is what lets the slot use the window; the key is
    # named after the architecture, which a shell cannot read from the file.
    [ -n "${SPEAR_SERVER_ARCH:-}" ] || {
        echo "serve.sh: SPEAR_SERVER_ARCH is required past the trained context" >&2
        echo "  (the GGUF general.architecture, e.g. qwen3next)" >&2
        exit 78
    }
    ARGS+=(--rope-scaling yarn --rope-scale "$SCALE"
           --yarn-orig-ctx "$SPEAR_SERVER_NATIVE_CTX"
           --override-kv "$SPEAR_SERVER_ARCH.context_length=int:$SPEAR_SERVER_CTX")
    echo "context $SPEAR_SERVER_CTX > trained $SPEAR_SERVER_NATIVE_CTX: YaRN x$SCALE" >&2
fi

exec "$SPEAR_SERVER_LLAMA_BIN" "${ARGS[@]}" "$@"
