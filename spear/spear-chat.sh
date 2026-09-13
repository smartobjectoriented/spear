#!/bin/bash
# HEIG-VD/REDS AI — launcher with a LOCAL / REMOTE backend selector.
#
#   spear-chat              # LOCAL  : laptop llama-server on :8080 (default)
#   spear-chat --local      #   (explicit)
#   spear-chat --remote     # REMOTE : pod vLLM via SSH tunnel on :8081
#
# Local and remote use DIFFERENT local ports (8080 vs 8081) so they never
# clash — you can even keep a local model and the pod tunnel up at once.
# Remote reads pod.conf (host/port/key/model); update it when the pod's SSH
# port changes. Any other flags (--bypass-permissions, -y, …) pass through.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Call the venv interpreter by path instead of relying on `activate`: the venv
# hardcodes its own absolute path, so a relocated tree activates into a
# directory that no longer exists and `python3` silently falls through to the
# system one — which lacks every dependency.  A direct path cannot drift.
PY="$SCRIPT_DIR/bin/python3"
[ -x "$PY" ] || { echo "venv interpreter missing: $PY"; exit 1; }
export VIRTUAL_ENV="$SCRIPT_DIR"
export PATH="$SCRIPT_DIR/bin:$PATH"

# Settings specific to THIS machine (untracked). Used to point at an inference
# endpoint that is already up: on a shared host the --local :8080 may well be
# taken by something else entirely.
[ -r "$SCRIPT_DIR/machine.env" ] && . "$SCRIPT_DIR/machine.env"

# ── move under the user manager when needed ──
# The sandbox's resource control needs delegated cgroup controllers, and
# systemd delegates them only to user@<uid>.service. An SSH session lands in
# user-<uid>.slice/session-N.scope — a SIBLING of user@.service, not a child:
# no controller delegated, and the bash tool fails closed ("cgroup controllers
# not delegated"). In a graphical session the problem does not exist, hence a
# breakage that only shows over ssh.
# So we re-exec into a transient scope under user@.service. No root, and
# nothing already running is touched.
if [ -z "${SPEAR_IN_USER_SCOPE:-}" ] \
   && ! grep -q "user@$(id -u)\.service" /proc/self/cgroup 2>/dev/null \
   && command -v systemd-run >/dev/null 2>&1 \
   && systemd-run --user --scope --quiet -- true >/dev/null 2>&1; then
    export SPEAR_IN_USER_SCOPE=1     # recursion guard
    exec systemd-run --user --scope --quiet -- "$0" "$@"
fi

# ── parse the backend selector + optional pod overrides, keep the rest for
#    rag_chat.py. --pod-host / --pod-port let you point at a pod on the fly
#    (e.g. after a restart changed the port) without editing pod.conf.
MODE=""
ARGS=()
CLI_HOST=""; CLI_PORT=""
while [ $# -gt 0 ]; do
    case "$1" in
        # Answered here and now: --help must not index a corpus, start
        # llama-server, or open an SSH tunnel on the way to printing text.
        --help|-h)      exec "$PY" "$SCRIPT_DIR/rag_chat.py" --help ;;
        --reds)         MODE=reds ;;
        --remote|--pod) MODE=remote ;;
        --local)        MODE=local ;;
        --pod-host)     CLI_HOST="$2"; MODE=remote; shift ;;
        --pod-port)     CLI_PORT="$2"; MODE=remote; shift ;;
        *)              ARGS+=("$1") ;;
    esac
    shift
done

# ── endpoint already designated? provision nothing over it ──
# Same rule as for CUDA_VISIBLE_DEVICES: what the caller set explicitly is a
# choice, not a default to correct. Without this, --local would start a
# llama-server on an already-taken :8080, or the reds mode would open a tunnel
# to the very machine we are running on.
if [ -n "${SPEAR_API_BASE:-}" ] && [ -z "$MODE" ]; then
    echo "→ endpoint pinned: $SPEAR_API_BASE (model ${SPEAR_MODEL_NAME:-?}) — nothing to start"
    exec "$PY" "$SCRIPT_DIR/rag_chat.py" "${ARGS[@]}"
fi

# ── which backend? an explicit flag wins; otherwise ask on a TTY, or reuse the
#    last choice. The picker writes its own persistence and prints ONE token.
CHOICE=$("$PY" "$SCRIPT_DIR/backend_select.py" "${ARGS[@]}" ${MODE:+--$MODE})
BACKEND="${CHOICE%%:*}"
CHOSEN_MODEL="${CHOICE#*:}"; [ "$CHOSEN_MODEL" = "$CHOICE" ] && CHOSEN_MODEL=""
case "$BACKEND" in
    remote) MODE=remote ;;
    reds)   MODE=reds ;;
    local)  MODE=local ;;
    anthropic)
        MODE=anthropic
        # Only add the flags the user did not already pass, so an explicit
        # --model is never overridden by the remembered one.
        case " ${ARGS[*]} " in *" --provider"*) ;; *) ARGS+=(--provider anthropic) ;; esac
        if [ -n "$CHOSEN_MODEL" ]; then
            case " ${ARGS[*]} " in *" --model"*) ;; *) ARGS+=(--model "$CHOSEN_MODEL") ;; esac
        fi ;;
    # Fail closed. Anything else -- an empty answer because the picker died, a
    # token with the interactive prompt glued to it -- used to match no case,
    # leave MODE empty, and fall through to the local branch, which started a
    # llama-server nobody asked for. Silently choosing a different model than
    # the one selected is the worst possible way to be wrong here.
    *)  echo "backend selector returned an unusable answer: '$CHOICE'" >&2
        echo "Refusing to guess. Rerun with --reds, --local, --remote or" >&2
        echo "--provider anthropic, or report this." >&2
        exit 1 ;;
esac

DB_PATH="$SCRIPT_DIR/chromadb"
if [ ! -d "$DB_PATH" ]; then
    echo "First launch — indexing the corpus..."
    "$PY" "$SCRIPT_DIR/index_corpus.py"; echo ""
fi

if [ "$MODE" = anthropic ]; then
    # Nothing local to prepare: the model lives at Anthropic. Starting
    # llama-server or opening the pod tunnel here would provision a backend
    # rag_chat is about to ignore entirely.
    echo "→ ANTHROPIC ${CHOSEN_MODEL:-(default model)} — no local server started"
elif [ "$MODE" = reds ]; then
    # ── REDS: stable host, tunnel only ──
    # Deliberately NOT the pod path: reds-server keeps its SSH port across
    # restarts, so there is nothing to re-enter, and we never install, upload
    # or start anything there — start the inference server yourself.
    # shellcheck disable=SC1090
    [ -f "$SCRIPT_DIR/reds.conf" ] && source "$SCRIPT_DIR/reds.conf"
    REDS_HOST="${REDS_HOST:-reds-server}"; REDS_PORT="${REDS_PORT:-8000}"
    # Extra ssh options from reds.conf, so the target is self-contained rather
    # than depending on the operator's personal ~/.ssh/config. Needed on hosts
    # with a low MaxAuthTries: without IdentitiesOnly, ssh offers every key in
    # ~/.ssh and is disconnected before it reaches the right one.
    read -r -a REDS_SSH_ARGS <<< "${REDS_SSH_OPTS:-}"
    LPORT=8082                       # 8080 local, 8081 pod, 8082 reds
    export SPEAR_API_BASE="http://127.0.0.1:$LPORT/v1"
    export SPEAR_MODEL_NAME="${REDS_MODEL:-qwen3}"
    if curl -sf "http://127.0.0.1:$LPORT/v1/models" >/dev/null 2>&1; then
        echo "REDS tunnel already up on :$LPORT → $REDS_HOST"
    else
        echo "Opening SSH tunnel :$LPORT → $REDS_HOST:$REDS_PORT …"
        ssh -o ConnectTimeout=12 -o ExitOnForwardFailure=yes \
            -o ServerAliveInterval=30 "${REDS_SSH_ARGS[@]}" \
            -L "$LPORT:localhost:$REDS_PORT" \
            -N -f "$REDS_HOST" || {
                echo "tunnel failed — check that '$REDS_HOST' resolves in"
                echo "~/.ssh/config and that the host is reachable."; exit 1; }
    fi
    if ! curl -sf "http://127.0.0.1:$LPORT/v1/models" >/dev/null 2>&1; then
        echo "  tunnel open, but nothing is serving on $REDS_HOST:$REDS_PORT."
        echo "  Start the inference server there, then relaunch. (REDS_PORT in"
        echo "  reds.conf: 8000 for vLLM, 8080 for llama-server.)"
        exit 1
    fi
    # Name the host we actually tunnelled to, and hand it to the banner —
    # rag_chat cannot infer it from a port number.
    export SPEAR_BACKEND_LABEL="${REDS_HOST##*@} (tunnel :$LPORT)"
    # Take the context window from the server rather than from a local
    # default. The harness trimmed every prompt to 32k while the remote
    # llama-server was serving 65k, silently wasting half the window — and a
    # value copied into reds.conf would drift the day the server is restarted
    # with different flags. An explicit SPEAR_CTX still wins.
    if [ -z "${SPEAR_CTX:-}" ]; then
        SRV_CTX=$(curl -sf --max-time 5 "http://127.0.0.1:$LPORT/props" 2>/dev/null \
            | "$PY" -c 'import sys,json;print(json.load(sys.stdin).get("default_generation_settings",{}).get("n_ctx",""))' 2>/dev/null)
        case "$SRV_CTX" in
            ''|*[!0-9]*) ;;                       # no answer, or not a number
            *) export SPEAR_CTX="$SRV_CTX"
               echo "  context window: $SPEAR_CTX (from the server)" ;;
        esac
    fi
    echo "→ REDS model '$SPEAR_MODEL_NAME' on $REDS_HOST"
elif [ "$MODE" = remote ]; then
    # ── REMOTE: SSH tunnel laptop:8081 -> pod:8080 (vLLM) ──
    # pod.conf holds defaults; --pod-host/--pod-port override; if the pod is
    # unreachable we PROMPT for host/port (so a restarted pod's new port never
    # forces a manual pod.conf edit) and offer to save them.
    # shellcheck disable=SC1090
    [ -f "$SCRIPT_DIR/pod.conf" ] && source "$SCRIPT_DIR/pod.conf"
    # No default key name here: which identity reaches a pod is a
    # property of whoever runs it, and pod.conf above is where it is
    # said. Unset, ssh uses the caller's own ~/.ssh/config.
    POD_KEY=$(eval echo "${POD_KEY:-}")   # expand $HOME / ~ if set
    # `-i ""` is an error, not a no-op, so the flag appears only with a value.
    POD_ID=(); [ -n "$POD_KEY" ] && POD_ID=(-i "$POD_KEY")
    [ -n "$CLI_HOST" ] && POD_HOST="$CLI_HOST"
    [ -n "$CLI_PORT" ] && POD_PORT="$CLI_PORT"
    LPORT=8081
    export SPEAR_API_BASE="http://127.0.0.1:$LPORT/v1"
    export SPEAR_MODEL_NAME="${POD_MODEL:-qwen3}"

    norm_host() {           # accept user@host:port / host:port / ip, add root@
        case "$1" in *:*) POD_PORT="${1##*:}"; POD_HOST="${1%:*}";; *) POD_HOST="$1";; esac
        case "$POD_HOST" in *@*) ;; *) POD_HOST="root@$POD_HOST";; esac
    }
    norm_host "$POD_HOST"
    SSH_OPTS=(-n -o StrictHostKeyChecking=no -o ConnectTimeout=12 "${POD_ID[@]}" -p "$POD_PORT")

    # pre-flight: reachable? if not (or unset), ask for IP/port interactively.
    if [ -z "$POD_HOST" ] || [ -z "$POD_PORT" ] || \
       ! ssh "${SSH_OPTS[@]}" -o BatchMode=yes "$POD_HOST" true 2>/dev/null; then
        echo "Pod unreachable at ${POD_HOST:-?}:${POD_PORT:-?} — enter its SSH access"
        echo "(from the RunPod console 'Connect' button):"
        read -r -p "  host (user@ip or ip:port): " _h
        [ -n "$_h" ] && norm_host "$_h"
        read -r -p "  port [$POD_PORT]: " _p
        [ -n "$_p" ] && POD_PORT="$_p"
        SSH_OPTS=(-n -o StrictHostKeyChecking=no -o ConnectTimeout=12 "${POD_ID[@]}" -p "$POD_PORT")
        if ssh "${SSH_OPTS[@]}" -o BatchMode=yes "$POD_HOST" true 2>/dev/null; then
            read -r -p "  reachable ✓ — save to pod.conf? [Y/n]: " _s
            case "$_s" in n|N) ;; *)
                sed -i "s|^POD_HOST=.*|POD_HOST=$POD_HOST|; s|^POD_PORT=.*|POD_PORT=$POD_PORT|" \
                    "$SCRIPT_DIR/pod.conf" 2>/dev/null && echo "  saved." ;;
            esac
        else
            echo "  still unreachable — check the access and the pod state."; exit 1
        fi
    fi

    # 1) tunnel laptop:8081 -> pod:8080. Opened even before vLLM is up — the
    #    forward is lazy, curl just gets "refused" until the server listens.
    if curl -sf "http://127.0.0.1:$LPORT/health" >/dev/null 2>&1; then
        echo "Remote tunnel already up on :$LPORT → $POD_HOST"
    else
        echo "Opening SSH tunnel :$LPORT → $POD_HOST:$POD_PORT …"
        ssh "${SSH_OPTS[@]}" -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
            -L "$LPORT:localhost:8080" -N -f "$POD_HOST" || {
                echo "tunnel failed — the SSH port changes on every pod restart:"
                echo "update POD_PORT in pod.conf, and check the pod is running."
                exit 1; }
    fi

    # 2) PREPARE THE POD + auto-start vLLM if it isn't serving yet (first
    #    launch on a fresh or restarted pod). On a brand-new pod the rootfs is
    #    empty, so push the serve script if it's missing (prefer the persistent
    #    /workspace volume), then launch it. vLLM install + cached-weight load
    #    is ~8-10 min; a warm pod is instant.
    if ! curl -sf "http://127.0.0.1:$LPORT/health" >/dev/null 2>&1; then
        echo "vLLM not serving — preparing the pod (first launch)…"
        # The script that starts the inference server ON the pod. It is
        # deployment glue -- it names a model, a port and a vLLM invocation --
        # so it is configured (pod.conf: POD_SERVE_SCRIPT) rather than assumed
        # to sit at one path in one person's ~/.local/bin, which is where it
        # used to be looked for after that file had been moved elsewhere.
        SERVE_LOCAL="${POD_SERVE_SCRIPT:-}"
        SERVE_NAME="$(basename "${SERVE_LOCAL:-serve-pod.sh}")"
        DEST=$(ssh "${SSH_OPTS[@]}" "$POD_HOST" '[ -d /workspace ] && echo /workspace || echo /root' 2>/dev/null)
        DEST=${DEST:-/root}
        if ! ssh "${SSH_OPTS[@]}" "$POD_HOST" "test -x $DEST/$SERVE_NAME" 2>/dev/null; then
            if [ -z "$SERVE_LOCAL" ] || [ ! -r "$SERVE_LOCAL" ]; then
                echo "  (no serve script on the pod, and POD_SERVE_SCRIPT names"
                echo "   none here — set it in pod.conf, or start the server on"
                echo "   the pod yourself)"
            else
                echo "  · serve script missing on the pod — pushing it → $DEST"
                scp -o StrictHostKeyChecking=no -o ConnectTimeout=15 "${POD_ID[@]}" \
                    -P "$POD_PORT" "$SERVE_LOCAL" \
                    "$POD_HOST:$DEST/$SERVE_NAME" >/dev/null 2>&1 \
                    && ssh "${SSH_OPTS[@]}" "$POD_HOST" "chmod +x $DEST/$SERVE_NAME" \
                    || echo "  (warning: could not copy the serve script)"
            fi
        fi
        echo "  · launching vLLM on the pod"
        ssh "${SSH_OPTS[@]}" "$POD_HOST" "
            tmux has-session -t vllm 2>/dev/null && exit 0
            tmux new-session -d -s vllm '$DEST/$SERVE_NAME 2>&1 | tee /root/vllm.log'
        " || echo "  (warning: could not start vLLM over SSH — check the pod)"
        echo -n "  Loading the model (~8-10 min on first start, cached weights)"
        for i in $(seq 1 240); do
            curl -sf "http://127.0.0.1:$LPORT/health" >/dev/null 2>&1 \
                && { echo " OK!"; break; }
            echo -n "."; sleep 5
        done; echo ""
    fi
    echo "→ REMOTE model '$SPEAR_MODEL_NAME' on the pod"
else
    # ── LOCAL: laptop llama-server on :8080 (persistent) ──
    export SPEAR_API_BASE="http://127.0.0.1:8080/v1"
    if curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1; then
        echo "Local llama-server already running on :8080"
    else
        echo "Starting local llama-server (active model profile)…"

        # The inference server is generic and takes everything from
        # SPEAR_SERVER_*; the client's own profiles ARE that configuration
        # here. Deciding which model and which adapter is a client matter --
        # `/model` and `spear-model` write these files -- so the mapping
        # happens on this side rather than the server growing an opinion
        # about a checkout it should know nothing about.
        SERVE="$SCRIPT_DIR/../server/inference/serve.sh"
        [ -x "$SERVE" ] || {
            echo "no inference server at $SERVE" >&2
            echo "  --local needs the server/ tree of this repository." >&2
            exit 1; }

        : "${SPEAR_SERVER_LLAMA_BIN:=$SCRIPT_DIR/../llama.cpp-next/build/bin/llama-server}"
        if [ -z "${SPEAR_SERVER_MODEL:-}" ]; then
            if [ -n "${SPEAR_MODEL:-}" ]; then
                SPEAR_SERVER_MODEL="$SPEAR_MODEL"
            elif [ -r "$SCRIPT_DIR/active-model.conf" ]; then
                SPEAR_SERVER_MODEL="$(cat "$SCRIPT_DIR/active-model.conf")"
            fi
        fi
        if [ -z "${SPEAR_SERVER_LORA:-}" ]; then
            if [ -n "${SPEAR_LORA+x}" ]; then
                SPEAR_SERVER_LORA="$SPEAR_LORA"
            elif [ -r "$SCRIPT_DIR/active-lora.conf" ]; then
                SPEAR_SERVER_LORA="$(cat "$SCRIPT_DIR/active-lora.conf")"
            fi
        fi
        if [ -z "${SPEAR_SERVER_GPU_UUID:-}" ] && [ -r "$SCRIPT_DIR/active-gpu.conf" ]; then
            SPEAR_SERVER_GPU_UUID="$(tr -d "[:space:]" < "$SCRIPT_DIR/active-gpu.conf")"
        fi
        # 32k is what a laptop holds; the health check below waits on :8080.
        : "${SPEAR_SERVER_CTX:=${SPEAR_CTX:-32768}}"
        : "${SPEAR_SERVER_PORT:=8080}"
        [ -n "${SPEAR_NCPUMOE:-}" ] && : "${SPEAR_SERVER_NCPUMOE:=$SPEAR_NCPUMOE}"
        [ -n "${SPEAR_THREADS:-}" ] && : "${SPEAR_SERVER_THREADS:=$SPEAR_THREADS}"
        [ -n "${SPEAR_NGL:-}" ] && : "${SPEAR_SERVER_NGL:=$SPEAR_NGL}"
        export SPEAR_SERVER_LLAMA_BIN SPEAR_SERVER_MODEL SPEAR_SERVER_CTX \
               SPEAR_SERVER_PORT
        [ -n "${SPEAR_SERVER_LORA:-}" ] && export SPEAR_SERVER_LORA
        [ -n "${SPEAR_SERVER_GPU_UUID:-}" ] && export SPEAR_SERVER_GPU_UUID
        [ -n "${SPEAR_SERVER_NCPUMOE:-}" ] && export SPEAR_SERVER_NCPUMOE
        [ -n "${SPEAR_SERVER_THREADS:-}" ] && export SPEAR_SERVER_THREADS
        [ -n "${SPEAR_SERVER_NGL:-}" ] && export SPEAR_SERVER_NGL

        sudo -n systemctl reset-failed spear-llm 2>/dev/null || true
        if ! sudo -n systemd-run --unit=spear-llm --collect \
                "$SERVE" --parallel 1 \
                > "$SCRIPT_DIR/llama-server.log" 2>&1; then
            echo "(systemd unavailable — starting detached)"
            setsid "$SERVE" --parallel 1 \
                > "$SCRIPT_DIR/llama-server.log" 2>&1 < /dev/null &
        fi
        echo -n "Waiting for the server (first start only)"
        for i in $(seq 1 300); do
            curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1 \
                && { echo " OK!"; break; }
            echo -n "."; sleep 1
        done; echo ""
    fi
    echo "→ LOCAL model on the laptop"
fi

exec "$PY" "$SCRIPT_DIR/rag_chat.py" "${ARGS[@]}"
