#!/bin/bash
# Run a published SPEAR image with nothing but docker: no checkout of this
# repository, no registry of your own. Made for the reds image, which carries
# its trees and their index; it serves any image whose corpora are baked.
#
#     spear-run.sh [--image IMAGE] [--pull]
#                  [--reds USER@HOST [--key FILE] [--port N] | --endpoint URL
#                   | --provider anthropic [--model NAME]]
#                  [--tree NAME | --here | --corpus PATH]
#                  [--mount HOST_DIR:PATH]... [--state DIR]
#                  [--] [harness args, e.g. --ask, --auto, --fresh]
#
#   --image     the image to run; default ghcr.io/smartobjectoriented/spear-reds
#               at this release
#   --pull      docker pull it first
#   --reds      the inference host, as ssh names it; a tunnel is opened from
#               this machine to its port (--port, default 8010) unless one is
#               already up. --key names the ssh identity when the host needs
#               one named.
#   --endpoint  any OpenAI-compatible URL instead of the tunnel
#   --provider  a hosted model instead of an endpoint: anthropic, with --model
#               naming the model. Its key is ANTHROPIC_API_KEY.
#   --tree      the current directory is your checkout of the image's corpus
#               NAME: it is bound over the baked tree and the session opens
#               there, on that corpus's index
#   --here      the current directory is a tree the image does not know: it
#               is bound at its own path and the session opens there
#   --corpus    where the session opens, as a path under /corpora
#               (e.g. opencn/opencn-ng); the image lists them with --list
#   --mount     bind your own checkout over a baked tree: HOST_DIR:PATH, with
#               PATH the tree's path under /corpora. Repeatable.
#   --state     where conversations, knowledge and audit land on this machine
#               (default ~/.spear/state)
#   --list      print the image's corpora and exit
#
# What does not change from one run to the next goes in a configuration file,
# read before the options: client/local.conf of the repository this script is
# in, else ~/.config/spear/local.conf. It is shell, one assignment per line:
#
#     SPEAR_IMAGE=registry.example/spear:1.0-acme
#     SPEAR_PROVIDER=anthropic
#     SPEAR_MODEL=<model name>
#     ANTHROPIC_API_KEY=<key>
#
# or REDS_HOST and SPEAR_REDS_KEY for an inference host. An option overrides
# the file.
#
# spear-docker.sh is the other launcher: for a workstation that has the
# repository and mounts its own registered trees. This one is for everyone else.
set -e

# The image this script was published with. From the repository the release
# script answers; from a copy of this file alone, the constant does.
HERE="$(cd "$(dirname "$0")" && pwd)"

for conf in "$HERE/../../client/local.conf" "${XDG_CONFIG_HOME:-$HOME/.config}/spear/local.conf"; do
    if [ -r "$conf" ]; then
        # shellcheck disable=SC1090
        . "$conf"
        break
    fi
done

VERSION="$("$HERE/../spearversion.sh" 2>/dev/null || echo 0.3.0)"
IMAGE="${SPEAR_IMAGE:-ghcr.io/smartobjectoriented/spear-reds:$VERSION-reds}"
REDS_HOST="${REDS_HOST:-}"
REDS_KEY="${SPEAR_REDS_KEY:-}"
REDS_PORT="${REDS_PORT:-8010}"
LPORT="${SPEAR_TUNNEL_PORT:-8082}"
ENDPOINT="${SPEAR_API_BASE:-}"
MODEL_NAME="${SPEAR_MODEL_NAME:-qwen3}"
PROVIDER="${SPEAR_PROVIDER:-}"
MODEL="${SPEAR_MODEL:-}"
STATE="${SPEAR_STATE_DIR:-$HOME/.spear/state}"
CORPUS=""
TREE=""
OPEN_HERE=0
PULL=0
LIST=0
MOUNTS=()
ARGS=()

usage() { sed -n '2,52p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }
die() { echo "spear-run: $*" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --image)    IMAGE="$2"; shift 2 ;;
        --pull)     PULL=1; shift ;;
        --reds)     REDS_HOST="$2"; PROVIDER=""; shift 2 ;;
        --key)      REDS_KEY="$2"; shift 2 ;;
        --port)     REDS_PORT="$2"; shift 2 ;;
        --endpoint) ENDPOINT="$2"; PROVIDER=""; shift 2 ;;
        --provider) PROVIDER="$2"; shift 2 ;;
        --model)    MODEL="$2"; shift 2 ;;
        --tree)     TREE="$2"; shift 2 ;;
        --here)     OPEN_HERE=1; shift ;;
        --corpus)   CORPUS="$2"; shift 2 ;;
        --mount)    MOUNTS+=("$2"); shift 2 ;;
        --state)    STATE="$2"; shift 2 ;;
        --list)     LIST=1; shift ;;
        -h|--help)  usage ;;
        --)         shift; ARGS=("$@"); break ;;
        *)          ARGS+=("$1"); shift ;;
    esac
done

command -v docker >/dev/null || die "docker is not installed"
[ "$PULL" = 1 ] && docker pull "$IMAGE"
docker image inspect "$IMAGE" >/dev/null 2>&1 || die "no image $IMAGE here; add --pull, or docker login first"

if [ "$LIST" = 1 ]; then
    docker run --rm --entrypoint "/opt/spear/client/bin/python" "$IMAGE" -c '
import json
for name, spec in sorted(json.load(open("/opt/spear/client/projects.json")).items()):
    print(name.ljust(18), "/corpora/" + spec["path"])'
    exit 0
fi

# The model. The tunnel stays on this machine: the image carries no key, and
# with host networking 127.0.0.1 is the same thing on both sides.
BACKEND=()

if [ -n "$PROVIDER" ]; then
    [ "$PROVIDER" = anthropic ] || die "--provider knows anthropic, not '$PROVIDER'"
    [ -n "${ANTHROPIC_API_KEY:-}" ] \
        || die "--provider anthropic needs ANTHROPIC_API_KEY; set it in the configuration file (see --help)"
    export ANTHROPIC_API_KEY

    # The name alone: the value reaches the container from this environment
    # and never appears in a process list.

    BACKEND=(-e ANTHROPIC_API_KEY)
    ARGS=(--provider anthropic ${MODEL:+--model "$MODEL"} "${ARGS[@]}")
elif [ -z "$ENDPOINT" ]; then
    [ -n "$REDS_HOST" ] || die "name the model: --reds USER@HOST, --endpoint URL or --provider anthropic"
    ENDPOINT="http://127.0.0.1:$LPORT/v1"

    if ! curl -sf -m 3 "$ENDPOINT/models" >/dev/null 2>&1; then
        SSH_OPTS=(-o ConnectTimeout=12 -o ExitOnForwardFailure=yes -o ServerAliveInterval=30)

        # A host with a low MaxAuthTries disconnects ssh before it reaches the
        # right key unless that key is the only one offered.
        [ -n "$REDS_KEY" ] && SSH_OPTS+=(-o IdentitiesOnly=yes -i "$REDS_KEY")
        echo "opening the tunnel :$LPORT -> $REDS_HOST:$REDS_PORT" >&2
        ssh "${SSH_OPTS[@]}" -L "$LPORT:localhost:$REDS_PORT" -N -f "$REDS_HOST" \
            || die "tunnel failed: is $REDS_HOST reachable (VPN?) and your key authorised?"
    fi

    curl -sf -m 5 "$ENDPOINT/models" >/dev/null 2>&1 \
        || die "the tunnel is up but nothing answers on $REDS_HOST:$REDS_PORT"
fi

[ -z "$PROVIDER" ] && BACKEND=(-e "SPEAR_API_BASE=$ENDPOINT" -e "SPEAR_MODEL_NAME=$MODEL_NAME")

# Your own checkouts over the baked trees, at the baked paths.
BINDS=()
for spec in "${MOUNTS[@]}"; do
    host="${spec%%:*}"; rel="${spec#*:}"
    [ "$host" != "$spec" ] && [ -d "$host" ] || die "--mount wants HOST_DIR:PATH with an existing HOST_DIR, not '$spec'"
    BINDS+=(-v "$(readlink -f "$host"):/corpora/${rel#/corpora/}")
done

WORKDIR="/corpora${CORPUS:+/${CORPUS#/corpora/}}"

if [ -n "$TREE" ]; then
    rel="$(docker run --rm --entrypoint /opt/spear/client/bin/python "$IMAGE" -c '
import json, sys
spec = json.load(open("/opt/spear/client/projects.json")).get(sys.argv[1])
print(spec["path"] if isinstance(spec, dict) else "")' "$TREE")"
    [ -n "$rel" ] || die "the image has no corpus named '$TREE' (--list names them)"
    BINDS+=(-v "$PWD:/corpora/$rel")
    WORKDIR="/corpora/$rel"
elif [ "$OPEN_HERE" = 1 ]; then
    BINDS+=(-v "$PWD:$PWD")
    WORKDIR="$PWD"
fi

mkdir -p "$STATE"
TTY=(-i); [ -t 0 ] && TTY=(-i -t)

# The three --security-opt are what bubblewrap, the harness's sandbox, needs
# to start inside a container; without them it refuses every command.
exec docker run --rm "${TTY[@]}" \
    --security-opt seccomp=unconfined \
    --security-opt apparmor=unconfined \
    --security-opt systempaths=unconfined \
    --network host \
    --user "$(id -u):$(id -g)" \
    "${BACKEND[@]}" \
    -e "SPEAR_STATE_DIR=/state" \
    -e "HOME=/state/home" \
    -v "$STATE:/state" \
    "${BINDS[@]}" \
    -w "$WORKDIR" \
    "$IMAGE" "${ARGS[@]}"
