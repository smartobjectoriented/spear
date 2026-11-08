#!/bin/bash
# Run a published SPEAR image with nothing but docker: no checkout of this
# repository, no registry of your own. Made for the reds image, which carries
# its trees and their index; it serves any image whose corpora are baked.
#
#     spear-run.sh [--image IMAGE] [--pull]
#                  [--reds USER@HOST [--key FILE] [--port N] | --endpoint URL]
#                  [--corpus PATH] [--mount HOST_DIR:PATH]... [--state DIR]
#                  [--] [harness args, e.g. --ask, --auto, --fresh]
#
#   --image     the image to run; default ghcr.io/smartobjectoriented/spear-reds
#               at this release (SPEAR_IMAGE overrides)
#   --pull      docker pull it first
#   --reds      the inference host, as ssh names it; a tunnel is opened from
#               this machine to its port (--port, default 8010) unless one is
#               already up. --key names the ssh identity when the host needs
#               one named. SPEAR_REDS_HOST / SPEAR_REDS_KEY set the defaults.
#   --endpoint  any OpenAI-compatible URL instead of the tunnel
#   --corpus    where the session opens, as a path under /corpora
#               (e.g. opencn/opencn-ng); the image lists them with --list
#   --mount     bind your own checkout over a baked tree: HOST_DIR:PATH, with
#               PATH the tree's path under /corpora. Repeatable.
#   --state     where conversations, knowledge and audit land on this machine
#               (default ~/.spear/state)
#   --list      print the image's corpora and exit
#
# spear-docker.sh is the other launcher: for a workstation that has the
# repository and mounts its own registered trees. This one is for everyone else.
set -e

# The image this script was published with. From the repository the release
# script answers; from a copy of this file alone, the constant does.
HERE="$(cd "$(dirname "$0")" && pwd)"
VERSION="$("$HERE/../spearversion.sh" 2>/dev/null || echo 0.3.0)"
IMAGE="${SPEAR_IMAGE:-ghcr.io/smartobjectoriented/spear-reds:$VERSION-reds}"
REDS_HOST="${SPEAR_REDS_HOST:-}"
REDS_KEY="${SPEAR_REDS_KEY:-}"
REDS_PORT="${SPEAR_REDS_PORT:-8010}"
LPORT="${SPEAR_TUNNEL_PORT:-8082}"
ENDPOINT="${SPEAR_API_BASE:-}"
MODEL_NAME="${SPEAR_MODEL_NAME:-qwen3}"
STATE="${SPEAR_STATE_DIR:-$HOME/.spear/state}"
CORPUS=""
PULL=0
LIST=0
MOUNTS=()
ARGS=()

usage() { sed -n '2,33p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }
die() { echo "spear-run: $*" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --image)    IMAGE="$2"; shift 2 ;;
        --pull)     PULL=1; shift ;;
        --reds)     REDS_HOST="$2"; shift 2 ;;
        --key)      REDS_KEY="$2"; shift 2 ;;
        --port)     REDS_PORT="$2"; shift 2 ;;
        --endpoint) ENDPOINT="$2"; shift 2 ;;
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
if [ -z "$ENDPOINT" ]; then
    [ -n "$REDS_HOST" ] || die "name the model: --reds USER@HOST, or --endpoint URL"
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

# Your own checkouts over the baked trees, at the baked paths.
BINDS=()
for spec in "${MOUNTS[@]}"; do
    host="${spec%%:*}"; rel="${spec#*:}"
    [ "$host" != "$spec" ] && [ -d "$host" ] || die "--mount wants HOST_DIR:PATH with an existing HOST_DIR, not '$spec'"
    BINDS+=(-v "$(readlink -f "$host"):/corpora/${rel#/corpora/}")
done

WORKDIR="/corpora${CORPUS:+/${CORPUS#/corpora/}}"
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
    -e "SPEAR_API_BASE=$ENDPOINT" \
    -e "SPEAR_MODEL_NAME=$MODEL_NAME" \
    -e "SPEAR_STATE_DIR=/state" \
    -e "HOME=/state/home" \
    -v "$STATE:/state" \
    "${BINDS[@]}" \
    -w "$WORKDIR" \
    "$IMAGE" "${ARGS[@]}"
