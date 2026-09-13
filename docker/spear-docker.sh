#!/bin/bash
# Run the SPEAR harness container against local checkouts.
#
#   docker/spear-docker.sh [--reds] [--endpoint URL] [--state DIR]
#                          [--mount DIR]... [-- <harness args>]
#
# Every tree is bound at its OWN ABSOLUTE PATH, not remapped under a common
# root. That is not a preference: CMake caches, bitbake stamps and the
# toolchain paths they record are absolute, so a tree mounted elsewhere builds
# against compilers that are not there. ib.md states the same rule for
# dbuild.sh -- "-v $(pwd):$(pwd), the SAME absolute path, not /src" -- and it
# is also what the harness's own sandbox does (effective_mount_root binds each
# tree at its host path). An identity mount keeps all three in agreement, and
# makes the working directory need no translation at all.
#
# --mount adds a tree that is not a registered corpus but that a build needs;
# ~/soo/so3/build/tmp/toolchains is the standard example.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
APP="${SPEAR_APP:-$REPO/spear}"
IMAGE="${SPEAR_IMAGE:-spear:1.0}"
API_BASE="${SPEAR_API_BASE:-http://127.0.0.1:8082/v1}"
MODEL_NAME="${SPEAR_MODEL_NAME:-qwen3}"
STATE="${SPEAR_STATE_DIR:-$HOME/.spear/state}"
REDS=0
EXTRA=()
ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --endpoint) API_BASE="$2"; shift 2 ;;
        --state)    STATE="$2"; shift 2 ;;
        --mount)    EXTRA+=("$2"); shift 2 ;;
        --reds)     REDS=1; shift ;;
        --)         shift; ARGS=("$@"); break ;;
        *)          ARGS+=("$1"); shift ;;
    esac
done
mkdir -p "$STATE"

# The SSH identity, read-only: the container needs it for the remote embedder
# and for ssh by hand. One key, not ~/.ssh. The command policy refuses any path
# matching a credential store, so it is there for ssh and denied to the model.
# shellcheck disable=SC1090
. "$APP/reds.conf" 2>/dev/null || true
SSH_MOUNTS=()
if [ -n "${REDS_KEY:-}" ] && [ -f "$REDS_KEY" ]; then
    SSH_MOUNTS+=(-v "$(readlink -f "$REDS_KEY"):/ssh/id_reds:ro"
                 -e "SPEAR_REDS_KEY=/ssh/id_reds")
    [ -f "$HOME/.ssh/known_hosts" ] && SSH_MOUNTS+=(
        -v "$HOME/.ssh/known_hosts:/ssh/known_hosts:ro")
fi

# The SSH tunnel stays on the HOST: shipping keys into an image meant to be
# handed around is the opposite of the point, and --network host makes
# 127.0.0.1 the same thing on both sides.
if [ "$REDS" = 1 ]; then
    REDS_HOST="${REDS_HOST:-reds-server}"; REDS_PORT="${REDS_PORT:-8000}"
    read -r -a REDS_SSH_ARGS <<< "${REDS_SSH_OPTS:-}"
    LPORT=8082
    API_BASE="http://127.0.0.1:$LPORT/v1"
    MODEL_NAME="${REDS_MODEL:-qwen3}"
    if ! curl -sf "$API_BASE/models" >/dev/null 2>&1; then
        echo "Opening SSH tunnel :$LPORT -> $REDS_HOST:$REDS_PORT ..." >&2
        ssh -o ConnectTimeout=12 -o ExitOnForwardFailure=yes \
            -o ServerAliveInterval=30 "${REDS_SSH_ARGS[@]}" \
            -L "$LPORT:localhost:$REDS_PORT" -N -f "$REDS_HOST" || {
                echo "tunnel failed - is '$REDS_HOST' reachable?" >&2; exit 1; }
    fi
    curl -sf "$API_BASE/models" >/dev/null 2>&1 || {
        echo "tunnel open, but nothing is serving on $REDS_HOST:$REDS_PORT." >&2
        exit 1; }
fi

# Mount set: every registered corpus at its own path, the repository itself,
# the cross-toolchains, and whatever --mount named. Derived from the registry
# so a corpus added on the host needs no change here.
mapfile -t TREES < <(
    "$APP/bin/python" - "$APP" "$PWD" <<'PYTREES'
import os, sys
sys.path.insert(0, sys.argv[1])
import rag_chat
seen = {os.path.realpath(s["path"]) for s in rag_chat.load_projects().values()}
seen.add(os.path.realpath(rag_chat.ROOT_DIR))          # the repository
# The WORKING TREE, which is not the same list as the corpora. Corpora are
# components -- so3, u-boot, avz -- while the build system that drives them
# sits in the umbrella above: env.sh, scripts/build.sh, build/tmp/toolchains.
# Mounting only the corpora left the model unable to see build.sh at all, so
# it went spelunking in CMake caches instead. Corpus is what retrieval indexes;
# the workspace is what the tools need present.
seen.add(os.path.realpath(sys.argv[2]))
# Toolchains are mounted read-only, separately -- not here.
# Drop a tree already inside another: binding it twice buys nothing.
minimal = [p for p in sorted(seen)
           if os.path.isdir(p)
           and not any(p != q and p.startswith(q.rstrip("/") + "/") for q in seen)]
print("\n".join(minimal))
PYTREES
)
MOUNTS=()
for tree in "${TREES[@]}" "${EXTRA[@]}"; do
    [ -d "$tree" ] && MOUNTS+=(-v "$tree:$tree")
done
# Read-only, and at their own paths like everything else. /usr/local/bin is
# where ib.md tells the operator to symlink each cross-toolchain's bin/, and
# those symlinks point into /opt/toolchains -- so one without the other is
# useless. Without both, the SO3 kernel build dies on "aarch64-none-elf-gcc:
# not found" while the compiler is mounted two directories away. Read-only
# because a toolchain is read by a build, never written by one.
for ro in /usr/local/bin /opt/toolchains; do
    [ -d "$ro" ] && MOUNTS+=(-v "$ro:$ro:ro")
done
[ ${#MOUNTS[@]} -gt 0 ] || { echo "no tree to mount" >&2; exit 1; }
echo "mounting ${#MOUNTS[@]} trees at their own paths · state $STATE · $API_BASE" >&2

# The registry itself: the host's, so the paths inside it are the paths that
# exist. The image ships one as a fallback for a machine that has none.
# SPEAR_CORPUS_ROOT (set below) points at the mounted repository rather than
# the image's own copy of the harness: the corpora registered relatively live
# in the repository, and the image carries only the harness.
[ -f "$APP/projects.json" ] && MOUNTS+=(
    -v "$APP/projects.json:/opt/spear/spear/projects.json:ro")

# -t only when there IS a terminal: docker refuses to allocate one otherwise,
# which would make the container unusable from a script or a CI job.
TTY=(-i); [ -t 0 ] && TTY=(-i -t)

# --security-opt: bwrap must create a user namespace AND mount a fresh /proc
# inside it. seccomp/apparmor cover the namespace; systempaths covers the mount
# -- Docker masks paths under /proc and mounting proc over a masked proc fails
# with EPERM. Measured: --cap-add SYS_ADMIN does NOT fix it.
# --memory/--pids-limit/--cpus mirror DEFAULT_CGROUP_LIMITS. The harness cannot
# build a systemd scope in here and is fail-closed about that; the flags below
# are what make SPEAR_RESOURCE_CONTROL=delegated true rather than a claim.
exec docker run --rm "${TTY[@]}" \
    --security-opt seccomp=unconfined \
    --security-opt apparmor=unconfined \
    --security-opt systempaths=unconfined \
    --memory 2g --memory-swap 2g --pids-limit 256 --cpus 8 \
    -e SPEAR_RESOURCE_CONTROL=delegated \
    --network host \
    --user "$(id -u):$(id -g)" \
    -e "SPEAR_API_BASE=$API_BASE" \
    -e "SPEAR_MODEL_NAME=$MODEL_NAME" \
    -e "SPEAR_STATE_DIR=/state" \
    -e "SPEAR_CORPUS_ROOT=$REPO" \
    -e "HOME=/state/home" \
    -v "$STATE:/state" \
    -w "$PWD" \
    "${MOUNTS[@]}" "${SSH_MOUNTS[@]}" \
    "$IMAGE" "${ARGS[@]}"
