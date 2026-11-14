#!/bin/bash
# Turn a fresh SPEAR checkout into an image-build host.
#
#     scripts/bootstrap-buildhost.sh
#     scripts/bootstrap-buildhost.sh --full
#     scripts/bootstrap-buildhost.sh --index-from /path/to/chromadb
#     scripts/bootstrap-buildhost.sh --projects /path/to/projects.json
#
# `spear-image build` reuses the staging scripts, and those run `python3` from
# the PATH to read the corpus index -- so a checkout cannot build an image
# until it has three machine-local things the repository deliberately does not
# carry: a client venv, a corpus registry (client/projects.json) and a prebuilt
# index (client/chromadb). This sets them up, idempotently.
#
# What a build actually needs is narrow: the staging scripts import only
# chromadb (they COPY an existing index, they do not embed), so the default
# venv installs chromadb alone -- enough to build an image from an index that
# already exists. --full installs client/deploy/requirements.txt instead, which
# is what BUILDING an index or running spear-chat natively needs. Even --full
# does not install torch: it must come from the CUDA index matching the GPU
# (see the note in requirements.txt), so that stays a manual step.
#
# The index itself is not built here -- that is the corpora's embeddings, hours
# of GPU work. Point --index-from at one to copy (another host's, or a release
# artifact), or build it afterwards with `spear-corpus add --autoindex` /
# `/reindex` (offloading embeddings with SPEAR_EMBED_REMOTE). Likewise the
# registry: --projects copies one, or `spear-corpus add` writes it corpus by
# corpus.
#
# After this, `spear-configure` writes client/local.conf; make sure its PATH
# puts this checkout's client/bin first, and set SPEAR_ENTITY, SPEAR_COMMONS
# and the entity registry there (see client/local.conf.example).

set -e

HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
# The release this is: "[spear vX.Y.Z] <script> <args>" on stderr, once.
[ -r "$HERE/common/banner.sh" ] && . "$HERE/common/banner.sh"
REPO="$(cd "$HERE/.." && pwd)"
APP="$REPO/client"

usage() { sed -n '2,38p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }
die()   { echo "bootstrap-buildhost: $*" >&2; exit 1; }

PY=python3
FULL=0
FORCE=0
INDEX_FROM=""
PROJECTS=""

while [ $# -gt 0 ]; do
    case "$1" in
        --full)       FULL=1; shift ;;
        --force)      FORCE=1; shift ;;
        --index-from) INDEX_FROM="$2"; shift 2 ;;
        --projects)   PROJECTS="$2"; shift 2 ;;
        --python)     PY="$2"; shift 2 ;;
        -h|--help)    usage ;;
        *)            die "unknown option: $1 (see --help)" ;;
    esac
done

[ -f "$APP/cli/rag_chat.py" ] || die "no harness under $APP -- run this from a SPEAR checkout"

# ── the client venv ──────────────────────────────────────────────────────
# Idempotent: an existing venv is kept unless --force. The staging scripts
# find it through the PATH (client/local.conf), not from here.

if [ -x "$APP/bin/python3" ] && [ "$FORCE" = 0 ]; then
    echo "   venv      $APP/bin (kept; --force to recreate)"
else
    [ "$FORCE" = 1 ] && rm -rf "$APP/bin" "$APP/lib" "$APP/lib64" "$APP/include" "$APP/pyvenv.cfg"
    echo "   venv      creating $APP (with $PY)"
    "$PY" -m venv "$APP"
    "$APP/bin/pip" install --quiet --upgrade pip
fi

if [ "$FULL" = 1 ]; then
    echo "   deps      client/deploy/requirements.txt (full; torch is separate -- see the file)"
    "$APP/bin/pip" install --quiet -r "$APP/deploy/requirements.txt"
else
    # The one dependency the staging scripts import, pinned to the same version
    # as the full set so the index format matches.
    CHROMA="$(sed -n 's/^chromadb==/chromadb==/p' "$APP/deploy/requirements.txt" | head -1)"
    CHROMA="${CHROMA:-chromadb}"
    echo "   deps      $CHROMA (build-only; --full for indexing or native spear-chat)"
    "$APP/bin/pip" install --quiet "$CHROMA"
fi

# ── the corpus registry ──────────────────────────────────────────────────

if [ -n "$PROJECTS" ]; then
    [ -r "$PROJECTS" ] || die "--projects: cannot read $PROJECTS"
    cp "$PROJECTS" "$APP/projects.json"
    echo "   registry  copied $PROJECTS -> client/projects.json"
elif [ -r "$APP/projects.json" ]; then
    echo "   registry  client/projects.json (present)"
else
    echo "   registry  client/projects.json MISSING -- copy one (--projects FILE) or"
    echo "             register corpora with 'spear-corpus add'"
fi

# ── the index ────────────────────────────────────────────────────────────

if [ -n "$INDEX_FROM" ]; then
    [ -d "$INDEX_FROM" ] || die "--index-from: not a directory: $INDEX_FROM"
    if [ -d "$APP/chromadb" ] && [ "$FORCE" = 0 ]; then
        echo "   index     client/chromadb (present; --force to replace)"
    else
        [ "$FORCE" = 1 ] && rm -rf "$APP/chromadb"
        echo "   index     copying $INDEX_FROM -> client/chromadb"
        cp -a "$INDEX_FROM" "$APP/chromadb"
    fi
elif [ -d "$APP/chromadb" ]; then
    echo "   index     client/chromadb (present)"
else
    echo "   index     client/chromadb MISSING -- copy one (--index-from DIR), or build"
    echo "             it with 'spear-corpus add --autoindex' / '/reindex'"
fi

# ── report what a build would find ─────────────────────────────────────────

echo
"$APP/bin/python3" -c 'import chromadb; print("   chromadb", chromadb.__version__, "ok")' \
    || die "chromadb did not import from the venv"

ready=1
[ -r "$APP/projects.json" ] || ready=0
[ -d "$APP/chromadb" ]      || ready=0

if [ "$ready" = 1 ]; then
    cat <<MSG

This checkout can build images. Next:
  - spear-configure            # writes client/local.conf
  - in local.conf: PATH first entry = $APP/bin, then SPEAR_ENTITY,
    SPEAR_COMMONS and the entity registry (see client/local.conf.example)
  - spear-image build harness && spear-image build base
MSG
else
    cat <<MSG

Venv ready; the registry and/or index are still missing (see above). Supply
them, then re-run this to confirm, and configure with spear-configure.
MSG
fi
