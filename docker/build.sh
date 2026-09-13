#!/bin/bash
# Build the harness image.
#
# Two contexts, deliberately: the DEFAULT one is spear/ (harness code +
# the prebuilt index), and `repo` is this repository, from which only claude/
# and corpora/ are taken. Keeping the default context narrow is what lets the
# 4.5 GB embedder layer stay cached across rebuilds -- and what keeps models/
# and the served GGUFs (120 GB of weights) out of the build entirely.
#
# A named context is fetched lazily: BuildKit transfers only the paths actually
# COPYed from it, so pointing `repo` at a 130 GB tree costs nothing.
set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
APP="${SPEAR_APP:-$REPO/spear}"
TAG="${1:-spear:1.0}"
[ -d "$REPO/claude" ] || { echo "no claude/ under $REPO" >&2; exit 1; }
[ -f "$APP/rag_chat.py" ] || { echo "no harness under $APP — set SPEAR_APP" >&2; exit 1; }
# Regenerate the registry the image carries. Doing it here rather than by hand
# is what stops it drifting from projects.json: a corpus added on the host
# would otherwise be absent from the image with no sign but a "missing" line
# at startup.
"$REPO/docker/gen-registry.py" || echo "   (some corpora were skipped, see above)" >&2

echo "== building $TAG =="
echo "   harness context : $APP"
echo "   repo context    : $REPO  (claude/ + corpora/ only)"
exec docker build --build-context "repo=$REPO" \
    -f "$REPO/docker/Dockerfile" -t "$TAG" "$APP"
