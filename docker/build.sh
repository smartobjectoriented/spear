#!/bin/bash
# Build the harness image.
#
# Contexts, deliberately not the obvious ones: the DEFAULT one is spear/
# (harness code), and the rest are NAMED contexts, one per input that a clone
# does not necessarily carry. Keeping the default context narrow is what lets
# the 4.5 GB embedder layer stay cached across rebuilds -- and what keeps
# models/ and the served GGUFs (120 GB of weights) out of the build entirely.
#
# A named context is fetched lazily: BuildKit transfers only the paths actually
# COPYed from it, so pointing `repo` at a 130 GB tree costs nothing.
#
# THE OPTIONAL INPUTS. Four of the things this image would like to carry are
# not in the repository and cannot be: the retrieval index is built on the host
# and gitignored, and the rules, the skills, the benches and the shared notes
# corpus are a deployment's own content. Each resolves to the in-tree
# directory when there is one, to whatever its SPEAR_*_DIR names when the
# deployment keeps it elsewhere, and to an EMPTY directory otherwise -- so a
# clean public clone builds a working image that simply carries less. It says
# which, rather than failing on the first missing one.
set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
APP="${SPEAR_APP:-$REPO/spear}"
TAG="${1:-spear:1.0}"
[ -f "$APP/rag_chat.py" ] || { echo "no harness under $APP — set SPEAR_APP" >&2; exit 1; }

EMPTY="$(mktemp -d)"
trap 'rm -rf "$EMPTY"' EXIT

# name : environment override : in-tree default : what it is
OPTIONAL=(
    "index:SPEAR_INDEX_DIR:$APP/chromadb:prebuilt retrieval index"
    "rules:SPEAR_RULES_DIR:$APP/rules.d:global rules"
    "skills:SPEAR_SKILLS_DIR:$APP/skills:skill library"
    "benches:SPEAR_BENCH_DIR:$APP/benches:acceptance benches"
    "notes:SPEAR_NOTES_DIR:$REPO/claude:shared notes corpus"
)

CONTEXTS=()
for spec in "${OPTIONAL[@]}"; do
    IFS=: read -r name var fallback label <<<"$spec"
    dir="${!var:-$fallback}"

    if [ -d "$dir" ]; then
        printf '   %-8s %s\n' "$name" "$dir"
    else
        printf '   %-8s (absent — image carries no %s)\n' "$name" "$label"
        dir="$EMPTY"
    fi

    CONTEXTS+=(--build-context "$name=$dir")
done

# Generate the registry the image carries. It is BUILD OUTPUT and is not in the
# repository: a committed copy would be one machine's corpus graph, published,
# and stale the moment a corpus is added. Generating it here is also what stops
# it drifting from projects.json — a corpus added on the host would otherwise be
# absent from the image with no sign but a "missing" line at startup.
REGISTRY="$REPO/docker/projects.docker.json"
"$REPO/docker/gen-registry.py" || echo "   (some corpora were skipped, see above)" >&2
[ -f "$REGISTRY" ] || {
    echo "gen-registry.py produced no $REGISTRY — the image needs one" >&2
    exit 1
}

echo "== building $TAG =="
echo "   harness context : $APP"
echo "   repo context    : $REPO  (corpora/ + docker/ only)"
exec docker build --build-context "repo=$REPO" "${CONTEXTS[@]}" \
    -f "$REPO/docker/Dockerfile" -t "$TAG" "$APP"
