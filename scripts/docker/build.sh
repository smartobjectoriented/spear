#!/bin/bash
# Build the harness image.
#
# Contexts, deliberately not the obvious ones: the DEFAULT one is client/
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
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
# The release this is: "[spear vX.Y.Z] <script> <args>" on stderr, once.
[ -r "$REPO/scripts/common/banner.sh" ] && . "$REPO/scripts/common/banner.sh"
HERE="$REPO/scripts/docker"
APP="${SPEAR_APP:-$REPO/client}"

# THE PROFILE decides what the image is allowed to carry, and it defaults to
# the one that is safe to hand to anyone. `private` is the deliberate word
# for "this image carries licensed and customer material"; there is no way to
# get there by omission.
PROFILE=public
BAKE=""
KNOWLEDGE=""
TAG=""

while [ $# -gt 0 ]; do
    case "$1" in
        --profile) PROFILE="$2"; shift 2 ;;
        --profile=*) PROFILE="${1#*=}"; shift ;;
        --bake) BAKE="$2"; shift 2 ;;
        --bake=*) BAKE="${1#*=}"; shift ;;
        --knowledge) KNOWLEDGE="$2"; shift 2 ;;
        --knowledge=*) KNOWLEDGE="${1#*=}"; shift ;;
        -h|--help)
            cat <<'USAGE'
scripts/docker/build.sh [TAG] [--profile public|reds|private] [--bake name,name,...]
                        [--knowledge name,name,...|all]

  --profile public       (default) only normative documents that declare
                         themselves PUBLIC, and of rules, skills, benches and
                         notes only what the repository tracks; no retrieval
                         index, no baked corpus. Safe to hand over.
  --profile reds         for the people of the institute: the corpora named
                         by --bake (required), their collections of the index
                         and nothing else of it, the rules, skills and benches
                         their scopes admit, PUBLIC normative documents only.
                         Pushed to the institute's registry and refused
                         elsewhere unless you say --allow-push.
  --profile private      everything this machine has, licensed documents and
                         the original PDFs included. NOT redistributable; the
                         image is labelled so, and push is refused unless you
                         say --allow-push.
  --bake a,b,c           copy these registered corpora INTO the image, for a
                         container that has to work with nothing mounted.
                         Default: none, and they are expected at /corpora.
  --knowledge a,b|all    ship the active workspace knowledge of these projects
                         as the image's common state, read under each user's
                         own. Default: none.
USAGE
            exit 0 ;;
        -*) echo "unknown option: $1" >&2; exit 1 ;;
        *) TAG="$1"; shift ;;
    esac
done

case "$PROFILE" in
    public|reds|private) ;;
    *) echo "--profile must be public, reds or private, not '$PROFILE'" >&2; exit 1 ;;
esac

# A baked corpus is a tree off this host, and nothing records whether it may
# be passed on -- a customer checkout looks like any other. A public image
# cannot vouch for it, so it does not carry one.
if [ "$PROFILE" = public ] && [ -n "$BAKE" ]; then
    echo "--bake needs --profile private: a public image carries no corpus" \
         "tree, since nothing says which ones may be redistributed" >&2
    exit 1
fi

# Knowledge is what one deployment has learned about its own trees, customer
# code included, and nothing records which of it may be passed on.
if [ "$PROFILE" = public ] && [ -n "$KNOWLEDGE" ]; then
    echo "--knowledge needs --profile private: a public image carries no" \
         "workspace knowledge, since nothing says which of it may be redistributed" >&2
    exit 1
fi

# A reds image exists for people who have none of the trees: what it carries
# is what it bakes, and everything else it takes -- index, rules, skills -- is
# cut to those corpora. Without a bake list there is nothing to cut to.
if [ "$PROFILE" = reds ] && [ -z "$BAKE" ]; then
    echo "--profile reds needs --bake: the image carries the corpora it bakes," \
         "and its index and rules are cut to them" >&2
    exit 1
fi

# Tagged with the release the tree is, from the one place that says it: a
# hard-coded number here drifted to 1.0 while the release line was 0.2.
TAG="${TAG:-spear:$("$REPO/scripts/spearversion.sh")-$PROFILE}"
[ -f "$APP/cli/rag_chat.py" ] || { echo "no harness under $APP — set SPEAR_APP" >&2; exit 1; }

# The same machine settings the launcher reads, for the same reason: this is
# where a deployment says that its rules, skills and benches live outside the
# checkout. Without it the OPTIONAL table below resolves to the in-tree
# directories -- which hold a README and nothing else -- and the build reports
# them as found. The image then ships a harness with no rules and no skills,
# announcing neither, which is exactly what machine.env exists to prevent on
# the workstation.
if [ -r "$APP/machine.env" ]; then
    echo "   settings  $APP/machine.env"
    # shellcheck disable=SC1091
    . "$APP/machine.env"
fi

EMPTY="$(mktemp -d)"
trap 'rm -rf "$EMPTY"' EXIT

# Generate the registry the image carries. It is BUILD OUTPUT and is not in the
# repository: a committed copy would be one machine's corpus graph, published,
# and stale the moment a corpus is added. Generating it here is also what stops
# it drifting from projects.json — a corpus added on the host would otherwise be
# absent from the image with no sign but a "missing" line at startup.
REGISTRY="$REPO/docker/projects.docker.json"
"$HERE/gen-registry.py" --profile "$PROFILE" || echo "   (some corpora were skipped, see above)" >&2
[ -f "$REGISTRY" ] || {
    echo "gen-registry.py produced no $REGISTRY — the image needs one" >&2
    exit 1
}


# name : environment override : in-tree default : what it is
OPTIONAL=(
    "index:SPEAR_INDEX_DIR:$APP/chromadb:prebuilt retrieval index"
    "rules:SPEAR_RULES_DIR:$APP/rules.d:global rules"
    "skills:SPEAR_SKILLS_DIR:$APP/skills:skill library"
    "benches:SPEAR_BENCH_DIR:$APP/benches:acceptance benches"
    "notes:SPEAR_NOTES_DIR:$REPO/claude:shared notes corpus"
)

# The profile used to decide the normative documents and nothing else: rules,
# skills, benches, notes and the retrieval index were copied whatever it said.
# A deployment's machine.env points those at its private tree, and the index
# holds chunks of every corpus on the host, customer code included -- so a
# `public` image carried all of it under a redistributable=true label.
#
# A public image therefore carries, of these inputs, exactly what the
# repository publishes: the files git tracks in an in-tree directory. Outside
# the checkout or ignored by git means withheld, and the build says so. Not
# "whatever is in the in-tree directory": an untracked file dropped into
# skills/ is not published either.
STAGE="$(mktemp -d)"
trap 'rm -rf "$EMPTY" "$STAGE"' EXIT

publishable() {
    # The tracked files of $1, copied to $2. Fails if there are none, or if
    # $1 is not a directory of this checkout.
    local dir="$1" dest="$2" rel
    rel="$(realpath -m --relative-to="$REPO" "$dir")"
    case "$rel" in ../*|/*) return 1 ;; esac
    git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1 || return 1
    [ -n "$(git -C "$REPO" ls-files -- "$rel")" ] || return 1
    mkdir -p "$dest"
    git -C "$REPO/$rel" ls-files -z | tar -C "$REPO/$rel" --null -T - -cf - \
        | tar -C "$dest" -xf -
}

CONTEXTS=()
CONTENT_SRC=()
for spec in "${OPTIONAL[@]}"; do
    IFS=: read -r name var fallback label <<<"$spec"
    dir="${!var:-$fallback}"

    # A reds image's content is staged below, once every tree is known: the
    # machine's own directories when they exist, then SPEAR_CONTENT_TREES.
    if [ "$PROFILE" = reds ]; then
        case "$name" in
            rules|skills|benches)
                [ -d "$dir" ] && CONTENT_SRC+=("--$name" "$dir")
                continue ;;
        esac
    fi

    if [ ! -d "$dir" ]; then
        printf '   %-8s (absent — image carries no %s)\n' "$name" "$label"
        dir="$EMPTY"
    elif [ "$PROFILE" = reds ]; then
        # Cut to the baked corpora: the index to their collections. The notes
        # corpus is one deployment's own writing about every tree it has, and
        # declares no scope at all.
        case "$name" in
            index)
                "$HERE/stage-index.py" --bake "$BAKE" --index "$dir" "$STAGE/optional/index"
                dir="$STAGE/optional/index" ;;
            notes)
                printf '   %-8s %s  WITHHELD — a reds image carries no notes corpus\n' \
                    "$name" "$dir"
                dir="$EMPTY" ;;
        esac
    elif [ "$PROFILE" = public ]; then
        if publishable "$dir" "$STAGE/optional/$name"; then
            printf '   %-8s %s  (tracked files only)\n' "$name" "$dir"
            dir="$STAGE/optional/$name"
        else
            printf '   %-8s %s  WITHHELD — not published by the repository\n' \
                "$name" "$dir"
            dir="$EMPTY"
        fi
    else
        printf '   %-8s %s\n' "$name" "$dir"
    fi

    CONTEXTS+=(--build-context "$name=$dir")
done

# Further content trees, one per project the image serves: a checkout that
# carries its own rules.d/, skills/ and benches/ (an OpenCN tree's
# spear-private/, for one). Colon-separated, merged after the machine's own.
if [ "$PROFILE" = reds ] && [ -n "${SPEAR_CONTENT_TREES:-}" ]; then
    IFS=: read -r -a _trees <<<"$SPEAR_CONTENT_TREES"
    for tree in "${_trees[@]}"; do
        [ -d "$tree" ] && CONTENT_SRC+=(--tree "$tree") \
            || echo "   content  $tree  (absent — not a content tree)" >&2
    done
fi

if [ "$PROFILE" = reds ]; then
    "$HERE/stage-content.py" --bake "$BAKE" "${CONTENT_SRC[@]}" "$STAGE/content"
    CONTEXTS+=(--build-context "rules=$STAGE/content/rules.d"
               --build-context "skills=$STAGE/content/skills"
               --build-context "benches=$STAGE/content/benches")
fi

# The normative store and the baked trees are staged rather than pointed at:
# what may travel is a per-document decision (see stage-standards.py), and
# what gets baked is a named subset of a 300 GB registry. Neither is a
# directory that happens to be in the right shape already.

"$HERE/stage-standards.py" --profile "$PROFILE" "$STAGE/standards"
CONTEXTS+=(--build-context "standards=$STAGE/standards")

SPEAR_COMMON_STATE_DIR="${SPEAR_COMMON_STATE_DIR:-}" \
    "$HERE/stage-knowledge.py" --workspaces "$KNOWLEDGE" "$STAGE/common"
CONTEXTS+=(--build-context "common=$STAGE/common")

RESTRICT=()
[ -n "$BAKE" ] && RESTRICT=(--restrict-registry)
"$HERE/stage-corpora.py" --bake "$BAKE" "${RESTRICT[@]}" "$STAGE/baked" || true
CONTEXTS+=(--build-context "baked=$STAGE/baked")

# What the image says about itself. A tarball changes hands and a tag gets
# retyped; a label travels with the bytes, and `docker inspect` answers the
# only question that matters about a copy of this image: may it be passed on?
LABELS=(
    --label "ch.heig-vd.reds.spear.profile=$PROFILE"
    --label "ch.heig-vd.reds.spear.built=$(date -Iseconds)"
)

if [ "$PROFILE" = public ]; then
    LABELS+=(--label "ch.heig-vd.reds.spear.redistributable=true")
else
    LABELS+=(--label "ch.heig-vd.reds.spear.redistributable=false")
fi

echo "== building $TAG =="
echo "   profile         : $PROFILE"
echo "   harness context : $APP"
echo "   repo context    : $REPO  (corpora/ + docker/ only)"
docker build --build-context "repo=$REPO" "${CONTEXTS[@]}" "${LABELS[@]}" \
    -f "$REPO/docker/Dockerfile" -t "$TAG" "$APP"

echo
echo "built $TAG ($PROFILE)"

if [ "$PROFILE" = reds ]; then
    cat <<MSG

  This image carries the institute's trees and their index, for the institute's
  own registry. Tag it there and push; scripts/docker/push.sh refuses any other
  registry without --allow-push:

      docker tag $TAG reds-gitlab.heig-vd.ch:5050/<group>/spear:${TAG#spear:}
      scripts/docker/push.sh reds-gitlab.heig-vd.ch:5050/<group>/spear:${TAG#spear:}
MSG
fi

if [ "$PROFILE" = private ]; then
    cat <<MSG

  This image carries licensed normative material and customer trees. It is
  labelled redistributable=false and scripts/docker/push.sh will refuse to publish it
  without --allow-push. Hand it over as a file:

      docker save $TAG | zstd -T0 -19 -o spear-private.tar.zst
MSG
fi
