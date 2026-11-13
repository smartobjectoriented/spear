#!/bin/bash
# Build the tiered images: harness -> base -> project.
#
#   build-tiered.sh harness
#   build-tiered.sh base    --commons a,b,c [--entity NAME] [--licensed]
#                           [--knowledge p,q|all]
#   build-tiered.sh project --from spear:<ver>-base --tree DIR
#                           [--bake a,b] [--standards DIR] [--licensed] [--tag TAG]
#
# A project adds its own standards too: --licensed takes the licensed documents
# of --standards DIR (a store holding, say, only one private standard), appended over the
# base's. So an edgemtech project image = the reds base + a private standard.
#
# This is the FROM chain of PACKAGING-REWORK.md. The legacy flat profiles
# (public|reds|private) stay in build.sh, untouched; this script only knows the
# three new tiers and leaves everything else alone.
#
# It reuses the staging scripts as they are. gen-registry.py takes any non-
# "public" profile as "the whole map", and stage-standards.py's profile is
# mapped here (base -> public, or private with --licensed; project -> private),
# so neither needs a new profile value.

set -e

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
[ -r "$REPO/scripts/common/banner.sh" ] && . "$REPO/scripts/common/banner.sh"
HERE="$REPO/scripts/docker"
APP="${SPEAR_APP:-$REPO/client}"
VERSION="$("$REPO/scripts/spearversion.sh")"
LABEL=spear

die() { echo "build-tiered: $*" >&2; exit 1; }

TIER="${1:-}"; shift || true
case "$TIER" in
    harness|base|project) ;;
    ""|-h|--help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "tier must be harness, base or project, not '$TIER'" ;;
esac

COMMONS="" BAKE="" KNOWLEDGE="" ENTITY="local" FROM_BASE="" TAG=""
STANDARDS_STORE=""
LICENSED=0
TREES=()

while [ $# -gt 0 ]; do
    case "$1" in
        --commons)   COMMONS="$2"; shift 2 ;;
        --bake)      BAKE="$2"; shift 2 ;;
        --knowledge) KNOWLEDGE="$2"; shift 2 ;;
        --entity)    ENTITY="$2"; shift 2 ;;
        --from)      FROM_BASE="$2"; shift 2 ;;
        --tree)      TREES+=("$2"); shift 2 ;;
        --standards) STANDARDS_STORE="$2"; shift 2 ;;
        --licensed)  LICENSED=1; shift ;;
        --tag)       TAG="$2"; shift 2 ;;
        *) die "unknown option: $1" ;;
    esac
done

[ -f "$APP/cli/rag_chat.py" ] || die "no harness under $APP — set SPEAR_APP"

# The machine's own content locations and content trees, exactly as build.sh and
# the launchers read them. Without it a base ships no rules and no skills.

if [ -r "$APP/machine.env" ]; then
    echo "   settings  $APP/machine.env"
    # shellcheck disable=SC1091
    . "$APP/machine.env"
fi

EMPTY="$(mktemp -d)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$EMPTY" "$STAGE"' EXIT

exists() { docker image inspect "$1" >/dev/null 2>&1; }

# redistributable=true only if every baked corpus is public:true and no licensed
# standard travels. Reads projects.json; unknown or unmarked corpus => private.

all_public() {
    local list="$1"
    [ -n "$list" ] || return 0
    python3 -I - "$APP/projects.json" "$list" <<'PY'
import json, sys
reg, names = sys.argv[1], [n for n in sys.argv[2].split(",") if n]
try:
    data = json.load(open(reg))
except Exception:
    sys.exit(1)
for n in names:
    s = data.get(n)
    if not isinstance(s, dict) or s.get("public") is not True:
        sys.exit(1)
sys.exit(0)
PY
}

write_manifest() {
    # $1 = out file, $2 = corpus list
    local out="$1" corpora="$2"
    mkdir -p "$(dirname "$out")"
    python3 -I - "$out" "$corpora" <<PY
import json, sys, datetime
out, corpora = sys.argv[1], sys.argv[2]
doc = {
    "profile": "$TIER",
    "entity": "$ENTITY",
    "version": "$VERSION",
    "built": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    "corpora": [c for c in corpora.split(",") if c],
    "licensed": bool($LICENSED),
    "harness_image": "$HARNESS",
    "base_image": "$FROM_BASE",
}
json.dump(doc, open(out, "w"), indent=2)
print("   manifest  " + out)
PY
}

case "$TIER" in

harness)
    TAG="${TAG:-spear:$VERSION-harness}"
    echo "== building $TAG (harness) =="
    docker build -f "$REPO/docker/Dockerfile.harness" \
        --build-context "repo=$REPO" \
        --label "$LABEL.profile=harness" \
        --label "$LABEL.built=$(date -Iseconds)" \
        --label "$LABEL.redistributable=true" \
        -t "$TAG" "$APP"
    echo "built $TAG (harness)"
    ;;

base)
    [ -n "$COMMONS" ] || die "base needs --commons a,b,c (the common corpora, infrabase included)"
    HARNESS="spear:$VERSION-harness"
    exists "$HARNESS" || die "$HARNESS not found — build it first: spear-image build harness"
    TAG="${TAG:-spear:$VERSION-base}"

    # The registry (whole map), then the commons staged: index cut to them, the
    # content from the machine's dirs and content trees, the standards, the
    # knowledge seed, and the baked corpora.

    "$HERE/gen-registry.py" --profile private || echo "   (some corpora skipped)" >&2

    IDX="${SPEAR_INDEX_DIR:-$APP/chromadb}"
    if [ -d "$IDX" ]; then
        "$HERE/stage-index.py" --bake "$COMMONS" --index "$IDX" "$STAGE/index"
    else
        echo "   index    (absent — base carries no prebuilt index)"; mkdir -p "$STAGE/index"
    fi

    CONTENT=()
    [ -n "${SPEAR_RULES_DIR:-}" ]  && CONTENT+=(--rules "$SPEAR_RULES_DIR")
    [ -n "${SPEAR_SKILLS_DIR:-}" ] && CONTENT+=(--skills "$SPEAR_SKILLS_DIR")
    [ -n "${SPEAR_BENCH_DIR:-}" ]  && CONTENT+=(--benches "$SPEAR_BENCH_DIR")
    if [ -n "${SPEAR_CONTENT_TREES:-}" ]; then
        IFS=: read -r -a _t <<<"$SPEAR_CONTENT_TREES"
        for t in "${_t[@]}"; do [ -d "$t" ] && CONTENT+=(--tree "$t"); done
    fi
    "$HERE/stage-content.py" --bake "$COMMONS" "${CONTENT[@]}" "$STAGE/content"

    STD_PROFILE=public; [ "$LICENSED" = 1 ] && STD_PROFILE=private
    "$HERE/stage-standards.py" --profile "$STD_PROFILE" "$STAGE/standards"

    SPEAR_COMMON_STATE_DIR="${SPEAR_COMMON_STATE_DIR:-}" \
        "$HERE/stage-knowledge.py" --workspaces "$KNOWLEDGE" "$STAGE/common"

    "$HERE/stage-corpora.py" --bake "$COMMONS" --restrict-registry "$STAGE/baked" || true

    NOTES="${SPEAR_NOTES_DIR:-$REPO/claude}"; [ -d "$NOTES" ] || NOTES="$EMPTY"

    write_manifest "$STAGE/manifest/base-manifest.json" "$COMMONS"

    REDIST=false
    if [ "$LICENSED" = 0 ] && all_public "$COMMONS"; then REDIST=true; fi

    echo "== building $TAG (base, entity=$ENTITY, redistributable=$REDIST) =="
    docker build -f "$REPO/docker/Dockerfile.base" \
        --build-arg "HARNESS_IMAGE=$HARNESS" \
        --build-context "repo=$REPO" \
        --build-context "index=$STAGE/index" \
        --build-context "rules=$STAGE/content/rules.d" \
        --build-context "skills=$STAGE/content/skills" \
        --build-context "benches=$STAGE/content/benches" \
        --build-context "notes=$NOTES" \
        --build-context "standards=$STAGE/standards" \
        --build-context "common=$STAGE/common" \
        --build-context "baked=$STAGE/baked" \
        --build-context "manifest=$STAGE/manifest" \
        --label "$LABEL.profile=base" \
        --label "$LABEL.entity=$ENTITY" \
        --label "$LABEL.built=$(date -Iseconds)" \
        --label "$LABEL.redistributable=$REDIST" \
        -t "$TAG" "$APP"
    echo "built $TAG (base)"
    ;;

project)
    [ -n "$FROM_BASE" ] || FROM_BASE="spear:$VERSION-base"
    exists "$FROM_BASE" || die "$FROM_BASE not found — build the base first: spear-image build base ..."
    [ "${#TREES[@]}" -gt 0 ] || die "project needs --tree DIR (the project's spear-private content)"
    TAG="${TAG:-spear:$VERSION-project}"
    HARNESS=""

    "$HERE/gen-registry.py" --profile private || echo "   (some corpora skipped)" >&2

    # The index is the FULL set the base already has plus this project's corpora,
    # so it replaces the base's rather than merging into it.
    ALLBAKE="$BAKE"
    IDX="${SPEAR_INDEX_DIR:-$APP/chromadb}"
    if [ -d "$IDX" ] && [ -n "$ALLBAKE" ]; then
        "$HERE/stage-index.py" --bake "$ALLBAKE" --index "$IDX" "$STAGE/index"
    else
        mkdir -p "$STAGE/index"
    fi

    CONTENT=()
    for t in "${TREES[@]}"; do [ -d "$t" ] && CONTENT+=(--tree "$t") || die "not a directory: $t"; done
    "$HERE/stage-content.py" --bake "$BAKE" "${CONTENT[@]}" "$STAGE/content"

    if [ -n "$BAKE" ]; then
        "$HERE/stage-corpora.py" --bake "$BAKE" --restrict-registry "$STAGE/baked" || true
    else
        mkdir -p "$STAGE/baked"
    fi

    # The project's own standards: a licensed document it is entitled to
    # (--licensed), from its own store (--standards), appended over the base's.
    # Empty otherwise, so the base's standards are kept untouched.

    if [ -n "$STANDARDS_STORE" ] || [ "$LICENSED" = 1 ]; then
        STD_PROFILE=public; [ "$LICENSED" = 1 ] && STD_PROFILE=private
        STORE_ARG=(); [ -n "$STANDARDS_STORE" ] && STORE_ARG=(--store "$STANDARDS_STORE")
        "$HERE/stage-standards.py" --profile "$STD_PROFILE" "${STORE_ARG[@]}" "$STAGE/standards"
    else
        mkdir -p "$STAGE/standards"
    fi

    write_manifest "$STAGE/manifest/project-manifest.json" "$BAKE"

    echo "== building $TAG (project, FROM $FROM_BASE) =="
    docker build -f "$REPO/docker/Dockerfile.project" \
        --build-arg "BASE_IMAGE=$FROM_BASE" \
        --build-context "repo=$REPO" \
        --build-context "index=$STAGE/index" \
        --build-context "rules=$STAGE/content/rules.d" \
        --build-context "skills=$STAGE/content/skills" \
        --build-context "benches=$STAGE/content/benches" \
        --build-context "baked=$STAGE/baked" \
        --build-context "standards=$STAGE/standards" \
        --build-context "manifest=$STAGE/manifest" \
        --label "$LABEL.profile=project" \
        --label "$LABEL.entity=$ENTITY" \
        --label "$LABEL.built=$(date -Iseconds)" \
        --label "$LABEL.redistributable=false" \
        -t "$TAG" "$APP"
    echo "built $TAG (project)"
    ;;
esac
