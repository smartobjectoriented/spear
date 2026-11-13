#!/bin/bash
# Publish an image, unless it is one that must not be published.
#
# Whether an image's contents may leave is a LABEL written at build time, not
# the tag -- a tag gets retyped, shortened and reused. `redistributable=true`
# (the harness, a public image) goes anywhere. A private image (a base or a
# project, or a legacy reds/private) goes to its entity's own registry, and
# anywhere else only on --allow-push.
#
#     scripts/docker/push.sh ghcr.io/<org>/spear:<version>-harness
#     scripts/docker/push.sh reds-gitlab.heig-vd.ch:5050/<group>/spear:<version>-base
#     scripts/docker/push.sh --allow-push <registry>/<org>/spear:<version>-project
#
# An entity's registry is SPEAR_<ENTITY>_REGISTRY, with reds and edgemtech
# defaulted. --allow-push is not a formality: a private image carries a build
# system, a customer's source tree, or a licensed normative corpus, and
# publishing one is a decision about a licence and a contract.
set -e
# The release this is: "[spear vX.Y.Z] <script> <args>" on stderr, once.
_banner="$(dirname "$(readlink -f "$0")")/../common/banner.sh"
[ -r "$_banner" ] && . "$_banner"

ALLOW=0
ARGS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --allow-push) ALLOW=1; shift ;;
        -h|--help) sed -n '2,20p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *) ARGS+=("$1"); shift ;;
    esac
done

[ "${#ARGS[@]}" -eq 1 ] || { echo "usage: scripts/docker/push.sh [--allow-push] <image>" >&2; exit 1; }
IMAGE="${ARGS[0]}"

docker image inspect "$IMAGE" >/dev/null 2>&1 || {
    echo "no such image: $IMAGE" >&2; exit 1; }

# The label, from the neutral namespace, falling back to the old one so images
# built before the rename still read.

label() {
    local v
    v="$(docker image inspect "$IMAGE" \
        --format "{{ index .Config.Labels \"spear.$1\" }}" 2>/dev/null)"
    [ -n "$v" ] && { printf '%s' "$v"; return; }
    docker image inspect "$IMAGE" \
        --format "{{ index .Config.Labels \"ch.heig-vd.reds.spear.$1\" }}" 2>/dev/null
}

PROFILE="$(label profile)"
OK="$(label redistributable)"
ENTITY="$(label entity)"

[ -n "$PROFILE" ] || {
    echo "$IMAGE carries no SPEAR profile label." >&2
    echo "It was not built by SPEAR's build scripts; refusing to guess." >&2
    exit 1; }

# The registry a private image belongs to, if one is known: an entity's own
# (SPEAR_<ENTITY>_REGISTRY, reds and edgemtech defaulted), or the institute's
# for a legacy reds image. A legacy private image, or an entity with no
# registry, has none -- and then publishing always needs --allow-push.

entity_registry() {
    case "$1" in
        reds)      echo "${SPEAR_REDS_REGISTRY:-reds-gitlab.heig-vd.ch}" ;;
        edgemtech) echo "${SPEAR_EDGEMTECH_REGISTRY:-gitlab.edgemtech.ch}" ;;
        ""|local)  echo "" ;;
        *) local var="SPEAR_$(printf '%s' "$1" | tr '[:lower:]-' '[:upper:]_')_REGISTRY"
           echo "${!var:-}" ;;
    esac
}

case "$PROFILE" in
    reds)         REGISTRY="${SPEAR_REDS_REGISTRY:-reds-gitlab.heig-vd.ch}" ;;
    base|project) REGISTRY="$(entity_registry "$ENTITY")" ;;
    *)            REGISTRY="" ;;
esac

# Redistributable (harness, public) goes anywhere.

if [ "$OK" = true ]; then
    exec docker push "$IMAGE"
fi

# Private, but headed for its own registry: allowed. The registry is a prefix
# of the image name, so a retag anywhere else is what gets caught.

if [ -n "$REGISTRY" ]; then
    case "$IMAGE" in
        "$REGISTRY"/*|"$REGISTRY":*) exec docker push "$IMAGE" ;;
    esac
fi

# Private, headed elsewhere: only on --allow-push.

if [ "$ALLOW" != 1 ]; then
    cat >&2 <<MSG
Refusing to push $IMAGE.

  profile          $PROFILE
  entity           ${ENTITY:-<none>}
  redistributable  ${OK:-false}

This is a private SPEAR image: it carries a build system, a customer's source
tree, or a licensed normative corpus.
MSG
    if [ -n "$REGISTRY" ]; then
        cat >&2 <<MSG
It belongs to its entity's registry, $REGISTRY. Tag it there:

    docker tag $IMAGE $REGISTRY:5050/<group>/spear:${IMAGE##*:}
MSG
    fi
    cat >&2 <<MSG

Or publish it elsewhere on your own authority -- a decision about a licence and
a contract:

    scripts/docker/push.sh --allow-push $IMAGE

To hand it over without a registry:

    docker save $IMAGE | zstd -T0 -19 -o spear-$PROFILE.tar.zst
MSG
    exit 1
fi

echo "!! pushing a private image ($PROFILE, entity=${ENTITY:-none}) on --allow-push: $IMAGE" >&2
exec docker push "$IMAGE"
