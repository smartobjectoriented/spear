#!/bin/bash
# Publish an image, unless it is one that must not be published.
#
# The two profiles differ in exactly one way that matters once the image
# exists: whether its contents may leave. A tag does not carry that -- it gets
# retyped, shortened and reused -- so the answer is a LABEL, written at build
# time, travelling with the bytes through `docker save`, a registry and back.
#
#     scripts/docker/push.sh ghcr.io/<org>/spear:1.0-public
#     scripts/docker/push.sh --allow-push ghcr.io/<org>/spear-private:1.0-private
#
# --allow-push is not a formality. A private image carries a licensed
# normative corpus -- the original document included, where the store retained
# it -- and a customer's source tree. Publishing one is a decision about a
# licence and a contract, not about a registry, and it should cost a
# deliberate word.
set -e

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

label() {
    docker image inspect "$IMAGE" \
        --format "{{ index .Config.Labels \"ch.heig-vd.reds.spear.$1\" }}" 2>/dev/null
}

docker image inspect "$IMAGE" >/dev/null 2>&1 || {
    echo "no such image: $IMAGE" >&2; exit 1; }

PROFILE="$(label profile)"
OK="$(label redistributable)"

case "$PROFILE" in
    "")
        echo "$IMAGE carries no SPEAR profile label." >&2
        echo "It was not built by scripts/docker/build.sh; refusing to guess." >&2
        exit 1 ;;
    public) ;;
    private)
        if [ "$ALLOW" != 1 ]; then
            cat >&2 <<MSG
Refusing to push $IMAGE.

  profile          $PROFILE
  redistributable  $OK

This image was built with --profile private: it carries licensed normative
material and customer source. Publishing it puts both on the registry's
infrastructure, under whatever access policy that registry has today.

If that is covered by the agreement you are working under, say so explicitly:

    scripts/docker/push.sh --allow-push $IMAGE

To hand it over without a registry:

    docker save $IMAGE | zstd -T0 -19 -o spear-private.tar.zst
MSG
            exit 1
        fi
        echo "!! pushing a private image on --allow-push: $IMAGE" >&2 ;;
esac

exec docker push "$IMAGE"
