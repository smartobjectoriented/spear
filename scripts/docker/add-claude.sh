#!/bin/bash
# Add Claude Code to a SPEAR image that is already built:
#
#     add-claude.sh IMAGE [TAG]
#
# TAG defaults to IMAGE itself: the image gains a layer and keeps its name.
# Run it with spear-run.sh --claude.

set -e

HERE="$(cd "$(dirname "$0")" && pwd)"
IMAGE="$1"
TAG="${2:-$1}"

[ -n "$IMAGE" ] || { sed -n '2,7p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }

ROOT="$(cd "$HERE/../.." && pwd)"

docker build --build-arg "BASE=$IMAGE" --build-context "repo=$ROOT" -t "$TAG" \
    -f "$ROOT/docker/Dockerfile.claude" "$ROOT/docker"

echo "built $TAG"
