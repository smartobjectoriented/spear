#!/bin/bash
# Add Claude Code to a SPEAR image that is already built:
#
#     add-claude.sh IMAGE [TAG]
#
# TAG defaults to IMAGE itself: the image gains a layer and keeps its name.
# The image has to be built from a harness that has the `claude` command.

set -e

HERE="$(cd "$(dirname "$0")" && pwd)"
IMAGE="$1"
TAG="${2:-$1}"

[ -n "$IMAGE" ] || { sed -n '2,7p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }

docker build --build-arg "BASE=$IMAGE" -t "$TAG" - < "$HERE/../../docker/Dockerfile.claude"

echo "built $TAG"
