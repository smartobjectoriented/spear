#!/bin/sh
# Put this tree's commands on PATH:   . ./env.sh   (from the tree's root)
#
# Only scripts/ goes on PATH, not scripts/docker/. The latter holds a
# build.sh, and a shell that has also sourced another tree's env.sh with a
# build.sh of its own would then have two, the first one found winning. The
# scripts in scripts/docker/ are reached through `spear-image` instead.

if [ ! -f "$PWD/env.sh" ] || [ ! -d "$PWD/scripts/docker" ]; then
	echo "env.sh: source it from the SPEAR tree's root (. ./env.sh)" >&2
	return 1 2>/dev/null || exit 1
fi

# Strip the previous tree's contribution first, so `cd ../other-spear &&
# . ./env.sh` does not leave both on PATH with the older one winning.
if [ -n "$SPEAR_ROOT_DIR" ] && [ "$SPEAR_ROOT_DIR" != "$PWD" ]; then
	_escaped_root=$(printf '%s' "$SPEAR_ROOT_DIR" | sed 's|[][\.*^$()+?{}\|]|\\&|g')
	PATH=$(printf '%s:' "$PATH" | \
		sed -e "s|$_escaped_root/scripts:||g" \
		    -e 's/:$//')
	unset _escaped_root
fi

export SPEAR_ROOT_DIR=$PWD

# Prepend, and only once: re-sourcing in the same tree must not grow PATH.
case ":$PATH:" in
	*":$PWD/scripts:"*) ;;
	*) PATH="$PWD/scripts:$PATH" ;;
esac

export PATH
