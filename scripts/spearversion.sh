#!/bin/sh
# Print the SPEAR release version, e.g. "0.2.0".
#
# The version is derived from the git release tag (vX.Y.Z) of the tree this
# script belongs to, so it never has to be bumped by hand. Only the base
# version is kept: the "-<commits>-g<hash>" and "-dirty" suffixes git-describe
# adds on a commit past the tag are stripped, so both the tagged commit and
# development on top of v0.2.0 report "0.2.0". An "-rc" suffix is kept.
#
# When the tree carries no git metadata (a tarball export, a container image,
# a copy without .git), the SPEAR_VERSION_FALLBACK constant below is printed
# instead. It is bumped with every release -- see
# doc/source/contributing/release_process.rst.
#
# Used by the banner the entry points print (scripts/common/banner.sh) and by
# the documentation (doc/source/conf.py).
#
# Usage: spearversion.sh

SPEAR_VERSION_FALLBACK="0.3.0-rc1"

_tree=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)

if _v=$(git -C "$_tree" describe --tags --match 'v[0-9]*' 2>/dev/null) && [ -n "$_v" ]; then
	_v=$(printf '%s' "$_v" | sed -e 's/-[0-9][0-9]*-g[0-9a-f]*$//' -e 's/-dirty$//')
	printf '%s\n' "${_v#v}"
	exit 0
fi

printf '%s\n' "$SPEAR_VERSION_FALLBACK"
