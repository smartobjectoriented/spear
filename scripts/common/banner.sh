# Source me first thing from every user-facing SPEAR script.
#
# Prints which SPEAR release is running, on stderr so it never mixes with
# what a script writes on stdout:
#
#     [spear v0.2.0] spear-chat --reds --auto
#
# Once per invocation, not once per script: scripts call each other
# (spear-image runs scripts/docker/build.sh, spear-chat re-executes itself
# under systemd-run), so the first one to print exports SPEAR_BANNER_SHOWN and
# the nested ones stay quiet.
#
# The version comes from scripts/spearversion.sh (the git release tag). The
# callers live in three directories, so this file finds the helper from its
# own location, not from the caller's.

if [ -z "${SPEAR_BANNER_SHOWN:-}" ]; then
	_spear_banner_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

	printf '[spear v%s] %s%s\n' \
		"$(sh "$_spear_banner_dir/spearversion.sh")" \
		"$(basename "$0" .sh)" "${*:+ $*}" >&2

	SPEAR_BANNER_SHOWN=1
	export SPEAR_BANNER_SHOWN
	unset _spear_banner_dir
fi
