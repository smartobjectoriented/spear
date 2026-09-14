#!/bin/bash
# Snapshot the SO3 cross-toolchain headers into a stable corpus root.
#
# The toolchain lives under build/tmp/, which the build system deletes and
# rebuilds: a corpus pointing straight at it goes empty without warning, and
# its index silently answers nothing. The header set is an API surface that
# only changes with the toolchain version, so a copy is the honest form.
#
# Re-run after a toolchain rebuild, then reindex whatever it wrote:
#   ./refresh-musl-headers.sh
#   spear-index "$(./refresh-musl-headers.sh --where)"
#
#   refresh-musl-headers.sh [toolchain-include-dir] [destination]
#
# The destination defaults to musl-headers/ beside this script, which is where
# a plain checkout keeps it. It is a PARAMETER and not a fixed location: a
# corpus does not have to live inside the repository, and a deployment that
# keeps its corpora elsewhere should not have to edit tracked source to say
# so. SPEAR_MUSL_HEADERS_DIR sets it too, for when the caller is a cron line
# rather than a person.
set -euo pipefail

SRC="${1:-$HOME/sye/sye_sol/build/tmp/toolchains/arm-linux-musleabihf/arm-linux-musleabihf/include}"
DST="${2:-${SPEAR_MUSL_HEADERS_DIR:-$(dirname "$(readlink -f "$0")")/musl-headers}}"

# `--where` prints the destination and exits, so the reindex command above can
# name it without the caller repeating the default.
if [ "${1:-}" = "--where" ]; then echo "$DST"; exit 0; fi

[ -d "$SRC" ] || { echo "toolchain headers not found: $SRC" >&2; exit 1; }

# c++/ is libstdc++ (14 MB) and linux/ is kernel UAPI (6 MB): neither is the C
# library an SO3 application links against.
rsync -a --delete \
  --exclude 'c++/' --exclude 'linux/' --exclude 'xen/' --exclude 'drm/' \
  --exclude 'mtd/' --exclude 'scsi/' --exclude 'rdma/' --exclude 'sound/' \
  --exclude 'video/' --exclude 'asm/' --exclude 'asm-generic/' \
  "$SRC/" "$DST/"

echo "$(find "$DST" -name '*.h' | wc -l) headers in $DST"
