#!/bin/bash
# Snapshot the SO3 cross-toolchain headers into a stable corpus root.
#
# The toolchain lives under build/tmp/, which the build system deletes and
# rebuilds: a corpus pointing straight at it goes empty without warning, and
# its index silently answers nothing. The header set is an API surface that
# only changes with the toolchain version, so a copy is the honest form.
#
# Re-run after a toolchain rebuild, then reindex:
#   spear-index /opt/llm/spear/corpora/musl-headers
set -euo pipefail

SRC="${1:-$HOME/sye/sye_sol/build/tmp/toolchains/arm-linux-musleabihf/arm-linux-musleabihf/include}"
DST="$(dirname "$(readlink -f "$0")")/musl-headers"

[ -d "$SRC" ] || { echo "toolchain headers not found: $SRC" >&2; exit 1; }

# c++/ is libstdc++ (14 MB) and linux/ is kernel UAPI (6 MB): neither is the C
# library an SO3 application links against.
rsync -a --delete \
  --exclude 'c++/' --exclude 'linux/' --exclude 'xen/' --exclude 'drm/' \
  --exclude 'mtd/' --exclude 'scsi/' --exclude 'rdma/' --exclude 'sound/' \
  --exclude 'video/' --exclude 'asm/' --exclude 'asm-generic/' \
  "$SRC/" "$DST/"

echo "$(find "$DST" -name '*.h' | wc -l) headers in $DST"
