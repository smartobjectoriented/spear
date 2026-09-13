#!/bin/bash
# spear-corpus — manage the SPEAR corpus registry (projects.json).
#
#   spear-corpus [list]
#   spear-corpus add <name> [path] [--kind K] [--indexer I]
#                                  [--autoindex] [--prompt-file F]
#                                             (path defaults to the cwd)
#   spear-corpus rm  <name>
#   spear-corpus scan [path] [--min N]          split a workspace by file count
#
# The same thing as /corpus inside spear-chat, and now the same code. The
# registry used to have a SECOND implementation living in ~/.local/bin, with
# its own load/save, its own seed list, its own skip list and a hardcoded path
# to projects.json — so it could not see SPEAR_CORPUS_ROOT, and its notion of
# an indexable file had already drifted from the indexer's.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# The venv interpreter by absolute path, for the reason spear-chat.sh gives:
# `activate` hardcodes a path that a relocated tree no longer has, and the
# fallback to the system python3 lacks every dependency.
PY="$SCRIPT_DIR/bin/python3"
[ -x "$PY" ] || { echo "venv interpreter missing: $PY"; exit 1; }

exec "$PY" -c 'import sys
sys.path.insert(0, sys.argv[1])
import rag_chat
print(rag_chat.handle_corpus_command(sys.argv[2:]))' "$SCRIPT_DIR" "$@"
