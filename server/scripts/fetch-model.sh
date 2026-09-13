#!/bin/bash
# Fetch a GGUF, shards and all, into the models directory.
#
#     fetch-model.sh [repo] [file] [dest]
#
# with each argument falling back to SPEAR_SERVER_MODEL_REPO,
# SPEAR_SERVER_MODEL_FILE and SPEAR_SERVER_MODELS_DIR.
#
# Name any member of a shard set, or the first one, and all of them are
# fetched: llama.cpp opens the first and expects its siblings beside it, under
# their real names. A cache blob path will not do, which is why this downloads
# into the destination directly rather than linking out of a cache.
#
# wget rather than a hub client, and this is measured rather than assumed: on
# the GPU host used here a hub client sustained 1.8 MB/s against 40 MB/s for a
# plain HTTPS pull -- twelve hours against thirty minutes for the same 85 GB.
# It is also one fewer Python dependency on a machine that only has to serve.
#
# -c so an interrupted fetch resumes instead of starting over.
set -euo pipefail

REPO="${1:-${SPEAR_SERVER_MODEL_REPO:-}}"
FILE="${2:-${SPEAR_SERVER_MODEL_FILE:-}}"
DEST="${3:-${SPEAR_SERVER_MODELS_DIR:-}}"

missing=()
[ -n "$REPO" ] || missing+=("repo  (argument 1, or SPEAR_SERVER_MODEL_REPO)   e.g. owner/Model-GGUF")
[ -n "$FILE" ] || missing+=("file  (argument 2, or SPEAR_SERVER_MODEL_FILE)   e.g. Model-Q8_0-00001-of-00004.gguf")
[ -n "$DEST" ] || missing+=("dest  (argument 3, or SPEAR_SERVER_MODELS_DIR)   where the weights go")

if [ "${#missing[@]}" -gt 0 ]; then
    {
        echo "fetch-model.sh: nothing to fetch."
        printf '  missing: %s\n' "${missing[@]}"
    } >&2
    exit 78
fi

BASE="${SPEAR_SERVER_HF_ENDPOINT:-https://huggingface.co}/$REPO/resolve/main"

# <stem>-00001-of-000NN.gguf -> every member. Anything else is one file.
name="$(basename "$FILE")"
dir="$(dirname "$FILE")"
[ "$dir" = "." ] && dir=""
files=()
if [[ "$name" =~ ^(.+)-([0-9]{5})-of-([0-9]{5})\.gguf$ ]]; then
    stem="${BASH_REMATCH[1]}"
    total=$((10#${BASH_REMATCH[3]}))
    for i in $(seq 1 "$total"); do
        files+=("$(printf '%s-%05d-of-%05d.gguf' "$stem" "$i" "$total")")
    done
else
    files=("$name")
fi

mkdir -p "$DEST"
cd "$DEST"
echo "== $REPO -> $DEST  (${#files[@]} file(s))"

# In parallel: each shard is one connection, and the link is rarely saturated
# by one. wait fails the script if any of them did.
pids=()
for f in "${files[@]}"; do
    url="$BASE/${dir:+$dir/}$f"
    wget -c -q --show-progress --progress=dot:giga "$url" -O "$f" &
    pids+=($!)
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
[ "$status" -eq 0 ] || { echo "fetch-model.sh: at least one shard failed" >&2; exit 1; }

echo "== done"
ls -la "$DEST"
