#!/bin/bash
# Fail loudly on the two things that make this container useless in silence.
set -e
APP=/opt/spear/spear

# 1. bubblewrap. Docker's default seccomp profile blocks clone(CLONE_NEWUSER),
#    so bwrap cannot start -- and the harness then refuses EVERY command with
#    "sandbox unavailable", which reads like a harness bug rather than a
#    missing run flag. Say it here, once, with the fix.
if ! /opt/spear/spear/bin/python - <<'PYBWRAP' 2>/dev/null; then
import sys, tempfile
sys.path.insert(0, "/opt/spear/spear")
from tool_runtime import BubblewrapSandbox, Workspace
# The harness's OWN sandbox, not a simpler bwrap invocation. A weaker probe
# passed while the real thing failed: `bwrap --unshare-user true` needs no
# /proc, so it said OK inside a container where mounting proc is forbidden,
# and every command in the session then died on "sandbox unavailable" -- after
# which the model edited files blind.
with tempfile.TemporaryDirectory() as tmp:
    result = BubblewrapSandbox().run(Workspace.from_path(tmp), ["/bin/true"])
sys.exit(0 if result.ok else 1)
PYBWRAP
    cat >&2 <<'MSG'
bubblewrap cannot start in this container.

The harness runs every command inside bwrap and will refuse to run any
without it. Re-run with:

    --security-opt seccomp=unconfined \
    --security-opt apparmor=unconfined \
    --security-opt systempaths=unconfined

The first two let an UNPRIVILEGED namespace be created inside the container;
the third un-masks the /proc paths Docker hides, without which bwrap cannot
mount a fresh proc inside that namespace. None of them gives the container new
privileges on the host. scripts/docker/spear-docker.sh passes all three.
MSG
    exit 1
fi

# 2. The corpora. An empty mount point is the difference between "the model
#    answers from the index" and "the model cites files that do not exist".
if [ -z "$(ls -A "${SPEAR_CORPUS_ROOT:-/corpora}" 2>/dev/null)" ]; then
    echo "Nothing mounted at ${SPEAR_CORPUS_ROOT:-/corpora}." >&2
    echo "Mount your checkouts there -- see docker/README.md." >&2
    exit 1
fi

# Report what actually resolved, so a missing tree is visible at startup and
# not three questions later as an empty retrieval.
"$APP/bin/python" - <<'PY'
import os, sys
sys.path.insert(0, "/opt/spear/spear")
import rag_chat
found = miss = 0
for name, spec in sorted(rag_chat.load_projects().items()):
    if os.path.isdir(spec["path"]):
        found += 1
    else:
        miss += 1
        print(f"  missing: {name:16s} {spec['path']}")
print(f"corpora: {found} present, {miss} missing "
      f"(root {rag_chat.CORPORA_ROOT or '/'})")

# The image SHIPS its index, so a corpus whose collection is absent is a
# packaging fault, not a user choice. Unchecked, the harness says "no index
# for this corpus yet (optional)" and starts with no retrieval at all -- which
# is the entire reason the tool exists.
import textwrap

import chromadb
have = {c.name for c in chromadb.PersistentClient(path=rag_chat.DB_PATH).list_collections()}
absent = sorted(n for n, spec in rag_chat.load_projects().items()
                if os.path.isdir(spec["path"])
                and rag_chat.collection_name_for(spec) not in have)
if absent:
    # Every name, not the first six. Announcing 11 and listing 6 reads as a
    # bug in the count -- the reader cannot tell which five are missing from
    # the list, and the whole point of the line is to say which corpora will
    # answer without retrieval.
    print(f"  WARNING: {len(absent)} mounted corpora have no collection in the "
          f"baked index:")
    for line in textwrap.wrap(", ".join(absent), 72):
        print(f"    {line}")
    print("  They will answer without retrieval. Re-index on the host and "
          "rebuild (scripts/docker/build.sh).")
else:
    print(f"index: every mounted corpus has its collection ({len(have)} total)")
PY

# The normative store. An image that carries a standard nothing is bound to
# opens looking capable and answers every normative question from the source
# tree -- which is the failure this platform exists to prevent. So say what is
# there, and bind it when there is no ambiguity about which one is meant.
#
# The binding is computed by the harness, never written by hand: it carries
# fingerprints of the corpus and of the retrieval configuration, and a
# fabricated one passes inspection and fails later.
"$APP/bin/python" - <<'PYSTD'
import sys
sys.path.insert(0, "/opt/spear/spear")
import rag_chat

operator = rag_chat.STANDARD_OPERATOR

try:
    ingested = list(operator.store.list_standards())
except Exception as exc:
    print(f"standards: store unavailable ({type(exc).__name__})")
    raise SystemExit(0)

if not ingested:
    print("standards: none in this image — normative answers are unavailable")
    raise SystemExit(0)

try:
    bound = operator.active_binding()
except Exception:
    bound = None

if bound is not None:
    print(f"standards: bound {bound.standard_id} {bound.revision}")
elif len(ingested) == 1:
    standard_id, revision = ingested[0]
    try:
        bound = operator.use(standard_id, revision)
        print(f"standards: bound {bound.standard_id} {bound.revision} "
              f"(the only one in this image)")
    except Exception as exc:
        print(f"standards: {standard_id} {revision} present but could not be "
              f"bound ({type(exc).__name__})")
else:
    print(f"standards: {len(ingested)} available, none bound — "
          f"/standard use <id> <revision>")
    for standard_id, revision in ingested:
        print(f"           {standard_id} {revision}")
PYSTD

exec "$APP/bin/python" "$APP/rag_chat.py" "$@"
