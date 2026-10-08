# SPEAR harness, self-contained enough to hand to a colleague.
#
# The embedder (bge-m3, 4.3 GB) is always baked in. The ChromaDB index is baked
# in WHEN THE BUILDING HOST HAS ONE: retrieval then works the moment the
# container starts, with no re-indexing -- which takes hours and needs every
# corpus tree present. The index is built on the host and is not in the
# repository, so a clean clone produces an image that starts and answers, with
# retrieval empty until `spear-index` fills it. Only the source trees are
# mounted, because they are the user's own working copies and change daily.
#
# Corpora are addressed RELATIVE to SPEAR_CORPUS_ROOT (see projects.docker.json)
# so this image works for someone whose checkouts are not under /home/operator.
#
# Build: scripts/docker/build.sh -- the contexts are not the obvious ones, see there.
#     scripts/docker/build.sh                 # from anywhere; wraps both contexts
# Run:
#     scripts/docker/spear-docker.sh            # see that script for the flags and why
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
# bubblewrap is not optional: the harness runs every command in it and refuses
# to run any if it is missing. The build tools are there because a sandboxed
# command is often `make` -- the model must be able to test what it wrote.
#
# The second block is bitbake's host prerequisites, taken from ib.md's list.
# They are not optional either: without a generated en_US.UTF-8 bitbake
# refuses to start, and the message blames the host rather than the image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev \
        bubblewrap ca-certificates curl git openssh-client \
        build-essential cmake ninja-build pkg-config \
        ripgrep less file \
        locales \
        gawk diffstat chrpath socat zstd lz4 texinfo \
        python3-pexpect python3-git python3-jinja2 \
        bison flex bc cpio unzip rsync device-tree-compiler u-boot-tools \
    && sed -i "s/^# *en_US.UTF-8/en_US.UTF-8/" /etc/locale.gen \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

ARG APP=/opt/spear/client
WORKDIR $APP

# Dependencies first: they change far less often than the harness code, so an
# edit to rag_chat.py must not re-resolve the whole wheel set.
COPY deploy/requirements.txt deploy/requirements.txt
# CPU torch FIRST. sentence-transformers depends on torch, so letting pip
# resolve it from PyPI pulls the CUDA build and ~3 GB of nvidia_* wheels; a CPU
# wheel installed afterwards replaces torch but leaves those orphans behind in
# the image. Satisfying the dependency up front means they are never fetched.
# CPU is the right target anyway: the index ships prebuilt, so the only
# embedding this container ever does is one query at a time.
RUN python3 -m venv "$APP" \
    && "$APP/bin/pip" install --no-cache-dir --upgrade pip \
    && "$APP/bin/pip" install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu torch \
    && "$APP/bin/pip" install --no-cache-dir -r deploy/requirements.txt

# The embedder, downloaded at BUILD time. Left to first run it would be a
# 4.3 GB surprise on a machine that may have no HF access at all.
ENV HF_HOME=/opt/spear/hf
ENV HF_HUB_DISABLE_XET=1
RUN "$APP/bin/python" -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('BAAI/bge-m3')" \
    && find /opt/spear/hf -name '*.onnx' -delete 2>/dev/null || true

# Chroma's own default embedder (MiniLM, ONNX), which the skill library and the
# conversation archive still use. Chroma fetches it into ~/.cache on first use,
# and a container runs with a HOME of the caller's choosing: every fresh state
# directory meant a 79 MB download, and none at all offline. Fetched here into
# a fixed home; the entrypoint links it into whichever HOME the run has.
RUN HOME=/opt/spear/chroma-home "$APP/bin/python" -c "\
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2; \
ONNXMiniLM_L6_V2()(['warm'])" \
    && chmod -R a+rX /opt/spear/chroma-home

# The prebuilt index, then the code.
# --chmod on the COPY itself, not a later RUN: a separate chmod would rewrite
# every file into a second 4.2 GB layer. ChromaDB opens its sqlite read-write
# even to query it (WAL, migrations), so an index owned by root is unusable the
# moment the container runs as the host user -- which it must, or every file it
# leaves on a mounted volume is root-owned.
COPY --chmod=0777 --from=index . $APP/chromadb/
COPY *.md $APP/
COPY agent/ $APP/agent/
COPY cli/ $APP/cli/
COPY context/ $APP/context/
COPY evidence/ $APP/evidence/
COPY harness/ $APP/harness/
COPY models/ $APP/models/
COPY normative/ $APP/normative/
COPY retrieval/ $APP/retrieval/
COPY runtime/ $APP/runtime/
COPY standard/ $APP/standard/
COPY training/ $APP/training/
# rules.d/, skills/ and benches/ are CONTENT, and a deployment's own copy is
# usually not in the repository -- see build.sh, which resolves each to the
# in-tree directory, to its SPEAR_*_DIR, or to an empty one. An empty context
# still creates the directory, so the harness finds a directory with nothing
# in it rather than no directory at all.
COPY --from=rules . $APP/rules.d/
COPY --from=skills . $APP/skills/
COPY --from=benches . $APP/benches/
COPY eval/ $APP/eval/
COPY deploy/ $APP/deploy/
COPY --from=repo docker/projects.docker.json $APP/projects.json

# The cross-cutting parts of the spear repo, baked so the image is useful to
# someone who has not cloned it: the shared notes corpus, and `corpora/`, which
# holds the API references. They come from NAMED BUILD CONTEXTS rather than the
# main one, which keeps the build context at client/ and therefore keeps the
# expensive layers above cached.
#
# The notes corpus is a deployment's own writing and is not in this repository;
# `notes` is empty unless build.sh found one, and the corpus is then simply
# absent from the image.
#
# Deliberately NOT baked: llama.cpp-next, qwen3-finetune and src -- gigabytes
# of venvs and build artefacts whose retrieval value is small -- and models/
# and the served GGUFs, which are weights, not corpus.
#
# A bind mount on /corpora/spear shadows all of this, which is the intended
# behaviour: someone who HAS the repo works from their own copy.
COPY --from=notes . /corpora/spear/claude/
COPY --from=repo corpora/ /corpora/spear/corpora/

# The trees a build chose to bake, at the paths the generated registry already
# resolves them to. Empty unless --bake named some, and shadowed by a bind
# mount on /corpora, so a workstation keeps working from its own checkouts.
# This exists for the container that is handed over with nothing mounted,
# where an empty /corpora is not a stale corpus but no corpus at all.
COPY --from=baked --chmod=0777 . /corpora/

# THE COMMON STATE: what the image gives everyone who runs it, read under each
# user's own state and never written. The normative store, so that a harness
# that can bind a standard has one to bind (which documents travel is decided
# per document -- see scripts/docker/stage-standards.py), and the workspace
# knowledge the build was asked to ship (stage-knowledge.py).
COPY --from=common . /opt/spear/common/
ENV SPEAR_COMMON_STATE_DIR=/opt/spear/common
# The standards store stays writable, so that a standard ingested in the
# container can be bound there; it is kept apart from the state directory
# because every documented run mounts its own state over that one.
COPY --from=standards --chmod=0777 . /opt/spear/common/standards/
ENV SPEAR_STANDARDS_ROOT=/opt/spear/common/standards

# The user's own state, for a run that mounts none. 0777 because the session
# WRITES here: history, knowledge, the audit trail, and the container runs as
# whichever uid the operator passes.
RUN mkdir -p /opt/spear/state && chmod 0777 /opt/spear/state
ENV SPEAR_STATE_DIR=/opt/spear/state
# NOT in /usr/local/bin: the host's copy of that directory is mounted over
# it (189 toolchain symlinks that ib.md tells the operator to put there),
# which would hide the entrypoint and stop the container dead.
COPY --from=repo docker/entrypoint.sh /opt/spear/entrypoint.sh
RUN chmod +x /opt/spear/entrypoint.sh

# Where the mounted trees live. Every corpus path in projects.json is relative
# to this, which is the whole point: the colleague mounts THEIR checkouts here.
ENV SPEAR_CORPUS_ROOT=/corpora
ENV PATH=$APP/bin:$PATH
WORKDIR /corpora
ENTRYPOINT ["/opt/spear/entrypoint.sh"]
