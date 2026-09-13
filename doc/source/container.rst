=========
Container
=========

The container exists to hand the assistant to someone else: the harness, its
dependencies, the embedder and a prebuilt retrieval index, in one image.  They
mount their own checkouts and point it at a model endpoint.

Everything about it lives in ``docker/`` at the repository root.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - File
     - Purpose
   * - ``Dockerfile``
     - The image.  Do not build it by hand — see `Two build contexts`_.
   * - ``build.sh``
     - Builds, wrapping both contexts.
   * - ``spear-docker.sh``
     - Runs, deriving the mounts and carrying the two ``--security-opt`` flags.
   * - ``entrypoint.sh``
     - Refuses to start on the two failures that are otherwise silent.
   * - ``projects.docker.json``
     - The corpus registry, in **relative** paths.
   * - ``README.md``
     - The short version, for whoever receives the image.

Daily use
=========

.. code-block:: sh

   spear-docker --reds --auto        # what `spear-chat --reds --auto` does

``~/.local/bin/spear-docker`` is the sibling of ``spear-chat`` and takes the
same arguments; everything after them is passed to the harness unchanged.
Three things the runner does that the native launcher does not have to:

**The SSH tunnel stays on the host.** ``--reds`` opens it exactly as
``spear-chat.sh`` does, reading the same ``reds.conf``, then hands the
container ``SPEAR_API_BASE=http://127.0.0.1:8082/v1``.  Putting the tunnel
inside would mean shipping the keys and ``~/.ssh/config`` into an image meant
to be handed around; ``--network host`` makes ``127.0.0.1`` the same thing on
both sides anyway.

**The current directory is translated.**  The harness runs its tools in the
cwd, whatever the corpus (:doc:`retrieval`), so the host cwd is mapped to the
matching path under ``/corpora`` and passed as the container's working
directory.  Without it every session would start at the mount root and
``cd ~/soo/so3/so3`` would mean nothing.

**Session state is written outside the image**, as the host user.  History,
memories, trajectories and the audit trail accumulate; ``docker run --rm``
would throw them away, and a container running as root would leave them
owned by root and unreadable to the harness running natively.  The default is
``~/.spear/state``, overridable with ``--state DIR`` or
``SPEAR_STATE_DIR``.

That separation is also a change in the harness itself: ``STATE_DIR`` covers
every path that accumulates and defaults to the application directory, so a
workstation launch is unaffected.  Since the agent rework it covers thirteen,
and the five that were added are the ones that make a session reconstructible
rather than merely readable:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Path under ``STATE_DIR``
     - What is lost with it
   * - ``audit/sessions/``
     - the resumable session — an interrupted task cannot be picked up again
   * - ``audit/tool-results/``
     - the full tool output kept out of model context; only the previews the
       model saw would survive
   * - ``audit/checkpoints/``
     - the exact pre-mutation bytes, so a rollback has nothing to restore from
   * - ``audit/runtime-trace.jsonl``
     - the spans: which tool ran, how long, with what outcome
   * - ``audit/tool-actions.jsonl``
     - the metadata-only record of every mutating attempt
   * - ``history*.json``, ``history-archive.jsonl``, ``memories-*.md``,
       ``trajectories.jsonl``, ``.input_history``
     - conversation, durable knowledge, and the trajectories a future
       fine-tune would train on

Evidence inside the image is evidence lost with the container that produced
it, which is precisely the case it exists for.

What is baked, and what is not
==============================

.. list-table::
   :header-rows: 1
   :widths: 30 16 54

   * - Content
     - Where
     - Why
   * - harness + venv
     - image
     - pinned, reproducible
   * - embedder (bge-m3)
     - image, 4.5 G
     - fetching it on first run is a surprise on a machine that may have no
       Hugging Face access at all
   * - ChromaDB index
     - image, 4.5 G
     - re-indexing takes hours *and* needs every corpus tree present — the one
       thing a newcomer does not have
   * - ``claude/`` + ``corpora/``
     - image, 38 M
     - a shared retrieval corpus, attached to every session without duplicating
       the notes into each project index
   * - source trees
     - **mounted**
     - working copies that change daily; an image would be stale the next
       morning
   * - model weights
     - **neither**
     - the harness talks to an endpoint, it does not host a model

Two build contexts
==================

``build.sh`` passes two, and the reason is size:

``.`` (default) → ``spear/``
   Harness code and the prebuilt index.  Narrow on purpose: a wider context
   would be re-transferred on every build and would invalidate the 4.5 GB
   embedder layer.

``repo`` (named) → the repository root
   Only ``claude/``, ``corpora/``, ``docker/entrypoint.sh`` and
   ``docker/projects.docker.json`` are taken from it.  A named context is
   fetched lazily — BuildKit transfers only the paths actually ``COPY``-ed — so
   pointing it at a tree holding 122 GB of weights costs nothing.

A bare ``docker build`` therefore fails on the missing ``--from=repo``.

Relative corpus paths
=====================

``projects.docker.json`` registers every corpus **relative** to
``SPEAR_CORPUS_ROOT`` (``/corpora`` in the image), where the workstation
registry uses absolute paths under ``/home/operator``.  That is what makes one
image work for someone whose checkouts live elsewhere;
``resolve_corpus_path()`` leaves absolute paths untouched, so the workstation
keeps behaving exactly as before (:doc:`retrieval`).

A tree the host does not have is not an error.  The entrypoint prints what
resolved and what did not, at startup, rather than letting a missing corpus
surface three questions later as an empty retrieval.

Mounts are derived, not listed
==============================

Without ``--corpora DIR``, ``spear-docker.sh`` computes one bind per registered
corpus from the registry itself, dropping any path already inside another.

This is not tidiness.  Binding the whole of ``/opt/llm/spear`` — the obvious
single mount — would put ``models/`` (75 G) and the served GGUFs (47 G) inside
the container **read-write**, in the one tree the sandbox grants write access
to, for no retrieval value.  Weights are not corpus.  Deriving the list also
means a corpus added to the registry is mounted without touching this script.

The two flags that are not optional
===================================

.. code-block:: text

   --security-opt seccomp=unconfined --security-opt apparmor=unconfined

The harness runs **every** command inside bubblewrap and refuses to run any
without it (:doc:`sandbox`).  Docker's default seccomp profile blocks
``clone(CLONE_NEWUSER)``, so ``bwrap`` cannot start, and the failure surfaces
as ``sandbox unavailable`` — which reads like a broken harness rather than a
missing run flag.  ``spear-docker.sh`` passes both, and ``entrypoint.sh``
tests ``bwrap`` first and prints exactly this if it cannot.

Neither flag grants the container new privileges on the host: both are about
letting an *unprivileged* namespace be created inside it.  The sandbox is
still what confines the model's commands, and it is still doing its job.

Refreshing the index
====================

The baked index is a snapshot.  Re-index on the workstation, then rebuild:
``chromadb/`` is copied late in the ``Dockerfile``, so only that layer and the
ones after it are rebuilt.

.. code-block:: sh

   spear-index /path/to/tree     # updates chromadb/ on the workstation
   docker/build.sh spear:1.1

Why retrieval is not served from reds-ml
========================================

The obvious economy is to move the embedder and the index onto the machine
that already serves the model, and keep a thin client here.  It was costed on
2026-08-25 and declined; the numbers are worth keeping, because the idea comes
back every time someone looks at the image size.

What it would save, measured on this workstation:

.. list-table::
   :header-rows: 1
   :widths: 60 40

   * - Removed from the local install
     - Size
   * - ``nvidia/`` + ``torch/`` + ``transformers`` + sentence-transformers
     - 4.0 GB
   * - the bge-m3 weights
     - 4.3 GB
   * - the ChromaDB index
     - 4.3 GB
   * - **total**, leaving ~0.2 GB of client
     - **12.6 GB**

The image would fall from 17.1 GB to roughly 5 GB, and the ten-second embedder
load on the first query of a session would disappear, since the model would
stay resident server-side on a GPU.  Latency is not the objection: a round trip
through the SSH tunnel that is already open measures **17 ms**, against ~100 ms
for a local CPU embedding.  (The existing ``deploy/embed_worker.py`` could not
be used as it stands: it opens one SSH connection per batch, **350 ms**, which
is right for indexing 45 000 chunks once and wrong for three embeddings a
turn.  It would need a persistent service behind the tunnel.)

It was declined for three reasons, in increasing order of weight:

* **It ends offline operation.**  Retrieval is what turns 18 % into 90 % on the
  build-system audit.  Making it require a VPN and a reachable reds-ml means a
  session on a train is not a degraded session, it is a different assistant.
* **It empties the container of its purpose.**  The image exists so that
  retrieval works the moment it starts, on a machine that may have no Hugging
  Face access at all.  A 5 GB image that needs a tunnel to answer anything is a
  different product, not a smaller one.
* **reds-ml is a shared login.**  Collections are named from the corpus's
  absolute path, so two people indexing ``~/soo/so3`` under that account would
  write into the same collection.  Fixing that means per-user prefixes and a
  shared ChromaDB on a filesystem that already had 104 GB free against a 97 GB
  Hugging Face cache belonging to someone else.

The work itself is small -- about a day: a retrieve endpoint doing embedding
and query in one round trip, a second port forward in ``spear-chat.sh``, ten
``PersistentClient`` call sites and five embedding ones.  Cheapness was never
the question.
