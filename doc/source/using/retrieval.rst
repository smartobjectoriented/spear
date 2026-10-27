.. _retrieval:

=========
Retrieval
=========

The assistant is expected to answer questions about specific code bases —
SO3, BitBake-based build systems, LVGL, U-Boot — that no general model has memorised.
Retrieval is what closes that gap.

Corpus
======

The vector store lives in ``client/chromadb/``.  It is entirely
rebuildable: nothing in it is authoritative, it is a derived index of source
trees that live elsewhere.

Two ingestion entry points:

``index_corpus.py``
   Full corpus build across the configured project set.

``index_dir.py``
   Incremental ingestion of a single directory, for when one tree has moved on
   and a full rebuild would be wasteful.

Both honour ``SPEAR_DB_PATH`` (``--db-path``), so a test — or a container —
indexes into its own store rather than the one the assistant queries.  Both
answer ``--help`` with their usage and index nothing.

Which tree, and which collection
--------------------------------

``/reindex`` rebuilds the *corpus's* index — its tree and its collection — and
neither is the current directory whenever the session runs outside its corpus,
which is the normal case after a workspace split.  Registered corpora that sit
under that tree are excluded from it: a tree owning an index is never part of
another one, and re-walking them would spend the file cap on trees indexed
elsewhere.

The registry itself is one implementation with two front doors: ``/corpus``
(``list``/``add``/``rm``/``scan``) inside a session and the ``spear-corpus``
CLI, both calling ``handle_corpus_command``.

A truncated index is a refusal, not a warning
---------------------------------------------

``index_dir.py`` walks until it hits ``--max-files`` (60000 by default).
Reaching that cap is an **error**: exit 2, nothing embedded, nothing committed,
and the previous index left intact — the staged collection is only swapped in
at the very end.

The reason is that a partial index is not a sample.  It is whichever
directories ``os.walk`` reached first, and it answers confidently from that
fraction: a build-system tree that vendors QEMU and U-Boot can yield an index
of vendored source with **none** of the build system it exists for, and
nothing fails.  The refusal names the directories that consumed the budget, so
the fix — usually ``--exclude`` on a vendored tree that belongs in its own
corpus — is visible in the message.  ``--allow-partial`` still builds it, for
whoever has a reason.

Cross-cutting notes are registered as a separate ``"shared": true`` corpus.
They are indexed once and attached at retrieval time, so changing a note does
not require rebuilding every project collection.

Which trees are corpora, and how a path resolves, is the registry's business
(:ref:`projects`).

Federated and shared corpora
============================

A session retrieves from more than its own corpus when either applies:

``"corpora": ["a", "b"]`` on the project
   A :term:`federation` of one tree's parts — a kernel question, a userspace question
   and a bootloader question all belong to the same checkout.

``"shared": true`` on a corpus
   Attached to **every** session.  Cross-cutting knowledge belongs to no single
   tree: the build system is not the property of one product checkout, and
   without this it would have to be redeclared in every project.

``--with NAME`` adds one for a single session and ``--without NAME`` removes
any of them, so a shared corpus is never a sentence.

They are **not merged into one index**.  Each contributes its own top-k and
Reciprocal Rank Fusion merges the rankings, so the first hit of a small corpus
weighs as much as the first of a large one.  Merging would let size decide:
U-Boot holds 11298 indexable files against SO3's 1482, and kernel code would
compete 7-to-1 for the same twelve slots.

Federated and shared corpora *add to* the session's own index rather than replacing it.
The prefix rewrites each chunk's ``# File:`` header so the path is usable from
where the tools actually run — relative while the corpus sits under the launch
directory, **absolute** otherwise, because ``../../../srv/src/...`` is both
unusable and refused by the command policy.  Absolute paths outside the
workspace are readable (:doc:`/harness/security_model`), so what the model is shown is
what it can open.

.. _retrieval-measured-effect:

Measured effect
---------------

On 37 questions about a BitBake-based build system, scored on whether the answer
names the real identifiers:

.. list-table::
   :header-rows: 1
   :widths: 46 27 27

   * - Retrieval
     - Identifier recall
     - Fully answered
   * - none (cold)
     - 18–19 %
     - 3–5 / 37
   * - one corpus
     - 88–89 %
     - 31–32 / 37
   * - federation of three
     - **90 %**
     - **32 / 37**

The federation's gain on the headline is within noise; its real effect is a
redistribution — the storage section went from 60–70 % to 90 %, while two
others each lost one identifier to the increased competition for the twelve
slots.  The five questions that still fail all had a *full* context budget, so
what remains is the model not naming an identifier it was shown: a behaviour
gap, not a retrieval gap.

Prompt assembly
===============

On the :term:`normative runtime`, a turn's context is composed by ``ContextEngine``
from explicit layers (an implementation turn on the :term:`coding core` builds its own
request from the project's rules, :term:`workspace knowledge`, skills and admitted
external capabilities, and reads the tree through its tools instead of from
retrieved chunks):

#. system and tool-use guidance;
#. matching project rules;
#. the current ``WorkingState`` projection;
#. compacted and recent conversation;
#. the skills the workspace's scope admits, and retrieved corpus chunks;
#. bounded tool evidence.

Every item retains provenance, priority and a token estimate.  Stable rules do
not contain current plan, failure or verification state.  Those dynamic facts
come from the WorkingState layer.

Workspace knowledge, rules and skills
=====================================

What a workspace keeps across sessions is its workspace knowledge
(:ref:`knowledge`), selected by exact workspace and lifecycle, never by
resemblance.  Rules, skills and orientation maps are selected the same way,
deterministically (:ref:`context_selection`).

.. _embedding-offload:

Embedding on another machine
============================

Corpus embedding is the one stage worth moving.  Measured on identical real
chunks (median 1533 characters): 28 chunks/s on the laptop's RTX 4060 against
279 on an RTX PRO 6000.  Reading and chunking 984 files takes 0.17 s, so
indexing *is* embedding.  Only the compute moves — chunking and the Chroma
writes stay local, so there is no source tree to mirror and no index to copy
back.

Queries are never offloaded: one sentence per turn embeds in 44 ms here
against 58 ms there plus the round trip, so the laptop wins.

The client sends the semantics; the worker executes
---------------------------------------------------

The client and the worker speak a versioned protocol, defined in
``server/embed/protocol.py``.  Every field that decides what a vector *means*
travels in the request:

.. list-table::
   :header-rows: 1
   :widths: 34 22 44

   * - Decision
     - Decided by
     - Why
   * - model, prefix, sequence cap, normalisation, batch size
     - **client**
     - they define the collection, and the client owns the collection
   * - device, which card, dtype
     - **server**
     - properties of the machine, which a laptop cannot know

The worker has **no model registry**, no default prefix and no per-model
branch.  That is the correction this protocol exists to make: the worker it
replaces imported the client's module on the GPU host and read the client's
registry from whichever copy was installed there, so the document prefix, the
cap and the normalisation were decided by a file nobody was comparing against
the client's.  A collection filled with two different prefixes is inconsistent
in a way no later query reports — results merely get worse.

``spear_embed_protocol`` is carried and checked in both directions.  A
mismatch is refused, naming both numbers, and is never negotiated down: two
sides that still parse each other's bytes while disagreeing about who applies
the prefix would produce vectors that are subtly wrong and perfectly
well-formed.

``server/embed/README.md`` is the deployment contract — what to copy, how to
configure it, and how to verify it without indexing anything.

A configured destination is required, never preferred
-----------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Configuration
     - Behaviour
   * - no destination
     - local embedding, a valid deployment
   * - destination configured
     - remote, **or an error**.  Never a local substitute.

Falling back to local compute when the remote embedder fails looks like a
kindness — a slower index beats a failed one.  It is not.  The two devices
load the same weights and do not produce the same vectors, so a collection
half-filled from each is quietly inconsistent.  ``RemoteEmbeddingError`` is
raised and never swallowed.

Before a deployment switches workers, ``client/tests/test_embed_equivalence.py``
runs both on real weights and requires the vectors to be **bit-identical** —
not merely close.  Measured on ``BAAI/bge-m3``, max absolute difference **0.0**
for documents and for queries, on CPU/fp32 and on CUDA/fp16 alike.

The tolerance question is not hand-waved.  The same texts embedded singly and
in one batch differ by padding alone: ~2e-07 on CPU/fp32 and ~4.9e-04 on
CUDA/fp16.  A tolerance loose enough to cover the GPU figure would be large
enough to hide a genuinely different encoding, so bit-exactness is the
criterion — and it is available precisely because both paths reduce to the
same call.

That second number is worth knowing operationally too: **changing the batch
size changes the vectors**, on fp16 well above float32 noise, though at cosine
0.999999 it is far below anything retrieval can notice.

Retrieval budget
================

Retrieval competes with the conversation for the context window (asked of the
server at startup, 32 768 tokens when it cannot say).  Two mechanisms keep it
bounded:

* the number and size of retrieved chunks are capped at assembly time;
* tool output is truncated before it reaches the model — 10 000 characters on
  the normative runtime, 50 000 on the coding core.

The second one matters more than it looks.  A single ``make`` on a failing
build can emit hundreds of kilobytes; without truncation one tool call would
evict the entire conversation and the assistant would lose the thread of what
it was doing.
