=========
Retrieval
=========

The assistant is expected to answer questions about specific code bases —
SO3, the EDGE-M1 products, LVGL, U-Boot — that no general model has memorised.
Retrieval is what closes that gap.

Corpus
======

The vector store lives in ``spear/chromadb/``.  It is entirely
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
answer ``--help`` with their usage: ``spear-index --help`` used to fall through
the parser as an unrecognised flag and start indexing the current directory.

Which tree, and which collection
--------------------------------

``/reindex`` rebuilds the *corpus's* index — its tree and its collection — and
neither is the current directory whenever the session runs outside its corpus,
which is the normal case after a workspace split.  Registered corpora that sit
under that tree are excluded from it: a tree owning an index is never part of
another one, and re-walking four of them is how a 60000-file cap was reached.

The registry itself is one implementation with two front doors: ``/corpus``
(``list``/``add``/``rm``/``scan``) inside a session and the ``spear-corpus``
CLI, both calling ``handle_corpus_command``.

A truncated index is a refusal, not a warning
---------------------------------------------

``index_dir.py`` walks until it hits ``--max-files`` (60000 by default).
Reaching that cap used to print a warning and index the prefix anyway.  It is
now an **error**: exit 2, nothing embedded, nothing committed, and the previous
index left intact — the staged collection is only swapped in at the very end.

The reason is that a partial index is not a sample.  It is whichever
directories ``os.walk`` reached first, and it answers confidently from that
fraction.  An ``infrabase`` index built this way came out 99.8 % vendored QEMU
and U-Boot source, with **none** of the build system it existed for, and
nothing failed.  The refusal names the directories that consumed the budget, so
the fix — usually ``--exclude`` on a vendored tree that belongs in its own
corpus — is visible in the message.  ``--allow-partial`` still builds it, for
whoever has a reason.

Cross-cutting notes are registered as a separate ``"shared": true`` corpus.
They are indexed once and attached at retrieval time, so changing a note does
not require rebuilding every project collection.

For the tracked ``claude/`` store, exclude its archived snapshots and preserve
the registry's portable collection name:

.. code-block:: console

   $ cd spear
   $ bin/python index_dir.py ../claude --collection claude_memories \
       --exclude _originals_backup

Projects
========

``projects.json`` maps short names to absolute paths and a project *kind*:

.. code-block:: json

   {
     "so3":       { "path": "/home/operator/soo/so3/so3",  "kind": "generic" },
     "verdin":    { "path": ".../verdin", "kind": "buildsystem",
                    "collection": "edgem1_verdin", "indexer": "buildsystem",
                    "autoindex": true, "prompt_file": "system-prompt.md" },
     "virt64":    { "path": ".../virt64", "kind": "buildsystem",
                    "collection": "edgem1_virt64", "indexer": "buildsystem",
                    "autoindex": true, "prompt_file": "system-prompt.md" },
     "lvgl":      { "path": "/home/operator/work/lvgl", "kind": "generic" },
     "spear":{ "path": "/opt/llm/spear/spear",         "kind": "generic" }
   }

The ``kind`` selects which rule fragments are injected into the prompt.
``kind`` is a label: it scopes skills and prints in the listing. What a corpus
DOES is declared key by key -- ``indexer`` (``buildsystem`` for the curated
BitBake/Yocto walk, else generic), ``autoindex``, ``prompt_file`` and
``collection``. A ``generic``
ones do not.

A ``path`` is either **absolute** — a tree of yours, genuinely machine-specific
— or **relative**, in which case it resolves against ``SPEAR_CORPUS_ROOT``.
That root defaults to the **repository root**, so the six corpora that live
inside the repository (``llama.cpp-next``, ``qwen3-finetune``, ``src``,
``spear`` and the two under ``corpora/``) are found wherever the
repository is cloned, and a container overrides the root with its mount point
(:doc:`container`).

Federated and shared corpora
============================

A session retrieves from more than its own corpus when either applies:

``"corpora": ["a", "b"]`` on the project
   A federation of one tree's parts — a kernel question, a userspace question
   and a bootloader question all belong to the same checkout.

``"shared": true`` on a corpus
   Attached to **every** session.  Cross-cutting knowledge belongs to no single
   tree: the build system is not the property of one product checkout, and
   without this it would have to be redeclared in all 22 projects.

``--with NAME`` adds one for a single session and ``--without NAME`` removes
any of them, so a shared corpus is never a sentence.

They are **not merged into one index**.  Each contributes its own top-k and
Reciprocal Rank Fusion merges the rankings, so the first hit of a small corpus
weighs as much as the first of a large one.  Merging would let size decide:
U-Boot holds 11298 indexable files against SO3's 1482, and kernel code would
compete 7-to-1 for the same twelve slots.

Attached corpora *add to* the session's own index rather than replacing it.
The prefix rewrites each chunk's ``# File:`` header so the path is usable from
where the tools actually run — relative while the corpus sits under the launch
directory, **absolute** otherwise, because ``../../../opt/llm/...`` is both
unusable and refused by the command policy.  Absolute paths outside the
workspace are readable (:doc:`security_model`), so what the model is shown is
what it can open.

.. _retrieval-measured-effect:

Measured effect
---------------

On 37 questions about the Infrabase build system, scored on whether the answer
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

Corpus and workspace are two roots, not one
===========================================

``set_project`` binds two distinct roots, and the distinction is the whole
safety story:

``CORPUS_ROOT``
   The registered tree the RAG index, history and memories belong to — an
   *identity*.  Selected by the cwd, or forced with ``--corpus``.

``PROJECT_ROOT``
   Where the **tools** run: the real current directory, always.  It is the
   workspace root the execution harness validates every filesystem path
   against (:doc:`security_model`).

Only the cwd moves ``PROJECT_ROOT``.  Nothing in a question does — not a
``--corpus`` flag, and certainly not a corpus name appearing in a sentence.
Letting natural language relocate the sandbox root would hand the workspace
boundary to the model's reading of a phrase, which is precisely what the four
containment layers exist to prevent.

That has an ergonomic cost, so the harness pays it in information instead.
When a question names a *registered corpus other than the current one*, it says
so once and does nothing:

.. code-block:: text

   > generate a simple ping.c to run in so3
     ⎿  'so3' is a registered corpus (/home/operator/soo/so3/so3), but your
        tools run in /opt/llm/spear/spear. To work there:
        cd /home/operator/soo/so3/so3 && spear-chat

Matching is on whole words, longest registered name first (so ``micropython-so3``
wins over ``so3``), once per name per session.  Names that are ordinary
directory words — ``src``, ``lib``, ``build``… — are never treated as mentions,
nor is a name used as a path fragment (``so3/usr/main.c``), nor a registry entry
whose tree has disappeared.

Selecting a corpus without moving the cwd is legitimate but asymmetric: the
model then reads about one tree and acts on another.  The harness warns it
explicitly with a *Working directory* note in the system prompt, telling it to
adapt the retrieved paths.  Prefer ``cd`` to ``--corpus`` when you intend to
edit.

Prompt assembly
===============

A turn's context is composed by ``ContextEngine`` from explicit layers:

#. system and tool-use guidance;
#. matching project rules;
#. selected active durable memories;
#. the current ``WorkingState`` projection;
#. compacted and recent conversation;
#. task-relevant skills and retrieved corpus chunks;
#. bounded tool evidence.

Every item retains provenance, priority and a token estimate.  Stable rules do
not contain current plan, failure or verification state.  Those dynamic facts
come from the WorkingState layer.

Memory selection
================

Project memory remains in human-readable ``memories-*.md`` files.  The
``MarkdownMemoryStore`` imports those lines, applies optional sidecar metadata
for scope and supersession, excludes inactive records, and selects a bounded
lexically relevant set for the current request.  It does not introduce another
vector database or treat memory as grounded task evidence.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Fragment
     - Content
   * - ``rules.d/10-rootfs-packages.md``
     - How to add a package to the target root filesystem.
   * - ``rules.d/20-conventions.md``
     - Coding conventions, including the copyright header rule.
   * - ``rules.d/30-doc-build.md``
     - How the project documentation is built.
   * - ``skills/debug-build-failures.md``
     - Procedure for a failing build.
   * - ``skills/debug-buildsystem-build.md``
     - The Infrabase/bitbake specific variant.
   * - ``skills/improve-c-code-quality.md``
     - Refactoring guidance for C.

Keeping these as files rather than as strings in ``rag_chat.py`` means the
behaviour of the assistant can be adjusted without touching code, and the
diff of a behavioural change is readable.

The skill library
-----------------

``skill_library.py`` owns the format and the file/index contract; the
directory is the source of truth and the ``edgem_skills`` collection is a
derived index of it.  The two are reconciled on the first lookup of a session
by comparing a stored digest, so only what changed is embedded.  Before that,
only ``save_skill`` ever wrote to the collection: a skill added or edited by
hand was listed by ``/skills`` and never injected, and a deleted one kept
being injected.

A skill may open with a ``SKILL.md`` frontmatter block -- the convention
agentskills.io and Hermes Agent use -- which the harness reads and the model
never sees, because it is stripped before injection:

.. code-block:: text

   ---
   name: debug-buildsystem-build
   description: Read the failing bitbake task log.
   scope: [buildsystem]
   requires: [bitbake]
   version: 3
   created: 2026-06-23T13:26:00
   updated: 2026-09-07T15:12:41
   ---

``description``
   One line saying what the procedure is *for*.  It is embedded with the body,
   because a question resembles a purpose more than it resembles steps.  A
   skill that declares none is indexed exactly as before, so the library
   already on disk keeps matching without a re-embed.

``scope``
   ``any`` by default, and then nothing narrows.  Named otherwise, the skill
   is injected only in a corpus whose name or kind it lists -- a bitbake
   procedure has no business in an LVGL session.

``requires``
   Commands that must exist for the procedure to be runnable.  A skill whose
   requirement is absent is withheld rather than injected, and ``/skills``
   says which command is missing: why a procedure was *not* used is worth more
   than the fact that it exists.

``version``/``created``/``updated``
   Written by the harness.  An overwrite bumps the version and keeps the first
   stamp, so a procedure the model rewrote three times says so.  A rewrite
   that does not restate ``scope`` or ``requires`` inherits them: the model
   rewriting a body must not silently widen what an operator narrowed.

A file with no frontmatter, or with a block this reader does not understand,
is a valid skill whose body is the whole file.  Losing a procedure to a stray
colon would be worse than ignoring metadata nobody set.

Teaching a rule at runtime
--------------------------

Three levels exist, and choosing between them is the whole question:

.. list-table::
   :header-rows: 1
   :widths: 22 22 56

   * - Command / file
     - Scope
     - Use it for
   * - ``/remember``
     - one corpus
     - ``memories-<corpus>.md`` — a fact about *this* tree.
   * - ``rules.d/corpora/<name>.md``
     - one corpus
     - The orientation map: where things live, what to never edit.
   * - ``/recall``
     - **every** corpus
     - ``rules-learned.md`` — a working rule true everywhere.

``/recall`` exists because ``/remember`` was the wrong home for something like
*never rewrite an existing copyright header, only extend its year range*: that
is true in SO3, in the EDGE-M1 trees and in ``pos_sol`` alike, and written per
corpus it would be invisible in all the others.  A recalled line is dated and
appended to ``rules-learned.md``, which ``load_rules()`` injects after
``rules.d/*.md`` on **every** request.

That reach is also the cost, so it is budgeted rather than unbounded: the whole
injected rule set is checked against ``RULES_BUDGET`` (8 000 characters) and
both the write and the load say so when it is exceeded.  Nothing is truncated
silently — the operator is told to trim, because deciding which rule stops
being injected is not a decision to take automatically.  ``/recall`` with no
argument prints what has been taught so far.

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

Before a deployment switches workers, ``spear/tests/test_embed_equivalence.py``
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

Retrieval competes with the conversation for the 32 768-token context.  Two
mechanisms keep it bounded:

* the number and size of retrieved chunks are capped at assembly time;
* tool output is truncated to ``max_output_chars`` (10 000) before it ever
  reaches the model.

The second one matters more than it looks.  A single ``make`` on a failing
build can emit hundreds of kilobytes; without truncation one tool call would
evict the entire conversation and the assistant would lose the thread of what
it was doing.

Orientation maps
================

Each corpus may have a short prose map — where user apps live, which build
command to run, what never to edit — injected into the system prompt on every
request.  It is the highest-leverage context the assistant has, and the one
most easily wrong: a stale path in it sent the model looking for a build
script deleted months earlier, while a stale *chunk* would merely have been
outranked.

They live in ``spear/rules.d/corpora/<corpus>.md``, **not** in the tree
they describe.  Two reasons:

* they say how the *assistant* should work, which is this repository's concern,
  not the product's — a prompt tweak should not go through a product's review;
* the in-tree form did not survive the trip.  A blanket ``.*`` line in so3's
  ``.gitignore`` swallowed ``.edgem-rules.md``, so a colleague cloning that
  tree got no rules at all and never knew.

An in-tree ``.edgem-rules.md`` still **wins** when present: a tree someone else
owns may carry its own map, and theirs should beat ours.  Nothing is invented
for a corpus that has neither.
