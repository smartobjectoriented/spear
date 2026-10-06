.. _architecture:

============
Architecture
============

.. _infrastructure_overview:

Infrastructure overview
=======================

Before the internals, the whole platform on one page.  Read it from the left:
an operator's question or change request enters through ``spear-chat`` and
reaches the reasoning and control plane, which draws on two sources of
evidence kept apart on purpose — the authoritative, normative one (standards,
in warm) and the implementation one (repositories, corpora, local tools and
build outputs, in blue).  The model backends only reason; everything that
touches a tree goes through the execution environment and stays inside its
workspace and read/write boundaries.  What comes out on the right is either
grounded — an answer, a plan, a change, a validation result — or an explicit
refusal to answer when the evidence is insufficient.

.. figure:: /img/SPEAR-Infrastructure.drawio.png
   :width: 100%
   :alt: SPEAR infrastructure: user and session, normative and implementation
         evidence, model backends, the runtime and control plane, the
         execution environment, and the outputs

   The SPEAR infrastructure.  The two sources of truth — specification and
   implementation — are kept separate and reconciled only in the runtime.

.. _request_classes:

Request classes and execution paths
===================================

Every request is classified before anything runs, from the request's own words
and from whether a standard is engaged for the session:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Class
     - What it is
   * - ``GENERAL``
     - names no subject of its own, and no standard is engaged
   * - ``IMPLEMENTATION``
     - about the working tree: read it, change it, build it, test it
   * - ``NORMATIVE``
     - about what a bound standard requires
   * - ``MIXED``
     - a change to the tree that must satisfy the bound standard

.. figure:: /img/SPEAR-Requests.drawio.png
   :width: 100%
   :alt: Request routing: the task controller sends unbound requests to the
         coding core behind SpearHost, normative requests to the normative
         runtime, and MIXED requests through the three-pass orchestration

   One request, one class, one path.

The class decides the path:

**The coding core, behind SpearHost** — every request with no standard engaged.
A standalone tool-calling loop with six tools (``read_file``,
``search_files``, ``patch``, ``write_file``, ``delete_file``, ``terminal``)
whose every call crosses SpearHost, the control plane. The turn ends with
implementation evidence computed from the record of what the calls did.
See :doc:`/reasoning/implementation`.

**The normative runtime** — requests in a session where a standard is engaged.
The provider-neutral ``AgentRuntime`` loop with the standard's tools, provision
records and the answer guards. It answers normative questions, and holds a
change asked for in such a session that is not MIXED to a five-stage workflow.
See :doc:`/reasoning/standards` and :doc:`/reasoning/workflow`.

**The MIXED orchestration** — a change that must satisfy the bound standard.
The normative runtime runs first, read-only; its cited provisions become a
constraint packet; the coding core makes the change; the final source is judged
against the packet by deterministic evidence providers. See
:doc:`/reasoning/mixed`.

The three paths share the harness underneath — workspace, command policy,
sandbox, audit — and none of them lets a model's statement stand as evidence.

Runtime components
==================

.. code-block:: text

   CLI / application (rag_chat.py)
              |
              v
       TaskController --- request class, standard binding
          |                  |                      |
          v                  v                      v
     coding core       AgentRuntime           MixedOrchestrator
     (agent/)          (normative runtime)      normative pre-pass
          |              ModelBackend           NormativeConstraintSet
          v              WorkingState           coding core
     SpearHost           ContextEngine          evidence providers
     (control_plane)     VerificationPolicy     composite verdict
          |              CheckpointManager
          |              SessionStore
          v                  |
       ToolRouter -- ToolRegistry -- ResultStore
          |
          v
       CommandRunner -- Bubblewrap

The coding core (``spear/agent/``) imports nothing of SPEAR: it calls a
``Host`` interface for every read, write and command, and ``SpearHost``
implements that interface with SPEAR's policy. Its evidence is turned into the
implementation verdict by ``completion.py``. The MIXED orchestration
(``mixed_orchestration.py``) composes the two runtimes and the normative
evidence modules (``normative_constraints``, ``normative_coverage``,
``normative_predicates``, ``normative_evidence``); it changes neither.

The ``AgentRuntime`` can also run Explorer and Reviewer roles with isolated
contexts and read-only tools; they are experimental and off by default.

Task ownership
==============

On the normative runtime, ``WorkingState`` is authoritative task truth.  It changes only through typed,
grounded events.  Conversation, retrieved context, durable memory and compacted
summaries are context sources, not alternative task-state stores.

``TaskController`` owns the bounded synchronous lifecycle: optional planning,
Main execution, verification, and checkpoint finalization by default. Explorer,
Reviewer and their bounded repair cycle are explicit experimental opt-ins. It
receives an already constructed
``AgentContext`` and injected callbacks for project bench execution, diff
evidence, child sessions, and the remaining code-block compatibility path.  It
does not read terminal input, render output, choose a provider, or construct a
filesystem persistence implementation.

Context and memory
==================

On the normative runtime, ``ContextEngine`` composes the context. The coding
core builds its own request from the project's rules, memories and skills and
reads the tree through its tools; it does not compact, it stops at half the
window and asks for a summary.  It accounts for
system rules, project rules, active durable memories, WorkingState projection,
conversation summaries, recent conversation, retrieval and tool evidence.

``MarkdownMemoryStore`` keeps ``memories-*.md`` as the human-editable source of
truth.  A ``.metadata.json`` sidecar adds IDs, scopes, provenance, tags,
confidence and explicit supersession.  The sidecar does not duplicate memory
content and may be deleted safely; active sessions then lose metadata
enrichment but retain every Markdown memory.  Memory selection is bounded and
lexically relevant when no semantic service is involved.  Memory never
overrides WorkingState.

Persistence ownership and retention
===================================

Persistent artifacts have distinct owners:

* ``audit/runtime-trace.jsonl`` is optional observability and may be rotated or
  deleted without affecting resume.
* ``audit/sessions/`` contains resumable state and references tool results and
  checkpoints.  Do not remove an active session's referenced evidence.
* ``audit/tool-results/`` holds full bounded-out-of-context tool evidence.
* ``audit/checkpoints/`` holds exact pre-mutation bytes for safe rollback.
* ``trajectories.jsonl`` and history files are training and user-facing
  projections, not resumable runtime state.  Recording is deliberately wider
  than judging: a turn the project's bench judged is recorded ``pass`` or
  ``fail``, and a turn nothing judged is recorded ``unrated`` rather than
  dropped.  Gating the recording on a verdict is what once left the dataset
  holding a single trajectory — one project declares a bench — so the width is
  the point, not an oversight.  ``/good`` promotes what was right.
* ``memories-*.md`` and their optional sidecars are durable project knowledge.

No automatic garbage collector currently runs.  Closed session bundles may be
archived or deleted manually as a unit after their checkpoint and result
references are no longer needed.  Deleting a trace or trajectory never repairs
or invalidates a session; deleting referenced results/checkpoints makes the
associated evidence unavailable.

Training data as a by-product
=============================

Fifteen of the modules belong to a subsystem the rest of the harness feeds
rather than calls: canonical episode capture, SFT and preference curation,
governance, readiness, the frozen bundle, and the operator control plane that
launches a job.  It sits outside the turn — nothing in it is model-visible, no
tool reaches it, and freezing a bundle executes no training.  It has its own
chapter, :doc:`/model/training`.

The coupling that does exist runs one way and is deliberate: ``AgentRuntime``
and ``TaskController`` emit trajectories, ``rag_chat`` owns the ``/finetune``
operator command, and ``training_handoff`` is allowed to stop the inference
service because on a single-GPU host the card that trains is the card that
serves.

Dependency direction
====================

Lower runtime modules do not import ``rag_chat``.  ``rag_chat`` owns backend
construction, project selection, terminal commands, human confirmation and
presentation.  ``TaskController`` depends on provider-neutral policies and
protocols; ``AgentRuntime`` does not depend on ``TaskController`` or on
Explorer/Reviewer orchestration.

Model capability boundary
=========================

``ModelBackend.complete`` remains the intentionally small required protocol.
Canonical ``ModelTurn`` carries stop reason and optional usage, while context
and output limits are explicit ``AgentContext`` configuration.  Cancellation
is checked immediately before and after calls for every backend.  No mandatory
capability object was added during consolidation: the current adapters cannot
truthfully promise transport-level cancellation or exact token counting, and
making those flags mandatory would couple the runtime to provider behavior.
Future adapters may expose optional native capabilities without changing the
canonical turn contract.

Default profile
===============

Requests with no standard engaged run on the coding core. Requests in a
standard-bound session run on the normative runtime, or through the MIXED
orchestration when they ask for a change in the standard's terms. On the
normative runtime, planning remains deterministic and conservative; Explorer,
Reviewer and reviewer repair are disabled unless the caller explicitly enables
them.

.. _provenance:

Provenance
==========

The coding core is SPEAR's own module, built by porting the coding loop and the
coding tools of Hermes Agent (Nous Research, MIT license) function by function,
and verifying the port against results captured from Hermes itself. What sits
around it — SpearHost, the command policy and sandbox, the evidence plane, the
normative runtime and the MIXED orchestration — is SPEAR's. Every file copied
or adapted, with its upstream revision, its license and its destination, is
listed in ``THIRD_PARTY_NOTICES.md`` at the repository root.
