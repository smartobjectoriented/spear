.. _harness_internals:

==================
Runtime internals
==================

What a contributor needs to know about who owns what inside the runtime. A
user of SPEAR can skip this page; :doc:`/overview/architecture` is the picture
from the outside.

Task ownership
==============

On the :term:`normative runtime`, ``WorkingState`` is authoritative task truth.
It changes only through typed, grounded events.  Conversation, retrieved
context, durable memory and compacted summaries are context sources, not
alternative task-state stores.

``TaskController`` owns the bounded synchronous lifecycle: optional planning,
Main execution, verification, and checkpoint finalization by default. Explorer,
Reviewer and their bounded repair cycle are explicit experimental opt-ins. It
receives an already constructed ``AgentContext`` and injected callbacks for
project bench execution, diff evidence, child sessions, and the remaining
code-block compatibility path.  It does not read terminal input, render
output, choose a provider, or construct a filesystem persistence
implementation.

Context and memory
==================

On the normative runtime, ``ContextEngine`` composes the context.  It accounts
for system rules, project rules, WorkingState projection, conversation
summaries, recent conversation, retrieval and tool evidence.

The :term:`coding core` builds its own request from the selected rules, project
metadata, skills, :term:`workspace knowledge` and admitted capabilities, and
reads the tree through its tools; it does not compact, it stops at half the
window and asks for a summary.  Each of its model responses is bounded by
``SPEAR_RESPONSE_MAX_TOKENS`` (:doc:`/using/configuration`).

Workspace knowledge is owned by ``workspace_knowledge.py`` and described in
:doc:`/using/knowledge`; external capabilities by ``capabilities.py``,
``capability_gateway.py`` and ``mcp_provider.py``, described in
:doc:`/using/capabilities`.

Persistence and retention
=========================

Where each persistent artifact lives is listed in
:doc:`/operations/directory_layout`.  No automatic garbage collector runs.
Closed session bundles may be archived or deleted manually as a unit once
their checkpoint and result references are no longer needed.  Deleting a trace
or trajectory never repairs or invalidates a session; deleting referenced
results or checkpoints makes the associated evidence unavailable.

Dependency direction
====================

Lower runtime modules do not import ``rag_chat`` or the ``cli/`` modules it is
built from.  The client owns backend construction, project selection, terminal
commands, human confirmation and presentation.  ``TaskController`` depends on
provider-neutral policies and protocols; ``AgentRuntime`` does not depend on
``TaskController`` or on Explorer/Reviewer orchestration.

The training subsystem (:doc:`/model/training`) is fed rather than called, and
its coupling runs one way: ``AgentRuntime`` and ``TaskController`` emit
trajectories, ``rag_chat`` owns the ``/finetune`` operator command, and
``training_handoff`` may stop the inference service, because on a single-GPU
host the card that trains is the card that serves.

Model capability boundary
=========================

``ModelBackend.complete`` is the intentionally small required protocol.
Canonical ``ModelTurn`` carries stop reason and optional usage, while context
and output limits are explicit ``AgentContext`` configuration.  Cancellation
is checked immediately before and after calls for every backend.  There is no
mandatory capability object: the adapters cannot truthfully promise
transport-level cancellation or exact token counting, and making those flags
mandatory would couple the runtime to provider behavior.  An adapter may
expose optional native capabilities without changing the canonical turn
contract.

Default profile
===============

Requests with no standard engaged run on the coding core. ``--fresh`` starts a
session without its stored conversation; knowledge, rules and configuration
still apply. Requests in a standard-bound session run on the normative runtime,
or through the :term:`MIXED` orchestration when they ask for a change in the
standard's terms. On the normative runtime, planning is deterministic and
conservative; Explorer, Reviewer and reviewer repair are disabled unless the
caller explicitly enables them.
