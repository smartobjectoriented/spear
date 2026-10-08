.. _final_harness_audit:

=====================
Final harness audit
=====================

This page records the final evidence-driven production profile. It is a
description of the current implementation, not a promise of model quality.

.. figure:: /img/SPEAR-Architecture_dark.drawio.png
   :width: 100%
   :alt: The agent harness in one picture: layers, roles and boundaries

   The whole harness on one sheet — the layers, what is on by default, what is
   optional, what is experimental, and where the security boundary runs.
   :doc:`/overview/architecture` shows how the request classes route through
   it.

Every box in the picture can be checked against the tree.  It replaced a hand-drawn
version of the same figure that had ``TaskController`` in two bands and drew
``RetryPolicy`` beside ``FailurePolicy`` although the second contains the first
(``failure_policy.py``).  That drawing is kept as the
``Final_architecture`` page of ``source/img/spear.drawio`` and in ``_static/``,
because it is a fine overview.

Production maturity
===================

CORE / DEFAULT ON

* request classification and routing (``TaskController``, ``answer_scope``)
* the :term:`coding core` and ``SpearHost``, its :term:`control plane`
* the evidence plane: canonical evidence, source epochs, the implementation
  verdict and project validation
* the :term:`MIXED` orchestration: :term:`constraint packet`, :term:`structural coverage`,
  applicability, source predicates and bound conformance checks
* ``AgentRuntime`` (the normative and general runtime)
* ``WorkingState`` and ``ContextEngine``
* transactional semantic compaction
* selected ``MarkdownMemoryStore`` memories and :term:`workspace knowledge`
  (``KnowledgeStore``)
* ``ToolRegistry`` / ``ToolRouter`` / ``ResultStore``
* ``SessionStore`` and cancellation
* ``VerificationPolicy`` and ``CheckpointManager``
* ``BudgetManager``, ``FailurePolicy`` and ``ProgressMonitor``
* role-aware tool exposure and tracing

OPTIONAL

* deterministic planning when ``PlanningPolicy`` identifies a complex task
* external capabilities, when a deployment registers MCP providers
  (``capability_gateway.py``)
* compatibility code-block mutation fallback
* legacy ``chat_once`` and ``execute_tool`` APIs

EXPERIMENTAL / DEFAULT OFF

* Explorer
* Reviewer
* reviewer-driven repair

OPERATOR-ONLY, OUTSIDE THE TURN

* the training subsystem — capture, curation, governance, readiness, frozen
  bundles, and the launcher.  Nothing in it is model-visible and freezing runs
  no training; its own readiness vocabulary tops out at ``READY_FOR_EXPERIMENT``
  (:doc:`/model/training`).

Explorer and Reviewer remain isolated, read-only roles and can be enabled by
an explicit ``TaskRequest`` or benchmark configuration. They are not
constructed, budgeted or given child sessions on the normal default path.
With the measured Qwen3-Coder-Next configuration, targeted child runs
started but exhausted their eight-turn contract without a valid structured
deliverable; no quality benefit is claimed.

Evidence
========

The initial real-model benchmark achieved 15/18 deterministic fixture tasks
(83.3 percent). Selected memory achieved 4/4 both with and without injection,
with lower context/call cost when selected. Role-aware views expose two or
five tools instead of seven in the tested roles. Targeted context fixtures
caused compaction to reduce an approximately 65K-token estimate to roughly
10K in completed runs. These are finite Qwen3-Coder-Next measurements, not
statistical or universal claims.

Ownership invariants
====================

``WorkingState`` is task truth; ``SessionStore`` is resumable runtime state;
``CheckpointManager`` is filesystem recovery; ``ResultStore`` owns large tool
evidence; ``MarkdownMemoryStore`` owns durable memories and ``KnowledgeStore``
the workspace's knowledge; ``ContextEngine`` is the
only production context composer; and ``BudgetManager`` owns agent-level
budgets. Legacy projections remain only where CLI/history compatibility or
weaker local models require them.

Security and provider boundary
==============================

All command execution still passes ``ToolRouter``'s authorization and the
CommandPolicy/CommandRunner/Bubblewrap boundary — the coding core's included,
through :term:`SpearHost`. The two host commands, ``spear-capability`` and
``spear-knowledge``, are answered by SpearHost and never reach a shell; the MCP
providers behind the first run on the host, outside the sandbox, under the
gateway's read/write policy. Read-only child roles receive
no checkpoint, memory-write, web or mutating-tool capability. Lower runtime
modules do not import the CLI, and provider-specific protocol details remain
inside backend adapters.

Known limitations and future candidates
========================================

The code-block fallback, ``chat_once``/``enforce_ctx_budget`` compatibility
path and string ``execute_tool`` facade remain maintenance debt because tests
and legacy callers still reach them.

``ToolSpec.execution_modes`` is enforced by ``ToolRouter``: a tool called in a
mode its spec does not declare is denied (``execution_mode_denied``), and a
mutating spec must declare its modes and may not declare ``safe``.  It does not
filter what the registry exposes, and it adds to, rather than replaces, the
authorization in ``CapabilityPolicy`` and ``CommandPolicy``
(:doc:`/harness/security_model`). Explorer/Reviewer contract reliability,
long-context benchmark scoring and repair quality need more model evidence.
ToolSearch, parallel agents, model routing, new retrieval, and new planning or
review capabilities are deliberately not implemented.
