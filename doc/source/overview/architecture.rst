.. _architecture:

============
Architecture
============

.. _infrastructure_overview:

Infrastructure overview
=======================

Before the internals, the whole platform on one page.  Read it from the left:
an operator's question or change request enters through ``spear-chat`` and
reaches the reasoning and :term:`control plane`, which draws on two sources of
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

What a turn is told about its workspace — rules, skills, :term:`workspace knowledge`,
project metadata — and the external capabilities it may reach are selected by
the control plane per workspace and request class; they are described below
(:ref:`context_selection`, :doc:`/using/knowledge`, :doc:`/using/capabilities`).

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
A standalone tool-calling loop with six tools whose every call crosses
:term:`SpearHost`, the control plane. The turn ends with :term:`implementation
evidence` computed from the record of what the calls did, never from the
model's account of it (:doc:`/reasoning/evidence`); the checks that may verify
it are the project's declared and probed commands (:doc:`/using/projects`). The
:term:`coding core` needs an OpenAI-compatible endpoint; the Anthropic backend
serves the general and normative paths. See :doc:`/reasoning/implementation`.

**The normative runtime** — requests in a session where a standard is engaged.
The provider-neutral ``AgentRuntime`` loop with the standard's tools, provision
records and the answer guards. It answers normative questions, and holds a
change asked for in such a session that is not :term:`MIXED` to a five-stage workflow.
See :doc:`/reasoning/standards` and :doc:`/reasoning/workflow`.

**The MIXED orchestration** — a change that must satisfy the bound standard.
The :term:`normative runtime` runs first, read-only; its cited provisions become a
:term:`constraint packet`; the coding core makes the change; a post-check judges the
final source against the packet by deterministic evidence providers. The
verdict composes the implementation evidence with a normative status:
compliance only from :term:`normative evidence`, otherwise ``NOT_DEMONSTRATED``. See
:doc:`/reasoning/mixed`.

The three paths share the :term:`harness` underneath — workspace, command policy,
sandbox, audit — and none of them lets a model's statement stand as evidence.

Each path is also given only its own context. A turn starts from its workspace
— the registered project, or an unregistered tree on its own — and a
deterministic selector (``context_selection.py``) keeps the context its class
and pass call for, of this workspace or explicitly generic: rules, skills,
workspace knowledge, project metadata and external capabilities. Every
eligible skill is offered — there is no similarity gate, only a bound of eight
per turn — and knowledge and capabilities reach the passes that run on the
coding core, never a normative one. The selector also names the tool family;
every decision is traced and none is shown to the model
(:ref:`context_selection`).

Runtime components
==================

.. code-block:: text

   CLI / application (rag_chat.py)
              |
              v
       TaskController --- request class, standard binding
          |
       context selection --- rules, skills, knowledge, metadata, capabilities
          |                  |                      |
          v                  v                      v
     coding core       AgentRuntime           MixedOrchestrator
     (agent/)          (normative runtime)      normative pre-pass
          |              ModelBackend           NormativeConstraintSet
          v              WorkingState           coding core
     SpearHost           ContextEngine          post-check
     (control_plane)     VerificationPolicy     composite verdict
       target policy     CheckpointManager
       refusal breaker   SessionStore
       capability gateway --- MCP providers (stdio)
          |                  |
          v                  v
       ToolRouter -- ToolRegistry -- ResultStore
          |
          v
       CommandRunner -- Bubblewrap

The coding core (``client/agent/``) imports nothing of SPEAR: it calls a
``Host`` interface for every read, write and command, and ``SpearHost``
implements that interface with SPEAR's policy. Its evidence is turned into the
implementation verdict by ``completion.py``. In front of every mutation,
``target_policy.py`` applies the :term:`target policy`
(:doc:`/harness/security_model`), and ``refusal_breaker.py`` stops a turn that
keeps repeating a refused operation (:doc:`/harness/tool_harness`). The capability
gateway (``capability_gateway.py``) answers the ``spear-capability`` host
command itself, never through a shell. The MIXED orchestration
(``mixed_orchestration.py``) composes the two runtimes and the normative
evidence modules (``normative_constraints``, ``normative_coverage``,
``normative_predicates``, ``normative_evidence``); it changes neither.

The ``AgentRuntime`` can also run Explorer and Reviewer roles with isolated
contexts and read-only tools; they are experimental and off by default.

Where state is kept
===================

Sessions, tool results, checkpoints, traces and workspace knowledge live in the
state directory, each with its own owner (:doc:`/operations/directory_layout`;
retention in :doc:`/harness/internals`).  Turns are also recorded as
trajectories, which the training subsystem curates into datasets outside the
turn (:doc:`/model/training`).

Contributor internals — task ownership, context composition, dependency
direction and the model capability boundary — are described in
:doc:`/harness/internals`.

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
