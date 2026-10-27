.. _introduction:

============
Introduction
============

What SPEAR is
=============

.. figure:: /img/261008_SPEAR_Overview.png
   :width: 100%
   :alt: SPEAR general architecture

   General architecture: the control plane routes each task to its lane --
   general, coding, normative or MIXED -- with the context selected for its
   workspace; changes run on the coding core's six tools, external
   capabilities go through the gateway, and the project's own validation
   decides what is shown.

**SPEAR — Specification-driven Platform for Embedded Agentic Reasoning** — is
a platform for engineering work where an agent must reason from an
authoritative technical source, inspect an implementation, change it under
control, and keep the evidence for every conclusion it reports.

It is not a retrieval front-end, not a general coding assistant, and not tied
to one standard. Its design rests on keeping six things apart that a single
agent loop tends to merge:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Concern
     - Where it lives
   * - agentic implementation
     - the **coding core**: a standalone tool-calling loop that reads, edits
       and runs commands (:doc:`/reasoning/implementation`)
   * - control and policy
     - **SpearHost** and the harness: what each call may read, write and run,
       confined and audited (:doc:`/harness/security_model`)
   * - execution evidence
     - the structured record of what every call actually did, and the
       implementation verdict computed from it (:doc:`/reasoning/evidence`)
   * - normative reasoning
     - the **normative runtime**: provisions of a bound standard, retrieved,
       identified and cited (:doc:`/reasoning/standards`)
   * - compliance evidence
     - deterministic source predicates and project-bound conformance checks —
       never a model's opinion (:doc:`/reasoning/mixed`)
   * - final verdicts
     - computed from the evidence on the final source state, not written by
       the model

It is **specification-driven**. A specified system has two sources of truth —
the specification, which says what is required, and the implementation, which
says what the code does — and they are not interchangeable. A claim about what
is *required* may rest only on the authoritative source; the code may
illustrate, compare and contradict, never establish.

It is **workspace-aware**. A turn starts from its workspace — a registered
project, or an unregistered tree on its own — and is given only what belongs to
it or is explicitly generic: rules, skills, :term:`workspace knowledge`, project
metadata and external capabilities, selected deterministically by workspace and
request class, never by resemblance to the request (:ref:`context_selection`).

It is **self-hosted**. No prompt, no source file and no command output leaves
the machine unless a tool call is granted the ``network`` capability and routed
through the sandbox's own network stack.

Four request classes
====================

Every request is read for its class, and each class runs where its evidence
can be kept (:doc:`/overview/architecture`):

**GENERAL** and **IMPLEMENTATION**
   no standard engaged. The :term:`coding core` behind :term:`SpearHost`; the turn ends with
   :term:`implementation evidence` — ``VERIFIED``, ``UNVERIFIED`` or ``NO_CHANGE``.
   A change asked for in a standard-bound session that is not :term:`MIXED` runs on
   the :term:`normative runtime`'s guarded workflow instead (:ref:`workflow`).

**NORMATIVE**
   a question about a bound standard. The normative runtime answers from the
   document first and cites every normative claim, or withholds the answer and
   says why.

**MIXED**
   a change that must satisfy the bound standard. A normative pre-pass builds a
   :term:`constraint packet`, the coding core makes the change, and a post-check judges
   the final source against the packet on authoritative evidence alone. The
   verdict keeps both dimensions: the implementation evidence, and a normative
   status. Compliance is reported only when :term:`normative evidence` establishes it
   — otherwise ``NOT_DEMONSTRATED``, never compliant on a passing build.

The parts
=========

**A served model.**
   Any OpenAI-compatible endpoint — ``llama-server`` from ``llama.cpp-next``
   on ``127.0.0.1:8080`` by default — or the Anthropic API.  The coding core
   needs the OpenAI-compatible interface; the Anthropic backend serves the
   general and normative paths only.  See :doc:`/model/backends`.

**A retrieval corpus.**
   A vector store indexed from the source trees SPEAR is expected to reason
   about.  See :doc:`/using/retrieval` and :doc:`/using/projects`.

**A normative store.**
   The authoritative specifications a session can be bound to, held as
   provisions rather than as pages: each with its kind, its ordinal, its
   section and its page, so a claim can cite one and be checked against it.
   See :doc:`/reasoning/standards`.

**A workspace context.**
   What a turn is told about its workspace: the rules and skills that apply
   to it, its declared build and test commands, and its :term:`workspace
   knowledge`.  See :doc:`/using/context` and :doc:`/using/knowledge`.

**External capabilities.**
   Tools of MCP servers, registered per workspace and reached only through
   SPEAR's gateway.  See :doc:`/using/capabilities`.

**An execution harness.**
   The part that lets the model actually *do* things: read files, run builds,
   run tests.  Everything the model proposes is classified, authorized,
   confined and audited before it runs.  See :doc:`/harness/tool_harness`.

What SPEAR does
===============

**Authoritative-source grounding**
    A specification is ingested once and bound to the machine. A question about
    it is answered from the document first, and every normative claim carries
    the provision it rests on.

**Codebase-aware reasoning**
    Registered source trees are indexed and retrieved from, so questions about
    a project are answered from that project.

**Controlled code modification**
    Changes are made by the coding core inside a contained workspace: every
    read, write and command crosses the :term:`control plane`, and a shell command
    cannot write where the file tools may not.  Generated output and snapshot
    copies are refused to every tool (:term:`target policy`).

**Final-state verification**
    A change is ``VERIFIED`` only if a check ran and passed on the final
    source — not on an earlier state of it (:doc:`/reasoning/evidence`).

**Evidence-based compliance**
    A change that must satisfy a standard is judged constraint by constraint,
    on deterministic source predicates and project-bound conformance checks.
    Where that evidence is missing, the verdict says *compliance not
    demonstrated* — never a guess in either direction.

**Workspace knowledge and external capabilities**
    What is known about a workspace persists across sessions with its
    provenance, and goes stale when the source it was bound to changes.
    External tools are offered only to the workspaces and request classes they
    are registered for, and never to a normative pass.

**Multiple model backends**
    Any OpenAI-compatible endpoint, local or remote, and the Anthropic API
    for the general and normative paths.

**Confined execution**
    Everything the model proposes is classified, authorized, confined and
    audited before it runs, and a confinement that cannot be applied is an
    error, never a silent downgrade (:term:`fail-closed`).

The confined execution path
===========================

A model that can only talk is safe and not very useful; a model that can run
arbitrary commands is useful and not at all safe.  The :term:`harness` is how
SPEAR gets the second without the first, and it rests on three assumptions:
the model is untrusted input — it proposes tool calls, it does not authorize
them; a command's argv is untrusted data, never interpolated into a shell
string; and the workspace is the only writable host surface.

A coding-core call first crosses the :term:`control plane`; a command then
passes a decision layer (what may run at all), a boundary layer (where it runs)
and an execution layer (how it is confined: Bubblewrap, per-process and
whole-tree resource limits).  The layers
are described in :doc:`/harness/tool_harness`, the authorization rules and the
complete fail-closed list in :doc:`/harness/security_model`.

Project
=======

SPEAR is developed at the `REDS institute <https://reds.heig-vd.ch>`_ of
`HEIG-VD <https://heig-vd.ch/>`_, and is published under the Apache License
2.0.
