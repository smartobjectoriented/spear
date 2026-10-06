.. _introduction:

============
Introduction
============

What SPEAR is
=============

.. figure:: /img/SPEAR-Overview.drawio.png
   :width: 100%
   :alt: SPEAR overall architecture

   Overall architecture: entry points, the Python core, model serving and the
   confined tool execution path.

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

It is **self-hosted**. No prompt, no source file and no command output leaves
the machine unless a tool call is granted the ``network`` capability and routed
through the sandbox's own network stack.

Four request classes
====================

Every request is read for its class, and each class runs where its evidence
can be kept (:doc:`/overview/architecture`):

**GENERAL** and **IMPLEMENTATION**
   no standard engaged. The coding core behind SpearHost; the turn ends with
   implementation evidence — ``VERIFIED``, ``UNVERIFIED`` or ``NO_CHANGE``.

**NORMATIVE**
   a question about a bound standard. The normative runtime answers from the
   document first and cites every normative claim, or withholds the answer and
   says why.

**MIXED**
   a change that must satisfy the bound standard. A normative pre-pass builds a
   constraint packet, the coding core makes the change, and the final source is
   judged against the packet on authoritative evidence alone.

The parts
=========

**A served model.**
   Any OpenAI-compatible endpoint — ``llama-server`` from ``llama.cpp-next``
   on ``127.0.0.1:8080`` by default — or the Anthropic API.  See
   :doc:`/model/model_serving`.

**A retrieval corpus.**
   A vector store indexed from the source trees SPEAR is expected to reason
   about.  See :doc:`/using/retrieval` and :doc:`/using/projects`.

**A normative store.**
   The authoritative specifications a session can be bound to, held as
   provisions rather than as pages: each with its kind, its ordinal, its
   section and its page, so a claim can cite one and be checked against it.
   See :doc:`/reasoning/standards`.

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
    read, write and command crosses the control plane, and a shell command
    cannot write where the file tools may not.

**Final-state verification**
    A change is ``VERIFIED`` only if the checks that show what the answer
    claims ran and passed on the final source — not on an earlier state of it.

**Evidence-based compliance**
    A change that must satisfy a standard is judged constraint by constraint,
    on deterministic source predicates and project-bound conformance checks.
    Where that evidence is missing, the verdict says *compliance not
    demonstrated* — never a guess in either direction.

**Multiple model backends**
    Any OpenAI-compatible endpoint, local or remote, and the Anthropic API.

**Confined execution**
    One rule governs the whole execution path: **fail-closed** — a confinement
    that cannot be applied is an error, never a silent downgrade.

Why the harness is the interesting part
=======================================

A model that can only talk is safe and not very useful.  A model that can run
arbitrary commands is useful and not at all safe.  The harness is the entire
answer to "how do we get the second without the first".

Its design rests on one rule, applied without exception:

.. admonition:: Fail-closed
   :class: important

   Any mechanism that cannot be honoured is an **error**, never a downgrade.
   If the sandbox is unavailable, the command does not run unsandboxed.  If
   the resource-control mechanism is unavailable, the command does not run
   unlimited.  If the network helper cannot attach the way we require, the
   network command does not fall back to a weaker attachment.

That rule is why several code paths look more paranoid than they strictly
need to be, and why the test suite spends as much effort proving that things
*do not* happen as proving that they do.

Layers of confinement
=====================

Four independent layers apply to every sandboxed command.  They are
independent on purpose: each one assumes the others may fail.

.. list-table::
   :header-rows: 1
   :widths: 18 30 52

   * - Layer
     - Mechanism
     - What it bounds
   * - Authorization
     - ``CommandPolicy`` + ``CapabilityPolicy``
     - *whether* a command may run at all, and with which capabilities
   * - Filesystem / namespaces
     - Bubblewrap
     - what the command can see, write and reach
   * - Per-process limits
     - ``prlimit`` inside the sandbox
     - descriptors and core dumps of each process
   * - Whole-tree limits
     - cgroup v2 via a transient systemd scope
     - memory, task count and CPU of the command *and every descendant*

The order in which these wrap each other is fixed and is not an
implementation detail; see :doc:`/harness/sandbox`.

Trust boundaries
================

Three boundaries matter, and it is worth being explicit about which side of
each one the model sits on.

*The model is untrusted input.*
   It proposes tool calls; it does not authorize them.  Every proposal goes
   through classification and capability intersection before anything runs.

*The command argv is untrusted data.*
   It is never passed to a shell by the harness.  ``shell=False`` everywhere,
   structured argv everywhere.  When a command genuinely needs shell syntax,
   that is a distinct capability (``shell:complex``) and an explicit
   ``/bin/sh -c`` argv, not string interpolation.

*The workspace is the only writable host surface.*
   Everything else the command sees is read-only or private to the sandbox.

What this documentation covers
==============================

The chapters follow the life of a tool call: what may run
(:doc:`/harness/security_model`), where it runs (:doc:`/harness/sandbox`), how it reaches the
network if allowed (:doc:`/harness/network`), what resources it may consume
(:doc:`/harness/resource_control`), how all of that is verified (:doc:`/operations/testing`), and
what to look at when it misbehaves (:doc:`/operations/operations`).

Several sections quote real measurements.  Those numbers come from this
machine and are labelled as such; :doc:`/harness/resource_control` discusses which of
them are portable and which are not.

Project
=======

SPEAR is developed at the `REDS institute <https://reds.heig-vd.ch>`_ of
`HEIG-VD <https://heig-vd.ch/>`_, and is published under the Apache License
2.0.
