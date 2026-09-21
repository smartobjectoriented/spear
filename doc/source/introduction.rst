.. _introduction:

============
Introduction
============

What SPEAR is
================

.. figure:: img/spear_overview.svg
   :width: 100%
   :alt: SPEAR overall architecture

   Overall architecture: entry points, the Python core, model serving and the
   confined tool execution path.

SPEAR is a platform for engineering tasks where an agent must reason from an
authoritative technical source, inspect an implementation, make controlled
changes to it, and retain the evidence for its conclusions.

It is **specification-driven**. A specified system has two sources of truth —
the specification, which says what is required, and the implementation, which
says what the code does — and they are not interchangeable. SPEAR keeps the
two roles distinct throughout: a claim about what is *required* may rest only
on the authoritative source, while the code may illustrate, compare and
contradict but never establish. Where the evidence does not support a claim,
the answer is withheld with the reason rather than issued with a guess. See
:doc:`/standards`.

It is **self-hosted**. No prompt, no source file and no command output leaves
the machine unless a tool call is explicitly granted the ``network``
capability and routed through the sandbox's own network stack.

It has four parts:

**A served model.**
   ``llama-server`` from ``llama.cpp-next`` serves a quantised GGUF model over
   an OpenAI-compatible HTTP API on ``127.0.0.1:8080``.  See
   :doc:`/model/model_serving`.

**A retrieval corpus.**
   A vector store indexed from the source trees the platform is expected to
   reason about: an operating system, a build system, a UI stack, a
   bootloader, or any tree you register.  See :doc:`/retrieval` and
   :doc:`/projects`.

**A normative store.**
   The authoritative specifications a session can be bound to, held as
   provisions rather than as pages: each with its kind, its ordinal, its
   section and its page, so a claim can cite one and be checked against it.
   See :doc:`/standards`.

**An execution harness.**
   The part that lets the model actually *do* things: read files, run builds,
   run tests.  Everything the model proposes is classified, authorized,
   confined and audited before it runs.  This is where most of the engineering
   — and most of this documentation — lives.  See :doc:`/harness/tool_harness`.

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
     - :class:`CommandPolicy` + :class:`CapabilityPolicy`
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
(:doc:`/harness/resource_control`), how all of that is verified (:doc:`/testing`), and
what to look at when it misbehaves (:doc:`/operations`).

Several sections quote real measurements.  Those numbers come from this
machine and are labelled as such; :doc:`/harness/resource_control` discusses which of
them are portable and which are not.
