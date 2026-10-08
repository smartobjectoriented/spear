.. _getting_started:

===============
Getting started
===============

SPEAR — Specification-driven Platform for Embedded Agentic Reasoning — is an
engineering agent for specified systems. It changes code inside a confined
workspace and reports what it can *show* about the result; it answers questions
about a bound standard from the standard itself, with citations; and when a
change has to satisfy that standard, it judges the result constraint by
constraint on evidence the model cannot supply. :doc:`/overview/introduction`
explains why it is built that way.

This page goes from nothing to a first change, a first normative answer and a
first MIXED task.

Prerequisites
=============

* Linux with systemd and ``bubblewrap`` — the execution harness confines every
  command in a sandbox and refuses to run without one;
* Python 3.12 for a native installation, or Docker for the container;
* a model endpoint: an OpenAI-compatible server, local or remote
  (:ref:`backends`).

The full list is in :ref:`installation`.

Install
=======

The container path
------------------

If you have access to a model server and want to *use* SPEAR, Docker is the
only prerequisite:

.. code-block:: console

   $ git clone https://github.com/smartobjectoriented/spear ~/spear
   $ cd ~/spear
   $ scripts/docker/build.sh --profile private      # ~20 min, mostly the embedder
   $ scripts/docker/spear-docker.sh --reds --auto   # opens the tunnel, then chats

``spear-docker.sh`` runs the ``private`` image of this machine,
``spear:<version>-private``; a bare ``build.sh`` builds the ``public`` one,
which is the image to hand to someone else (:ref:`image-profiles`). The image
carries the harness and the embedder; you mount your own source trees. :doc:`/start/container_run` is the user's guide, and
:doc:`/start/container` describes what is baked, what is mounted, and the three
``--security-opt`` flags without which the harness refuses to run any command
at all.

The native path
---------------

To change the harness, re-index, or register projects of your own, install it
natively:

.. code-block:: console

   $ git clone https://github.com/smartobjectoriented/spear ~/spear
   $ cd ~/spear
   $ spear/deploy/install.sh        # the virtualenv, under spear/
   $ spear/deploy/preflight.sh      # what the harness needs, checked
   $ spear/spear-chat.sh --help     # every flag and setting

``spear/spear-chat.sh`` is the launcher; linking it onto your ``PATH`` as
``spear-chat`` is the usual arrangement, and the rest of this page assumes it.
See :ref:`installation` for the details.

Configure a model backend
=========================

``spear-chat`` talks to an OpenAI-compatible endpoint. Without a flag it asks
which backend to use and remembers the answer:

.. code-block:: console

   $ spear-chat --local                                    # llama-server on 127.0.0.1:8080
   $ spear-chat --api-base http://gpu-host:8080/v1         # any OpenAI-compatible endpoint
   $ spear-chat --provider anthropic --model <model-id>    # the Anthropic API

The Anthropic backend serves questions and normative answers; changes run on
the coding core, which needs an OpenAI-compatible endpoint. Serving a model
yourself is covered in :doc:`/model/model`.

Use a project
=============

Tools always run in the **current directory**. To work on a tree, start SPEAR
in it:

.. code-block:: console

   $ cd ~/src/acme-firmware
   $ spear-chat --ask              # confirm each change and command

That is an *ad-hoc* project. To register it — so it gets a name, a retrieval
index, declared build and test commands, and its own knowledge — add it to the
registry and index it:

.. code-block:: console

   $ spear-corpus add acme-firmware ~/src/acme-firmware
   $ spear-chat --corpus acme-firmware
   > /reindex

:doc:`/using/projects` documents every key a project can declare, including
``build_commands`` and ``test_commands``, which SPEAR runs on the final tree of
every change. A project that declares none is still checked: SPEAR probes the
tree for the usual build and test commands, and a declared kind always wins
over a probed one.

The permission mode is chosen at launch: ``--safe`` (the default) changes
nothing, ``--ask`` confirms every change and command, ``--auto`` runs without
asking. ``--no-network`` removes network access from every mode.

Make a change
=============

A request about the tree runs on the coding core (:ref:`implementation_mode`):

.. code-block:: text

   > Fix the off-by-one in ring_next() in src/ring.c, then run make.

The core reads, edits and runs ``make`` inside the workspace; every call
crosses the control plane, and in ``--ask`` you confirm each one. The answer
ends with SPEAR's own verdict on what was shown — ``VERIFIED`` if a build ran
and passed on the final source, ``UNVERIFIED`` (with the reason) if not. A
command sent to the background (``make &``), or a Makefile that only prints its
help, does not count as a check.

Tell SPEAR what it should keep knowing about the workspace; later coding and
general turns in that workspace, and no other, are given it:

.. code-block:: text

   > /remember The board's console is on UART2, not UART0.
   > /knowledge list

What the model itself offers to remember stays a proposal until you accept it
(:ref:`knowledge`). ``spear-chat --fresh`` starts without the stored
conversation; knowledge, rules and configuration still apply. Tools of external
MCP servers are made available per workspace in ``capabilities.json``
(:ref:`capabilities`).

Bind a standard
===============

A specification is ingested once and then bound; the binding is shared by every
session on the machine:

.. code-block:: text

   > /standard ingest ~/docs/acme-frame.pdf --id ACME-FRAME --revision 2024 --origin PUBLIC
   > /standard use ACME-FRAME 2024
   > /standard status

``--origin`` defaults to ``LICENSED_STANDARD``, the safe answer for a document
nobody classified. Then ask about it:

.. code-block:: text

   > What does Rule 7.1-3 require of the descriptor word?

That is a NORMATIVE request (:ref:`standards`): it is answered from the
document first, every normative claim carries its provision, and an answer the
evidence does not support is withheld with the reason.

Make a change the standard governs
==================================

Ask for a change in the standard's terms:

.. code-block:: text

   > Update parse_descriptor() in src/frame.c so it reads the descriptor words
     the way the standard requires.

That is a MIXED request (:ref:`mixed_mode`). SPEAR identifies the governing
provisions first, read-only; turns them into a compact constraint packet; lets
the coding core make the change; and judges the final source against the
packet.

Read the result
===============

A MIXED turn ends with one line that keeps both dimensions apart:

.. code-block:: text

   MIXED VERDICT: COMPLIANCE NOT DEMONSTRATED — ACME-FRAME 2024, …
   Coverage: COMPLETE — 3 cited, 1 added by the document's structure, …
   Required: 4 — applicable 0 (…), applicability unresolved 4, …
   Implementation evidence: VERIFIED. Normative: NOT_DEMONSTRATED.

**Implementation evidence** says whether the change was shown to work.
**Normative** says whether SPEAR holds authoritative evidence that each
applicable requirement holds. ``COMPLIANCE NOT DEMONSTRATED`` does *not* mean
non-compliant: it means that evidence is missing — and that the project can
supply it, by declaring which provisions apply and binding its own conformance
checks to them (:ref:`normative_checks`). :doc:`/reasoning/evidence` explains
every verdict.

Retrieval is not a detail
=========================

Measured on 37 build-system questions, the same model answers **18–19 %** of
them cold and **90 %** with the project's corpus injected. Index the projects
you ask about (:ref:`retrieval-measured-effect`).

Where next
==========

* :doc:`/using/usage` — the commands, inside and outside the chat;
* :doc:`/reasoning/index` — the request modes and their evidence;
* :doc:`/harness/harness` — how every command is confined.
