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
first :term:`MIXED` task.

Prerequisites
=============

* Linux with systemd and ``bubblewrap`` — the execution :term:`harness` confines every
  command in a sandbox and refuses to run without one;
* Python 3.12 or later for a native installation, or Docker for the container;
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
   $ scripts/docker/build.sh        # spear:<version>-public, ~20 min, mostly the embedder

Then run it as :doc:`/start/container_run` shows: the image carries the harness
and the embedder, and you mount your own checkouts and point it at a model
endpoint.  Its ``docker run`` line carries three ``--security-opt`` flags
without which the harness refuses to run any command (:ref:`container`).

On a workstation that also has a native installation,
``scripts/docker/build.sh --profile private`` builds the image
``scripts/docker/spear-docker.sh`` runs, and ``spear-docker.sh --reds --auto``
opens the tunnel and chats (:ref:`image-profiles`, :ref:`container`).

At the institute there is a shorter path still: the ``reds`` image carries the
shared trees and their index, and ``spear-run.sh`` runs it with nothing but
Docker (:ref:`container_run_reds`).

The native path
---------------

To change the harness, re-index, or register projects of your own, install it
natively as :ref:`installation` describes.  The launcher it leaves is
``client/spear-chat.sh``; linking it onto your ``PATH`` as ``spear-chat`` is
the usual arrangement, and the rest of this page assumes it.

Configure a model backend
=========================

``spear-chat`` talks to an OpenAI-compatible endpoint. Without a flag it asks
which backend to use and remembers the answer:

.. code-block:: console

   $ spear-chat --local                                    # llama-server on 127.0.0.1:8080
   $ spear-chat --api-base http://gpu-host:8080/v1         # any OpenAI-compatible endpoint
   $ spear-chat --provider anthropic --model <model-id>    # the Anthropic API

The Anthropic backend serves questions and normative answers; changes run on
the :term:`coding core`, which needs an OpenAI-compatible endpoint. Serving a model
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
every change; a project that declares none is probed for the usual ones.

The permission mode is chosen at launch — ``--safe`` (the default), ``--ask``
or ``--auto``; :doc:`/harness/security_model` says what each allows, and
:doc:`/using/configuration` lists the other flags.

Make a change
=============

A request about the tree runs on the coding core (:ref:`implementation_mode`):

.. code-block:: text

   > Fix the off-by-one in ring_next() in src/ring.c, then run make.

The core reads, edits and runs ``make`` inside the workspace; every call
crosses the :term:`control plane`, and in ``--ask`` you confirm each one. The answer
ends with SPEAR's own verdict on what was shown — ``VERIFIED`` only if a check
ran and passed on the final source (:doc:`/reasoning/evidence`).

Tell SPEAR what it should keep knowing about the workspace with
``/remember``; later turns in that workspace are given it (:ref:`knowledge`).
Tools of external MCP servers are made available per workspace
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
provisions first, read-only; turns them into a compact :term:`constraint packet`; lets
the coding core make the change; and judges the final source against the
packet.

Read the result
===============

A MIXED turn ends with a verdict that keeps two dimensions apart: whether the
change was shown to work, and whether SPEAR holds authoritative evidence that
each applicable requirement holds.  ``COMPLIANCE NOT DEMONSTRATED`` does *not*
mean non-compliant: it means that evidence is missing, and a project can
supply it (:ref:`normative_checks`).  :doc:`/reasoning/evidence` explains every
verdict and :ref:`mixed_mode` shows a complete one.

Retrieval is not a detail
=========================

Answers about a project are far better with its corpus indexed than without
(:ref:`retrieval-measured-effect`): index the projects you ask about.

Where next
==========

* :doc:`/using/usage` — the commands, inside and outside the chat;
* :doc:`/reasoning/index` — the request modes and their evidence;
* :doc:`/harness/harness` — how every command is confined.
