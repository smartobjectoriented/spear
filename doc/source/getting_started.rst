.. _getting_started:

===============
Getting started
===============

There are two ways to run SPEAR: a container that carries everything, and a
native installation.  The container is the right choice if you want to *use*
the assistant; the native one if you intend to change the harness, re-index, or
register a corpus of your own.

The container path
==================

If you have access to a model server and want to use the assistant, Docker is
the only prerequisite:

.. code-block:: console

   $ git clone https://github.com/smartobjectoriented/spear ~/spear
   $ cd ~/spear
   $ docker/build.sh                        # ~20 min, mostly the embedder
   $ docker/spear-docker.sh --reds --auto   # opens the tunnel, then chats

That image carries the harness and the embedder; you mount your own source
trees and it talks to the model served on the inference host over an SSH
tunnel.  :doc:`/container` describes what is baked, what is mounted, and the two
``--security-opt`` flags without which the harness refuses to run any command
at all.

It also bakes in whatever **retrieval index**, rules, skills and benches the
building host has — and a fresh clone has none of them, so the build says which
it found and produces a working image that simply carries less
(:ref:`optional-build-inputs`).  Filling that index is the first thing worth
doing, for the reason below.

The clone location is free
==========================

The corpora that live inside the repository are registered *relatively* and
resolved against the checkout, so the tree can sit anywhere.  Only your own
source trees are registered by absolute path, because they genuinely are
machine-specific.  See :ref:`relative-corpus-paths` for how the two kinds of
path are told apart.

Retrieval is not a detail
=========================

Measured on 37 build-system questions, the same model answers **18–19 %** of
them cold and **90 %** with the corpus injected.  Running without an index is
not running a slightly weaker assistant, it is running a different and much
worse one.  The full numbers, including what the federation of three corpora
changes, are in :ref:`retrieval-measured-effect`.

The native path
===============

To run it natively instead — which is what you want if you intend to change
the harness, re-index, or register a corpus — start from
``spear/deploy/install.sh`` and :doc:`/operations`.

Native installation puts the entry points on ``PATH`` through
``~/.local/bin``; :doc:`/usage` is the tour of what they do, and
:doc:`/directory_layout` explains what lives where.
