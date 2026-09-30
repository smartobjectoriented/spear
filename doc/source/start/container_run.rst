.. _container_run:

==========================
Running the public image
==========================

This chapter is for a colleague who has **never built SPEAR** and wants to use
it: pull the image, give it your checkouts and a model endpoint, and start
asking.  How the image is made is in :ref:`container`; nothing here needs it.

What you need
=============

* **Docker** (Engine 24 or later), able to run as your user.
* **The checkouts you want to ask about.**  The public image knows the SO3
  family of trees — ``so3``, its documentation, ``u-boot``, ``avz``, ``qemu``,
  ``atf`` and the libraries vendored under ``so3/usr`` — and expects them where
  `Laying out your checkouts`_ says.
* **A model endpoint**: any OpenAI-compatible server (``llama-server``, vLLM…)
  reachable from your machine.  The image does not contain a model.

Getting the image
=================

Public images are published on the GitHub container registry, one tag per
release:

.. code-block:: console

   $ docker pull ghcr.io/smartobjectoriented/spear:<version>-public

``<version>`` is the release you want, e.g. ``0.2.0``.  The image is labelled
``redistributable=true``: you may pass it on.

.. _container_run_layout:

Laying out your checkouts
=========================

Inside the container every corpus lives under ``/corpora``, at a path
**relative to your home directory** on the machine that built the image.  For
the SO3 family that is ``soo/so3``:

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Corpus
     - Path under ``/corpora``
     - Comes from
   * - ``so3``
     - ``soo/so3/so3``
     - the SO3 repository
   * - ``so3-doc``
     - ``soo/so3/doc``
     - the SO3 repository
   * - ``u-boot``, ``avz``, ``qemu``, ``atf``
     - ``soo/so3/<name>``
     - fetched by the SO3 build
   * - ``lvgl-so3``, ``micropython-so3``
     - ``soo/so3/so3/usr/lib/lvgl``, ``…/usr/src/micropython``
     - vendored in SO3

So one clone is enough, wherever you keep it:

.. code-block:: console

   $ git clone https://github.com/smartobjectoriented/so3.git ~/soo/so3

and it is mounted with ``-v ~/soo:/corpora/soo``.  A tree you do not have is
not an error: the container prints which corpora it found and which it did
not, then works with what is there.

To see the list an image expects, ask it:

.. code-block:: console

   $ docker run --rm --entrypoint cat ghcr.io/smartobjectoriented/spear:<version>-public \
         /opt/spear/spear/projects.json

Running it
==========

.. code-block:: console

   $ mkdir -p ~/.spear/state
   $ docker run --rm -it \
         --security-opt seccomp=unconfined \
         --security-opt apparmor=unconfined \
         --security-opt systempaths=unconfined \
         --network host \
         --user "$(id -u):$(id -g)" \
         -e SPEAR_API_BASE=http://127.0.0.1:8080/v1 \
         -e SPEAR_STATE_DIR=/state -e HOME=/state/home \
         -e SPEAR_DB_PATH=/state/chromadb \
         -v ~/.spear/state:/state \
         -v ~/soo:/corpora/soo \
         -w /corpora/soo/so3/so3 \
         ghcr.io/smartobjectoriented/spear:<version>-public

Each line has a reason:

``--security-opt`` ×3
   SPEAR runs every command the model asks for inside ``bubblewrap``, and
   refuses to run any without it.  Docker's defaults forbid the unprivileged
   namespaces ``bubblewrap`` needs; these three flags allow them *inside* the
   container and give it nothing on the host.  Without them the container
   stops at once and prints exactly this.

``--network host`` and ``SPEAR_API_BASE``
   The model endpoint.  With host networking ``127.0.0.1`` means your machine,
   so a server — or an SSH tunnel to one — listening locally is reachable as is.

``--user`` and the state volume
   Conversations, memories, the audit trail and — through
   ``SPEAR_DB_PATH`` — the retrieval index you build are written to
   ``~/.spear/state`` as you, and survive the container.  Without the volume
   they are lost when it exits.

``-v ~/soo:/corpora/soo`` and ``-w``
   Your checkouts, read-write: the assistant edits files when you ask it to.
   The working directory picks the corpus the session opens on.

Anything after the image name goes to ``spear-chat`` unchanged, e.g.
``--auto`` or ``--temp 0.1`` (:ref:`usage`).

The first session: indexing
===========================

A public image carries **no retrieval index**: an index holds chunks of every
tree it was built from, and nothing says those may be redistributed.  The
first session on a corpus therefore answers without retrieval until you build
its index:

.. code-block:: text

   /reindex

It runs inside the container with the embedder the image carries, and writes
into ``/state/chromadb`` — your state volume — so it is done once per corpus,
not once per run.
On a CPU it takes a while for the larger trees; until then the assistant still
works, by searching the files with its tools instead of the index.

When something is wrong
=======================

``WARNING: 8 mounted corpora have no collection in the baked index``
   Expected with a public image, which carries no index: see
   `The first session: indexing`_.  It disappears for each corpus once you
   have run ``/reindex`` on it.

``bubblewrap cannot start in this container``
   One of the three ``--security-opt`` flags is missing.

``Nothing mounted at /corpora``
   The ``-v`` line is missing or points at an empty directory.

``missing: so3 soo/so3/so3``
   The tree is not at the expected path; compare with
   `Laying out your checkouts`_.

The chat opens but every answer fails
   The endpoint is not reachable from the container: check
   ``curl $SPEAR_API_BASE/models`` on the host.
