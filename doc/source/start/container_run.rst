.. _container_run:

==================
Running the images
==================

This chapter is for a colleague who has **never built SPEAR** and wants to use
it: pull the image, give it your checkouts and a model endpoint, and start
asking.  How the image is made is in :ref:`container`; nothing here needs it.
At the institute, skip to `Running the reds image`_: that image brings its
own trees and index.

What you need
=============

* **Docker** (Engine 24 or later), able to run as your user.
* **The checkouts you want to ask about.**  The public image knows the SO3
  family of trees — ``so3``, its documentation, ``u-boot``, ``avz``, ``qemu``,
  ``atf`` and the libraries vendored under ``so3/usr`` — and expects them where
  `Laying out your checkouts`_ says.
* **A model endpoint**: any OpenAI-compatible server (``llama-server``, vLLM…)
  reachable from your machine.  The image does not contain a model.  The
  Anthropic API can answer questions and normative requests, but changes run
  on the :term:`coding core`, which needs an OpenAI-compatible endpoint.

Getting the image
=================

Public images are published on the GitHub container registry, one tag per
release:

.. code-block:: console

   $ docker pull ghcr.io/smartobjectoriented/spear:<version>-public

``<version>`` is the release you want, e.g. ``0.3.0-rc1``.  The image is labelled
``redistributable=true``: you may pass it on.

**At the institute**, the image to use is the ``reds`` one: it carries the
shared trees themselves, their retrieval index and the rules written for them,
so nothing below about laying out checkouts is needed to start.  It lives in
the institute's own GitLab registry and is for its members; log in with your
GitLab account (a personal access token with the ``read_registry`` scope as
the password) and pull:

.. code-block:: console

   $ docker login reds-gitlab.heig-vd.ch:5050
   $ docker pull reds-gitlab.heig-vd.ch:5050/reds-general/spear/spear-registry/spear:<version>-reds

Until that registry's certificate is renewed, the same image is on the GitHub
container registry as a private package of the organisation, for its members:

.. code-block:: console

   $ echo <token with read:packages> | docker login ghcr.io -u <github user> --password-stdin
   $ docker pull ghcr.io/smartobjectoriented/spear-reds:<version>-reds

It is labelled ``redistributable=false``: it stays within the institute.  A
checkout of your own mounted at ``/corpora`` shadows the baked copy of the same
tree, so you can work on your branch and keep the index.

.. _container_run_reds:

Running the ``reds`` image
==========================

The ``reds`` image carries the trees, their index and the rules, so there is
nothing to lay out and nothing to index: the steps below are all of it.  What
it does not carry is a model, which the institute serves on its GPU host; the
container reaches it through an SSH tunnel you open on your machine.

1. **Access.** Be on the HEIG-VD VPN, and have an SSH key authorised on the
   inference host's account (ask whoever administers it, giving them your
   ``~/.ssh/id_reds.pub``).  The host has a low ``MaxAuthTries``: always name
   the key and ``IdentitiesOnly``, or ssh is disconnected before it offers the
   right one.

2. **Pull** the image, as above.

3. **Get the launcher.** ``spear-run.sh`` needs only docker, ssh and curl; it
   opens the tunnel, mounts your state and runs the image.  It is in the
   repository under ``scripts/docker/``, and inside the image:

   .. code-block:: console

      $ docker run --rm --entrypoint cat \
            ghcr.io/smartobjectoriented/spear-reds:<version>-reds /opt/spear/spear-run.sh \
            > ~/bin/spear-run.sh && chmod +x ~/bin/spear-run.sh

4. **Run** it on the corpus you want to work in, by its path under
   ``/corpora``.  The tunnel to the inference server (port 8010 there, 8082
   here) is opened the first time and reused while it lasts; the image prints
   the corpora it found (29) and the standard it bound, then opens the chat.

   .. code-block:: console

      $ spear-run.sh --reds <account>@<inference host> --key ~/.ssh/id_reds \
            --corpus opencn/opencn-ng --ask
      $ spear-run.sh --list                       # every corpus and its path

   ``SPEAR_REDS_HOST`` and ``SPEAR_REDS_KEY`` in your environment save the two
   options.  Without the script, the same run by hand:

   .. code-block:: console

      $ ssh -o IdentitiesOnly=yes -i ~/.ssh/id_reds -N -f \
            -L 8082:localhost:8010 <account>@<inference host>

      $ mkdir -p ~/.spear/state
      $ docker run --rm -it \
            --security-opt seccomp=unconfined \
            --security-opt apparmor=unconfined \
            --security-opt systempaths=unconfined \
            --network host \
            --user "$(id -u):$(id -g)" \
            -e SPEAR_API_BASE=http://127.0.0.1:8082/v1 \
            -e SPEAR_MODEL_NAME=qwen3 \
            -e SPEAR_STATE_DIR=/state -e HOME=/state/home \
            -v ~/.spear/state:/state \
            -w /corpora/opencn/opencn-ng \
            ghcr.io/smartobjectoriented/spear-reds:<version>-reds --ask

   The paths under ``/corpora`` are those of the building machine, relative to
   its home: ``opencn/opencn-ng`` (and ``opencn/opencn-ng/linux/usr``),
   ``soo/so3/so3``, ``soo/so3/doc``, ``sye/sye_sol``,
   ``pos/pos_sol/so3`` and so on; ``spear-run.sh --list`` prints them all.

5. **Work on your own checkout** rather than the baked copy: mount it at the
   same path, and it shadows the copy while the index keeps serving.

   .. code-block:: console

      $ spear-run.sh --reds <account>@<inference host> --key ~/.ssh/id_reds \
            --mount ~/opencn/opencn-ng:opencn/opencn-ng --corpus opencn/opencn-ng --ask

   The index describes the baked revision; after large changes to your tree,
   ``/reindex`` inside the session rebuilds its collection into your state
   volume, on the CPU (minutes for a small tree, an hour for a large one).

The ``--ask`` at the end is the permission mode (:ref:`usage`): confirm each
edit and command.  ``--safe`` is read-only, ``--auto`` runs without asking.
Everything a session accumulates stays in ``~/.spear/state`` on your machine.

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
         /opt/spear/client/projects.json

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
   Conversations, :term:`workspace knowledge`, the audit trail and — through
   ``SPEAR_DB_PATH`` — the retrieval index you build are written to
   ``~/.spear/state`` as you, and survive the container.  Without the volume
   they are lost when it exits.

``-v ~/soo:/corpora/soo`` and ``-w``
   Your checkouts, read-write: the assistant edits files when you ask it to.
   The working directory picks the corpus the session opens on.

Anything after the image name goes to the :term:`harness` unchanged, as it would to
``spear-chat``, e.g. ``--auto``, ``--fresh`` or ``--temp 0.1`` (:ref:`usage`).
The launcher's own backend flags — ``--reds``, ``--local`` — do not apply: the
endpoint is ``SPEAR_API_BASE``.

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
