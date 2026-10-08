.. _installation:

Installation
############

There are two ways to run SPEAR. Which one you want depends on whether you
intend to *use* it or to *change* it.

Prerequisites
*************

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Requirement
     - Note
   * - Linux with systemd
     - the execution harness places each tool call in a transient user scope;
       without a user manager it cannot apply resource control
   * - Python 3.12 or later
     - the client runs from its own virtualenv under ``spear/``
   * - An inference endpoint
     - local, remote or hosted — see :ref:`Model backends <backends>`
   * - ``bubblewrap``
     - the sandbox layer for untrusted commands
   * - Docker *(container path only)*
     - with the three ``--security-opt`` flags the harness requires

A GPU is a property of the *backend*, not of the client. The client itself is
undemanding; a machine that cannot serve a model can still run SPEAR against
one that can.

Native installation
*******************

This is the path to take if you will modify the platform, re-index a corpus or
register a project.

.. code-block:: console

   $ git clone https://github.com/smartobjectoriented/spear ~/spear
   $ cd ~/spear
   $ spear/deploy/install.sh

The installer creates the virtualenv under ``spear/``, installs PyTorch and the
client requirements from ``spear/deploy/requirements.txt``, and caches the
active embedding model. It is idempotent. Check what the :term:`harness` needs before
first use:

.. code-block:: console

   $ spear/deploy/preflight.sh

On Ubuntu 24.04 and later, unprivileged user namespaces are restricted by
AppArmor and ``bwrap`` cannot start; ``sudo spear/deploy/enable-sandbox.sh``
installs the profile that allows it.

The launcher is ``spear/spear-chat.sh``; putting it on your ``PATH`` as
``spear-chat`` is the usual arrangement.

.. note::

   The virtualenv records its own absolute path, so the tree cannot simply be
   moved. Re-run the installer after relocating it.

Container installation
**********************

If you have an inference endpoint and only want to use SPEAR, Docker is enough:

.. code-block:: console

   $ git clone https://github.com/smartobjectoriented/spear ~/spear
   $ cd ~/spear
   $ scripts/docker/build.sh

The build takes a while, mostly for the embedding model. See :ref:`Container
<container>` for what the image carries, what it expects mounted, and the
three security options without which the harness refuses to run any command.
Without ``--profile`` it builds the ``public`` image; ``--profile private``
builds the one ``spear-docker.sh`` runs on this machine.

The inference runtime
*********************

The client does not build or download a model. Where you also serve the model
on this machine, the runtime is reconstructed from a pinned manifest rather
than from whatever is current:

.. code-block:: console

   $ scripts/bootstrap-runtime.sh --root ~/spear-runtime --dry-run
   $ scripts/bootstrap-runtime.sh --root ~/spear-runtime
   $ scripts/bootstrap-runtime.sh --root ~/spear-runtime --verify

Every version installed comes from ``server/runtime/manifest.json``: an
immutable upstream commit, exact model shards with their publisher's hashes,
and a tested constraints file. There is no "latest" and no branch name — a
moving reference would mean the binary serving today differs from the one that
was measured, with nothing recording the change.

What ends up where
******************

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Path
     - Holds
   * - ``spear/``
     - the client: chat, agent runtime, retrieval, execution harness
   * - ``server/``
     - the inference server component
   * - ``doc/``
     - this documentation
   * - ``docker/``
     - the container image: ``Dockerfile`` and entrypoint
   * - ``scripts/``
     - the operator scripts: ``spear-configure``, ``spear-image``, the
       container build and launcher under ``scripts/docker/``, the version
       helper
   * - ``spear/projects.json``
     - the corpus registry for **this** machine (untracked)
   * - ``spear/machine.env``
     - machine-specific settings (untracked), written by ``spear-configure``
   * - ``spear/capabilities.json``
     - the external capabilities (MCP providers) of **this** machine
       (untracked, optional; ``SPEAR_CAPABILITIES_FILE`` moves it)
   * - ``$SPEAR_STATE_DIR``
     - everything a session accumulates — history, audit trail, workspace
       knowledge (``knowledge.sqlite3``), ingested standards;
       ``spear-configure`` sets it to ``~/.local/state/spear``, and without
       it the harness falls back to ``spear/``

The untracked files are the boundary between the platform and the machine.
Nothing machine-specific belongs in a tracked file — see
:ref:`Configuration reference <configuration>`.

``machine.env`` is not written by hand. From the repository root:

.. code-block:: console

   $ . ./env.sh                 # puts scripts/ on PATH
   $ spear-configure            # writes spear/machine.env
   $ spear-configure --check    # does it still match this machine?

It finds the state directory, a ``spear-private/`` tree beside the checkout,
and the embedding host with the model revision both ends' caches agree on —
refusing to pin one they disagree on. A differing file is shown as a diff and
replaced only on confirmation, kept as ``machine.env.bak``; settings added
below its local-additions marker survive regeneration.

Checking the installation
*************************

.. code-block:: console

   $ spear/spear-chat.sh --help          # the CLI, its flags and its settings
   $ cd spear && PYTHONPATH=. ./bin/python -m unittest discover -s tests

The suite runs offline and takes a few minutes (:ref:`testing`).

.. seealso::

   :ref:`Getting started <getting_started>` · :ref:`Model backends <backends>`
   · :ref:`Configuration reference <configuration>`
