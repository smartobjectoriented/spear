.. _inference_host:

=============================
Setting up an inference host
=============================

This chapter builds, from an empty machine, the host that serves the model to
every SPEAR client: ``llama-server`` with the profile's model, supervised by
systemd, reachable through SSH, with the embedding worker beside it.  Follow
it in order; each step ends with a check, and the next one assumes it passed.

The figures quoted are those of the reference deployment.  :ref:`runtime_bootstrap` explains *why* the runtime is built the
way it is; this page is the procedure.

.. contents:: Steps
   :local:
   :depth: 1

What you are building
=====================

.. code-block:: text

   GPU host                                         each client
   ─────────────────────────────────────────        ───────────────────────
   ~/spear/            checkout (tens of MB)        spear-chat --reds
   ~/spear-runtime/                                   │  SSH tunnel
     llama.cpp/        pinned build                   │  127.0.0.1:8082
     models/gguf/      the weights, 79 GiB            ▼
     embed/            embedding worker venv   ◄──── ssh … run-worker.sh
     config/           server.conf, gpu.conf
     log/
   spear-inference.service ─► serve.sh ─► llama-server 127.0.0.1:<port>

The server listens on ``127.0.0.1`` only.  Clients never reach it over the
network: they open an SSH tunnel, and the embedder is started per request
over SSH.  Nothing on the host is exposed but ``sshd``.

1. Host prerequisites
=====================

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Item
     - Requirement
     - Reference deployment
   * - GPU
     - enough VRAM for the weights **and** the KV cache (below)
     - RTX PRO 6000 Blackwell, 96 GB (97.9 GB usable)
   * - NVIDIA driver
     - CUDA 13 capable (the embedder's torch is a ``cu130`` build)
     - 595.84
   * - CUDA toolkit
     - ``nvcc`` ≥ 12.8, which is the first to target Blackwell (``sm_120``)
     - 13.0, at ``/usr/local/cuda``
   * - OS
     - a systemd Linux
     - Ubuntu 24.04
   * - Packages
     - ``git``, ``cmake``, ``build-essential``, ``python3-venv``,
       ``python3-dev``, ``curl``, ``wget``
     -
   * - Disk, under the account's home
     - ~100 GB: model 85 GB (79 GiB), embedder venv 5 GB, embedder weights
       4.3 GB, llama.cpp build 1 GB
     -
   * - RAM
     - not critical once the weights are on the GPU
     - 46 GB
   * - Account
     - one account that owns the checkout and the runtime
     - see below

**Sizing the GPU.**  The profile serves Qwen3-Coder-Next Q8_0: 79 GiB of
weights.  Its KV cache is small because only 12 of its 48 layers use full
attention, with 2 KV heads of 256: at ``q8_0`` that is about 13 KB per token,
6.8 GB for the 524 288-token window.  Measured: 89.5 of 97.9 GB in use at
512K.  On a smaller card, lower ``SPEAR_SERVER_CTX`` (step 4); on a card that
cannot hold the weights at all, ``SPEAR_SERVER_NCPUMOE`` keeps the
mixture-of-experts weights in system RAM, at a large cost in speed.

**The account.**  Everything below lives in one account's home and needs no
root, except installing the system unit (step 6).  On a machine shared with
other people *under the same account*, the isolation is a matter of
discipline rather than permissions: pin the card by UUID (step 3), keep every
cache inside ``~/spear-runtime`` (step 7), and never stop a process you did
not start by pattern — ``pkill -f`` matches other people's command lines, and
its own.

Check:

.. code-block:: console

   $ nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv
   $ /usr/local/cuda/bin/nvcc --version | tail -1
   $ df -h ~

2. The checkout
===============

.. code-block:: console

   $ git clone https://github.com/smartobjectoriented/spear.git ~/spear
   $ cd ~/spear && git checkout <release>

Deploy a **release**, not whatever ``main`` holds (:ref:`release_process`).
The host runs its launcher from this checkout, so it must stay a clean
checkout of a known commit: never edit a file in it on the host.  A launcher
that matches no commit is a deployment nobody can reproduce.

3. The runtime
==============

Find the UUID of the card this deployment may use, then let the bootstrap
build everything the manifest describes:

.. code-block:: console

   $ nvidia-smi -L
   GPU 0: NVIDIA RTX PRO 6000 Blackwell … (UUID: GPU-xxxxxxxx-…)

   $ cd ~/spear
   $ scripts/bootstrap-runtime.sh --root ~/spear-runtime --gpu GPU-xxxxxxxx-… --dry-run
   $ scripts/bootstrap-runtime.sh --root ~/spear-runtime --gpu GPU-xxxxxxxx-…

The dry run says what it would do and writes nothing.  The real run clones
and builds the pinned llama.cpp, downloads the four model shards (budget
about an hour per 60 GB), builds the embedder's virtualenv, and writes
``config/server.conf`` and ``config/gpu.conf``.  It is idempotent: interrupt
it, run it again, and it keeps what is already correct.

**Pin the card**, even on a single-GPU host.  Unpinned, ``llama-server``
spreads its layers over every visible card, which on a shared host means
somebody else's.

Check, and prove the weights byte for byte once:

.. code-block:: console

   $ scripts/bootstrap-runtime.sh --root ~/spear-runtime --verify
   $ scripts/bootstrap-runtime.sh --root ~/spear-runtime --verify --checksum

``--verify`` exits 0 only when the runtime satisfies the manifest, and ends
with the exact command line the launcher would run.

4. The configuration
====================

``~/spear-runtime/config/server.conf`` is generated once from the profile and
is the deployment's from then on; no later bootstrap overwrites it.  Every key
is described in ``server/config/server.conf.example``.  The ones to review:

``SPEAR_SERVER_CTX=524288``
   The context window.  The model is trained at 262 144 tokens, so the
   profile serves twice that with YaRN, enabled by the next two keys.
   Lower it if the card is smaller: at or below 262 144 no scaling is used.

``SPEAR_SERVER_NATIVE_CTX=262144`` and ``SPEAR_SERVER_ARCH=qwen3next``
   The trained length and the model architecture.  When ``SPEAR_SERVER_CTX``
   exceeds the first, ``serve.sh`` adds ``--rope-scaling yarn --rope-scale 2
   --yarn-orig-ctx 262144`` **and** overrides
   ``qwen3next.context_length``.  The override is not optional:
   ``llama-server`` caps each slot at the trained length it reads from the
   GGUF, so without it a 512K request serves 256K and says so only in its log.
   YaRN is static — it applies to short prompts too — which is why it is
   switched on only past the trained length.

``SPEAR_SERVER_PORT=8010``, ``SPEAR_SERVER_HOST=127.0.0.1``
   Keep the host on loopback.  Clients use the port in their tunnel (step 8).

``SPEAR_SERVER_NCPUMOE=0``
   Everything on the GPU.  Unset, it is guessed from the model's file name.

5. First run, in the foreground
===============================

Get it serving by hand before supervising it — a unit that restarts a
misconfigured server hides the error behind a restart loop:

.. code-block:: console

   $ SPEAR_SERVER_ROOT=~/spear-runtime ~/spear/server/inference/serve.sh

It prints the card it pinned and, past the trained length, the YaRN factor.
Loading takes a minute or two.  From a second shell:

.. code-block:: console

   $ until curl -sf http://127.0.0.1:8010/health >/dev/null; do sleep 2; done
   $ curl -s http://127.0.0.1:8010/props | python3 -c \
       'import sys,json;print(json.load(sys.stdin)["default_generation_settings"]["n_ctx"])'
   524288

If that prints 262144, the architecture override is missing (step 4).  Stop
the server with Ctrl-C.

6. Supervision
==============

A **system** unit is the one that survives reboots and logouts without
changing how the machine treats the account.  ``render-unit.sh`` adapts the
template for it — account, home, boot target — and prints the result:

.. code-block:: console

   $ ~/spear/server/inference/render-unit.sh --system --account "$USER" \
         | sudo tee /etc/systemd/system/spear-inference.service >/dev/null
   $ sudo systemctl daemon-reload
   $ sudo systemctl enable --now spear-inference

Add ``--log ~/spear-runtime/log/serve.log`` to send the output to a file
instead of the journal.  The unit holds **no serving decision**: model, port,
context and card all stay in ``config/``, so changing one is an edit there and
a restart, never an edit of the unit.

A user unit (``render-unit.sh --user`` into ``~/.config/systemd/user/``) needs
no root, but survives a logout only with ``loginctl enable-linger`` — a
decision for the host's administrator.  ``server/inference/README.md`` weighs
the two.

``active`` means *started*, not *serving*; wait on ``/health`` as in step 5.
Day to day:

.. code-block:: console

   $ sudo systemctl restart spear-inference
   $ systemctl status spear-inference
   $ sudo journalctl -u spear-inference -f          # or tail the --log file

7. The embedding worker
=======================

Clients offload corpus indexing to this host's GPU.  The bootstrap already
built the worker's virtualenv and wrote its launcher,
``~/spear-runtime/embed/run-worker.sh``, which pins the same card and keeps
the embedder's weights in ``~/spear-runtime/embed/hf`` rather than in the
account-wide Hugging Face cache.  It is not a daemon: each client request
starts it over SSH and it exits when done.

Prove it end to end — the real launcher, the real protocol, one text:

.. code-block:: console

   $ scripts/bootstrap-runtime.sh --root ~/spear-runtime --verify --probe

The first probe downloads ``BAAI/bge-m3`` (4.3 GB).  The client side of
the worker is in step 8; the protocol is in ``server/embed/README.md``.

8. Client access
================

Each colleague needs an SSH key accepted by the host account
(``~/.ssh/authorized_keys``) and a host entry on their machine:

.. code-block:: text

   # ~/.ssh/config
   Host spear-host
       HostName <host address>
       User <account>
       IdentityFile ~/.ssh/<key>
       IdentitiesOnly yes

``IdentitiesOnly`` matters on hosts with a low ``MaxAuthTries``: without it
ssh offers every key it has and is disconnected before the right one.

**The model.**  In the client checkout, ``client/reds.conf`` (from
``reds.conf.example``):

.. code-block:: sh

   REDS_HOST=spear-host
   REDS_PORT=8010
   REDS_MODEL=qwen3

``spear-chat --reds`` then opens the tunnel ``127.0.0.1:8082 → host:8010``
itself and checks that something answers.  A container client uses the same
tunnel with ``SPEAR_API_BASE=http://127.0.0.1:8082/v1``
(:ref:`container_run`).  By hand:

.. code-block:: console

   $ ssh -o ExitOnForwardFailure=yes -L 8082:localhost:8010 -N -f spear-host
   $ curl -s http://127.0.0.1:8082/v1/models

**The embedder.**  Two files in the ``client/`` directory:

.. code-block:: text

   active-embed-remote.conf       line 1: spear-host
                                  line 2: extra ssh options, if any
   active-embed-remote-cmd.conf   ~/spear-runtime/embed/run-worker.sh

A leading ``~/`` is expanded on the host.  Indexing (``/reindex``,
``spear-index``) then encodes on the host's GPU.

9. Verifying the whole
======================

.. code-block:: console

   host   $ scripts/bootstrap-runtime.sh --root ~/spear-runtime --verify --probe
   host   $ curl -s http://127.0.0.1:8010/props | grep -o '"n_ctx":[0-9]*'
   client $ spear-chat --reds          # the banner shows the model and the window

What the reference deployment measured after its move to 512K:

.. list-table::
   :widths: 50 50

   * - VRAM in use
     - 89.5 of 97.9 GB
   * - Generation
     - 156 tokens/s (157 at 98K: YaRN costs nothing measurable)
   * - Prompt processing
     - ~2 100 tokens/s — a 512K prompt takes about four minutes
   * - Retrieval past the trained length
     - a fact placed 40 % into a 311 744-token prompt, recalled exactly

10. Updating and rolling back
=============================

An update is a new release of the checkout, then whatever the manifest now
says, then a restart:

.. code-block:: console

   $ cd ~/spear && git fetch && git checkout <new-release>
   $ scripts/bootstrap-runtime.sh --root ~/spear-runtime --dry-run
   $ scripts/bootstrap-runtime.sh --root ~/spear-runtime
   $ sudo systemctl restart spear-inference

The dry run shows whether the release moved the llama.cpp pin or the model;
if it did not, the bootstrap only verifies.  ``config/`` is never rewritten,
so a key a new release introduces (as ``SPEAR_SERVER_ARCH`` was) has to be
added by hand — its release notes say so, and ``--verify`` reports the command
line it would now produce.

**Before editing** ``server.conf``, copy it: ``cp -p server.conf
server.conf.bak-<what>-<date>``.

**To roll back**, check out the previous release, run the bootstrap (a moved
pin is rebuilt, a kept one is left alone), restore the configuration backup
if the change touched it, and restart.  Weights of a previous model are not
kept: a rollback across a model change downloads them again.

If the harness itself also runs on this host
============================================

Everything above is the server side.  A host that *also* runs ``spear-chat``
needs what any client needs to sandbox commands — unprivileged user namespaces
for ``bubblewrap`` (Ubuntu 24.04 restricts them through AppArmor and needs a
``bwrap`` profile) and delegated cgroup controllers, which a plain SSH session
does not have.  :ref:`operations` lists the symptoms, and
:doc:`/harness/sandbox` and :doc:`/harness/resource_control` the mechanisms.
