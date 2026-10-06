.. _sandbox:

=======
Sandbox
=======

.. figure:: /img/SPEAR-Sandbox.drawio.png
   :width: 100%
   :alt: Wrapper order and confinement

   The wrapper order is fixed: the cgroup scope outside, ``prlimit`` inside.

Wrapper order
=============

.. code-block:: text

   systemd-run --scope        (cgroup limits: the whole tree)
     └── bwrap                (namespaces, mounts, environment)
           └── prlimit        (per-process rlimits)
                 └── COMMAND  (untrusted argv)

This order is a contract, not an accident, and three inversions are explicitly
forbidden:

``prlimit → systemd-run``
   would apply the descriptor limit to systemd's client rather than to the
   workload.

``bwrap → systemd-run``
   would put the scope *inside* the sandbox, where the sandboxed process could
   observe and interfere with it.

``systemd-run → prlimit → bwrap``
   would apply rlimits to ``bwrap`` itself, not to the command, and would break
   bwrap's own setup.

The reason ``prlimit`` must stay inside is simple: rlimits are per-process and
inherited across ``fork``.  Applied inside, they land on the command and every
child it spawns.  Applied outside, they land on ``bwrap``, which is not the
process anyone is trying to constrain.

Namespaces
==========

.. code-block:: text

   --unshare-user  --unshare-pid  --unshare-ipc  --unshare-uts  --unshare-net
   --clearenv  --die-with-parent  --new-session

``--unshare-net`` is unconditional — including for the network profile.  The
sandbox always starts with no network at all; connectivity, when granted, is
added afterwards by attaching a helper to the private namespace
(:doc:`/harness/network`).  There is no code path in which the command shares the host
network namespace.

``--die-with-parent``
   Anchors the sandbox lifetime to the Python supervisor via ``PR_SET_PDEATHSIG``.
   This still holds under the systemd scope: ``systemd-run --scope`` *execs* in
   place, so ``bwrap``'s parent remains the supervisor.

``--new-session``
   Detaches from the controlling terminal, so a sandboxed process cannot
   inject input into the user's terminal with ``TIOCSTI``.

``--clearenv``
   The environment boundary.  It is what allows the supervisor to pass
   ``XDG_RUNTIME_DIR`` to ``systemd-run`` without any risk of that variable
   reaching the command.

Filesystem
==========

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Mount
     - Nature
   * - ``/usr``
     - read-only bind of the host ``/usr``
   * - ``/bin``, ``/lib``, ``/lib64``
     - symlinks into ``/usr`` (merged-usr layout)
   * - ``/proc``
     - fresh procfs for the private PID namespace
   * - ``/dev``
     - minimal device set
   * - ``/tmp``
     - a private per-session directory, outside the workspace, removed when
       the session ends (``SPEAR_SANDBOX_EPHEMERAL_TMP=1``: a fresh tmpfs per
       command)
   * - ``/home``, ``/home/sandbox``
     - empty directories
   * - ``/etc``
     - private tmpfs, **sealed read-only**, containing only ``alternatives``,
       ``passwd`` and ``group`` (plus the resolver files on the network
       profile)
   * - ``/opt/toolchains``
     - read-only, when the host has it — cross-toolchains linked from
       ``/usr/local/bin``
   * - the workspace
     - **the only host-writable mount**, at its own host path by default (see
       below); the registered corpora beside it unless ``--single-root``

Environment:

.. code-block:: text

   PATH=/usr/local/bin:/usr/bin:/bin   LANG=LC_ALL=en_US.UTF-8
   HOME=/home/sandbox   TMPDIR=/tmp    cwd=<the workspace mount>

Workspace binding follows the profile
-------------------------------------

.. code-block:: text

   profile.workspace_write  →  --bind     <host workspace>  <mount>
   profile.workspace_read   →  --ro-bind  <host workspace>  <mount>
   neither                  →  --dir      /workspace

A read-only profile therefore cannot write even by accident: the restriction
is a mount option, not a check in Python.

``<mount>`` is the workspace's own host path, so a build tree configured
outside the sandbox keeps working inside it — CMake caches, generated
Makefiles and ``compile_commands.json`` all embed absolute paths.
``/workspace`` is used instead when the host path would be unsafe to mirror,
or when ``SPEAR_SANDBOX_IDENTITY_MOUNT=0``.

``/etc`` and the alternatives system
------------------------------------

The host ``/etc`` is never exposed.  The sandbox gets its own tmpfs, into
which one host directory and two world-readable identity files are bound
read-only:

.. code-block:: text

   --tmpfs /etc
   --ro-bind /etc/alternatives /etc/alternatives
   --ro-bind /etc/passwd /etc/passwd   --ro-bind /etc/group /etc/group
   …                                              (resolver files, if network)
   --remount-ro /etc

``passwd`` and ``group`` are there because build tools resolve uids and gids
through them — BitBake refuses to start without them.

``/etc/alternatives`` is there because ``cc``, ``awk``, ``editor`` and a
number of other commands are symlinks through it.  Without it they dangle, and
a plain ``make`` fails with ``cc: No such file or directory`` while ``gcc``
works — a confusing way to lose the C toolchain.  The directory holds no data
of its own: every entry is a symlink into ``/usr``, which is already visible
read-only.

The ``--remount-ro`` is what keeps the strong property intact.  A ``--dir`` or
a plain tmpfs would be *writable*, so a command could create files under
``/etc`` inside its sandbox.  Nothing would reach the host, but "nothing
outside the workspace is writable" would no longer be literally true.  A
tmpfs is a real mount point and can therefore be sealed once every bind is in
place — which is why the seal is emitted last, after the network profile's
resolver files.

Verified from inside:

.. code-block:: console

   $ ls -A /etc
   alternatives  group  passwd
   $ cc --version | head -1
   cc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
   $ touch /etc/x
   touch: cannot touch '/etc/x': Read-only file system

Per-process limits
==================

.. code-block:: python

   DEFAULT_RESOURCE_LIMITS = ResourceLimits(nofile=4096, core_bytes=0)

Only two limits are on by default, and the omissions are deliberate:

``nofile=4096``
   Bounds descriptor fan-out.

``core_bytes=0``
   No core dumps.

``cpu_seconds``
   Left unset: a wall-clock timeout already exists, and a CPU-second limit
   kills legitimate long compiles.

``file_size_bytes``
   Left unset: ``RLIMIT_FSIZE`` is not a workspace quota — it caps *individual
   file* size, which breaks legitimate link and archive steps while doing
   nothing about a workload writing a million small files.

Hard and soft values are set equal, so an unprivileged child cannot raise its
own soft limit back up.

Execution paths
===============

Closed network
--------------

``run()`` builds the bwrap argv, wraps it in a scope if cgroup limits are
active, spawns it with ``env={}`` (or the supervisor environment when scoped),
and waits with a bounded ``communicate(timeout=...)``.

On timeout it terminates the scope first — systemd reaches descendants a
``Popen`` handle cannot — then stops the process without waiting for EOF.
Every wait on that path is bounded; a stuck tree cannot block the supervisor.

Network
-------

``_run_with_slirp()`` coordinates bwrap's private network namespace with the
helper's ready/exit protocol.  It is described in full in :doc:`/harness/network`.

Preflight
=========

``preflight()`` runs a real mini-sandbox rather than probing a version:

.. code-block:: sh

   test "$PWD" = <mount> && test -w <mount> \
     && test "$HOME" = /home/sandbox && test "$TMPDIR" = /tmp

where ``<mount>`` is the path the workspace answers to inside the sandbox.

If that fails, availability becomes ``REFUSED`` and stays there.  A version
number cannot tell you whether unprivileged user namespaces are permitted on
this kernel; running the thing can.

``preflight_network()`` exercises the complete pidfd + slirp path with
``/bin/true`` before the first real network use, so the first failure is
observed on a trivial command rather than in the middle of the user's work.
