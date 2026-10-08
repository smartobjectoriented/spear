.. _testing:

=======
Testing
=======

.. figure:: /img/SPEAR-Testing.drawio.png
   :width: 100%
   :alt: Test topology

   Always-on suites, opt-in suites, and what each one needs.

Running the suites
==================

Everything uses the application's own virtualenv:

.. code-block:: console

   $ cd ~/spear/spear

   $ ./bin/python -m compileall -q agent cli context evidence harness models normative retrieval runtime standard training
   $ PYTHONPATH=. ./bin/python -m unittest discover -s tests -p "test_*.py"
   Ran … tests — OK (skipped=…)

The discovery command is authoritative: about 4400 tests, a few minutes.  It
covers the :term:`coding core` and its :term:`control plane`, the evidence plane and its
verdicts, the :term:`normative runtime` and the :term:`MIXED` orchestration, workspace context,
knowledge and capabilities, sessions, verification, checkpoints, the tool
architecture and the security substrate — with scripted backends; it does not
require a real model API.

A test never touches the operator's own state.  One that names no state
directory is given a temporary one (``state_paths.test_state_root``), removed
when the run exits, and the knowledge store refuses to open its default path
under test at all.

Most skips are the opt-in suites below.  The others are tests that need what a
clean install does not have: the bubblewrap sandbox, or a licensed standard in
the store.  A checkout with no Internet, no systemd user bus and no patience
still runs the full always-on set.

The server component has its own suite — the runtime manifest, the bootstrap,
the launcher and the embedding worker — run from the same directory:

.. code-block:: console

   $ ./bin/python -m unittest discover -s ../server/tests -p "test_*.py"

Opt-in suites
=============

.. list-table::
   :header-rows: 1
   :widths: 32 40 28

   * - Flag
     - What it exercises
     - Requires
   * - ``SPEAR_TEST_NETWORK=1``
     - outbound DNS and TLS through slirp4netns
     - Internet access
   * - ``SPEAR_TEST_CGROUP=1``
     - real transient scopes: memory, pids, cpu, timeout cleanup, membership
     - a systemd user bus with cpu/memory/pids delegation
   * - ``SPEAR_TEST_NETWORK_RACE=1``
     - 40 repetitions per arm with injected delays, plus a CPU-load arm
     - a few minutes
   * - ``SPEAR_TEST_EMBED_EQUIVALENCE=1``
     - the server's embedding worker reproduces, bit for bit, the vectors
       of the worker it replaced
     - the embedding model's weights (``BAAI/bge-m3``)

.. code-block:: console

   $ SPEAR_TEST_NETWORK=1     ./bin/python -m unittest tests.test_sandbox_network.Slirp4netnsNetworkTests
   Ran 29 tests — OK
   $ SPEAR_TEST_CGROUP=1      ./bin/python -m unittest tests.test_resource_control.OptInRealCgroupTests
   Ran 6 tests — OK
   $ SPEAR_TEST_NETWORK_RACE=1 ./bin/python -m unittest tests.test_sandbox_network.OptInNetworkRaceTests
   Ran 4 tests — OK

What the suites assert
======================

Negative properties
-------------------

Much of the value is in asserting that things **do not** happen.  These are
easy to satisfy by accident with a test that only checks a return status, so
they are checked at the mechanism level instead:

* ``popen.assert_not_called()`` — a :term:`fail-closed` decision spawned nothing;
* ``self.assertEqual(writes, [])`` — no release write followed a setup
  failure, i.e. the sandboxed command never started;
* the marker file the command would have created does not exist;
* the supervisor's descriptor count did not grow;
* no residual ``spear-tool-*`` unit, no orphan ``bwrap`` or ``slirp4netns``.

Descriptor and namespace hygiene
--------------------------------

``PinnedNamespaceAttachmentTests`` is always on and fast (~0.2 s).  It checks
the helper argv and its exact ``pass_fds``, that the command's descriptor table
contains no namespace handle, that the supervisor leaks nothing on success, on
setup failure or on a Python exception, and that both fail-closed paths — an
incapable helper, and an inode mismatch — refuse without releasing.

The characterization test
-------------------------

``test_characterization_old_pid_based_attachment_still_races`` deliberately
asserts that the **old** PID-based attachment *still fails* (10 out of 10 with
a 20 ms delay).

That looks backwards until you consider what it protects.  The pinned
attachment exists to work around a real interaction between bwrap's two-step
user namespace and slirp4netns.  If a future kernel or helper made the simple
form safe, this test turns red — and tells us the workaround can be revisited
— instead of leaving a permanent unexplained complication in the code.

Determinism
===========

The always-on suites are deterministic.  One property required care:

.. admonition:: Do not instrument the critical window
   :class: warning

   Any synchronous work between reading bwrap's info-fd and spawning the
   network helper breaks the network path (:doc:`/harness/network`).  A test that wants
   to observe the live scope must sample from a **side thread**, and that
   thread must be cheap — an early version scanned all of ``/proc`` every
   10 ms and made the suite flaky through GIL contention alone.

   ``OptInRealCgroupTests`` samples via ``/proc/self/task/*/children`` rather
   than a full ``/proc`` walk, and stops the watcher between attempts.

A second, pre-existing property is worth knowing: under heavy artificial CPU
load the network suites can still hit ``slirp4netns failed to become ready``,
because the readiness wait is a wall-clock timeout.  That is a property of the
timeout, not of the attachment.

Residual state
==============

After a full run, all four must hold:

.. code-block:: console

   $ systemctl --user list-units --all 'spear-tool-*' --no-legend | wc -l
   0
   $ pgrep -a bwrap ; pgrep -a slirp4netns
   (nothing)
   $ ls /sys/fs/cgroup/user.slice/user-*.slice/user@*.service/app.slice/spear-tool-* 2>/dev/null
   (nothing)

and the supervisor's descriptor count is unchanged across repeated runs.

.. note::

   Orphan ``bwrap`` processes observed during development came from ad-hoc
   diagnostic scripts that killed ``bwrap`` without going through the pidfd
   sequence — not from the runtime.  They are recognisable by their workspace
   path: the runtime always uses a ``/tmp/spear-slirp-`` prefix.

Live smoke test
===============

The end-to-end check that serving and the backend contract are both healthy is
in :doc:`/model/model_serving`; the expected answer is ``end_turn`` /
``SPEAR-BACKEND-OK`` / empty tool tuple.
