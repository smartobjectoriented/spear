==========
Operations
==========

What to look at when something misbehaves, and where each state is reported.

Health check
============

.. code-block:: console

   $ systemctl --user list-units 'edgem*'          # or the system unit
   $ curl -s http://127.0.0.1:8080/v1/models | head -c 200
   $ cat /proc/$(pgrep -x llama-server)/cgroup
   0::/system.slice/spear-llm.service

   $ cd /opt/llm/spear/spear && ./bin/python -m unittest tests.test_tool_runtime
   Ran 307 tests — OK

Reading a failure
=================

Every harness failure surfaces as a ``ToolResult`` whose ``summary`` names the
mechanism that refused.  The vocabulary is deliberately narrow.

.. list-table::
   :header-rows: 1
   :widths: 46 54

   * - Summary contains
     - Meaning and first thing to check
   * - ``bubblewrap sandbox unavailable: bwrap is absent``
     - ``bwrap`` not on ``PATH`` or not executable.
   * - ``bubblewrap sandbox refused by kernel or runtime``
     - Preflight ran and failed.  Usually unprivileged user namespaces are
       disabled: ``sysctl kernel.unprivileged_userns_clone``.
   * - ``resource limits unavailable: prlimit is absent``
     - ``/usr/bin/prlimit`` missing while ``ResourceLimits`` are active.
   * - ``resource control unavailable: systemd-run is absent``
     - Cgroup limits active, ``systemd-run`` missing.
   * - ``resource control unavailable: user bus is absent``
     - No ``$XDG_RUNTIME_DIR/bus`` socket.  Typical in a non-interactive ssh
       session or cron.  See below.
   * - ``resource control unavailable: cgroup controllers not delegated: …``
     - ``user@<uid>.service`` does not delegate the controller the contract
       needs.
   * - ``slirp4netns network backend unavailable: helper is absent``
     - ``slirp4netns`` not found.
   * - ``… pinned namespace attachment requires --netns-type and --userns-path``
     - The helper is too old.  There is no fallback; see :doc:`network`.
   * - ``… containment failure``
     - The backend is poisoned: a previous run could not confirm the sandbox
       child died.  Restart the chat session.
   * - ``network sandbox returned invalid namespace info: …``
     - The pin or the type check failed.  The command never ran.
   * - ``slirp4netns failed to become ready``
     - The helper did not signal within ``network_ready_timeout_seconds``.
       Under heavy CPU load this can be the wall-clock timeout rather than a
       real fault.
   * - ``bubblewrap sandbox timed out after Ns``
     - Wall-clock timeout.  The scope was killed as a tree.

No user bus
-----------

.. code-block:: text

   Failed to connect to bus: No medium found

The harness fails closed, by design.  If SPEAR must run without an
interactive session, the deployment decision is
``loginctl enable-linger <user>`` — taken by an administrator.  The runtime
never does it (:doc:`resource_control`).

Inspecting a live scope
=======================

Scopes are short-lived and ``--collect`` removes them promptly, so catching one
alive means sampling while the command runs:

.. code-block:: console

   $ watch -n0.1 "systemctl --user list-units --all 'spear-tool-*' --no-legend"

   # from a known bwrap pid
   $ cat /proc/<pid>/cgroup
   $ base=/sys/fs/cgroup$(cut -d: -f3 /proc/<pid>/cgroup)
   $ cat $base/memory.peak $base/pids.peak $base/cpu.stat

``memory.peak`` and ``pids.peak`` are monotonic high-water marks, so the last
read before the cgroup disappears is the meaningful one.

Cleaning up after an interrupted session
========================================

.. code-block:: console

   $ systemctl --user list-units --all 'spear-tool-*' --no-legend
   $ systemctl --user kill --kill-whom=all --signal=KILL <unit>

   $ pgrep -a bwrap
   $ pgrep -a slirp4netns

An orphan ``bwrap`` blocked on its ``block-fd`` is the signature of a
supervisor that died between spawning the sandbox and releasing the command.
It is harmless — the command never started — but it holds a namespace, so kill
it.

Switching model
===============

.. code-block:: console

   $ spear-model <name>          # rewrites active-model.conf
   $ spear-model adapter none    # rewrites active-lora.conf
   $ systemctl --user restart spear-llm.service    # or re-run spear-server

Or, without persisting:

.. code-block:: console

   $ SPEAR_SERVER_MODEL=/path/to/other.gguf spear-server

Logs and audit
==============

``llama-server.log``
   Server-side: model load, context, slot activity.

``audit/tool-actions.jsonl``
   One record per mutating tool attempt.  Metadata only — no file contents, no
   command output, no environment values.  This is the file to read to answer
   "what did the assistant change", and it is intentionally useless for
   answering "what was in it".

``history*.json``
   Conversation transcripts, per project.

Runtime tracing
---------------

Tracing is **disabled by default**.  Enable provider-neutral JSONL traces for a
benchmark run with:

.. code-block:: console

   $ SPEAR_TRACE=1 spear-chat

Events are appended to ``audit/runtime-trace.jsonl``; ``SPEAR_TRACE_FILE``
chooses another location.  Traces record timing, counts, provider and model
identifiers, normalized outcomes and safe tool metadata.  They do **not** record
raw prompts, model responses, command strings, tool content, query or note
values, environment variables, or authorization data.  Workspace-relative file
paths and tool argument *names* are kept, because without them a trace cannot
be used to debug or to analyse a benchmark run.

Known environment quirks
========================

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Symptom
     - Explanation
   * - ``Read-only file system`` writing under ``/etc``
     - Expected.  ``/etc`` is a sealed tmpfs holding only ``alternatives``;
       only ``/workspace`` is writable (:doc:`sandbox`).
   * - A tool needs a file from the host ``/etc``
     - It will not find it.  Only ``/etc/alternatives`` is bound, plus the
       generated resolver files on the network profile.
   * - ``drawio`` CLI export fails on every file
     - snap 30.4.1 raises ``ReferenceError: next is not defined`` from its own
       ``electron.js``.  The documentation renders its SVGs directly instead
       (:doc:`directory_layout`).
   * - Network tests flaky under heavy load
     - the slirp readiness wait is a wall-clock timeout; the attachment itself
       is timing-independent (:doc:`network`).
   * - Suspend breaks a running CUDA job
     - unrelated to the harness, but a recurring loss on this workstation:
       do not suspend during a fine-tuning run.

Training-data capture
---------------------

Normal harness tasks capture a provider-neutral, redacted training episode in
``$SPEAR_STATE_DIR/audit/training-data``.  Capture is enabled by default and
can be disabled with ``SPEAR_TRAINING_CAPTURE=0``.  Drafts are updated only at
model/tool boundaries; finalized episodes are immutable, checksummed JSON with
an append-only metadata manifest.  These records are source evidence for a
future selective exporter, not ready-to-train SFT or preference data.
