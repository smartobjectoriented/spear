.. _operations:

==============================
Operations and troubleshooting
==============================

What to look at when something misbehaves, and where each state is reported.
Checking that a deployment is healthy end to end is the last step of setting
up the inference host (:doc:`/model/inference_host`, *Verifying the whole*);
changing the model it serves is in :doc:`/model/model_serving`.

.. _troubleshooting:

Symptoms
========

The symptoms below are the ones that come up in practice, with what each one
actually means.

No corpus index
---------------

*Symptom* — answers are vague, name nothing from the tree, or say the project
could not be searched.

Retrieval is not a refinement. The same model answers a small fraction of
build-system questions cold and the large majority with the corpus injected;
running without an index is running a different and much worse assistant.

.. code-block:: console

   $ spear-chat
   > /corpus                    # is this tree registered, and indexed?
   > /reindex                   # build or rebuild the index

For a tree you are passing through, ``--here`` registers it ad hoc. For one you
return to, register it properly with ``spear-corpus add`` and set
``autoindex`` so a missing index is built on sight.

Backend unavailable
-------------------

*Symptom* — the session starts and the first turn never returns, or fails with
a connection error.

Check the endpoint directly:

.. code-block:: console

   $ curl -s "$SPEAR_API_BASE/models"

``SPEAR_API_BASE`` is the endpoint the session talks to, ending in ``/v1``;
the port each backend listens on is in :doc:`/model/backends`.  The banner
names the backend the session is talking to. If it names one you did
not expect, a remembered choice in ``active-backend.conf`` is being reused —
pass the backend flag explicitly.

Where the backend is reached over an SSH tunnel, the flag opens the tunnel but
does not start the server on the far side.

Requests refused whole
----------------------

*Symptom* — a turn fails with an error from the server rather than a bad answer.

Usually the context window. ``--ctx`` is what the client budgets against, not a
request to the server; setting it above what the server serves produces
requests the server rejects entirely. Lower it to the server's real window, or
lower ``--max-tokens``.

No standard bound
-----------------

*Symptom* — a question about a specification is answered from the source tree,
with no citations.

.. code-block:: text

   /standard status          what is bound right now
   /standard list            what has been ingested
   /standard use <id> <rev>  bind one

The binding is machine-wide, so another session may have changed it. A bound
turn announces itself:

.. code-block:: text

   ⎿  bound standard: ACME-1234.5 2019-R2023 — normative claims are grounded
      in it and cited

If that line is absent on a question you meant as normative, the turn ran
unbound. Rephrase it as a question about the document — asking what something
*means*, how it *works* or how it should be *read* — rather than as a request
to do something.

The answer was withheld
-----------------------

*Symptom* — instead of an answer, a line saying the evidence does not support
it, and a list of the clauses that were read.

This is a result, not a fault. The named condition tells you what to supply;
:ref:`Normative answer guards <guards>` lists each one and the usual next step. The
most common are an ambiguous citation — name the provision kind as well as the
ordinal — and an unsupported count, where the document may simply not state a
bound.

Stale or missing evidence
-------------------------

*Symptom* — the agent is told a file is unchanged since it last read it, or that
a retrieval returned something it has already seen.

Both are deliberate: re-reading what has not changed spends the context window
for nothing, and a write puts the file back in play. If the
agent is looping over the same area without progress, it is usually short of a
*different* file rather than short of that one: say which.

A ``source_id`` from an earlier turn cannot be fetched. Evidence is
dereferenced only by the turn that retrieved it.

Build or test failure during agent work
---------------------------------------

*Symptom* — the agent's own validation fails repeatedly.

If the loop persists, the usual causes are a test command in ``projects.json``
that does not run in this environment, or a requirement that cannot be reached
by an automated test at all — which is an acceptable thing to state.

Network disabled
----------------

*Symptom* — a fetch or a search fails, and nothing suggests a proxy.

Network access is a property of the permission mode, and ``--no-network``
takes it away from every mode (:doc:`/harness/security_model`, *Execution
modes*): a refusal means the session's mode has no network, or the session was
launched with ``--no-network``.

Wrong writable root
-------------------

*Symptom* — an edit is refused although the file plainly exists.

Tools run in the directory the session was launched from, and no flag moves
the workspace: to work on a tree, launch from it.  Which paths are writable —
``/workspaces/<name>`` for a registered corpus, and what ``--single-root``
takes away — is in :doc:`/harness/security_model` (*Roots*).

Where to look next
------------------

.. code-block:: console

   $ ls "${SPEAR_STATE_DIR:-$HOME/spear/client}/audit/sessions"

The state directory and its defaults are in :ref:`persistent-state`.

The newest session directory holds ``events.jsonl`` — every tool call in order,
and whether a :term:`standard binding` was set — and ``snapshot.json``, the
conversation the turn ran on. Between them they answer most "why did it do
that?" questions without guesswork.

.. seealso::

   :ref:`Normative answer guards <guards>` · :ref:`Authoritative standards
   <standards>` · :ref:`Configuration reference <configuration>`

Reading a harness failure
=========================

Every :term:`harness` failure surfaces as a ``ToolResult`` whose ``summary`` names the
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
     - The helper is too old.  There is no fallback; see :doc:`/harness/network`.
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
never does it (:doc:`/harness/resource_control`).

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

Logs and audit
==============

The paths below are relative to the state directory
(:ref:`persistent-state`).

Server log
   Server-side: model load, context, slot activity.  In the journal
   (``sudo journalctl -u spear-inference``), or in the file the unit was
   rendered with ``--log``.

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

   $ spear-chat --trace                  # or SPEAR_TRACE=1

Events are appended to ``audit/runtime-trace.jsonl``; ``--trace-file``
(``SPEAR_TRACE_FILE``) chooses another location.  Traces record timing, counts, provider and model
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
     - Expected.  ``/etc`` is a sealed tmpfs holding only ``alternatives``,
       ``passwd`` and ``group``; only the workspace is writable
       (:doc:`/harness/sandbox`).
   * - A tool needs a file from the host ``/etc``
     - It will not find it.  Only ``/etc/alternatives``, ``/etc/passwd`` and
       ``/etc/group`` are bound, plus the generated resolver files on the
       network profile.
   * - Network tests flaky under heavy load
     - the slirp readiness wait is a wall-clock timeout; the attachment itself
       is timing-independent (:doc:`/harness/network`).
   * - Suspend breaks a running CUDA job
     - unrelated to the harness: do not let the machine suspend during a
       fine-tuning run.

Training-data capture and what it records are described in
:doc:`/model/training`.
