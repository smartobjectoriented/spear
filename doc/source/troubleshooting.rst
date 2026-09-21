.. _troubleshooting:

Troubleshooting
###############

The symptoms below are the ones that come up in practice, with what each one
actually means.

No corpus index
***************

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
*******************

*Symptom* — the session starts and the first turn never returns, or fails with
a connection error.

Check the endpoint directly:

.. code-block:: console

   $ curl -s http://localhost:8080/v1/models

The banner names the backend the session is talking to. If it names one you did
not expect, a remembered choice in ``spear/active-backend.conf`` is being
reused — pass the backend flag explicitly.

Where the backend is reached over an SSH tunnel, the flag opens the tunnel but
does not start the server on the far side.

Requests refused whole
**********************

*Symptom* — a turn fails with an error from the server rather than a bad answer.

Usually the context window. ``--ctx`` is what the client budgets against, not a
request to the server; setting it above what the server serves produces
requests the server rejects entirely. Lower it to the server's real window, or
lower ``--max-tokens``.

No standard bound
*****************

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
***********************

*Symptom* — instead of an answer, a line saying the evidence does not support
it, and a list of the clauses that were read.

This is a result, not a fault. The named condition tells you what to supply;
:ref:`Evidence and guards <guards>` lists each one and the usual next step. The
most common are an ambiguous citation — name the provision kind as well as the
ordinal — and an unsupported count, where the document may simply not state a
bound.

Stale or missing evidence
*************************

*Symptom* — the agent says a file region is already in evidence, or that a
retrieval returned something it has already seen.

Both are deliberate. A window of source already read is answered from evidence
rather than read again, and a write puts every region back in play. If the
agent is looping over the same area without progress, it is usually short of a
*different* file rather than short of that one: say which.

A ``source_id`` from an earlier turn cannot be fetched. Evidence is
dereferenced only by the turn that retrieved it.

Build or test failure during agent work
***************************************

*Symptom* — the agent's own validation fails repeatedly.

A validation that fails the same way twice reopens planning rather than being
patched a third time; two identical failures are evidence about the design. If
the loop persists, the usual causes are a test command in ``projects.json``
that does not run in this environment, or a requirement that cannot be reached
by an automated test at all — which is an acceptable thing to state.

Network disabled
****************

*Symptom* — a fetch or a search fails, and nothing suggests a proxy.

Network access is a property of the permission mode, and this is intentional:

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Mode
     - Writes
     - Network
   * - ``--safe``
     - refused
     - none
   * - ``--ask``
     - confirmed one by one
     - **available**
   * - ``--auto``
     - unattended
     - none

``--ask`` is the only mode with network access, and ``--no-network`` removes it
there too. An unattended run with network is the combination deliberately not
offered.

Wrong writable root
*******************

*Symptom* — an edit is refused although the file plainly exists.

Tools always run in the directory the session was launched from. A registered
corpus is reachable at ``/workspaces/<name>``; a relative path always resolves
in the launch directory and never reaches another tree. ``--single-root``
restricts writes further, to the launch directory alone.

To work on a tree, launch from it. No flag moves the workspace.

Where to look next
******************

.. code-block:: console

   $ ls "${SPEAR_STATE_DIR:-$HOME/.local/state/spear}/audit/sessions"

The newest session directory holds ``events.jsonl`` — every tool call in order,
and whether a standard binding was set — and ``snapshot.json``, the
conversation the turn ran on. Between them they answer most "why did it do
that?" questions without guesswork.

.. seealso::

   :ref:`Evidence and guards <guards>` · :ref:`Authoritative standards
   <standards>` · :ref:`Configuration reference <configuration>`
