.. _guards:

Evidence and guards
###################

SPEAR records what a turn was actually shown, and checks the answer against it
before the answer reaches you. This page explains what is recorded, what is
checked, and what to do when a check fires.

Three kinds of evidence
***********************

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Kind
     - Where it comes from
     - What it can establish
   * - Normative evidence
     - the bound authoritative source, through the ``standard.*`` tools
     - what is **required**, at the force the provision carries
   * - Implementation evidence
     - what a code-reading tool **returned** this turn
     - what the code **does**, and that a name exists
   * - Corpus evidence
     - retrieval over the indexed project corpora
     - where to look; it is a pointer, not a citation

The second row is narrower than it looks. What grounds a name is content a tool
put in front of the model — not a file somewhere in the repository that happens
to contain it. A name the model never saw is a name the model invented,
whatever an unopened file holds. Results that list only paths (``find``,
``ls``, ``grep -l``) ground nothing, because a path is not content.

What the guards check
*********************

.. list-table::
   :header-rows: 1
   :widths: 26 38 36

   * - Condition
     - Meaning
     - Typical action
   * - Ungrounded identifier
     - the answer names a field or symbol that is in neither the retrieved
       document nor anything a tool returned
     - ask again naming the area, so the turn retrieves it; or check whether
       the name is real
   * - Strengthened modality
     - the conclusion is stronger than the provision it cites — a *shall* on
       top of a *should*
     - re-read the cited provision; the weaker reading is usually the correct
       answer
   * - Incoherent conclusion
     - the answer contradicts its own evidence, or itself
     - ask for the specific provision; the draft was arguing with itself
   * - Unsupported cardinality
     - a maximum, minimum or exact count with no clause that states one
     - the document may simply not state a bound; ask what it *does* state
   * - Ambiguous citation
     - the cited label resolves to more than one provision
     - name the kind as well as the ordinal (Rule, Permission, Observation)
   * - Misattributed force
     - a named provision is credited with a force its role cannot carry
     - check what the provision is: an Observation describing an obligation
       is not the obligation
   * - No normative call
     - a normative turn is about to answer having retrieved nothing
     - nothing to do — SPEAR issues the retrieval itself and continues

The first six are checks on the answer. The last is a repair: the turn is sent
to the document before it is allowed to finish.

Why SPEAR withholds
*******************

When a check fires and the draft cannot be corrected from what was retrieved,
SPEAR does not publish it with a hedge. It withholds it, names the condition,
and lists the clauses the turn did read:

.. code-block:: text

   The retrieved normative evidence does not support this answer as written,
   so it is withheld rather than shown.
     - Rule 9.9.9-1 names more than one provision; which one is meant
       decides the answer.
   Clauses read this turn: §4.2.1, §4.3, §9.9.

A hedged wrong answer is worse than no answer, because the hedge is read as
politeness and the claim is read as fact. The withheld form tells you exactly
what to supply.

Why "incomplete" is a result
****************************

On a turn that changes code, the closing report is generated from the
requirement ledger. A requirement the turn could not establish is reported as
undetermined; one deliberately excluded is reported as out of scope, with the
reason; one whose code changed but whose validation has not run is reported as
awaiting validation.

None of these is a failure of the run. They are the run telling you what it
knows, and the alternative — a summary that rounds all three up to "done" — is
the failure mode this replaces.

.. note::

   If the model's own closing sentence claims completion over an open ledger,
   the report says so explicitly and directs you to the matrix. The ledger is
   the record.

Auditing a session
******************

Every tool action is recorded with metadata, and every session keeps a snapshot
of the conversation it was generated from:

.. code-block:: console

   $ ls "${SPEAR_STATE_DIR:-$HOME/.local/state/spear}/audit"
   checkpoints  sessions  tool-actions.jsonl  tool-results

The per-session ``events.jsonl`` carries the turn-by-turn event stream —
including which tools ran, in which order, and whether a standard binding was
set for the turn. That is the record to read when an answer surprises you.

.. seealso::

   :ref:`Authoritative standards <standards>` · :ref:`The engineering workflow
   <workflow>` · :ref:`Troubleshooting <troubleshooting>`
