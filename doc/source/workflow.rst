.. _workflow:

The engineering workflow
########################

An answer to a question is one thing. A change to a codebase is another, and
SPEAR does not treat them the same way.

When a turn is going to modify a project, it runs through five stages. Each one
opens on a condition the runtime can check, and a stage that cannot open says
so rather than proceeding on an assumption.

.. figure:: /img/spear_workflow.svg
   :width: 100%

   The five stages, and the condition that opens each gate.

Why the stages are separate
***************************

Each separation answers a way the single-pass version fails.

**INVESTIGATE before PLAN.** A requirement nobody read cannot be planned
against. The turn must hold both sides — what the authoritative source requires
and what the implementation currently does — before a plan is accepted. Reading
only one side produces a plan that is either unmoored from the specification or
unmoored from the code.

**PLAN before EDIT.** A change described only as it is being made is a change
nobody can review. Each plan item is recorded separately and names the file it
will touch, which is also what opens the write gate for that file.

**Validation designed before the code.** A plan item carries the validation
that will prove it. Deciding this *after* writing the code produces a test that
describes what was written instead of checking what was required; deciding it
first makes the test a check.

**TEST reaches the changed behaviour.** Running an existing suite that never
enters the changed path proves nothing about it. A green run of two hundred
tests is a true statement and an irrelevant one.

**REVIEW reads the ledger.** The closing report is generated from the
requirement ledger, not from the model's own summary of its work.

INVESTIGATE
***********

The turn reads. Where an authoritative source is bound, that means the document
as well as the tree.

Two things make investigation converge rather than wander:

* a window of source already in evidence is answered *from* evidence instead of
  being read again — the common waste is not calling the same command twice, it
  is reading the same lines through different windows;
* a short barren sequence — tool calls that add no new file and no new
  provision — prompts the turn to synthesise what it has rather than keep
  looking.

A write puts every region back in play, because after an edit the file is no
longer the file that was read.

PLAN
****

Each requirement becomes one recorded item:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Field
     - What it holds
   * - requirement
     - what is required, in the document's terms
   * - requirement evidence
     - the provision it comes from
   * - current behaviour
     - what the implementation does today
   * - implementation evidence
     - the file that shows it
   * - gap
     - the difference between the two
   * - correction
     - the change, naming the file it lands in
   * - validation
     - how the corrected behaviour will be proved

An item whose validation is "the existing project test suite" is refused at
this point, not at the end. So is an item with a requirement and no evidence
for it.

Saying plainly that no automated test can reach a behaviour — because it needs
real hardware, for example — is accepted. Silence is not.

EDIT
****

The write gate is scoped to the plan. A file becomes writable because a planned
item named it, and only that file. This is deliberately per-item rather than
per-contract: waiting for a whole contract to be planned before anything may be
written was correct and unusable.

An edit outside the tree the session opened is refused whatever the plan says.

TEST
****

The validation designed in PLAN is run. A requirement whose code changed and
whose validation has not run is not "done" — it is recorded as *code changed,
awaiting validation*, and the closing report says so.

A validation that fails the same way twice does not get patched a third time.
It reopens PLAN, because two identical failures against the same design are
evidence about the design.

REVIEW
******

The requirement ledger is published as a matrix: one row per requirement, with
its evidence, its disposition and whether its validation ran.

Three dispositions matter more than the rest, because they are the ones a
summary tends to lose:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Disposition
     - Meaning
   * - ``satisfied_already``
     - the implementation already met it; no change was needed
   * - ``explicitly_out_of_scope``
     - deliberately not addressed, with the reason recorded
   * - ``undetermined``
     - could not be established from what the turn read

A requirement that was reviewed and excluded is named in the report together
with why. Narrowing the work is a visible decision, not a silence.

.. note::

   If the closing prose claims the work is complete while the ledger still has
   an open row, the report says to take the matrix and not the claim. The
   ledger has the last word, and "incomplete" is a correct result when it is
   the true one.

.. seealso::

   :ref:`Evidence and guards <guards>` · :ref:`Authoritative standards
   <standards>`
