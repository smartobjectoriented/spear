.. _evidence_verdicts:

Evidence and verdicts
#####################

A turn that changes code ends with a verdict that SPEAR computes, not one the
model writes. There are two kinds of evidence behind it, and they are never
allowed to stand in for each other:

**Implementation evidence**
   what the turn's own tools did to the tree and what its checks showed —
   builds, tests, a look at the result. It says whether the change *works as
   written*.

**Normative evidence**
   evidence that a constraint taken from the bound standard holds in the final
   source: a deterministic reading of the source, or a project check bound to
   that provision. It says whether the change *does what the standard
   requires*.

A passing build is not :term:`normative evidence`, and a model's reading of a
provision is neither.

Implementation evidence
***********************

Every change-making turn on the :term:`coding core` ends with one of three states:

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - State
     - Meaning
   * - ``VERIFIED``
     - source changed, and the checks that show what the answer claims ran and
       passed on the **final** source
   * - ``UNVERIFIED``
     - source changed, and nothing that ran on the final source shows it works
       — or something that ran on it failed
   * - ``NO_CHANGE``
     - no source changed

``VERIFIED`` is exactly as strong as the checks that ran: the configured
validation passed on the final source, and nothing more is claimed. A project
whose tests do not exercise a property can be ``VERIFIED`` while that property
is wrong -- an external oracle may still find it. Whether a change does what a
bound standard requires is the normative status, which stays
``NOT_DEMONSTRATED`` until normative evidence establishes it; a passing build
never does.

An ``UNVERIFIED`` answer opens with the verdict and its reason, and every
sentence in it that claims success without hedging is marked
*(not verified)* where it stands, so the answer cannot say one thing while the
verdict says another.

The final-state principle
=========================

.. admonition:: A validation proves the source state that existed when it ran
   :class: important

   Every successful change to delivered source opens a new **source epoch**. A
   check counts for the verdict only if it ran in the final epoch — after the
   last change. ``edit → build PASS → edit`` does not verify the final tree: the
   build proved a tree that no longer exists.

Consequences that follow directly:

* **A failed project build verifies nothing.** SPEAR runs the project's own
  build and test commands on the final tree. A failure there makes the turn
  ``UNVERIFIED`` whatever else passed; a build that could not run proves
  nothing either way.
* **Declared commands win over probed ones.** A project needs no configuration
  to be checked: its build and test commands are probed from the files it has
  (a CMake tree, a Makefile, cargo, go, npm, a Python package). What the
  project declares (:ref:`projects`) takes precedence kind by kind — a declared
  build is the build, and a failing one is never replaced by an easier guess;
  a probe only fills a kind the project left undeclared. Each run is recorded
  with its origin, ``configured`` or ``probed``.
* **A Makefile that only prints its help is not a build.** When a bare
  ``make`` would print the help (as a generated Sphinx Makefile does), it is
  not probed as the build: it would have "passed" every change.
* **Evidence is claim-specific.** "It builds" needs a build; "the tests pass"
  needs a test run; "the link survives a clean and a rebuild" needs a clean, a
  build and a look at the result, in that order, all in the final epoch. A
  build alone does not show persistence across a clean.
* **An individual BitBake task is not a full build.** ``bitbake -c configure
  x`` validates that task; it is not the ``bitbake x`` (or ``-c compile``) a
  build claim needs, and ``bitbake -n``, ``-e`` or ``-g`` run nothing at all.
* **A failure behind a filter is still a failure.** ``make | tail`` returns the
  exit status of ``tail``; without ``pipefail``, what the output says decides.
* **A command sent to the background shows nothing.** ``make &`` returns at
  once with the shell's 0, before ``make`` has done anything, so it is not a
  check -- nor is any command of a list that ends with ``&``.
* **Compiling loose files is not the project's verification.** When the
  project has its own build or test command, declared or probed, a turn whose
  only check was compiling the files it touched is ``UNVERIFIED``.
* **Build outputs are not source changes.** What a build writes into its own
  output areas (``build/tmp``, ``generated/``, git-ignored files) does not open
  a new epoch, so a build that writes its artefacts still counts.

Technically correct code can still be ``UNVERIFIED``. The verdict is about what
was *shown*, not about what is true — and an honest "not shown" is the result
the operator can act on.

Normative evidence
******************

On a :term:`MIXED` turn (:ref:`mixed_mode`), each constraint of the packet receives an
authoritative status:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Status
     - Meaning
   * - ``SATISFIED``
     - decisive authoritative evidence on the final source shows it holds
   * - ``VIOLATED``
     - decisive authoritative evidence on the final source shows it does not
   * - ``NOT_DEMONSTRATED``
     - nothing authoritative establishes it either way
   * - ``AMBIGUOUS``
     - the provision itself could not be resolved in the standard as stored
   * - ``EVIDENCE_CONFLICT``
     - two authoritative providers disagree about the same final source

Only two providers are authoritative today: deterministic **source
predicates** and **conformance checks** a project has explicitly bound to a
provision (:ref:`mixed_mode`). A model's opinion of a constraint — and a second
model's opinion of that opinion — is recorded as advice and decides nothing.

Composite verdicts
******************

A MIXED turn reports both dimensions in one line:

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Verdict
     - When
   * - ``VERIFIED + COMPLIANT``
     - every applicable requirement established, implementation verified
   * - ``COMPLIANT BUT IMPLEMENTATION UNVERIFIED``
     - every applicable requirement established; the change is not shown to
       work
   * - ``COMPLIANT, NO CHANGE MADE``
     - every applicable requirement established on an unchanged tree
   * - ``NOT COMPLIANT``
     - an applicable requirement is established as violated
   * - ``COMPLIANCE NOT DEMONSTRATED``
     - not every applicable requirement is established, or applicability or
       coverage is incomplete
   * - ``NORMATIVE AMBIGUITY``
     - an applicable requirement is ambiguous in the standard as stored
   * - ``EVIDENCE CONFLICT``
     - authoritative evidence disagrees about a requirement

``COMPLIANT`` needs all of the following at once: complete :term:`structural coverage`,
every applicable SHALL constraint established ``SATISFIED``, none ``VIOLATED``,
no requirement whose applicability is unresolved, no evidence conflict, and all
of that on the final :term:`source epoch`. SHOULD and MAY constraints keep their force:
they are reported, and they never block.

.. _compliance_not_demonstrated:

COMPLIANCE NOT DEMONSTRATED
***************************

.. admonition:: Not demonstrated is not non-compliant
   :class: important

   A project may compile, pass its tests, and even be correct — and still
   receive ``COMPLIANCE NOT DEMONSTRATED``. It means SPEAR does not hold
   authoritative evidence for every applicable requirement. It does **not**
   mean SPEAR found a violation.

This is the intended result whenever the evidence is insufficient, and in
practice the most common one for real protocol and register requirements:
"one acknowledgement packet per requested type" or "program the frequency
register during boot" cannot be read off source literals, and no model
judgement is allowed to stand in for the missing evidence.

The verdict comes with its counts, so it is clear what is missing:

.. code-block:: text

   Coverage: COMPLETE — 5 cited, 1 added by the document's structure,
   0 excluded as not applicable.
   Required: 6 — applicable 0 (satisfied 0, violated 0, not demonstrated 0),
   applicability unresolved 6, evidence conflicts 0.

There are two ways to turn it into ``COMPLIANT``, and both are explicit project
decisions: declare which provisions apply to the project, and bind a
:term:`conformance check` to each requirement that the source cannot show by itself
(:ref:`projects`).

.. _evidence_conflict:

Evidence conflict
*****************

When two authoritative providers disagree about the same final source — say a
:term:`source predicate` reads the count as four (``SATISFIED``) while a decisive bound
check fails (``VIOLATED``) — the constraint is ``EVIDENCE_CONFLICT`` and the
verdict is ``EVIDENCE CONFLICT``. Neither provider wins because it ran later,
and no repair is attempted: one of the two is wrong about the project, and only
a person can say which.

Examples
********

**An unverified change.**

.. code-block:: text

   **UNVERIFIED** — src/ring.c changed, and the final change was not
   revalidated after the last source modification: what passed ran on an
   earlier state of the tree.

   ring_next() now wraps at size, and the build succeeds *(not verified)*.

**Compliance not demonstrated.** A synthetic standard requires a field to carry
exactly four entries and a mode to be 2 when a flag is set; the turn also cites
a timing rule nothing deterministic can check.

.. code-block:: text

   MIXED VERDICT: COMPLIANCE NOT DEMONSTRATED — SYNTH-STD 1, constraint set
   ncs-… (3 constraint(s); compliance is judged against these only).
   - C1 Rule 4.2.1-1 (SHALL, when flag X is set; applicability APPLICABLE):
     SATISFIED — CONDITIONAL_VALUE: the provision states 2, the final source
     gives 2 (record.py:2 `mode = 2 if flag_x else 1`) [record.py:2]
   - C2 Rule 4.2.1-2 (SHALL; applicability APPLICABLE): SATISFIED —
     EXACT_COUNT: the provision states 4, the final source gives 4
     (record.py:3 `count = [0, 0, 0, 0]`) [record.py:3]
   - C3 Rule 4.3-1 (SHALL; applicability APPLICABLE): NOT_DEMONSTRATED —
     not independently established (the model check found a possible
     satisfaction: …)
   Implementation evidence: VERIFIED. Normative: NOT_DEMONSTRATED.

**An evidence conflict.**

.. code-block:: text

   - C2 Rule 4.2.1-2 (SHALL; applicability APPLICABLE): EVIDENCE_CONFLICT —
     authoritative evidence disagrees: EXACT_COUNT: the provision states 4,
     the final source gives 4 (record.py:3 `count = [0, 0, 0, 0]`); project
     check count-four (`python3 -B tests/check_count.py`) failed (exit 1);
     its binding makes a failure decisive [record.py:3]

.. seealso::

   :ref:`implementation_mode` · :ref:`mixed_mode` · :ref:`projects`
