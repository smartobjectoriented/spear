.. _mixed_mode:

Mixed mode
##########

A **MIXED** request asks for a change that has to satisfy a bound standard:
"make the encoder emit one acknowledgement per requested type, as the standard
requires". It needs both sides — what the document requires and what the code
does — and a change. SPEAR does not answer it with one loop doing two jobs. It
runs three passes around the :term:`coding core` and judges the result on evidence the
model cannot supply.

.. figure:: /img/SPEAR-Mixed.drawio.png
   :width: 100%
   :alt: The MIXED pipeline: normative pre-pass, cited provisions, structural
         coverage, applicability, constraint packet, coding core, final source
         state, authoritative evidence, statuses and the composite verdict

   The MIXED pipeline. Advisory model findings are recorded beside it and
   decide nothing; a single bounded repair loops back only on a decisive
   violation.

When a request is MIXED
***********************

Three things must hold together:

* a standard is **bound and engaged** for the session (:ref:`standards`);
* the request asks for a **change** to the tree;
* it asks in the **standard's terms** — it names the document as its authority
  ("as the specification requires"), a provision, or compliance itself.

A write in a session that merely has a standard bound is not :term:`MIXED` because of
the binding: "rename this function" stays a change request, and a question
about a provision stays a normative question. The class is read from the
request's own words, by its syntax — a file name, a function call, a causative
"make X do Y" — not by a list of one project's vocabulary.

The three passes
****************

1. The normative pre-pass
=========================

The established :term:`normative runtime` (:ref:`standards`) runs first, read-only,
with the standard's own tools and nothing else. It is asked to identify the
provisions that govern the change and to state each one exactly — its force,
its condition, its counts, the identifiers and values it names — citing every
one. It writes nothing and reads no code.

It investigates with those tools, then finalizes with none: when the tool window
closes the model is told that no tool remains, and a tool call it then writes
out as text is never run and never taken as the answer. It is asked once more;
a pass that still gives no answer ends as not completed, and says so.

2. The constraint packet
========================

What the pre-pass *cited* becomes the starting point of a packet; nothing is
taken from its prose. Three sets are kept apart, and every move between them is
recorded with its reason:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Set
     - What it holds
   * - Retrieved candidates
     - every provision the pre-pass looked at
   * - Coverage set
     - the provisions it cited — by printed label, by source id, or by section
       and a verbatim quotation of the provision's own words — **plus** the
       provisions the document itself groups with them
   * - Active constraint set
     - the coverage set minus what is established as not applicable

Structural coverage
-------------------

Selecting one requirement of a group can leave its siblings out — and a
compliance picture with half a list in it is not a compliance picture. So each
cited provision is closed over the relationships the document's own structure
records, and only those:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Relation
     - Adds
   * - ``SAME_NORMATIVE_LIST``
     - the other normative declarations of the same numbered list: same
       section, same kind (Rule, Permission, …), same heading
   * - ``SAME_TABLE``
     - the other normative rows of the same table
   * - ``EXPLICIT_REFERENCE``
     - a provision the cited one names outright — one step, never onward

Every added provision records the relation, the provision it was reached from
and the structure it was reached through. Closure is bounded: a group larger
than its limit, or a closure that would add too many provisions, is not
absorbed — the coverage is marked **INCOMPLETE**, and an incomplete coverage
can never yield ``COMPLIANT``. A pre-pass that cites nothing identifiable also
leaves coverage incomplete; the packet is never filled from whatever retrieval
happened to return. The coverage reason says which of three things happened:
the pass answered and cited no provision, its answer was withheld because the
evidence did not support it, or it did not complete.

Applicability
-------------

Being retrieved, cited or structurally adjacent does not make a provision apply
to this change. Each provision in the coverage set is:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Applicability
     - Established by
   * - ``APPLICABLE``
     - the request naming the provision or its section; the project declaring
       it applicable; or a conformance check the project binds to it
   * - ``NOT_APPLICABLE``
     - the project declaring it not applicable, with a reason
   * - ``UNRESOLVED``
     - nothing deterministic — the default

The model's view (*possibly applicable* because the pre-pass cited it,
*possibly not applicable* from the post-check) is kept in the audit record as
advice. It never discards a requirement and never adds one.

An ``UNRESOLVED`` requirement can be neither met nor failed: it holds the
verdict at ``COMPLIANCE NOT DEMONSTRATED``, it is never reported as
``VIOLATED``, and it is never repaired.

The packet the coding core receives
-----------------------------------

The active set becomes a ``NormativeConstraintSet``: one constraint per
provision, each with the provision's identity (``ProvisionKey`` and
``ProvisionInstanceId``), its modality (SHALL, SHOULD, MAY), the requirement in
the document's words, its condition, any exact count, any stated values and
identifiers, and a one-line implication — "permitted, not required: the
implementation must still work when it is absent" for a MAY, "applies only when
…" for a conditional requirement. The set has a stable identity that the audit
trail and the verdict refer to.

The coding core receives the objective and this packet, compactly, and nothing
else of the standard — no surrounding sections, no normative tools.

The workspace's knowledge and its external capabilities reach this pass and
no other. The pre-pass answers from the standard alone, and the post-check
reads only the packet and the final source: what someone once noted about a
project is not evidence of what a standard requires.

3. Implementation, then the final source
========================================

The frozen coding core (:ref:`implementation_mode`) makes the change, with its
six tools and :term:`SpearHost` around it — exactly as for any implementation request.
The constraints are guidance to it, not a policy it is held to. Then the
**final source state** is fixed: its fingerprint is taken, and the
:term:`implementation evidence` of the turn is computed (:ref:`evidence_verdicts`).

The normative authority model
*****************************

.. admonition:: A model's interpretation is advisory
   :class: important

   A model's judgement of a constraint cannot by itself establish
   ``SATISFIED``, ``VIOLATED``, ``COMPLIANT`` or ``NOT COMPLIANT``. A second
   call to the same model — or to another — is not independent proof either.

After the change, a tool-less model check reads the final source against the
packet. What it returns is a set of **candidate findings**: for each
constraint, the source facts it quotes (only those the final files really
contain — path, line, excerpt and its hash) and its interpretation of them,
stored separately. A candidate violation may also receive a second reading.
Both are recorded; neither decides.

Authority belongs to **evidence providers** that do not depend on a model:

Source predicates
=================

A small set of deterministic predicates reads a constraint off the final source
when, and only when, both sides are certain: the packet states the value, and
the source assigns the thing the constraint names in a form that can be read
without interpretation.

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Predicate
     - The provision says
     - The source must show
   * - ``EXACT_COUNT``
     - "the *count field* shall contain exactly four entries"
     - a literal list, array or initializer assigned to that field
   * - ``VALUE_EQUALS``
     - "the *mode* shall be 2"
     - a literal assigned to it
   * - ``CONDITIONAL_VALUE``
     - "the *mode* shall be 2 when *flag X* is set"
     - a literal, or a conditional expression on that one flag

They are deliberately narrow. A predicate decides nothing — leaving the
constraint ``NOT_DEMONSTRATED`` — when the subject names several things; when
the count is not exact; when the statement goes on past the value; when the
source assigns the subject anywhere in a way a literal cannot show (a compound
assignment, an increment, a method call, its address taken, a call argument, an
unpacking, a branch or a loop); or when two sites disagree. Comments and
strings are ignored. SPEAR does not claim to prove arbitrary C or C++
semantics: a source shape it cannot read safely is *not demonstrated*, by
design.

Conformance checks
==================

A project can declare that one of its checks is evidence for a provision — a
``ConstraintCheckBinding``, written in the project's ``normative_checks``
(:ref:`normative_checks`). Only a bound check carries normative weight:

.. admonition:: A passing test proves nothing normative on its own
   :class: important

   ``make test`` passing says the tests pass. Only a check explicitly bound to
   a provision says that provision holds.

A bound check runs in the same sandbox as the project's own verification,
against the final source, and is recorded as ``PASSED``, ``FAILED``,
``NOT_RUN`` or ``ERROR`` with its exit code and a hash of its output. Its
meaning comes from the binding's semantics:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Semantics
     - A pass / a failure
   * - ``PASS_ESTABLISHES_SATISFIED``
     - a pass establishes ``SATISFIED``; a failure establishes nothing
   * - ``PASS_AND_FAIL_DECISIVE``
     - a pass establishes ``SATISFIED``; a failure establishes ``VIOLATED``

``ERROR`` (the command could not be started, could not be found, or timed out)
and ``NOT_RUN`` establish nothing. A result counts only for the :term:`source epoch` it
ran on: a check that changes the source it was run against, or any later edit,
makes its result stale. A pass establishes the bound provisions and nothing
else — not the section, not the standard, not the applicability of a provision
it is not bound to.

When providers disagree
=======================

If a :term:`source predicate` and a decisive bound check disagree about the same final
source, the constraint is ``EVIDENCE_CONFLICT`` (:ref:`evidence_conflict`).
Neither wins.

One bounded repair
******************

If an **applicable SHALL** constraint is ``VIOLATED`` on decisive authoritative
evidence, the coding core gets exactly one repair: the constraint, its
provision, and the contradiction the evidence found — never a model's
speculation. The source epoch moves, every earlier check becomes stale, the
bound checks run again, and the verdict is computed afresh from the new final
source. A model-only finding, an ambiguity, an unresolved applicability or a
MAY never triggers a repair.

The verdict
***********

The turn ends with the composite verdict (:ref:`evidence_verdicts`) and the
coverage report behind it: how many provisions were cited, how many structure
added, how many were excluded as not applicable, and for the required ones how
many are satisfied, violated, not demonstrated, of unresolved applicability, or
in conflict. Every compliance claim in the coding core's own prose that the
verdict does not support is marked *(compliance not established)* where it
stands.

Everything above is recorded in the audit trail — the pre-pass, the packet and
its coverage, each applicability decision, each binding loaded, each check run
and its result, each constraint's candidate and authoritative status, any
conflict, the repair and the final verdict — and none of it is shown to the
coding core.

Examples
********

**A MIXED task.** The synthetic standard states, in section 4.2.1, that the
*mode* field shall be 2 when flag X is set (Rule 4.2.1-1), that the *count*
field shall contain exactly four entries (Rule 4.2.1-2), and that the
*metadata* field may be absent (Permission 4.2.1-3).

.. code-block:: text

   > /standard use SYNTH-STD 1
   > Update the header implementation in record.py so it complies with the
     Rule 4.2.1 provisions of the standard.

   MIXED VERDICT: VERIFIED + COMPLIANT — SYNTH-STD 1, constraint set ncs-…
   (3 constraint(s); compliance is judged against these only).
   Coverage: COMPLETE — 3 cited, 0 added by the document's structure,
   0 excluded as not applicable.
   Required: 2 — applicable 2 (satisfied 2, violated 0, not demonstrated 0),
   applicability unresolved 0, evidence conflicts 0.
   - C1 Rule 4.2.1-1 (SHALL, when flag X is set; applicability APPLICABLE):
     SATISFIED — CONDITIONAL_VALUE: the provision states 2, the final source
     gives 2 (record.py:2 `mode = 2 if flag_x else 1`) [record.py:2]
   - C2 Rule 4.2.1-2 (SHALL; applicability APPLICABLE): SATISFIED —
     EXACT_COUNT: the provision states 4, the final source gives 4
     (record.py:3 `count = [0, 0, 0, 0]`) [record.py:3]
   - C3 Permission 4.2.1-3 (MAY; applicability APPLICABLE): NOT_DEMONSTRATED
   Implementation evidence: VERIFIED. Normative: SATISFIED.

The request named "the Rule 4.2.1 provisions", which is what makes all three
applicable; the permission is reported and does not block.

**A project-bound check.** The same requirement, implemented as
``count = [0] * 4`` — a shape no predicate reads — with the project binding
its own check to Rule 4.2.1-2 (:ref:`normative_checks`):

.. code-block:: text

   - C2 Rule 4.2.1-2 (SHALL; applicability APPLICABLE): SATISFIED —
     project check count-four (`python3 -B tests/check_count.py`) passed

Without the binding the same source, with the same passing tests, would report
C2 ``NOT_DEMONSTRATED``.

.. seealso::

   :ref:`evidence_verdicts` · :ref:`standards` · :ref:`normative_checks`
