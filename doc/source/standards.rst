.. _standards:

Authoritative standards and normative evidence
##############################################

This is the part of SPEAR that is not a coding assistant feature.

An engineering answer about a specified system has two kinds of source, and
they are not interchangeable. The **specification** says what is required. The
**implementation** says what the code currently does. An answer that takes the
first from the second is wrong even when every sentence in it is true of the
code, because it reports a project's decisions as the document's requirements.

SPEAR keeps the two roles apart:

.. figure:: /img/spear_evidence.svg
   :width: 100%

   Two sources, two roles. Only the document establishes what is required.

.. rubric:: The rule

**Authoritative source = normative authority. Implementation = implementation
evidence.** A claim about what is *required* may rest only on the first. The
second may illustrate, compare and contradict — it may never establish.

Binding an authoritative source
*******************************

An authoritative source is *ingested* once and then *bound*. Binding is a
property of the machine, not of a conversation: every session on the host
answers against the same document and revision, so two people cannot get
answers from two different editions without noticing.

.. code-block:: text

   /standard list                     the ingested documents and revisions
   /standard status                   what is bound right now
   /standard use <id> <revision>      bind one
   /standard ingest <path>            ingest a document
   /standard verify                   check the store against its manifest

.. important::

   Check ``/standard status`` before trusting a normative answer. The binding
   decides how every normative claim in the session is grounded, and it is
   shared machine-wide.

``/standard`` is an *operator* command. It is typed in the session and is never
reachable by the model: the agent cannot rebind the document it is being held
to.

What binding changes
====================

A bound session does not behave differently on every turn. SPEAR decides per
turn whether the turn *engages* the document — a question that asks what
something means, how a structure is composed or how it should be read engages
it; a request to fetch, install or change something does not.

On a turn that engages it:

* an opening retrieval against the document is issued **before** the model's
  first round, so the turn starts from the source rather than arriving at it;
* the four normative tools are put on the table;
* on a turn whose scope is purely normative, the tools that read a working
  tree are taken **off** the table — not by instruction, but by not offering
  them;
* every normative claim in the answer is checked against what was retrieved.

The normative tools
*******************

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Tool
     - What it does
   * - ``standard.search``
     - Searches the bound document with local hybrid retrieval. Every result
       carries a ``source_id``.
   * - ``standard.fetch``
     - Reads one unit by its ``source_id``, with its neighbours, its section
       and its page.
   * - ``standard.get_structure``
     - Reads a declared structure (a layout, a table of fields) as a
       structure, rather than as prose that happens to contain numbers.
   * - ``standard.cite``
     - Produces the citation for a unit already retrieved.

The handles are deliberately not guessable and not durable across turns: a
turn may dereference only evidence it retrieved itself. A ``source_id`` seen in
an earlier turn's transcript cannot be fetched to answer a new question from an
old question's clauses.

Provision identity
******************

A section is not a provision. One retrieved unit routinely carries a Rule, a
Permission and an Observation that share an ordinal, and a claim grounded by
one of them is not grounded by the others.

SPEAR therefore records provisions, not section numbers: each carries its
**kind** (Rule, Permission, Observation, …), its **ordinal**, its **section**,
its **page** and its **source id**. Two provisions that print the same label
are two records, and an answer that cites the label without saying which one it
means is ambiguous — which the guards report rather than resolve by guessing.

Normative force
***************

Provisions are not equal. SPEAR orders what a claim may assert against what its
evidence supports:

.. code-block:: text

   informative  <  permission  <  recommendation  <  requirement
                   (may)          (should)          (shall)

A conclusion may sit at or below the level of the provision it cites, never
above it. "The field **shall** be present" resting on a provision that says
*should* is a strengthened modality, and is reported as one. This also catches
the case with no modal verb in it at all: a bare "Yes" to "is this required?"
asserts the obligation just as plainly.

The force a provision may carry is capped by its printed role. An Observation
that *describes* an obligation is not the obligation.

Why an implementation comment is not authority
**********************************************

A comment in a source file is written by the same people who wrote the code. It
may restate a rule correctly, restate it as it was understood five years ago,
or describe a local convention that the document never mentions. Nothing about
its position in the tree makes it normative.

SPEAR grounds a name read out of a repository for exactly one purpose:
**existence**. It establishes that the name was really in front of the model
and was not invented. It carries no modality, no authority and no clause, and
it cannot raise the level a claim is entitled to.

.. warning::

   This is the failure the separation exists for: a self-contained question
   about a specification, answered from the implementation's comments and
   presented as the specification. Every sentence can be accurate about the
   code and the answer still be wrong about the document.

When SPEAR withholds an answer
******************************

Where the retrieved evidence does not support the draft, SPEAR does not publish
the draft with a caveat. It withholds it and says which check failed:

.. code-block:: text

   The retrieved normative evidence does not support this answer as written,
   so it is withheld rather than shown.
     - Rule 9.9.9-1 names more than one provision (…); which one is meant
       decides the answer.
   Clauses read this turn: §4.2.1, §4.3, §9.9.

That is a result, not an error. The correct next step is usually to ask a
narrower question, or to name which provision is meant. See :ref:`Evidence and
guards <guards>` for the full list of conditions and what each one asks of you.

An example, end to end
**********************

The document below is invented; no licensed text appears in this
documentation.

.. code-block:: text

   > /standard status
     bound: ACME-1234.5 2019-R2023

   > Explain how the frame descriptor words work and how they should be parsed
     ⎿  bound standard: ACME-1234.5 2019-R2023 — normative claims are
        grounded in it and cited
     ↪ reading the bound standard before answering — standard.search

   Descriptor words are enabled by the first word of the set (§7.1, p.44).
   Setting bit *n* of that word designates that word *n* is present
   (Rule 7.1-3, p.44). Fields are read in descending bit order within a word
   and in ascending word order (Rule 7.4.1-2, p.61).

Three things are true of that turn and are worth naming:

#. the first tool call was against the document, not against the source tree;
#. no code-reading tool was offered, because the question asked about the
   document and nothing else;
#. every sentence carries the provision it rests on, so a reader can check it
   without trusting the agent.

Mixed questions
***************

"Compare our parser with what the standard requires" asks about both, and gets
both: the normative tools and the code tools are on the table together. The
answer is then expected to keep the halves visibly distinct — this is what the
document requires, this is what the implementation does, and here is where they
differ.

Nothing in SPEAR merges them for you. The distinction is the product.

.. seealso::

   :ref:`Evidence and guards <guards>` · :ref:`Conversation context
   <context>` · :ref:`Agent workflow <workflow>`
