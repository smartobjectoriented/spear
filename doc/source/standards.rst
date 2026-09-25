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

Which documents are supported
*****************************

There is no list of supported standards, and that is deliberate. Extraction is
generic: any specification-style PDF can be ingested, and what SPEAR derives
from it — provisions, their kinds, their ordinals, their sections and pages —
is read from the document's own structure.

What *is* document-specific is smaller, and it is the only thing that ever
needs declaring: **what this document's provision roles mean.**

Most specifications follow the conventional reading, which is the default:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Role
     - May establish at most
   * - Rule, Requirement
     - a requirement
   * - Recommendation
     - a recommendation
   * - Permission
     - a permission
   * - Observation, Definition, Informative
     - nothing — they explain, they do not oblige
   * - unlabelled normative prose, and tables
     - a requirement

A **profile** is written only where a document's own front matter says
something different, or where it is worth recording that the conventional
reading was checked against the document rather than assumed. Profiles live in
``standard_profiles.py``, one short declaration each.

.. note::

   The last row is not a detail. A document's bit-assignment tables usually
   *are* the requirement a Rule points at, so capping them would discard the
   thing the Rule refers to.

   The row above it is the one profiles get written for. A document whose
   Observations describe an obligation in the words of that obligation will,
   without a profile, have those Observations read as binding — and an answer
   then reports the Observation as the requirement instead of the Rule that
   imposes it.

So: to find out whether a document works, ingest it and look at what came out.
The manifest records what was extracted and what failed, and ``/standard
verify`` checks the store against it.

Ingesting a document
********************

.. code-block:: text

   /standard ingest <pdf> --id <id> --revision <revision>
                          [--retain-pdf]
                          [--origin LICENSED_STANDARD|PUBLIC]
                          [--allow-offload]

The input is a PDF. ``--id`` and ``--revision`` are how the document is named
afterwards, and they are what ``/standard use`` takes.

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Flag
     - Meaning
   * - ``--origin``
     - what this document *is*. Defaults to ``LICENSED_STANDARD``, the safe
       answer, so a document nobody classified is treated as restricted.
   * - ``--retain-pdf``
     - keep the original file beside the extracted corpus. Without it the
       store holds the extraction only.
   * - ``--allow-offload``
     - embed this licensed corpus on the configured GPU host anyway, for this
       one command.

``--origin`` is not bookkeeping. A licensed corpus is **not** sent to a shared
embedding host: that is refused, and ``--allow-offload`` is the operator
saying otherwise once. It is a flag rather than a stored property on purpose —
relabelling the corpus ``PUBLIC`` would buy the same offload *and* tell
training governance the text is exportable, which is a much larger claim. It
is also what decides whether the document may travel in a container image
(:ref:`image-profiles`).

What ingestion produces
=======================

.. code-block:: text

   <store>/<id>/<revision>/
     manifest.json          what was extracted, and what failed
     corpus/                the canonical units
     source/                the original PDF, with --retain-pdf
     indexes/lexical/       always built
     indexes/vector/        when an embedding model is configured: the
                            vectors as a float32 matrix (vectors.npy)
     verified.json          what was last checked in full, and against which files
     indexes/crossrefs/     references between provisions
     retrieval.json         how this document is searched, once set

A document is checked in full once — every unit rehashed, every index's
fingerprint recomputed — and the stamp of the files it was checked against
(size, mtime, ctime, inode) is recorded in ``verified.json``. Until one of
them changes, binding and searching trust that check instead of repeating it:
for a document of 300 000 units the difference is five minutes against under
a second. Any write moves a stamp, so a changed file is always checked again;
``/standard verify`` checks everything regardless.

The vector index is optional: with no embedding model configured, retrieval
falls back to lexical and says so rather than failing. Where one is
configured, its revision must be pinned — an unpinned model makes the index
fingerprint depend on whatever the cache happened to hold.

A corpus is never embedded on the local CPU. A ``PUBLIC`` document is embedded
on the host ``SPEAR_STANDARD_EMBED_REMOTE`` names; a licensed one only with
``--allow-offload``. Otherwise the vector index is skipped, as if no model were
configured, unless ``SPEAR_STANDARD_EMBED_DEVICE`` names a local accelerator.

How a document is searched
==========================

Retrieval mode (``lexical``, ``vector`` or ``hybrid``) and structural
completion (how many companion units a search may add) are recorded per
document, not per machine: what is measured is one document, and binding
another must not inherit its settings.

.. code-block:: text

   /standard retrieval                              the bound document's settings
   /standard retrieval lexical --completion 2       set them
   /standard retrieval <id> <revision> hybrid       for a document not bound

A document that declares nothing gets ``hybrid`` with completion 0.
``SPEAR_STANDARD_RETRIEVAL_MODE`` and ``SPEAR_STANDARD_EVIDENCE_COMPLETION``
still override the document for one session; ``/standard status`` and the
startup banner say when they do.

The manifest is the report:

.. code-block:: text

   page_count               121
   canonical_unit_count    1109
   requirement_count         42
   recommendation_count      54
   definition_count           2
   extraction_errors         []
   human_validation_status  NOT_REVIEWED
   extractor_version        poppler-structure-v2

``extraction_errors`` and the counts are how you judge whether a document came
out usable. A specification of 121 pages yielding four provisions did not
extract; one yielding eleven hundred units with an empty error list did.

Clause numbers are read in one of two schemes, decided once per document:
digits only (``7.1.5``), or with a part letter (``A2.2.5``, ``D24.2.67``) as
the Arm architecture manuals number them. In a lettered document only lettered
numbers of at least two levels open a section — ``A64`` and ``T32`` are names
— and the running header each page repeats (``A2.2 Armv8-A …``, set with one
space where the heading has several) is page furniture, not a heading. A
digits-only document is extracted exactly as before; the scheme only widens
what a lettered one can be read as.

``human_validation_status`` starts at ``NOT_REVIEWED`` and stays there until
somebody says otherwise — extraction is not review, and the field does not
pretend it is.

Seeing what is in the store
===========================

``/standard list`` reports every ingested document *with its parameters*, and
marks the one currently bound:

.. code-block:: text

   Available standards:

     ACME-1234.5 2019-R2023  <- bound
       origin      LICENSED_STANDARD  ·  the document itself is retained
       extractor   hybrid-span-canonical-v1
       corpus      13878 units  ·  899 requirements  ·  95 recommendations  ·  355 pages
       validation  NOT_REVIEWED

     NIST-RS274NGC NISTIR6556
       origin      PUBLIC  ·  extraction only
       extractor   poppler-structure-v2
       corpus      1109 units  ·  42 requirements  ·  54 recommendations  ·  121 pages
       validation  NOT_REVIEWED

That is the answer to *what do I have, and what is each one*. ``origin`` says
whether a document may be embedded on a shared host and whether it may travel
in an image; ``the document itself is retained`` says the original PDF is in
the store beside the extraction; the corpus counts say whether the extraction
worked.

``/standard status`` reports the same and much more — index states, cross
references, retrieval fingerprints, candidates — but **only for the bound
document**. The binding is shared by every session on the machine, so reading
``list`` is how you look at an unbound one without changing what everyone else
is answering from.

A public standard
=================

``--origin PUBLIC`` is the whole difference, and it buys two things:

* the corpus may be embedded on a configured GPU host without
  ``--allow-offload``, because there is nothing to keep off it;
* the document may travel in a ``--profile public`` container image
  (:ref:`image-profiles`).

Nothing else changes. The extraction, the provisions, the guards and the
citations are identical — a public document is not a lesser one, it is one
without a restriction.

.. code-block:: text

   /standard ingest ~/NISTIR6556.pdf --id NIST-RS274NGC --revision NISTIR6556 --origin PUBLIC

A document with no role labels
==============================

Most documents do not label their provisions the way a VITA-style standard
does, and they do not need to. Where there are no labels, the kind is read from
the **modal verb**: *must* makes a requirement, *should* a recommendation,
*may* a permission, and prose with no modal verb stays informative.

Ingesting NISTIR 6556, which carries no taxonomy of its own, gives:

.. code-block:: text

   1109 units    42 REQUIREMENT    54 RECOMMENDATION    55 MAY
                 588 UNKNOWN      227 PAGE_FURNITURE   150 FRONT_MATTER

That is a working corpus. ``UNKNOWN`` is not a failure — it is prose that
obliges nobody, which in a specification is most of it.

Datasheets and other non-prose documents
========================================

A datasheet ingests, and the result has a different shape, because a datasheet
**states facts rather than obligations**. Expect few requirement units and a
great many tables. Three things follow.

**The force machinery simply does not fire.** There are no modalities to
strengthen or contradict, so the guards that compare a claim against the force
of its provision have nothing to say. They do not misfire; they stand down.

**Tables carry requirement force**, through the ``unlabelled`` ceiling. For a
datasheet that is usually the right reading: an absolute-maximum rating or a
register's reset value binds as firmly as any *shall*.

**The register maps need approving before they can be cited.** A table becomes
a *candidate* structure at ingestion, and ``standard.get_structure`` serves
only structures a person has approved:

.. code-block:: text

   /standard candidates <id> <revision>
   /standard approve-bitfield <id> <revision> <candidate-id> <verdict> <role,role,...>
   /standard build-structure <id> <revision>

``build-structure`` promotes the approved candidates and only those. An
unapproved one has no route to an answer however it is asked for — which is
the point: a bit layout SPEAR guessed at is exactly the kind of thing that
looks authoritative and is not.

.. important::

   A PDF has no other way in. ``.pdf`` is not among the extensions a corpus
   indexes, so ``spear-index`` will not read one. If all you want is retrieval
   over a document — no citations, no force, no approval step — convert it to
   text or Markdown first and register *that* as a corpus. That is what the
   ``posix-api`` corpus is: man pages as ``.txt``.

   Use the normative store when you want claims about the document to be
   **citable and checked**. Use a corpus when you want to find things in it.

The rest of the operator commands
=================================

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Command
     - What it does
   * - ``/standard list``
     - every ingested document with the parameters it was ingested with
   * - ``/standard status``
     - what is bound right now
   * - ``/standard use <id> <revision>``
     - bind one
   * - ``/standard unbind``
     - bind nothing
   * - ``/standard verify [<id> <revision>]``
     - check the corpus and every index in full, then the contents table
   * - ``/standard rebuild [<id> <revision>] [--allow-offload]``
     - rebuild the indexes without re-extracting
   * - ``/standard retrieval [<id> <revision>] [<mode>] [--completion <n>]``
     - show or set how a document is searched
   * - ``/standard candidates [<id> <revision>]``
     - alternative extractions held beside the promoted one
   * - ``/standard promote-candidate <id> <revision> <candidate>``
     - make a candidate the corpus that answers
   * - ``/standard approve <id> <revision> [<reviewer>]``
     - record human validation
   * - ``/standard build-structure [<id> <revision>]``
     - build the structure registry ``standard.get_structure`` reads

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
