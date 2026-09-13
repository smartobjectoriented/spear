STD1 local hybrid standard retrieval
====================================

Normative source of truth
-------------------------

A technical standard is not model memory. SPEAR treats the local canonical
corpus, an operator-pinned standard/revision, retrieval, and stable citations as
the normative chain of evidence. Fine-tuning may later teach application
patterns, but it must not replace inference-time retrieval of normative sources.

Private canonical store
-----------------------

The default store is ``<SPEAR_STATE_DIR>/standards``. It is private runtime
state and is gitignored. Each ``<standard-id>/<revision>`` contains:

* ``manifest.json`` and ``corpus/<source-id>.json``: authoritative state;
* ``source/metadata.json``: logical filename, PDF SHA-256, page count, and
  retention status;
* ``source/original.pdf`` only when the operator explicitly uses
  ``--retain-pdf``;
* ``indexes/lexical``: derived, replaceable BM25 state and manifest;
* ``indexes/vector``: derived local normalized vectors keyed by canonical
  ``source_id`` and a model/configuration manifest;
* ``indexes/crossrefs``: derived conservative one-hop section resolution;
* ``indexes/embedding-cache``: disposable content-addressed vectors.

Deleting every index leaves the manifest and canonical corpus intact. All
retrieval artifacts can be rebuilt from those authoritative files.

Writes are atomic and private. Identifiers are validated, traversal and symlink
escapes are rejected, and a revision label cannot be silently overwritten by a
different PDF or corpus. Canonical corpus use does not depend on the original
external path remaining available.

Canonical units and stable IDs
------------------------------

``StandardDocumentUnit`` records schema version, standard/revision, physical
page, detected section and heading path, original text, content type, modality,
parent source, literal cross-references, PDF/content hashes, extractor version,
and review warnings. Content types cover requirements, recommendations,
definitions, informative text, examples, tables, figures, formulae, and
unknown material. Modalities preserve explicit SHALL, SHOULD, MAY, and MUST.

The source ID is ``std-`` plus the first 128 bits of a canonical SHA-256 over
standard ID, revision, page, detected section, canonical unit position, and the
unit's complete content SHA-256. It contains no path or source text and never
depends on an index row. Exact repeated ingestion therefore preserves IDs;
material canonical content changes change identity.

Ingestion
---------

Ingestion is an operator-only local path using Poppler ``pdfinfo`` and
``pdftotext -layout``. It preserves physical page boundaries and extracted
ordering without OCR. Numbered headings and blank-line paragraphs drive
conservative semantic segmentation; canonical units are not fixed-token
chunks. Uncertain or empty extraction is warned, and a document with no text
fails with an explicit statement that OCR is disabled.

Tables and figures retain their extracted text, page, section, and hashes but
are marked ``needs_structured_review``. STD0 never derives bit offsets or
claims lossless visual interpretation. Explicit references such as ``see
5.3.2`` are stored literally. STD1 resolves only conservative explicit section
forms. Missing and multiply anchored targets remain unresolved or ambiguous;
the resolver never guesses.

The ingestion manifest records PDF identity, logical filename, timestamp,
extractor/schema versions, page/unit/section and classification counts,
warnings/errors, validation status, source origin, retention choice, and the
canonical corpus fingerprint. The timestamp is audit metadata; source IDs and
the corpus fingerprint depend only on canonical semantic state.

Pinned binding and tools
------------------------

``StandardBinding`` contains standard ID, revision, PDF SHA-256, canonical
corpus fingerprint, lexical/vector/cross-reference fingerprints, an overall
retrieval fingerprint, extractor version, corpus schema version, optional bind
timestamp, and data origin. One binding is supported per task/session. The
operator selects it; a tool argument may repeat the ID or revision only when it
exactly equals the binding.

The operator's choice arms the binding; it does not apply it to every prompt.
``standard_scope.engages()`` decides per turn, and the turn is announced when
it binds.  A turn engages the standard when it names it — the terms are derived
from the binding's own identifier and revision, not from a maintained list — or
carries a structure identifier, or uses the concrete vocabulary of a
data-format standard (bit, octet, offset, field, width, encoding…).  The
generic words *standard*, *revision*, *rule* and *section* are excluded on
purpose: bound to every prompt, the frame turned "get the RS274/NGC PDF so we
can inject it as a standard" into eleven turns explaining that the system is
bound to ANSI-VITA-49.2.  Erring towards binding remains the safe direction —
an unnecessary binding costs a system rule and two tool schemas, a missing one
costs grounding on a licensed source.

The main agent receives these read-only tools only while bound:

* ``standard.search`` defaults to bounded local hybrid retrieval and returns
  source identity/metadata, snippet, lexical/vector/fused ranks, match reason,
  and bounded direct cross-reference expansion;
* ``standard.fetch`` resolves one validated source ID and returns the full
  canonical unit, optional parent, literal cross-references, and citation;
* ``standard.cite`` renders the citation from stored metadata, for example
  ``[TEST-STD TEST-1 §3.1, p.1, source std-...]``.

Large responses use the existing ``ResultStore``. Search, fetch, and cite have
no path, shell, network, ingestion, replacement, or revision-switch capability.
Explorer, Reviewer, and Planning roles do not receive them. Ingestion,
binding, unbinding, and rebuild remain slash-command operator actions and are
not registered model tools.

Context, sessions, and resume
-----------------------------

When bound, ``TaskController`` adds only the pinned identity and concise
behavioral guidance to the protected context: retrieve normative sources, cite
conclusions, distinguish modalities and informative/examples, do not invent
values, and report insufficient/conflicting evidence. Standard text enters
model context only through tools. Ordinary unbound tasks retain their prior
context and tool view.

``SessionSnapshot`` persists the binding and source IDs observed from grounded
tool results. Standard session events contain hashes, IDs, counts, citations,
query hashes, and ``ResultStore`` references—not long normative text.

Resume compares the persisted binding with current canonical state:

* revision, PDF, or corpus mismatch blocks continuation and never rebinds;
* any retrieval-only change is allowed because indexes are derived. The binding
  is refreshed, ``standard_retrieval_changed`` is journaled with component
  fingerprints, and cached retrieval tied to the old retrieval fingerprint is
  invalidated. ``standard_index_changed`` remains for lexical compatibility.

Local vector retrieval and fusion
---------------------------------

The standards adapter reuses SPEAR's supported SentenceTransformer model
catalog and asymmetric query/document prefixes, but not ordinary RAG's Chroma
collections or optional SSH offload. Operators set
``SPEAR_STANDARD_EMBED_MODEL`` and
``SPEAR_STANDARD_EMBED_REVISION``. The model must already be a local path or in
the local cache; loading uses ``local_files_only`` and offline mode. Normative
text and queries never reach a remote embedding provider.

Retrieval text is deterministic ``section + heading path + canonical text``;
it excludes source IDs, hashes, timestamps, and paths. Vectors are L2-normalized
against canonical source IDs. Model/revision, dimension, normalization, corpus
hash, configuration hash, count, format, and fingerprint are validated. A
missing, corrupt, or mismatched vector index yields explicit
``lexical_fallback`` mode, never a claim that hybrid ran.

Hybrid search uses reciprocal-rank fusion (RRF, constant 60) over BM25 and
cosine candidates, with source-ID tie breaking. Exact section queries are
structurally promoted so ``section 5.3.2`` is not degraded by similarity.
Duplicate hits collapse to one canonical source. A primary hit may add at most
two direct cross-reference targets and never recurses. Parent and bounded
same-section neighbors retain their own source IDs.

Citation rendering is independent of retrieval mode. The same ``source_id``
produces identical citation metadata whether found lexically, semantically,
through RRF, or through cross-reference expansion.

Retrieval evaluation
--------------------

``standard_retrieval_eval`` evaluates operator/human labels without an LLM
judge. Items bind a query/category to expected canonical IDs or a section.
Lexical, vector, and hybrid modes report Recall@1/3/5, MRR, exact-section hit
rate, median/p95 latency, and per-category metrics. Reports have machine JSON
and concise text forms. Committed synthetic fixtures cover structural,
exact-term, acronym, conceptual, requirement, definition, cross-reference, and
no-answer queries. Licensed evaluation sets belong under private state, not Git.

Training and licensing governance
---------------------------------

Training episodes retain binding identity, corpus hash, revision, and grounded
source IDs. They do not duplicate long normative passages in metadata. Bounded
tool excerpts remain in a turn only because the model actually saw them.

``LICENSED_STANDARD`` is excluded by the default training governance profile
with ``licensed_standard_review_required``. Test fixtures remain
``TEST_FIXTURE`` and excluded. Raw local episodes are retained for audit, while
automatic SFT/preference derivation is skipped for standard-bound episodes.
Explicit future governance may allow reviewed local use; task success alone
never approves trainer export. No standard content is uploaded, embedded by an
external provider, published, or routed through ordinary project RAG.

Operator flow
-------------

::

  /standard ingest ./authorized.pdf --id ANSI-VITA-49.2 --revision 2017-R2024
  /standard use ANSI-VITA-49.2 2017-R2024
  /standard status
  /standard rebuild
  /standard unbind

Classification and where the embedding runs
-------------------------------------------

``--origin LICENSED_STANDARD|PUBLIC`` records what the corpus is.  It defaults
to ``LICENSED_STANDARD``, and it is not decoration: ``training_data`` reads it
for ``standard_export_prohibited`` and for training governance.  Passing
``--origin`` over an already-ingested revision reclassifies it — re-ingesting an
identical PDF otherwise returns the stored manifest untouched, so the flag used
to be silently discarded.

The classification decides where the vector index may be built.  A ``PUBLIC``
corpus may be embedded on the GPU host named by ``--standard-embed-remote``
(seconds, against minutes of saturated local CPU); a licensed one may not,
because indexing sends the *standard's text* to a host shared under a common
login.  Answering sends only the user's question, so ``embed_query`` is local
whatever the origin.

``--allow-offload`` is the operator overriding that for one command.  It is a
flag and never a stored property: relabelling the corpus ``PUBLIC`` would buy
the same offload *and* claim the text is exportable.  When a licensed corpus
does go, the fact is appended beside the index and reported by
``/standard status``, because the question it answers has to survive the next
rebuild.

Where the model ran is part of the embedding config and therefore of the index
fingerprint.  Measured on NISTIR 6556 with ``BAAI/bge-m3`` at one pinned
revision, the GPU host's vectors differ from the local ones by up to 2.8e-4 per
component — reduced precision, not float32 rounding.  The pinned revision is
checked on both ends before anything is sent, because the remote worker takes a
model name and no revision.

``/standard list`` enumerates locally available ID/revision pairs. Raw PDF
retention requires the explicit ``--retain-pdf`` ingestion option.

Re-extracting an already-ingested revision goes through a candidate rather than
over the active corpus::

  /standard ingest ./authorized.pdf --id ANSI-VITA-49.2 --revision 2017-R2024 --candidate --layout
  /standard candidates ANSI-VITA-49.2 2017-R2024
  /standard promote-candidate ANSI-VITA-49.2 2017-R2024 cand-<id>
  /standard rebuild ANSI-VITA-49.2 2017-R2024
  /standard use ANSI-VITA-49.2 2017-R2024

Promotion archives the previous corpus as a recoverable generation, drops the
derived indexes, and clears the binding so existing sessions fail closed.

Checking extraction against the document itself
-----------------------------------------------

A standard's contents table names each clause and the page it opens on. That is
the one statement about structure that does not come from the extractor, so it is
what an operator should trust over any internal verdict::

  /standard verify ANSI-VITA-49.2 2017-R2024

The check measures the offset between printed and extracted page numbers from the
clauses that are not in dispute, then reports how many headings land on the page
the contents table gives, naming any that do not. The same evidence resolves a
contested clause number during extraction: where two lines claim one number, the
one on the page the contents table names wins.


Human review and approval
-------------------------

An automated extraction review is evidence, not approval. The operator walks the
private review file row by row and records a human verdict beside the automated
one::

  python -m standard_review --standard ANSI-VITA-49.2 --revision 2017-R2024

Keys are ``p`` PASS, ``w`` ACCEPTABLE_WARNING, ``f`` FAIL, ``n``
NEEDS_FOLLOWUP, ``s`` skip, ``u`` clear a verdict, ``b`` back, ``q`` save and
quit, ``?`` help. Each row shows its page, section, content and layout kind,
stratum, warnings, the automated verdict, and a bounded excerpt with one unit of
surrounding context, all read through ``StandardStore`` -- never through a path
recorded in the review file. Normative text is rendered to the terminal only; the
file stores identifiers, verdicts and timestamps.

The tool resumes at the first row still awaiting a verdict, so the same command
continues an interrupted review. ``--only-unreviewed``, ``--only-warning``,
``--stratum <name>`` narrow the worklist and ``--summary`` prints counts without
reviewing. The reviewer identity comes from ``--reviewer`` or ``$SPEAR_OPERATOR``
once per session, never per row.

A review file is pinned to the corpus fingerprint it was written against. If the
active corpus has been replaced, loading and saving both fail closed rather than
apply an old review to a new generation.

Completing the review does not approve the corpus. Approval is a separate
operator action, refused unless every row is human-reviewed with no ``FAIL`` and
no ``NEEDS_FOLLOWUP``::

  /standard approve ANSI-VITA-49.2 2017-R2024

It sets ``human_validation_status`` to ``APPROVED`` in the canonical manifest and
writes reviewer, timestamp and summary to a private ``approval.json`` beside it.
Corpus identity is untouched. Neither review nor approval is exposed as a model
tool.

STD1 limitations and next phase
-------------------------------

STD1 does not add a learned reranker, recursive cross-reference traversal,
semantic table/figure interpretation, OCR, ``standard.get_structure``,
``standard.validate``, compliance checking, normative task generation, or
fine-tuning. It does not treat vector similarity as normative truth.

STD2 may add reviewed structured tables/bitfields/packets,
deterministic masks and validators, ``standard.validate``, and narrow
``VerificationPolicy`` integration. Future SFT should train tasks supplied with
normative sources to produce grounded outputs and citations—not memorize the
standard—and retrieval should remain available at inference time.
