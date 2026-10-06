.. _glossary:

Glossary
########

.. glossary::

   Request class
      What a request is, read from its own words and from whether a standard
      is engaged: ``GENERAL``, ``IMPLEMENTATION``, ``NORMATIVE`` or ``MIXED``.
      The class decides which path runs it.

   MIXED
      A request for a change that must satisfy the bound standard. It runs as
      a normative pre-pass, a constraint packet, the coding core, and a final
      adjudication on authoritative evidence.

   Coding core
      The standalone tool-calling loop that runs implementation requests, with
      exactly six tools: ``read_file``, ``search_files``, ``patch``,
      ``write_file``, ``delete_file`` and ``terminal``. It decides what to try
      next; it neither authorizes its own calls nor judges its own result.

   SpearHost
      The control plane the coding core runs against: every read, write and
      command it asks for is authorized, resolved inside the workspace,
      confined, checkpointed and audited here.

   Control plane
      Everything that decides what a call may do and records what it did:
      SpearHost in front of the coding core, the command and capability
      policies, the workspace, and the audit trail.

   Normative runtime
      The ``AgentRuntime`` loop with the standard's tools, provision records
      and answer guards, which answers questions about a bound standard.

   Implementation evidence
      The structured record of what a turn's tools did and what its checks
      showed, and the verdict computed from it: ``VERIFIED``,
      ``UNVERIFIED`` or ``NO_CHANGE``.

   Final source state
      The tree as it stands after a turn's last change. Only a check that ran
      on it — in the final source epoch — counts for the verdict.

   Source epoch
      The stretch of a turn between two changes to delivered source. A check
      proves the epoch it ran in, and no later one.

   Normative evidence
      Evidence that a constraint taken from the bound standard holds in the
      final source: a source predicate, or a conformance check bound to the
      provision. A model's reading is never normative evidence.

   Constraint packet
      The ``NormativeConstraintSet`` a MIXED request is implemented and judged
      against: one constraint per applicable provision, each with its force,
      condition, counts, values, identifiers and provision identity.

   Structural coverage
      The closure of the cited provisions over the groups the document itself
      records — the same numbered list, the same table, a provision named
      outright — bounded, and incomplete when a bound is reached.

   Applicability
      Whether a covered provision applies to this change: ``APPLICABLE``,
      ``NOT_APPLICABLE`` or ``UNRESOLVED``, decided only from the request's
      words or the project's own declarations.

   Conformance check
      A project check bound to a provision in ``normative_checks``. Only a
      bound check carries normative weight; an ordinary passing test does not.

   Source predicate
      A deterministic reading of a constraint off the final source —
      ``EXACT_COUNT``, ``VALUE_EQUALS``, ``CONDITIONAL_VALUE`` — that decides
      only when nothing has to be interpreted.

   Corpus
      A source tree registered for retrieval, declared in ``projects.json``
      and managed with ``spear-corpus add/list/rm``. A corpus is a *scope*,
      not a copy: the tree stays where it is, and only its index lives in
      SPEAR's state directory.

   Collection
      The Chroma collection holding one corpus's embeddings, named
      ``adhoc_<md5(realpath)[:8]>`` unless the declaration names one
      explicitly. Deriving the name from the absolute path is what lets a
      corpus be re-indexed in place without a registry entry going stale;
      naming it explicitly is what lets an index travel with the registry, as
      in the container.

   Federation
      A corpus that attaches others to the same session, through its
      ``corpora`` key. A question about a build system then reaches its
      component trees without switching session. Federation is a reading
      convenience, not an indexing one: each federated corpus keeps its own
      collection.

   File cap
      The 60 000-file ceiling on a single indexing walk
      (``SPEAR_INDEX_MAX_FILES``). Reaching it is an **error**, not a
      warning: a truncated index is an arbitrary prefix of a tree, and it
      answers confidently from whichever part the walk happened to reach
      first. The cap is a runaway guard; ``--exclude`` is the tool for scope.

   Harness
      Everything between the model's request to act and the act itself: tool
      selection, authorization, and the confinement layers. The model
      is a component of SPEAR; the harness is what makes running it on a real
      machine defensible.

   Fail-closed
      The rule the whole harness is built on: a mechanism that cannot be
      honoured is an error, never a silent downgrade to a less confined
      execution. A model that asks for something the harness cannot confine
      gets a refusal, not a shortcut.

   Tool floor
      The set of tools a role keeps whatever the objective looks like, so
      that a turn is never left without a way to read, search, edit and run.
      The floor guarantees a way to work; it does not grant one the request
      refused — a turn told not to edit keeps reading and searching, and
      keeps no write tool.

   Answer scope
      What *this* turn asked about, derived from the message rather than
      stored. History may say what a message's words refer to; it does not
      add a second piece of work to a question that asked for one thing. A
      turn that asks about a bound document is not offered the tools that
      reach a working tree.

   Standard binding
      The identification of the normative document a session answers
      against: a standard identifier and a revision, pinned to the
      fingerprint of the corpus extracted from it. Every normative tool
      refuses an argument that names anything else.

   Source id
      A dereferenceable handle to one unit of a bound standard, of the form
      ``std-<32 hex>``. It is issued by a retrieval in the current turn and
      is fetchable only in that turn: a handle printed earlier is refused as
      stale, without the corpus being consulted, so that asking twice cannot
      reveal which units exist.

   Provision key
      The printed identity of a requirement in a standard — the number a
      human would cite. It is stable across extractions, which a source id is
      not: a source id binds to one extraction of one document, a provision
      key names the thing the document says.

   Guard
      A check applied to an answer before it is shown, each reporting a named
      finding rather than editing the text: an unsupported identifier, a
      strengthened modality, a citation that could mean two provisions. A
      guard that cannot decide says so; none of them quietly rewrites.

   Retained PDF
      The licensed source document kept beside the units extracted from it,
      so a later re-extraction can be checked against the original. Retention
      is derived from the store's path — refused inside a repository, allowed
      in private state — because a repository is cloned and a licensed
      document should not be.

   Attached corpus
      A corpus declared ``shared``, attached to every session rather than
      chosen per project. The C library reference is the usual case: it is
      cross-cutting, and a question that needs it rarely announces that it
      does.
