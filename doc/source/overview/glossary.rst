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

   AgentRuntime
      The provider-neutral execution loop for one task: it owns model and
      tool iteration and the task's context state, and knows nothing of
      terminal rendering, project selection or persistence
      (:doc:`/overview/architecture`).

   WorkingState
      The state of one task, built only from events the harness and the tool
      runtime emit — never from the model's account of itself — and replayed
      from that log: the objective, its budgets, the plan, failures and
      verification. Context assembly reads the current task's facts from it
      (:doc:`/overview/architecture`).

   Evidence plane
      The part of SPEAR that records what each tool call actually did and
      computes the implementation verdict from that record alone
      (``client/evidence/``; :doc:`/reasoning/evidence`).

   Canonical evidence
      The structured record of one tool call: what it changed, which command
      it ran and how that ended, whether it was refused. It is taken from the
      call's own fields, never from a rendering of them, and is what the
      verdict is decided from.

   Implementation evidence
      The structured record of what a turn's tools did and what its checks
      showed, and the verdict computed from it: ``VERIFIED``,
      ``UNVERIFIED`` or ``NO_CHANGE``. ``VERIFIED`` requires a build or test
      that ran and passed after the last change to the source; a command sent
      to the background, or a Makefile that only prints its help, is no check.

   Declared and probed commands
      A project's build and test commands, either declared in
      ``projects.json`` (``build_commands``, ``test_commands``) or probed from
      the files in the tree. Both may verify a change; a declared kind wins,
      and a probe only fills a kind the project left undeclared.

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

   Composite verdict
      The one-line verdict of a MIXED turn, combining the implementation
      verdict with the compliance status: ``VERIFIED + COMPLIANT``,
      ``NOT COMPLIANT``, ``COMPLIANCE NOT DEMONSTRATED`` and the like.
      ``NORMATIVE AMBIGUITY`` means an applicable requirement is ambiguous in
      the standard as stored; ``EVIDENCE CONFLICT`` means two authoritative
      providers disagree about the same final source, and only a person can
      say which is wrong (:doc:`/reasoning/evidence`).

   Compliance not demonstrated
      The normative status of a MIXED change when the bound standard's own
      evidence does not establish that each applicable constraint holds.
      Compliance is reported only from normative evidence — never from a
      passing build — so ``NOT_DEMONSTRATED`` is the honest default, not a
      finding of non-compliance.

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

   Workspace
      What a turn starts from: a registered project, or an unregistered tree on
      its own. It decides which rules, skills, knowledge, metadata and
      capabilities the turn may be given, and nothing scoped to another
      workspace reaches it.

   Context selection
      The deterministic choice of what a turn is shown, made per workspace and
      per request class or MIXED pass, never by resemblance to the request.
      Every eligible skill is offered, up to a bound per turn; every decision
      is audited and none is shown to the model (:ref:`context_selection`).

   Workspace knowledge
      Typed, persistent facts about one workspace, each with its provenance,
      its verification and a lifecycle — proposed, active, stale, revoked —
      kept in ``knowledge.sqlite3`` under the state directory. ``/remember``
      and ``/knowledge`` record it as the operator's; what a model offers stays
      a proposal until the operator accepts it. It describes; it never grants a
      permission or outranks the request, the rules or the configuration.

   External capability
      An operation of an external provider — a tool of an MCP server over
      stdio — registered in ``capabilities.json`` for the workspaces and
      request classes it applies to. It is reached only through the capability
      gateway, which applies the read/write policy; a small family is listed
      whole, a large one as an index read with ``spear-capability``
      (:ref:`capabilities`).

   Target policy
      The rule that a refusal attaches to the file, not to the tool: generated
      output, snapshot copies and git metadata refused to ``write_file`` and
      ``patch`` are refused to every other mutation as well, ``delete_file``
      and shell redirection included.

   Refusal breaker
      The guard that stops a turn which keeps asking for an operation already
      refused — five identical refusals by default
      (``SPEAR_REFUSAL_REPEATS``).

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

   Permission mode
      What a session may do to files and the network: ``SAFE`` (the
      default, read-only, no network), ``ASK`` (each mutation and network use
      confirmed) or ``AUTO`` (no confirmation). ``--no-network`` removes the
      network from any of them (:doc:`/harness/security_model`).

   State directory
      Where a session accumulates — history, audit trail, checkpoints,
      learned rules, workspace knowledge, the normative store:
      ``SPEAR_STATE_DIR``, or ``client/`` when it is not set
      (:doc:`/operations/directory_layout`).

   Host command
      A command the coding core types in its terminal that SPEAR answers
      itself and never hands to a shell: ``spear-capability`` for external
      capabilities, ``spear-knowledge`` for workspace knowledge. It is
      recognised only where a shell would run it, and must stand alone on one
      line (:ref:`capabilities`, :ref:`knowledge`).

   Fail-closed
      The rule the whole harness is built on: a mechanism that cannot be
      honoured is an error, never a silent downgrade to a less confined
      execution. A model that asks for something the harness cannot confine
      gets a refusal, not a shortcut.

   Normative store
      Where ingested standards are kept, one directory per standard and
      revision: ``standards/`` under the state directory, or wherever
      ``SPEAR_STANDARDS_ROOT`` points (:doc:`/reasoning/standards`).

   Provision
      One statement of a bound standard — a Rule, a Permission, an
      Observation, … — recorded with its kind, ordinal, section, page and
      source id. Two provisions that print the same label are two records
      (:doc:`/reasoning/standards`).

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
      reveal which units exist (:doc:`/reasoning/standards`).

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
      document should not be (:doc:`/reasoning/standards`).

   Shared corpus
      A corpus declared ``"shared": true``, attached to every session rather
      than chosen per project; ``--without`` drops it for one session. A C
      library reference is the usual case: it is cross-cutting, and a question
      that needs it rarely announces that it does (:ref:`retrieval`).

   Ad-hoc corpus
      A tree that was never registered: the current directory, indexed under
      a collection name derived from its path. It behaves like any other
      corpus, inherits nothing from a registered project, and is the normal
      way to ask about a tree you are passing through (:ref:`projects`).

   Project family
      A name a registered project declares in its ``families`` key. Rules,
      skills and external capabilities scoped ``family <name>`` reach every
      project declaring it. Membership is exactly what the key says, never
      inferred from the tree (:ref:`projects`).

   Orientation map
      A short prose map of one corpus — where things live, which build
      command to run, what never to edit — kept in
      ``rules.d/corpora/<name>.md`` or, winning when present, in an in-tree
      ``.edgem-rules.md``, and shown only to that corpus's turns
      (:ref:`orientation_maps`).

   Image profile
      What a container image may carry, chosen per build: ``public`` carries
      only documents that declare themselves public and only the files the
      repository tracks; ``private`` carries everything the building host
      has (:doc:`/start/container`).
