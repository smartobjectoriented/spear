.. _directory_layout:

================
Directory layout
================

The code lives in the checkout — ``~/spear`` in the examples; the tree can sit
anywhere.  Two directories outside it matter as well: the state directory,
which may sit outside the checkout (`Persistent state`_), and, on an inference
host, the runtime root that ``scripts/bootstrap-runtime.sh`` populates
(``~/spear-runtime`` in the examples, :doc:`/model/runtime_bootstrap`).  The
checkout itself mixes three quite different kinds of content — application
code, multi-gigabyte model weights, and persistent state — so it is worth
knowing which is which before running a ``find`` over it.

.. figure:: /img/SPEAR-Layout.drawio.png
   :width: 100%
   :alt: SPEAR checkout directory layout

   The checkout at a glance: application, models, state.

Top level
=========

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Path
     - Content
   * - ``client/``
     - The application **and** its Python virtualenv.  ``bin/python`` here is
       the interpreter every command in this documentation uses.
   * - ``server/``
     - The generic inference server component (see `The server component`_).
   * - ``scripts/``
     - Release, configuration and image scripts (``spear-configure``,
       ``spear-image``, ``spear-consolidate``, ``spearversion.sh``) and the
       inference host's ``bootstrap-runtime.sh``.  ``scripts/docker/`` builds
       the image (``build.sh``, ``push.sh`` and the staging helpers) and runs
       it (``spear-docker.sh``).
   * - ``docker/``
     - The container image: ``Dockerfile``, the entrypoint and the
       relative-path corpus registry.  It sits at the root rather than under
       ``client/`` because it draws on both (:doc:`/start/container`).
   * - ``models/``
     - Model weights, when a model is served locally.  Not tracked.
   * - ``llama.cpp-next/``
     - The ``llama.cpp`` checkout and build providing ``llama-server``, when
       built locally.  Not tracked.
   * - ``qwen3-finetune/``
     - The fine-tuning working area: LoRA trainers, GPU-host scripts, corpus
       builders and the load preflight.  The *governed* training path lives in
       the application instead (:doc:`/model/training`).
   * - ``corpora/``
     - Small trees indexed as their own corpora (``musl-headers``,
       ``posix-api``) — they answer questions no product tree contains.  Only
       ``refresh-musl-headers.sh``, which snapshots a toolchain's headers into
       ``musl-headers/``, is tracked; the trees themselves are not.
   * - ``env.sh``
     - ``. ./env.sh`` from the tree's root puts ``scripts/`` on ``PATH``
       (``spear-configure``, ``spear-image``).
   * - ``doc/``
     - This documentation (:doc:`/contributing/documentation`).
   * - ``.github/workflows/docs.yml``
     - The only CI workflow: it builds this documentation and publishes it.

.. note::

   ``client/`` being both the application directory and the virtualenv root
   is why ``bin/``, ``lib/`` and ``include/`` sit next to the packages.
   It is unusual but deliberate: the app and its exact dependency set travel
   together.

The application
===============

The code is one package per concern, imported from ``client/`` as their common
root (``from evidence import completion``).  ``rag_chat`` and the other
commands are run as files (``client/bin/python client/cli/rag_chat.py``); the
launchers and the image do that for you.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Package
     - Role
   * - ``cli/``
     - The commands.  The chat client's entry point is ``rag_chat.py``
       (startup, argument handling and the REPL loop); the rest of the client
       is ``chat_settings.py`` (session flags and the paths derived from
       them), ``corpus_registry.py`` (``projects.json`` and ``/corpus``),
       ``session_workspace.py`` (the bound project, execution mode, command
       policy and the file and command primitives), ``corpus_search.py``
       (retrieval), ``tool_handlers.py`` and ``tool_routing.py`` (the tool
       handlers, the registry and the coding core's host ports),
       ``turn_context.py`` (rules and per-turn context selection),
       ``knowledge_commands.py``, ``standard_session.py``,
       ``session_history.py``, ``project_checks.py``, ``model_io.py``,
       ``reply_edits.py``, ``operator_input.py``, ``startup_banner.py`` and
       ``terminal_ui.py``.  Beside it: ``backend_select.py`` (the startup
       backend picker), ``machine_config.py`` (``spear-configure``) and
       ``finetune_commands.py``.
   * - ``agent/``
     - The coding core: its loop, tool dispatch, tools and prompt.  It imports
       nothing of SPEAR and talks to a ``Host`` interface
       (:doc:`/reasoning/implementation`).
   * - ``harness/``
     - The execution harness: ``control_plane.py`` (``SpearHost``),
       the tool-execution substrate (``tool_primitives.py``,
       ``command_policy.py``, ``workspace.py``, ``sandbox.py``,
       ``resource_control.py``, and ``tool_runtime.py`` for the runner and
       the audit log; none imports ``rag_chat``, so they can be tested in
       isolation), ``tool_registry.py`` / ``tool_router.py`` (declarative
       exposure and structured lifecycle), ``target_policy.py``,
       ``refusal_breaker.py``, ``checkpoint.py``, and the external
       capabilities (``capabilities.py``, ``capability_gateway.py``,
       ``mcp_provider.py``).
   * - ``runtime/``
     - The provider-neutral model/tool loop of the normative and general
       runtime (``agent_runtime.py``) and the task it runs
       (``agent_context.py``); the rest of ``AgentRuntime`` comes from one
       mixin module each: the coding-core turn wrapper
       (``agent_core_turn.py``), the model call (``agent_model_turn.py``), the
       end of a turn (``agent_finalization.py``) and what it records
       (``agent_recording.py``), with the loop's notes and demands in
       ``agent_notes.py`` and its change verification in
       ``agent_verification.py``.  Also the UI-free task orchestration
       (``task_controller.py``), grounded task truth and compaction (``working_state.py``, ``compaction.py``), sessions,
       budgets and the runtime trace.
   * - ``evidence/``
     - The evidence plane: canonical evidence, source epochs and the
       implementation verdict (``completion.py``), the project's own build and
       test commands (``project_build.py``), and the evidence guards.
   * - ``context/``
     - What a turn is given: deterministic context selection, the workspace
       context, workspace knowledge (``workspace_knowledge.py``, with
       ``knowledge_migration.py`` for legacy Markdown memories), skills, and
       the request's class and scope (``answer_scope.py``,
       ``request_scope.py``).
   * - ``normative/``
     - The MIXED pipeline (``mixed_orchestration.py``,
       :doc:`/reasoning/mixed`), the constraint packet, structural coverage and
       applicability, the source predicates and conformance-check bindings.
   * - ``standard/``
     - The normative store: ingestion, structure, semantic records, retrieval,
       review and the standard's tools.
   * - ``models/``
     - Provider-neutral model turns (``ModelTurn``) and raw OpenAI-compatible
       turns (``RawTurn``) for the coding core, in ``model_backend.py``;
       operator-only control of the serving process and of a training host.
   * - ``retrieval/``
     - Embedding and corpus ingestion into Chroma (``index_corpus.py``,
       ``index_dir.py``).
   * - ``training/``
     - The fine-tuning subsystem: capture, curation, governance, readiness,
       frozen bundles and the operator control plane, none of it
       model-visible (:doc:`/model/training`); ``python -m training`` is its
       command line.
   * - ``tests/``
     - The test suite; see :doc:`/operations/testing`.
   * - ``eval/``
     - Evaluation harnesses that call no model, the retrieval one first
       (``run_eval.py``): they measure what lands in the prompt.
   * - ``benchmarks/``
     - The real-model benchmark runner and its ablations
       (``python -m benchmarks.runner``).
   * - ``deploy/``
     - Installing the application on a machine: ``preflight.sh`` (a survey
       that changes nothing), ``install.sh`` (the virtualenv, idempotent),
       ``enable-sandbox.sh`` and its AppArmor profile for ``bwrap``.
   * - ``doc/``
     - Two standalone notes kept beside the code, outside this manual.

Beside the packages, ``client/`` holds the launchers ``spear-chat.sh`` and
``spear-corpus.sh`` (`Entry points`_), ``tool-guide.md`` (the tool usage guide
handed to the model) and ``TRAINING_DATA.md`` (the operator's notes
on the training-data workflow, :doc:`/model/training`).

The server component
====================

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Path
     - Content
   * - ``server/inference/``
     - ``serve.sh`` (runs ``llama-server`` from configuration),
       ``install-llamacpp.sh``, and the ``spear-inference`` systemd unit with
       ``render-unit.sh``.
   * - ``server/config/``
     - ``server.conf.example`` and ``gpu.conf.example``: what a deployment
       must say.  The real files live under the runtime root, never here.
   * - ``server/embed/``
     - The embedding worker (``worker.py``, ``protocol.py``) and its pinned
       requirements.
   * - ``server/runtime/``
     - ``manifest.json``, what ``bootstrap-runtime.sh`` installs.
   * - ``server/scripts/``
     - ``fetch-model.sh``, which fetches a GGUF, shards included.
   * - ``server/tests/``
     - The server's own suite; it needs no GPU (:doc:`/operations/testing`).

Configuration
=============

These files sit in ``client/``.  The ones marked *ignored* are git-ignored:
each machine writes its own, usually from the ``.example`` file beside it.

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - File
     - In git
     - Meaning
   * - ``active-model.conf``
     - ignored
     - Absolute path of the GGUF that ``spear-chat --local`` serves when it
       starts the local server; ``SPEAR_MODEL`` in the environment overrides
       it.  An inference host chooses its model in its own ``server.conf``
       instead (:doc:`/model/model_serving`).
   * - ``active-lora.conf``
     - tracked
     - LoRA adapter path, or the literal ``none``.  ``SPEAR_LORA`` overrides.
   * - ``active-embedder.conf``
     - tracked
     - The embedding model that indexing and retrieval use.
       ``SPEAR_EMBED_MODEL`` overrides.
   * - ``active-embed-remote-cmd.conf``
     - ignored
     - The command that runs the embedding worker on a remote host.
       ``SPEAR_EMBED_REMOTE_CMD`` overrides.
   * - ``pod.conf``
     - ignored
     - Connection details for the remote vLLM pod used by
       ``spear-chat --remote``.  The SSH port changes on every pod restart.
   * - ``reds.conf``
     - ignored
     - Host, port and model of a shared GPU server used by
       ``spear-chat --reds``.  Stable host: the launcher only tunnels to it.
   * - ``machine.env``
     - ignored
     - Settings specific to THIS machine: the ``SPEAR_*`` path overrides a
       deployment needs and nothing a clone should inherit.  Every
       ``spear-*`` launcher sources it if it is there, and runs on the
       defaults if it is not.  ``spear-configure`` writes it.
   * - ``projects.json``
     - ignored
     - Named workspaces the chat can open, mapping a short name to an
       absolute path and a project kind.
   * - ``capabilities.json``
     - ignored
     - External capability providers registered per workspace (MCP servers
       over stdio).  ``SPEAR_CAPABILITIES_FILE`` overrides the path.
   * - ``tool-guide.md``
     - tracked
     - The tool usage guide handed to the model.
   * - ``rules.d/``
     - tracked
     - Prompt fragments.  Each one applies where the header it opens with
       scopes it, and a rule without a header applies nowhere
       (:doc:`/using/projects`).  ``rules.d/corpora/<name>.md`` holds a
       corpus's orientation map (:ref:`orientation_maps`).  The directory
       ships with only its ``README``.
   * - ``skills/``
     - tracked
     - Task recipes, each optionally opening with a ``SKILL.md`` frontmatter
       block read by ``skill_library.py`` (:ref:`skill_library`).  The
       directory ships with only its ``README``.

``active-backend.conf`` is state, not configuration: see `Persistent state`_.

.. _resource-directories:

Content the deployment owns
===========================

``rules.d/``, ``skills/`` and ``benches/`` are *content*, not code — and a
deployment's own rules, its learned skills and the bench that rates it are
exactly the material that does not belong in a public tree.  Each therefore
answers to an environment variable:

.. list-table::
   :header-rows: 1
   :widths: 26 30 44

   * - Variable
     - Default
     - What it holds
   * - ``SPEAR_RULES_DIR``
     - ``client/rules.d``
     - the rules, and ``corpora/<name>.md`` under it
   * - ``SPEAR_SKILLS_DIR``
     - ``client/skills``
     - the skill library, read *and written* — ``save_skill`` lands here
   * - ``SPEAR_BENCH_DIR``
     - ``client/benches``
     - the acceptance benches a project declares by bare name
   * - ``SPEAR_CONTENT_TREES``
     - unset
     - further content trees, colon-separated, each holding ``rules.d/``,
       ``skills/`` and ``benches/`` (a project checkout's ``spear-private/``);
       read by ``build.sh`` for a ``reds`` image and merged after the three
       above, same-named files replacing earlier ones

The default is the in-tree directory; **no default points outside the
checkout**, which is what keeps a private path out of public source.  An empty
value counts as unset rather than naming the filesystem root.

A directory that is not there reads as empty, without a warning: a variable
that points at the wrong place gives a session with no rules and no skills.
The same three variables are read by ``scripts/docker/build.sh`` when it bakes
an image (:ref:`optional-build-inputs`), so one setting covers both.

.. _persistent-state:

Persistent state
================

What a session accumulates goes to the *state directory*:
``SPEAR_STATE_DIR`` (``--state-dir``), or ``client/`` when it is not set.
``spear-configure`` writes ``SPEAR_STATE_DIR`` into ``machine.env``, by default
``~/.local/state/spear``; the container sets it to its mounted state volume
(:doc:`/start/container`).

Not everything follows that rule.  The vector index and the local server's log
stay in ``client/`` whatever the state directory is.  The knowledge store,
and the standard store as the standard review commands open it, fall back to
``~/.local/state/spear`` rather than ``client/`` when ``SPEAR_STATE_DIR`` is
unset.  Setting ``SPEAR_STATE_DIR`` therefore puts everything but the first
two rows below in one place.

.. list-table::
   :header-rows: 1
   :widths: 30 26 44

   * - Path
     - Where, when nothing is set
     - Content
   * - ``chromadb/``
     - ``client/``
     - The vector store.  ``SPEAR_DB_PATH`` (``--db-path``) moves it;
       ``SPEAR_STATE_DIR`` does not.  Rebuildable from the sources it
       indexes.
   * - ``llama-server.log``
     - ``client/``, always
     - Log of the local ``llama-server`` that ``spear-chat --local`` starts.
   * - ``knowledge.sqlite3``
     - ``~/.local/state/spear``
     - Workspace knowledge (``/remember``, ``/knowledge``).
       ``SPEAR_KNOWLEDGE_DB`` names the file; otherwise it is under
       ``SPEAR_STATE_DIR``.  A read-only team copy may sit under
       ``SPEAR_COMMON_STATE_DIR`` (:ref:`knowledge`).
   * - ``standards/``
     - ``client/`` for the chat; ``~/.local/state/spear`` for the standard
       review commands
     - The normative-standard store.  ``SPEAR_STANDARDS_ROOT`` moves it
       alone; otherwise it is under ``SPEAR_STATE_DIR``.
   * - ``audit/tool-actions.jsonl``
     - state directory
     - Append-only audit trail of mutating tool attempts.  Metadata only:
       assignments that look like secrets are redacted before the record is
       written.
   * - ``audit/runtime-trace.jsonl``
     - state directory
     - The runtime trace, written only with ``--trace``; ``--trace-file``
       (``SPEAR_TRACE_FILE``) moves it.
   * - ``audit/sessions/`` · ``audit/tool-results/`` · ``audit/checkpoints/``
     - state directory
     - Resumable task snapshots, referenced large results, and pre-mutation
       rollback bytes.  Active sessions may reference the latter two.
   * - ``audit/training-data/``
     - state directory
     - Canonical training episodes, the source a bundle freezes from
       (:doc:`/model/training`).  ``SPEAR_TRAINING_CAPTURE=0`` turns capture
       off.
   * - ``rules-learned.md``
     - state directory
     - Rules taught at runtime with ``/recall``.  They apply in every
       workspace, on every turn but a normative one.
   * - ``history-adhoc-<tag>.json`` · ``history-archive.jsonl``
     - state directory
     - Conversation transcripts, one per corpus, and their archive.
   * - ``memories-adhoc-<tag>.md`` · ``….md.metadata.json``
     - state directory
     - Legacy remembered notes and their metadata.  Not shown to a turn;
       their content reaches one only once migrated into workspace knowledge.
       ``SPEAR_MEMORIES_FILE`` names the file.  ``client/memories-so3.md`` is
       a tracked file of this kind.
   * - ``trajectories.jsonl`` · ``experience.jsonl``
     - state directory
     - Rated agent trajectories and the samples ``/good`` records, kept for
       fine-tuning.  ``SPEAR_TRAJECTORY_FILE`` and ``SPEAR_EXPERIENCE_FILE``
       move them.
   * - ``projects/<name>/``
     - state directory
     - The cached result of probing a project's build and test commands.
   * - ``training-execution.json``
     - state directory
     - The training host the operator control plane launches on; an example
       ships as ``client/training-execution.json.example``.
   * - ``active-backend.conf``
     - state directory
     - The backend chosen last, preselected by the startup picker.
   * - ``.input_history``
     - state directory
     - The prompt's line-editing history.

``<tag>`` is derived from the corpus root, so history follows the tree rather
than the directory a session was launched from.

.. _entry-points:

Entry points
============

The repository provides these scripts; linking them onto ``PATH`` under the
names below — a one-line wrapper in ``~/.local/bin`` each — is the usual
arrangement, and the rest of this documentation uses the names:

.. list-table::
   :header-rows: 1
   :widths: 24 40 36

   * - Name
     - Script
     - What it does
   * - ``spear-chat``
     - ``client/spear-chat.sh``
     - the assistant
   * - ``spear-corpus``
     - ``client/spear-corpus.sh``
     - the corpus registry (``/corpus`` in the chat)
   * - ``spear-index``
     - ``client/bin/python client/retrieval/index_dir.py``
     - index any tree
   * - ``spear-reindex``
     - ``client/bin/python client/retrieval/index_corpus.py``
     - rebuild a build-system corpus with the curated walk
   * - ``spear-server``
     - ``server/inference/serve.sh``
     - ``llama-server`` alone
   * - ``spear-docker``
     - ``scripts/docker/spear-docker.sh``
     - the assistant in a container, this machine's trees mounted
       (:doc:`/start/container`)
   * - ``spear-run.sh``
     - ``scripts/docker/spear-run.sh``, also ``/opt/spear/spear-run.sh`` in the image
     - a published image on a machine that has docker and nothing else
       (:ref:`container_run_reds`)

``spear-chat --help`` lists every flag — permissions, model, corpus — and is
answered before any server is started or any corpus resolved, so it costs
nothing.  The launcher invokes the venv interpreter by path rather than
sourcing ``bin/activate``: a virtualenv hardcodes its own absolute path, so a
relocated tree activates into a directory that no longer exists and
``python3`` silently falls through to the system interpreter.

``spear-corpus`` is a wrapper around
``corpus_registry.handle_corpus_command``, which is also what ``/corpus`` calls
inside the chat: one registry, one implementation, and one environment — it
reads ``machine.env`` exactly as ``spear-chat`` does.
