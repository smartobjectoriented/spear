.. _directory_layout:

================
Directory layout
================

Everything lives in the checkout — ``~/spear`` in the examples; the tree can
sit anywhere.  It mixes three quite different
kinds of content — application code, multi-gigabyte model weights, and
persistent state — so it is worth knowing which is which before running a
``find`` over it.

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
   * - ``spear/``
     - The application **and** its Python virtualenv.  ``bin/python`` here is
       the interpreter every command in this documentation uses.
   * - ``server/``
     - The generic inference server component: the ``llama-server`` launcher,
       model fetching, embedding and the runtime bootstrap.
   * - ``scripts/``
     - Release, configuration and image scripts (``spear-configure``,
       ``spear-image``, ``spear-consolidate``, ``spearversion.sh``);
       ``scripts/docker/`` builds and runs the container.
   * - ``docker/``
     - The container image: ``Dockerfile``, the entrypoint and the
       relative-path corpus registry.  It sits at the root rather than under
       ``spear/`` because it draws on both (:doc:`/start/container`).
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
     - This documentation.

.. note::

   ``spear`` being both the application directory and the virtualenv root
   is why ``bin/``, ``lib/`` and ``include/`` sit next to the packages.
   It is unusual but deliberate: the app and its exact dependency set travel
   together.

The application
===============

The code is one package per concern, imported from ``spear/`` as their common
root (``from evidence import completion``).  ``rag_chat`` and the other
commands are run as files (``spear/bin/python spear/cli/rag_chat.py``); the
launchers and the image do that for you.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Package
     - Role
   * - ``cli/``
     - The commands: ``rag_chat.py`` (REPL startup, project selection,
       retrieval helpers, tool handlers, the coding core's host ports, history
       and rendering), ``backend_select.py`` (the startup backend picker),
       ``machine_config.py`` (``spear-configure``) and
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
       runtime and the coding-core turn wrapper (``agent_runtime.py``), the
       UI-free task orchestration (``task_controller.py``), grounded task truth
       and compaction (``working_state.py``, ``compaction.py``), sessions,
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
   * - ``eval/`` · ``benchmarks/``
     - Evaluation harnesses and the benchmark runner.

Configuration
=============

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - File
     - Meaning
   * - ``active-model.conf``
     - Absolute path of the GGUF to serve, read by ``spear-chat --local`` when
       it starts the local server; ``SPEAR_MODEL`` in the environment
       overrides it.
   * - ``active-lora.conf``
     - LoRA adapter path, or the literal ``none``.  ``SPEAR_LORA`` overrides.
   * - ``pod.conf``
     - Connection details for the remote vLLM pod used by
       ``spear-chat --remote``.  The SSH port changes on every pod restart.
   * - ``reds.conf``
     - Host, port and model of a shared GPU server used by
       ``spear-chat --reds``.  Stable host: the launcher only tunnels to it.
   * - ``machine.env``
     - Settings specific to THIS machine, untracked: the ``SPEAR_*`` path
       overrides a deployment needs and nothing a clone should inherit.  Every
       ``spear-*`` launcher sources it if it is there, and runs on the defaults
       if it is not.
   * - ``active-backend.conf``
     - The backend chosen last, preselected by the startup picker.  Runtime
       state, not versioned.
   * - ``projects.json``
     - Named workspaces the chat can open, mapping a short name to an absolute
       path and a project kind.
   * - ``capabilities.json``
     - External capability providers registered per workspace (MCP servers
       over stdio).  ``SPEAR_CAPABILITIES_FILE`` overrides the path.
   * - ``tool-guide.md``
     - The tool usage guide handed to the model.
   * - ``rules.d/``
     - Prompt fragments injected per task kind.  ``rules.d/corpora/<name>.md``
       holds a corpus's orientation map (:doc:`/using/retrieval`).
   * - ``skills/``
     - Task recipes (build debugging, rootfs packages, code quality…), each
       optionally opening with a ``SKILL.md`` frontmatter block read by
       ``skill_library.py`` (:doc:`/using/retrieval`).

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
     - ``<app>/rules.d``
     - the always-injected rules, and ``corpora/<name>.md`` under it
   * - ``SPEAR_SKILLS_DIR``
     - ``<app>/skills``
     - the skill library, read *and written* — ``save_skill`` lands here
   * - ``SPEAR_BENCH_DIR``
     - ``<app>/benches``
     - the acceptance benches a project declares by bare name

The default is the in-tree directory, so a plain checkout behaves exactly as it
did; **no default points outside the checkout**, which is what keeps a private
path out of public source.  An empty value counts as unset rather than naming
the filesystem root.

The failure mode these replace is quiet: ``load_rules()`` returns ``""`` for a
directory that is not there and the skill library returns ``[]``, so a session
whose content had moved ran with none of it and said nothing.  The same three
variables are read by ``scripts/docker/build.sh`` when it bakes an image
(:ref:`optional-build-inputs`), so one setting covers both.

Persistent state
================

These live in the state directory: ``SPEAR_STATE_DIR``, or ``spear/`` when it
is not set.  The knowledge store and the standard store are the exception:
without ``SPEAR_STATE_DIR`` they default to ``~/.local/state/spear``.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Path
     - Content
   * - ``chromadb/``
     - The vector store.  Rebuildable from the sources it indexes.
   * - ``audit/tool-actions.jsonl``
     - Append-only audit trail of mutating tool attempts.  Metadata only:
       assignments that look like secrets are redacted before the record is
       written.
   * - ``audit/runtime-trace.jsonl``
     - Optional privacy-bounded runtime observability.
   * - ``audit/sessions/`` / ``audit/tool-results/`` / ``audit/checkpoints/``
     - Resumable task snapshots, referenced large results, and pre-mutation
       rollback bytes. Active sessions may reference the latter two.
   * - ``audit/training-data/``
     - Canonical training episodes, the source a bundle freezes from.
   * - ``rules-learned.md``
     - Rules taught at runtime with ``/recall``.  Injected into every session,
       under a token budget.
   * - ``knowledge.sqlite3``
     - Workspace knowledge (``/remember``, ``/knowledge``).
       ``SPEAR_KNOWLEDGE_DB`` overrides the path.
   * - ``standards/``
     - The normative-standard store.  ``SPEAR_STANDARDS_ROOT`` moves it alone.
   * - ``memories-*.md`` / ``memories-*.md.metadata.json``
     - Legacy remembered notes.  No longer shown to a turn; their content
       reaches one only once migrated into workspace knowledge.
   * - ``history*.json`` / ``history-archive.jsonl``
     - Conversation transcripts, including per-project ones.
   * - ``llama-server.log``
     - Log of the local ``llama-server`` that ``spear-chat --local`` starts.

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
     - ``spear/spear-chat.sh``
     - the assistant
   * - ``spear-corpus``
     - ``spear/spear-corpus.sh``
     - the corpus registry (``/corpus`` in the chat)
   * - ``spear-index``
     - ``spear/bin/python spear/retrieval/index_dir.py``
     - index any tree
   * - ``spear-reindex``
     - ``spear/bin/python spear/retrieval/index_corpus.py``
     - rebuild a build-system corpus with the curated walk
   * - ``spear-server``
     - ``server/inference/serve.sh``
     - ``llama-server`` alone
   * - ``spear-docker``
     - ``scripts/docker/spear-docker.sh``
     - the assistant in a container (:doc:`/start/container`)

``spear-chat --help`` lists every flag — permissions, model, corpus — and is
answered before any server is started or any corpus resolved, so it costs
nothing.  The launcher invokes the venv interpreter by path rather than
sourcing ``bin/activate``: a virtualenv hardcodes its own absolute path, so a
relocated tree activates into a directory that no longer exists and
``python3`` silently falls through to the system interpreter.

``spear-corpus`` is a wrapper around ``rag_chat.handle_corpus_command``, which
is also what ``/corpus`` calls inside the chat: one registry, one
implementation, and one environment — it reads ``machine.env`` exactly as
``spear-chat`` does.

.. _doc-diagrams:

This documentation
==================

::

   doc/
     Makefile                       make html | latexpdf | …
     requirements.txt               Sphinx toolchain (local and CI)
     source/
       conf.py                      Sphinx configuration
       rstFlatTable.py              the ``flat-table`` directive
       *.rst                        the chapters
       _static/theme_overrides.css  small readability overrides
       img/
         spear.drawio               every diagram, one page each (source)
         SPEAR-<Page>.drawio.png    the pages exported for the HTML build
         261008_SPEAR_Overview.png  the overview figure, a standalone image

Diagrams follow the convention of the sibling projects.  ``spear.drawio`` is
the source of every diagram but the overview, and is edited directly in
draw.io or the VS Code extension.  Each page the documentation uses is
exported to ``SPEAR-<Page>.drawio.png`` (one file per page, named after the
page), and the exported PNG is committed next to the ``.drawio`` in the same
change.  The overview figure of the introduction is a standalone raster
image, not exported from ``spear.drawio``, and is replaced as a whole.  From
the command line, with the drawio snap:

.. code-block:: console

   $ cd doc/source/img
   $ mkdir -p ~/snap/drawio/common/x && cp spear.drawio ~/snap/drawio/common/x/
   $ xvfb-run -a drawio -x -f png --scale 1.5 --border 10 -p 6 \
         -o ~/snap/drawio/common/x/SPEAR-Layout.drawio.png \
         ~/snap/drawio/common/x/spear.drawio --no-sandbox --disable-gpu

``-p`` is the 1-based page index (7 is the *Layout* page).  The snap is
confined: it cannot read ``/opt`` nor any hidden directory in ``$HOME``
(``~/.cache`` included), which is why the file is staged under
``~/snap/drawio/common``.

Building
========

.. code-block:: console

   $ cd ~/spear/doc
   $ make html
   $ xdg-open build/html/index.html

Sphinx and the RTD theme come from the system Python (``/usr/bin/sphinx-build``),
not from the ``spear`` virtualenv, which deliberately carries only the
application's runtime dependencies.  ``doc/requirements.txt`` lists the toolchain
for anyone who prefers an isolated environment:

.. code-block:: console

   $ pip install -r doc/requirements.txt

The same file is what ``.github/workflows/docs.yml`` installs.  That workflow
builds these pages on every push to ``main`` and publishes them to GitHub
Pages, so the published site is always the documentation of the current
``main``; a pull request builds the documentation but never publishes it.  The
build must stay dependency-free with respect to the application — the
documentation imports no project module, which is why a Sphinx toolchain and a
checkout are the whole of what CI needs.
