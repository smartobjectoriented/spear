.. _configuration:

Configuration reference
#######################

Every session setting below is both a command-line flag and an environment
variable. **The flag wins.** The authoritative list is the one the program
prints:

.. code-block:: console

   $ spear-chat --help

This page groups the same settings by what they govern, and records the files
they live in.

Files
*****

.. list-table::
   :header-rows: 1
   :widths: 34 20 46

   * - File
     - Tracked
     - Holds
   * - ``spear/projects.json``
     - no
     - the corpus registry for this machine
   * - ``spear/projects.example.json``
     - yes
     - the template for the above
   * - ``spear/machine.env``
     - no
     - machine-specific environment: the one file that names what this
       deployment keeps outside the checkout. Read by the launcher, by
       ``spear-corpus`` and by ``scripts/docker/build.sh``. Written by
       ``scripts/spear-configure``
   * - ``spear/machine.env.example``
     - yes
     - its shape, for reading; ``spear-configure`` writes the real one
   * - ``spear/active-backend.conf``
     - no
     - the backend chosen last
   * - ``spear/active-model.conf``
     - no
     - the served model
   * - ``spear/capabilities.json``
     - no
     - the external capability providers (:ref:`capabilities`);
       ``SPEAR_CAPABILITIES_FILE`` names another file
   * - ``spear/rules.d/``
     - yes (a README)
     - the rules, each applied where its header scopes it
       (:ref:`context_selection`); ``SPEAR_RULES_DIR`` relocates it
   * - ``spear/skills/``
     - yes (a README)
     - the skill library; ``SPEAR_SKILLS_DIR`` relocates it
   * - ``server/runtime/manifest.json``
     - yes
     - the pinned inference runtime

.. important::

   The untracked files are the boundary between the platform and the machine.
   A host, an account, a card identifier or somebody's home directory in a
   tracked file is a disclosure — the public-boundary tests exist to catch
   exactly that.

Endpoint and model
******************

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Flag / variable
     - Meaning
   * - ``--api-base`` · ``SPEAR_API_BASE``
     - OpenAI-compatible endpoint URL
   * - ``--provider``
     - ``openai-compatible`` (default) or ``anthropic``
   * - ``--model``
     - model id
   * - ``--ctx`` · ``SPEAR_CTX``
     - context window in tokens (default: asked of the server, else 32768)
   * - ``--temp`` · ``SPEAR_TEMP``
     - sampling temperature (default 0.25)
   * - ``--max-tokens`` · ``SPEAR_MAX_TOKENS``
     - cap on one reply
   * - ``SPEAR_RESPONSE_MAX_TOKENS``
     - cap on one model response of a coding turn (default 16384); a response
       cut there is retried and then reported as truncated, never as complete
   * - ``ANTHROPIC_API_KEY``
     - Anthropic credential; an ``ant auth login`` session is used when it is
       unset

Corpora and retrieval
*********************

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Flag / variable
     - Meaning
   * - ``--corpus`` (``--project``, ``--checkout``)
     - force a registered corpus
   * - ``--here``
     - treat the current directory as an ad-hoc corpus
   * - ``--with`` / ``--without``
     - federate or drop a corpus, for one session
   * - ``--corpus-root`` · ``SPEAR_CORPUS_ROOT``
     - prefix for relative corpus paths
   * - ``--db-path`` · ``SPEAR_DB_PATH``
     - the vector store directory
   * - ``--embed-remote`` · ``SPEAR_EMBED_REMOTE``
     - ssh host that embeds the corpora

Authoritative standards
***********************

.. list-table::
   :header-rows: 1
   :widths: 44 56

   * - Flag / variable
     - Meaning
   * - ``--standard-embed-model`` · ``SPEAR_STANDARD_EMBED_MODEL``
     - embedding model for ``/standard`` indexes
   * - ``--standard-embed-revision`` · ``SPEAR_STANDARD_EMBED_REVISION``
     - its pinned revision (required with the model)
   * - ``--standard-embed-device`` · ``SPEAR_STANDARD_EMBED_DEVICE``
     - ``cpu`` (default) or ``cuda``, for a local build
   * - ``--standard-embed-remote`` · ``SPEAR_STANDARD_EMBED_REMOTE``
     - ssh host allowed to embed a **public** standard
   * - ``SPEAR_STANDARDS_ROOT``
     - the normative store

.. warning::

   ``SPEAR_STANDARD_EMBED_REMOTE`` sends document text to another host. It is
   for public specifications. A licensed document is embedded locally.

Permissions and execution
*************************

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Flag / variable
     - Meaning
   * - ``--safe``
     - read-only; the default when no mode is given. No network
   * - ``--ask`` (``--confirm``, ``--no-bypass``)
     - confirm each edit and command, network use included
   * - ``--auto`` (``-y``, ``--yolo``, ``--bypass-permissions``)
     - run without asking; network available
   * - ``--no-network``
     - no network in any mode
   * - ``--single-root``
     - restrict writes to the launch directory
   * - ``--allow-absolute-paths``
     - accept host absolute paths into the launch directory
   * - ``--max-commands`` · ``SPEAR_MAX_COMMANDS``
     - tool calls per task (default 500, doubled for a change request unless
       set)
   * - ``--max-tool-rounds`` · ``SPEAR_MAX_TOOL_ROUNDS``
     - model rounds per task (default 250, doubled for a change request unless
       set)
   * - ``SPEAR_REFUSAL_REPEATS``
     - how many times a turn may repeat an operation already refused before
       it is stopped (default 5)

Session state and audit
***********************

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Flag / variable
     - Meaning
   * - ``--state-dir`` · ``SPEAR_STATE_DIR``
     - where the session accumulates; ``machine.env`` sets it
       (``~/.local/state/spear`` in the example). Unset, the chat keeps its
       state in the ``spear/`` directory itself
   * - ``SPEAR_COMMON_STATE_DIR``
     - common state, read under the user's own and never written; unset by
       default, set by the image (:ref:`common-state`)
   * - ``--operator`` · ``SPEAR_OPERATOR``
     - operator name recorded in the audit trail
   * - ``--fresh`` · ``SPEAR_FRESH=1``
     - start without the corpus's stored conversation, and leave it as it is;
       workspace knowledge, rules and configuration still apply
   * - ``SPEAR_KNOWLEDGE_DB``
     - the workspace knowledge store (default ``knowledge.sqlite3`` under
       the state directory; :ref:`knowledge`)
   * - ``--trace`` · ``SPEAR_TRACE``
     - record a runtime JSONL trace
   * - ``--trace-file`` · ``SPEAR_TRACE_FILE``
     - trace destination
   * - ``--record`` · ``SPEAR_RECORD_TURNS``
     - record this session's model turns to a file
   * - ``--replay`` · ``SPEAR_REPLAY_TURNS``
     - replay a recorded file instead of calling the model

The state directory
===================

.. code-block:: text

   $SPEAR_STATE_DIR/
     audit/
       sessions/<id>/events.jsonl      the turn-by-turn event stream
       sessions/<id>/snapshot.json     the conversation it ran on
       tool-actions.jsonl              metadata-only action log
       runtime-trace.jsonl             the runtime trace
       checkpoints/                    file contents captured before a mutation
     standards/                        the normative store
     knowledge.sqlite3                 workspace knowledge
     rules-learned.md                  the rules taught with /recall
     history-adhoc-<tag>.json          per-corpus conversation history

``<tag>`` is derived from the corpus root, so history follows the tree rather
than the directory you happened to launch from. A session resumes
the stored conversation of its corpus unless it is started with ``--fresh``.

.. note::

   Pointing ``SPEAR_STATE_DIR`` at a fresh directory gives a session that
   carries nothing: no history, no workspace knowledge, no audit. That is the supported
   way to get a clean run.

Not configuration
*****************

Many other ``SPEAR_*`` variables exist in the source. They are internal
thresholds and test hooks, not a user-facing interface, and they are
deliberately absent from ``--help``. Treat what ``--help`` prints as the
supported surface; anything else may change without notice.

.. seealso::

   :ref:`Model backends <backends>` · :ref:`Projects and corpora <projects>`
