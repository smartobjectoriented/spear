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
   * - ``spear/rules.d/``
     - yes
     - project-independent rules injected into every session
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
     - context window in tokens (default 32768)
   * - ``--temp`` · ``SPEAR_TEMP``
     - sampling temperature (default 0.25)
   * - ``--max-tokens`` · ``SPEAR_MAX_TOKENS``
     - cap on one reply
   * - ``ANTHROPIC_API_KEY``
     - Anthropic credential

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
     - confirm each edit and command. The only mode with network access
   * - ``--auto`` (``-y``, ``--yolo``, ``--bypass-permissions``)
     - run without asking. Deliberately no network
   * - ``--no-network``
     - drop network even in ``--ask``
   * - ``--single-root``
     - restrict writes to the launch directory
   * - ``--allow-absolute-paths``
     - accept host absolute paths into the launch directory
   * - ``--max-commands`` · ``SPEAR_MAX_COMMANDS``
     - bash commands per task
   * - ``--max-tool-rounds`` · ``SPEAR_MAX_TOOL_ROUNDS``
     - tool rounds per task

Session state and audit
***********************

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Flag / variable
     - Meaning
   * - ``--state-dir`` · ``SPEAR_STATE_DIR``
     - where the session accumulates; defaults to
       ``~/.local/state/spear``
   * - ``--operator`` · ``SPEAR_OPERATOR``
     - operator name recorded in the audit trail
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
     standards/                        the normative store
     history-adhoc-<tag>.json          per-corpus conversation history
     memories-adhoc-<tag>.md           per-corpus durable memories

``<tag>`` is derived from the corpus root, so history and memories follow the
tree rather than the directory you happened to launch from.

.. note::

   Pointing ``SPEAR_STATE_DIR`` at a fresh directory gives a session that
   carries nothing: no history, no memories, no audit. That is the supported
   way to get a clean run.

Not configuration
*****************

Many other ``SPEAR_*`` variables exist in the source. They are internal
thresholds and test hooks, not a user-facing interface, and they are
deliberately absent from ``--help``. Treat what ``--help`` prints as the
supported surface; anything else may change without notice.

.. seealso::

   :ref:`Model backends <backends>` · :ref:`Projects and corpora <projects>`
