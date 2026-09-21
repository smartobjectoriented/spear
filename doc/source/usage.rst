.. _usage:

==========
spear-chat
==========

This page is the day-to-day surface: the commands outside the chat, the
commands inside it, where the assistant's knowledge comes from and on what
delay it changes, and the guards that will interrupt you.

The corpus is the unit
======================

Everything is a **corpus**: a working tree with its own retrieval index,
history and memories.  A corpus declares what it does — ``indexer:
buildsystem`` selects the curated BitBake/Yocto walk, ``autoindex`` builds a
missing index on sight, ``prompt_file`` gives it a domain prompt, ``collection``
names an existing index instead of deriving one from the path.  ``kind`` is a
free-form label and changes nothing.  Any other tree gets the generic indexer
and a collection derived from its path.

The active corpus is auto-detected from the current directory; otherwise a
picker lists them.  Launching in an unregistered multi-component workspace
offers to split it into one corpus per large sub-tree, so that a huge upstream
tree never dilutes the index.  :doc:`/retrieval` is the full account.

Permission modes
================

The first decision a session makes is what the assistant is allowed to do to
your files. One mode is always in force, and ``--safe`` is the default when
none is given.

.. list-table::
   :header-rows: 1
   :widths: 22 40 38

   * - Mode
     - Writes and commands
     - Network
   * - ``--safe``
     - refused, not proposed
     - none
   * - ``--ask``
     - each edit and each command is confirmed before it runs
     - **available**
   * - ``--auto``
     - run without asking
     - none

``--ask`` has the aliases ``--confirm`` and ``--no-bypass``; ``--auto`` has
``-y``, ``--yolo`` and ``--bypass-permissions``.

.. important::

   ``--ask`` is the only mode with network access, and ``--no-network``
   removes it there too. An unattended run *with* network is the combination
   deliberately not offered: confirmation is what makes reaching the network
   reviewable.

Two further flags bound where writes may land:

``--single-root``
   Restrict writes to the launch directory. By default the registered corpora
   are writable too, each mounted at ``/workspaces/<name>`` — a path that
   works in ``bash`` and in the file tools alike. Relative paths always
   resolve in the launch directory and never reach them.

``--allow-absolute-paths``
   Accept host absolute paths into the launch directory. Off by default;
   ``/workspace/...`` always works.

Command line
============

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Command
     - Purpose
   * - ``spear-chat``
     - the assistant; uses the corpus containing the current directory, else
       an ad-hoc one on it
   * - ``spear-chat --corpus <name>``
     - open a registered corpus by name (aliases ``--project``,
       ``--checkout``)
   * - ``spear-chat --here``
     - ad-hoc, on the current directory
   * - ``spear-chat --with <name>`` / ``--without <name>``
     - federate an extra corpus into this session, or drop one that would be
       attached
   * - ``spear-chat --ask`` / ``--auto`` / ``--safe``
     - the permission mode, as above
   * - ``spear-chat --no-network``
     - drop network even in ``--ask``
   * - ``spear-chat --local`` / ``--remote`` / ``--reds``
     - which backend to talk to; without one, an interactive launch shows a
       picker and preselects the last choice
   * - ``spear-chat --provider anthropic --model <id>``
     - use the Anthropic API instead of an OpenAI-compatible endpoint
   * - ``spear-chat --ctx 65536 --temp 0.1 …``
     - session settings, each also an environment variable; ``--help`` lists
       them all and the flag wins
   * - ``spear-chat --record FILE`` / ``--replay FILE``
     - write down every model turn, or answer from a recording while the
       tools, files and gates still run for real
   * - ``spear-corpus list|add|rm|scan``
     - manage the corpus registry (``/corpus`` in the chat)
   * - ``spear-server``
     - ``llama-server`` alone
   * - ``spear-reindex [path]``
     - rebuild a corpus with the curated walk
   * - ``spear-index [dir] [--max-files N] [--exclude D]``
     - index any tree

.. note::

   Tools always run in the **current directory**, whatever corpus is
   attached. To work on another tree, ``cd`` into it — no flag relocates the
   workspace. See :doc:`/projects`.

``spear-corpus scan <workspace>`` splits a multi-component tree into
per-component corpora by file count; the chat offers the same split
automatically when you open such a workspace.  The command-line tool and the
in-chat ``/corpus`` are one implementation — same registry, same walk.

:ref:`entry-points` documents each executable in more detail.

In-chat reference
=================

Direct tools, no model in the loop:

   ``!read`` ``!ls`` ``!grep`` ``!find`` ``!edit`` ``!run`` ``!web``

Session commands:

   ``/search <q>`` ``/reindex`` ``/history`` ``/skills`` ``/undo``
   ``/clear`` (``/new``) ``/tools`` ``/model`` (``/switch``)

``/model`` (or ``/switch``)
   Change backend or model without leaving the session.

``/clear`` (or ``/new``)
   Start a fresh conversation. The corpus, its memories and its index are
   unaffected; only the conversation is dropped.

``/corpus [list|add|rm|scan]``
   The registry, without leaving the session.  Registering does not switch
   corpus (``spear-chat --corpus <name>`` does), but a tree registered inside
   another is excluded from it at the next ``/reindex`` — which is the usual
   reason to register one.

``/remember <note>``
   Add a long-term memory for *this* corpus.  With no argument, list them.

``/recall <rule>``
   Add a rule for *every* corpus, injected into every request.  With no
   argument, list them.  Use it for what holds everywhere — "an existing
   copyright header is never rewritten" — and ``/remember`` for what is true of
   one tree only.

``/forget <regex>``
   Prune the history-search index.  The archive file itself is kept.

``/good`` / ``/bad``
   Save the last exchange as a fine-tuning sample, or log a bad one.  Bad ones
   are never trained on.  See :doc:`/model/training`.

Operator commands
-----------------

These two are typed in the session and are **never reachable by the model**.
The agent cannot rebind the document it is being held to, nor drive the
fine-tuning machinery.

``/standard [status|list|use|ingest|verify]``
   Ingest a specification, bind one, inspect the binding. The binding decides
   how normative answers are grounded and is shared by every session on the
   machine, so check it before trusting one: ``/standard status``. See
   :doc:`/standards`.

``/finetune``
   The fine-tuning control plane. See :doc:`/model/training`.

Multi-line paste is supported; ``ctrl+c`` interrupts generation; ``Enter``
confirms tool prompts; arrows, ``Home``/``End`` and ``ctrl+r`` come from
readline. ``quit``, ``exit`` or ``q`` ends the session.

Web access
==========

Four routes reach the network, and all of them disappear under
``--no-network``:

#. Freshness questions ("latest version of …") auto-trigger a search before
   the model sees the turn; results and sources are shown.
#. Explicit phrasings ("search the internet for …") auto-trigger the same way.
#. ``!web <query>`` — manual.
#. The model's own ``search_internet`` and ``fetch_url`` tool calls.

``search_internet`` finds pages and ``fetch_url`` reads one — HTML, PDF or
plain text — or saves it to disk.  Both run in the chat process rather than in
the sandboxed shell, so reading works even in safe mode; but ``fetch_url`` is
http/https only and refuses any host resolving to a private, loopback or
link-local address.  It reaches the internet, not the LAN the assistant happens
to sit on.  A long PDF comes back in slices, and the result names the ``pages``
range that continues it.  ``save_as=<path>`` downloads the file itself, never a
truncated one, and is a mutation — so it obeys the permission mode like any
write.  That is the route to a document you mean to ingest as a standard.  The
pair is described from the harness side in :ref:`the-web-pair`.

Knowledge layers, on different clocks
=====================================

.. list-table::
   :header-rows: 1
   :widths: 30 34 36

   * - Layer
     - Latency
     - Where
   * - memories (``remember`` tool, ``/remember``)
     - next turn
     - ``memories-*.md``
   * - learned rules (``/recall``)
     - next launch, every corpus
     - ``rules-learned.md`` under the state directory
   * - skills (``save_skill``, learned procedures)
     - next turn, similarity-injected, scope and prerequisite gated
     - ``skills/*.md``
   * - rules (project conventions)
     - next launch
     - ``rules.d/*.md``
   * - tool behaviour rules
     - next launch
     - ``tool-guide.md``
   * - retrieval corpus
     - after ``/reindex``
     - ``chromadb/``
   * - history search (cross-session recall)
     - continuous
     - the history archive collection

Two behaviours of the base model are worth knowing before you judge the
harness by them:

* Qwen3 models have a **thinking mode**.  The chat disables it through
  ``chat_template_kwargs.enable_thinking=false``; leaving it off is deliberate,
  since here it only burns tokens.
* The model *uses* its memory correctly but verbally **denies having one** when
  asked.  That is a pretraining reflex: judge it by behaviour, not by its
  self-description.  If a bad answer lands in history, ``/undo`` it — the model
  imitates its own past answers.

Safety guards
=============

Each of these was added after the failure it prevents:

* read-only commands are auto-approved; anything else asks, with ``Enter``
  meaning yes;
* ``write_file`` refuses to overwrite a file much larger than the proposed
  content, because a model's "full rewrite" of a big file is a hallucination
  magnet;
* a per-turn command budget (15), a round cap, and a result cache so nothing is
  run twice;
* anti-fabrication: invented ``[tool]`` output is stripped and the turn is
  retried;
* grounded edit errors: a failed exact-match edit returns the file's real tail
  and points the model at ``append_file``.

The authorization rules behind the first bullet are :doc:`/harness/security_model`; the
confinement behind all of them is :doc:`/harness/sandbox`, :doc:`/harness/network` and
:doc:`/harness/resource_control`.
