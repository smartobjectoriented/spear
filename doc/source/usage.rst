===================
Using the assistant
===================

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
tree never dilutes the index.  :doc:`retrieval` is the full account.

Command line
============

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Command
     - Purpose
   * - ``spear-chat``
     - the assistant; auto-detects the corpus from the current directory, and
       shows a picker otherwise
   * - ``spear-chat --corpus lvgl``
     - open a registered corpus by name
   * - ``spear-chat --here``
     - ad-hoc, on the current directory
   * - ``spear-chat -y``
     - bypass permissions: auto-accept tool actions, network included
   * - ``spear-chat --no-network``
     - offline — no network in the shell, and the web tools are not exposed
   * - ``spear-chat --ctx 65536 --temp 0.1 …``
     - session settings that used to be environment variables; ``--help``
       lists them all
   * - ``spear-corpus list|add|rm|scan``
     - manage the corpus registry (``/corpus`` in the chat)
   * - ``spear-server``
     - ``llama-server`` alone
   * - ``spear-reindex [path]``
     - rebuild a corpus with the curated walk
   * - ``spear-index [dir] [--max-files N] [--exclude D]``
     - index any tree

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

   ``/search <q>`` ``/reindex`` ``/history`` ``/skills`` ``/undo`` ``/clear``
   ``/tools``

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
   are never trained on.  See :doc:`training`.

Multi-line paste is supported; ``ctrl+c`` interrupts generation; ``Enter``
confirms tool prompts; arrows, ``Home``/``End`` and ``ctrl+r`` come from
readline.

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

The authorization rules behind the first bullet are :doc:`security_model`; the
confinement behind all of them is :doc:`sandbox`, :doc:`network` and
:doc:`resource_control`.
