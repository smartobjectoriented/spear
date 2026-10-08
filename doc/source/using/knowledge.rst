.. _knowledge:

Workspace knowledge
###################

*Workspace knowledge* is what SPEAR has been told, or has verified, about one
workspace -- where a driver lives, which recipe builds an image, how two
components relate, what the team decided -- kept across sessions. It is
explicit, scoped to one workspace, and auditable: every record says where it
came from, and nothing is recorded without someone asking for it. It is not a
conversational memory, and nothing is harvested from a conversation.

What it is not
**************

Knowledge describes. It never instructs, configures or decides compliance, and
everything else a turn is given outranks it:

.. list-table::
   :header-rows: 1
   :widths: 22 40 38

   * - Kind
     - What it is
     - Authority
   * - The current request
     - what the operator is asking now
     - highest: a fact it contradicts is out of date for this task
   * - Rules (``rules.d``)
     - how to work in a workspace: "use four spaces"
     - instruction (:ref:`context_selection`)
   * - Configuration (``projects.json``)
     - the build and test commands, the standard bound
     - configured fact (:ref:`projects`)
   * - The bound standard
     - the provisions and the constraint packet
     - normative (:ref:`standards`)
   * - **Workspace knowledge**
     - what is the case in this workspace
     - descriptive only
   * - The conversation
     - what was said this session
     - conversational

An instruction is not knowledge: "always run ``make test`` before committing"
is refused with a pointer to ``rules.d``, while "the validation command is
``make test``" is a fact. Knowledge never reaches a question about the bound
standard, the normative pre-pass or the compliance check, so a record can never
establish a provision, its applicability or a verdict. It grants no permission
and no external capability.

``/remember <fact>`` is a shorthand for recording one: the same record, in the
same store, with the same checks -- an instruction, or something true only for
the moment ("the last command failed once", "I think the parser is wrong"), is
refused. Recording the same fact twice keeps one record. When the model offers
a fact through its ``remember`` tool it only proposes it, and a proposal is
never shown to a turn until it is accepted.

Recording knowledge
*******************

.. code-block:: text

   /knowledge add --kind build --subject "boot image" The boot image is rootfs.cpio.
   /knowledge add --kind relation --subject "board link" --source build/rootfs.bb:48 \
       --quote "do_attach () {" The board link is re-pointed by the rootfs recipe.

Kinds are ``fact``, ``architecture``, ``build``, ``relation``, ``decision`` and
``command``. ``--tags`` and ``--path`` name what a record is about, so a
request that names the same component or path finds it. Everything after the
options is the statement, as written.

A record is one fact, with a stable id, and in one of four states:

``ACTIVE``
   what a turn may be shown. A fact the operator states is active at once,
   marked ``USER_CONFIRMED`` -- the operator's word, not a check: SPEAR does not
   test it against the source, and a turn may take it as given. When it matters
   that a fact stays true, record it with ``--source`` so it is bound to the
   file and goes stale when the file changes; a fact given with ``--source`` is active only if
   the file holds the quoted evidence (or the named line), and is marked
   ``SOURCE_VERIFIED`` and bound to that file's content.
``PROPOSED``
   recorded but never shown. Whatever a model, a tool or an external system
   suggests can only be proposed; ``/knowledge accept <id>`` makes it active.
``STALE``
   a source-verified fact whose file no longer reads as it did. It is left out
   until ``/knowledge check`` finds the evidence again.
``REVOKED``
   withdrawn by the operator, and kept in the history.

Two active records that disagree about the same subject are shown together as a
*knowledge conflict*, each with its source; the newer one does not quietly
win. ``/knowledge amend <id> <statement>`` records a correction as the next
version of the same record.

Inspecting and removing it
**************************

.. code-block:: text

   /knowledge list [--all|--proposed|--stale|--revoked]
   /knowledge show <id>        the record, its source and its earlier versions
   /knowledge revoke <id> [reason]
   /knowledge check            re-verify sources; report stale records and conflicts
   /knowledge export [FILE]    this workspace's records, with history, as JSON
   /knowledge purge            delete this workspace's records for good

Knowledge belongs to one exact workspace: the registered project, or an
unregistered tree by its own path. It is never shared with another workspace,
and a new tree inherits none. It is stored in SPEAR's state directory
(``knowledge.sqlite3``, or ``SPEAR_KNOWLEDGE_DB``), never in the project's
repository.

A team shares what it knows through an image's **common state**
(:ref:`common-state`): records read under the user's own and never written.
They are listed with ``common``, and are revoked, amended or found stale like
any other: the first change copies the record into the user's store, which
then shadows it, and the next user is still given the original. ``/knowledge
purge`` deletes the user's records only. ``spear-consolidate`` merges users'
records back into the common store for the next image.

What a turn is shown
********************

A turn that changes code, a general question, and the implementation pass of a
MIXED change are shown the workspace's active records under *Workspace
knowledge*, each with its provenance. Up to ten short records are shown whole.
A larger set is shown as an index -- one line a record, grouped by kind --
with the records the request names (by subject, tag or path) in full, and the
turn reads any other with ``spear-knowledge show <id>`` in its terminal. That
command is answered by SPEAR and never run by a shell, like
``spear-capability`` (:ref:`capabilities`). A turn may also propose a record
with ``spear-knowledge propose``; it stays a proposal until the operator
accepts it.

Every change to a record, every record a turn is shown, every stale source and
every conflict is in the audit trail (``knowledge_proposed``,
``knowledge_activated``, ``knowledge_selected``, ``knowledge_stale``,
``knowledge_conflict``, ``knowledge_revoked``, ``knowledge_duplicate``,
``knowledge_migrated``).

Legacy remembered notes
***********************

Before workspace knowledge, ``/remember`` and the ``remember`` tool appended
notes to a ``memories-*.md`` file per corpus. Those notes are no longer shown to
a turn, and SPEAR says so at start-up while a file is unmigrated. Nothing moves
them automatically:

.. code-block:: text

   /knowledge migrate-remember          a dry run: what would happen, by count
   /knowledge migrate-remember --apply  migrate

Each note gets only the authority its origin earns: one written with
``/remember`` becomes active and user-confirmed; one written by the model's
tool, or one whose origin the file does not record, becomes a *proposal* to
review with ``/knowledge list --proposed`` and ``/knowledge accept``. A note
that reads as an instruction is skipped -- move it to a rule yourself if it
still holds -- and so is one that describes a moment, or repeats a fact already
recorded. The report gives counts, never the notes' text. Running it again adds
nothing. The Markdown file is left as it was, with a marker beside it, so
nothing is lost if a migration has to be redone. Review the dry run before
applying it: a migrated note is a fact SPEAR will show to future turns.
