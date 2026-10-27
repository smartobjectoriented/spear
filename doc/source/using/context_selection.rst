.. _context_selection:

What a turn is shown
####################

A registered project is not context everywhere. Each turn starts from its
**workspace** — the registered project, or, for a tree nobody registered, that
tree alone, which inherits nothing from the project next to it or the one used
before — and is given only:

* what its :term:`request class` calls for: the coding rules, :term:`workspace knowledge`
  (:ref:`knowledge`), procedures and
  build commands for a change; the standard and its tools, and no
  implementation material, for a question about the standard; for a general
  question, the rules that name the ``general`` class, workspace knowledge and
  read-only external capabilities. A :term:`MIXED` change is selected once per pass;
* material of this workspace, or material explicitly declared generic;
* a rule scoped to paths only when the request names such a path.

The decision is deterministic — no model, no embedder — and recorded in the
audit trail with the reason for every item kept or left out
(``context_selected``, ``context_rejected``, ``toolset_selected``). It is never
shown to the model. When the context window is tight, optional material is left
out by priority; the request, the runtime's own constraints and the constraint
packet never are.

The tool family follows the same decision, at the family level only: a change
gets the :term:`coding core`'s six tools, a normative question the standard's tools,
and the MIXED post-check none. A turn on the coding core may also be offered
the external capabilities its workspace admits (:ref:`capabilities`).

Rules and skills are files
**************************

Rules (``rules.d/*.md``) and skills (``skills/*.md``) are a deployment's
content, not code: the repository ships the two directories with a README, and
each deployment fills them. Typical fragments look like this:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Fragment
     - Content
   * - ``rules.d/10-rootfs-packages.md``
     - How to add a package to the target root filesystem.
   * - ``rules.d/20-conventions.md``
     - Coding conventions, including the copyright header rule.
   * - ``rules.d/30-doc-build.md``
     - How the project documentation is built.
   * - ``skills/debug-build-failures.md``
     - Procedure for a failing build.
   * - ``skills/debug-buildsystem-build.md``
     - The BitBake-specific variant.
   * - ``skills/improve-c-code-quality.md``
     - Refactoring guidance for C.

Keeping these as files rather than as strings in ``rag_chat.py`` means the
behaviour of the assistant can be adjusted without touching code, and the
diff of a behavioural change is readable. Each one says where it applies — a
rule in its header, a skill in its ``scope`` — and one that says nothing is
not used. Where the two directories live is
:doc:`/operations/directory_layout`.

Rules declare where they apply
==============================

A rule file in ``rules.d/`` opens with a header saying where it applies. **A
rule without one applies nowhere**: a rule is generic because it says so, not
because nobody scoped it.

.. code-block:: text

   ---
   scope: global                     every workspace: the only generic scope
   scope: corpus acme-firmware       this registered project, or one of its parts
   scope: family acme-bsp            every registered project declaring that family
   scope: path /srv/src/acme         workspaces whose tree lies under this directory
   tasks: implementation, mixed      the request classes it serves (the default);
                                     also normative, general
   paths: doc/**                     only when the request names such a path
   priority: 80                      kept longer when the window is tight
   ---
   Build the documentation with `make -C doc html` and treat warnings as errors.

A skill is held to the same rule: its ``scope:`` must name the project or a
family (``[family:acme-bsp]``), or say ``[any]`` to be generic. ``/recall``
rules are generic by what that command means. A project's own
:ref:`orientation map <orientation_maps>` belongs to that project only.

.. _skill_library:

The skill library
=================

``skill_library.py`` owns the format, and the directory is the only source:
each turn is offered every skill whose ``scope`` admits its workspace and whose
required commands exist, read from the files at that moment, so a skill added,
edited or deleted by hand takes effect on the next turn.  Which of them fits
the request is left to the model, which is told to follow one only when it
does.  A library is expected to stay small; past eight skills for one
workspace, the rest are left out and the audit trail says so.

A skill may open with a ``SKILL.md`` frontmatter block -- the convention
agentskills.io and Hermes Agent use -- which the harness reads and the model
never sees, because it is stripped before injection:

.. code-block:: text

   ---
   name: debug-buildsystem-build
   description: Read the failing bitbake task log.
   scope: [bsp-a, bsp-b]
   requires: [bitbake]
   version: 3
   created: 2026-06-23T13:26:00
   updated: 2026-09-07T15:12:41
   ---

``description``
   One line saying what the procedure is *for*, shown with it.  ``/skills``
   lists it, and a skill that declares none is listed by its body's first
   line.

``scope``
   The registered projects the procedure belongs to -- a bitbake procedure has
   no business in an LVGL session -- or ``family:<name>`` for every project
   declaring that family.  ``[any]`` makes it generic, and must be
   written: a skill that states no scope is used nowhere.  A file saved by
   ``save_skill`` always states one.

``requires``
   Commands that must exist for the procedure to be runnable.  A skill whose
   requirement is absent is withheld rather than injected, and ``/skills``
   says which command is missing: why a procedure was *not* used is worth more
   than the fact that it exists.

``version``/``created``/``updated``
   Written by the harness.  An overwrite bumps the version and keeps the first
   stamp, so a procedure the model rewrote three times says so.  A rewrite
   that does not restate ``scope`` or ``requires`` inherits them: the model
   rewriting a body must not silently widen what an operator narrowed.

A file with no frontmatter, or with a block this reader does not understand,
is a valid skill whose body is the whole file.  Losing a procedure to a stray
colon would be worse than ignoring metadata nobody set.

.. _orientation_maps:

Orientation maps
****************

Each corpus may have a short prose map — where user apps live, which build
command to run, what never to edit — injected into the system prompt of every
request that changes or asks about that corpus's code.  It is the
highest-leverage context the assistant has, and the one most easily wrong: a
stale path in it sends the model looking for a script that no longer exists,
while a stale *chunk* would merely be outranked.

Maps live in ``$SPEAR_RULES_DIR/corpora/<corpus>.md`` (``client/rules.d/corpora/`` by
default), **not** in the tree
they describe.  Two reasons:

* they say how the *assistant* should work, which is this repository's concern,
  not the product's — a prompt tweak should not go through a product's review;
* an in-tree file is easily lost: a blanket ``.*`` line in a tree's
  ``.gitignore`` swallows it, and whoever clones that tree gets no map and
  never knows.

An in-tree ``.edgem-rules.md`` still **wins** when present: a tree someone else
owns may carry its own map, and theirs should beat ours.  Nothing is invented
for a corpus that has neither.

.. _learning_layers:

What the assistant learns, and when it takes effect
***************************************************

.. list-table::
   :header-rows: 1
   :widths: 30 34 36

   * - Layer
     - Takes effect
     - Where
   * - workspace knowledge (``/remember``, ``/knowledge``)
     - next turn, one workspace
     - ``knowledge.sqlite3`` under the state directory (:ref:`knowledge`)
   * - learned rules (``/recall``)
     - next turn, every corpus
     - ``rules-learned.md`` under the state directory
   * - orientation map
     - next turn, one corpus
     - ``rules.d/corpora/<name>.md``
   * - skills (``save_skill``, learned procedures)
     - next turn, scope and prerequisite gated
     - ``skills/*.md``
   * - rules (project conventions)
     - next turn, scope gated
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

Choosing between the first three is the usual question. ``/remember`` records
a fact about *this* tree, not a rule; the orientation map says where things
live in one corpus and what never to edit there; ``/recall`` records a working
rule true everywhere.

``/recall`` exists for rules such as *never rewrite an existing copyright
header, only extend its year range*: that is true in every tree alike, and
written per corpus it would be invisible in all the others.  A recalled line
is dated and appended to ``rules-learned.md``, which is injected after the
``rules.d`` rules on every request that changes or asks about a tree -- not on
a question about a bound standard, which no working rule bears on.

That reach is also the cost, so it is budgeted rather than unbounded: the whole
injected rule set is checked against ``RULES_BUDGET`` (8 000 characters) and
both the write and the load say so when it is exceeded.  Nothing is truncated
silently — the operator is told to trim, because deciding which rule stops
being injected is not a decision to take automatically.  ``/recall`` with no
argument prints what has been taught so far.

.. seealso::

   :ref:`Projects and corpora <projects>` · :ref:`Workspace knowledge
   <knowledge>` · :ref:`External capabilities <capabilities>`
