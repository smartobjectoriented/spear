.. _projects:

Projects and corpora
####################

A **corpus** is a source tree SPEAR can retrieve from. A **project** is a
registered corpus with a name and a declared behaviour. The two words are used
interchangeably in the interface, and the registry is one file.

The distinction that matters is a different one:

.. rubric:: Corpus is not workspace

A corpus is *what the assistant reads about*. The workspace is *where tools
run* — always the current directory, whatever corpus is attached. To work on
another tree, ``cd`` into it. No flag relocates the workspace.

The registry
************

``spear/projects.json`` holds the corpora registered on this machine. It is
untracked, because the paths in it are machine-specific;
``spear/projects.example.json`` is the tracked template.

Manage it from the shell or from inside a session:

.. code-block:: console

   $ spear-corpus                       # list
   $ spear-corpus add <name> [path] [--kind K] [--indexer I] [--autoindex]
   $ spear-corpus rm <name>
   $ spear-corpus scan [path] [--min N]      # split a workspace by file count

``/corpus`` inside ``spear-chat`` is the same code.

Entry format
************

.. code-block:: json

   {
     "acme-firmware": {
       "path": "/srv/src/acme-firmware",
       "kind": "generic",
       "exclude": ["./third_party", "./build/downloads"],
       "corpora": ["acme-shared-libs"],
       "test_commands": ["ctest --test-dir build"]
     }
   }

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Key
     - Meaning
   * - ``path``
     - absolute, or relative and resolved against ``SPEAR_CORPUS_ROOT``
       (which defaults to the repository root, so a corpus living inside the
       repository is found wherever the repository is cloned)
   * - ``kind``
     - a free label. It scopes skills and prints in the listing; it does not
       select behaviour
   * - ``indexer``
     - ``buildsystem`` or ``generic``
   * - ``autoindex``
     - build a missing index on sight rather than refusing
   * - ``prompt_file``
     - a domain prompt for this corpus
   * - ``collection``
     - name an existing index instead of deriving it from the path —
       required when the index travels with the registry
   * - ``exclude``
     - directories skipped at indexing
   * - ``include_build``
     - index build directories, minus the generated subtrees
   * - ``corpora``
     - federate these registered corpora into a session on this project
   * - ``shared``
     - attach this corpus to **every** session
   * - ``build_commands``, ``test_commands``
     - how this tree is built and tested. They win over every probe, and they
       are the **project's own verification**: SPEAR runs them on the final
       tree of a change-making turn, and a failure there makes the turn
       ``UNVERIFIED`` (:ref:`evidence_verdicts`)
   * - ``lint_commands``, ``acceptance_commands``
     - further commands recognised as verification when the agent runs them
   * - ``bench``
     - an acceptance script the harness runs to rate a turn — a bare name is
       looked up in the benches directory
   * - ``public``
     - mark the corpus as allowed in a public container image
       (:ref:`image-profiles`)
   * - ``normative_checks``
     - project checks bound to provisions of a bound standard
       (:ref:`normative_checks`)
   * - ``normative_applicability``
     - provisions of a bound standard declared applicable or not applicable to
       this project (:ref:`normative_checks`)
   * - ``standards``
     - the standards this project works against, as
       ``[{"id": …, "revision": …}]``; with several, a request picks one by
       naming it (:ref:`context_selection`)

The standard binding itself is **not** a project key: it is machine-wide, set
with ``/standard use`` (:ref:`standards`).

.. important::

   What a corpus *does* is declared, not inferred from its ``kind``. A label
   that silently selected an indexer, a prompt and three other behaviours was
   the source of corpora that behaved differently for no visible reason.

.. _normative_checks:

Normative checks and applicability
**********************************

When a change has to satisfy a bound standard (:ref:`mixed_mode`), SPEAR
reports a provision as satisfied or violated only on evidence it can
reproduce: a deterministic reading of the final source, or a check the project
itself declares to be evidence for that provision. Two keys carry those
declarations. They are the project's statements — written by whoever owns the
project, never by the model.

.. code-block:: json

   {
     "acme-firmware": {
       "path": "/srv/src/acme-firmware",
       "test_commands": ["make test"],
       "normative_checks": [{
         "id": "header-count",
         "standard": "SYNTH-STD", "revision": "1",
         "provisions": ["Rule 4.2.1-2"],
         "command": "make check-header-count",
         "evidence": {"kind": "test", "success": "exit_zero",
                      "semantics": "PASS_AND_FAIL_DECISIVE"},
         "enabled": true
       }],
       "normative_applicability": [{
         "provision": "Rule 4.2.1-3",
         "applicability": "NOT_APPLICABLE",
         "reason": "this device never receives headers"
       }]
     }
   }

``make test`` passing on this project says its tests pass, and nothing more.
``make check-header-count`` passing says Rule 4.2.1-2 holds — because the
binding says that is what this check checks. The binding, not the test, is
what carries normative meaning.

``normative_checks`` — one ``ConstraintCheckBinding`` per entry:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Field
     - Meaning
   * - ``id``
     - the binding's name, as the verdict and the audit trail report it
   * - ``standard``, ``revision``
     - the standard this binding is for. A binding for another standard or
       revision is reported and ignored; ``revision`` defaults to the bound one
   * - ``provisions``
     - the provisions this check is evidence for, by stable identity: an
       instance id (``Rule 4.2.1-2@<digest>``) or a printed label that names
       exactly one provision in the bound revision. A label that names none,
       or several, binds nothing and is reported
   * - ``command``
     - what to run, in the project's sandbox, from the workspace root
   * - ``evidence.kind``
     - ``test`` or ``check``
   * - ``evidence.success``
     - ``exit_zero`` — the only success criterion
   * - ``evidence.semantics``
     - ``PASS_ESTABLISHES_SATISFIED`` (default): a pass establishes the
       provisions, a failure establishes nothing. ``PASS_AND_FAIL_DECISIVE``:
       a failure also establishes a violation, and may trigger the one repair
   * - ``scope``
     - ``workspace`` (default)
   * - ``enabled``
     - ``false`` keeps the binding declared but unused

A bound check runs on the final source, in the same sandbox as the project's
own verification, and its result — ``PASSED``, ``FAILED``, ``NOT_RUN`` or
``ERROR`` — counts only for the source epoch it ran on. A check that could not
start, timed out, or modified the source it was run against establishes
nothing. Binding a check to a provision also declares that the provision
applies to the project.

``normative_applicability`` — one declaration per entry:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Field
     - Meaning
   * - ``provision``
     - the provision, by the same stable identity as above
   * - ``applicability``
     - ``APPLICABLE`` or ``NOT_APPLICABLE``
   * - ``reason``
     - required: why, in one line — the verdict quotes it
   * - ``standard``, ``revision``
     - optional; default to the bound standard

A provision declared ``NOT_APPLICABLE`` leaves the constraint packet. A
provision declared both ways is left unresolved.

.. _context_selection:

What a turn is shown
********************

A registered project is not context everywhere. Each turn starts from its
**workspace** — the registered project, or, for a tree nobody registered, that
tree alone, which inherits nothing from the project next to it or the one used
before — and is given only:

* what its request class calls for: the coding rules, workspace knowledge
  (:ref:`knowledge`), procedures and
  build commands for a change; the standard and its tools, and no
  implementation material, for a question about the standard; little more than
  the request for a general question. A MIXED change is selected once per pass;
* material of this workspace, or material explicitly declared generic;
* a rule scoped to paths only when the request names such a path.

The decision is deterministic — no model, no embedder — and recorded in the
audit trail with the reason for every item kept or left out
(``context_selected``, ``context_rejected``, ``toolset_selected``). It is never
shown to the model. When the context window is tight, optional material is left
out by priority; the request, the runtime's own constraints and the constraint
packet never are.

The tool family follows the same decision, at the family level only: a change
gets the coding core's six tools, a normative question the standard's tools,
and the MIXED post-check none. A turn on the coding core may also be offered
the external capabilities its workspace admits (:ref:`capabilities`).

Rules declare where they apply
==============================

A rule file in ``rules.d/`` opens with a header saying where it applies. **A
rule without one applies nowhere**: a rule is generic because it says so, not
because nobody scoped it.

.. code-block:: text

   ---
   scope: global                     every workspace: the only generic scope
   scope: corpus acme-firmware       this registered project, or one of its parts
   scope: path /srv/src/acme         workspaces whose tree lies under this directory
   tasks: implementation, mixed      the request classes it serves (the default);
                                     also normative, general
   paths: doc/**                     only when the request names such a path
   priority: 80                      kept longer when the window is tight
   ---
   Build the documentation with `make -C doc html` and treat warnings as errors.

A skill is held to the same rule: its ``scope:`` must name the project, or say
``[any]`` to be generic. ``/recall`` rules are generic by what that command
means. A project's own map (an in-tree ``.edgem-rules.md``, or
``rules.d/corpora/<name>.md``) belongs to that project only.

Exclusions have granularity
***************************

The common mistake is excluding too coarsely. A build-system tree that vendors
its components holds both generated material and tracked sources under the same
directory; excluding the directory wholesale takes the sources with it.

Prefer excluding the fetched and built subtrees one level down:

.. code-block:: json

   {
     "exclude": ["./linux/linux", "./linux/images", "./build/downloads"]
   }

Two things need no entry: snapshot directories ending in ``.back`` are skipped
by suffix wherever they appear, and ``include_build`` already drops the
generated build subdirectories by itself.

The leading ``./`` matters. A bare name matches *every* directory of that name
in the tree — including recipe directories elsewhere that you meant to keep.

Choosing a corpus for a session
*******************************

.. code-block:: console

   $ spear-chat                       # the corpus containing the cwd,
                                      # else an ad-hoc corpus on the cwd
   $ spear-chat --corpus acme-firmware
   $ spear-chat --here                # treat the cwd as an ad-hoc corpus
   $ spear-chat --with acme-shared-libs      # federate, for one session
   $ spear-chat --without some-shared-corpus

An **ad-hoc** corpus is one that was never registered: the current directory,
indexed under a name derived from its path. It behaves like any other corpus
and is the normal way to ask about a tree you are passing through.

Writable roots
**************

By default the launch directory is writable, and so is each registered corpus,
mounted at ``/workspaces/<name>`` — a path that works identically in ``bash``
and in the file tools. Relative paths always resolve in the launch directory
and never reach the others.

``--single-root`` restricts writes to the launch directory alone.
``--allow-absolute-paths`` accepts host absolute paths into the launch
directory; it is off by default, and ``/workspace/...`` always works.

Indexing
********

.. code-block:: text

   /reindex                  rebuild the index for the current corpus
   /corpus                   list and switch

Indexing is not optional in practice. See :ref:`Retrieval <retrieval>` for what
it measurably buys, and :ref:`Troubleshooting <troubleshooting>` for the
symptoms of a missing index.

.. seealso::

   :ref:`Retrieval <retrieval>` · :ref:`Configuration reference
   <configuration>`
