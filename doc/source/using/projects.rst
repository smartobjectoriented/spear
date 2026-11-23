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
another tree, ``cd`` into it. No flag relocates the workspace
(:ref:`two_roots`).

The registry
************

``client/projects.json`` holds the corpora registered on this machine. It is
untracked, because the paths in it are machine-specific;
``client/projects.example.json`` is the tracked template.

Manage it from the shell or from inside a session:

.. code-block:: console

   $ spear-corpus                       # list
   $ spear-corpus add <name> [path] [--kind K] [--indexer I] [--autoindex] \
                      [--public] [--prompt-file F]
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
     },
     "bsp-a": {
       "path": "/srv/src/bsp-a",
       "kind": "buildsystem",
       "indexer": "buildsystem",
       "collection": "bsp_a",
       "autoindex": true,
       "prompt_file": "bsp-prompt.md"
     },
     "spear": { "path": "client", "kind": "generic" }
   }

A ``path`` is either **absolute** — a tree of yours, genuinely
machine-specific — or **relative**, in which case it resolves against
``SPEAR_CORPUS_ROOT``. That root defaults to the repository root, so the
``spear`` entry above, and every corpus that lives inside the repository, is
found wherever the repository is cloned; a container overrides the root with
its mount point (:doc:`/start/container`).

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Key
     - Meaning
   * - ``path``
     - absolute, or relative and resolved against ``SPEAR_CORPUS_ROOT``
   * - ``kind``
     - a free label, printed in the listing. It selects no behaviour, and
       rules and skills are scoped by project name, not by kind
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
   * - ``families``
     - the project families this project belongs to, e.g. ``["acme-bsp"]``:
       it then receives the rules, skills and capabilities scoped to
       ``family acme-bsp``. Membership is exactly what this list says --
       never inferred from the tree -- and an unregistered tree has none.
       Lower-case names; a project may not itself be called ``family:...``
   * - ``build_commands``, ``test_commands``
     - how this tree is built and tested: the **project's own verification**.
       SPEAR runs them on the final tree of a change-making turn, and a failure
       there makes the turn ``UNVERIFIED`` (:ref:`evidence_verdicts`). A kind
       left undeclared is probed instead (CMake, a Makefile, cargo, ...), so a
       tree needs no configuration to be checked; a declared command always
       wins for its kind, and a declared build that fails is never replaced by
       a probe. Either kind of command can verify a change when it really runs
       and passes. A Makefile whose bare ``make`` only prints help, as a Sphinx
       one does, is not taken for a build. SPEAR runs the first entry of each
       list; the others are further forms that count as that kind of
       verification when the agent runs them. To require several checks,
       chain them into the first with ``&&``
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

The :term:`standard binding` itself is **not** a project key: it is machine-wide, set
with ``/standard use`` (:ref:`standards`).

.. important::

   What a corpus *does* is declared, not inferred from its ``kind``. A label
   that silently selected an indexer, a prompt and other behaviours would make
   corpora behave differently for no visible reason.

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
``ERROR`` — counts only for the :term:`source epoch` it ran on. A check that could not
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

A provision declared ``NOT_APPLICABLE`` leaves the :term:`constraint packet`. A
provision declared both ways is left unresolved.

What a build command proves
***************************

A project's build command proves what it exercises, on the tree it runs in.
Where that tree is a working copy derived from a source of truth kept
elsewhere -- a component fetched and patched by a build framework, with the
patches in git and the working copy not -- a build that passes shows the
working copy compiles, not that the change is kept: rebuilding from the
patches can lose it. Such a project covers both by chaining a check that the
patches reproduce the working copy after its build command; SPEAR's verdict
then reflects both, and its meaning is unchanged.

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

.. _two_roots:

Corpus and workspace are two roots
**********************************

``set_project`` binds two distinct roots, and the distinction is the whole
safety story:

``CORPUS_ROOT``
   The registered tree the retrieval index and the conversation history belong
   to — an *identity*.  Selected by the cwd, or forced with ``--corpus``.

``PROJECT_ROOT``
   Where the **tools** run: the real current directory, always.  It is the
   workspace root the execution :term:`harness` validates every filesystem path
   against (:doc:`/harness/security_model`).

Only the cwd moves ``PROJECT_ROOT``.  Nothing in a question does — not a
``--corpus`` flag, and certainly not a corpus name appearing in a sentence.
Letting natural language relocate the sandbox root would hand the workspace
boundary to the model's reading of a phrase, which is precisely what the
containment layers exist to prevent.

That has an ergonomic cost, so the harness pays it in information instead.
When a question names a *registered corpus other than the current one*, it says
so once and does nothing:

.. code-block:: text

   > generate a simple ping.c to run in so3
     ⎿  'so3' is a registered corpus (/srv/src/so3/so3), but your
        tools run in /home/me/spear/client. To work there:
        cd /srv/src/so3/so3 && spear-chat

Matching is on whole words, longest registered name first (so ``micropython-so3``
wins over ``so3``), once per name per session.  Names that are ordinary
directory words — ``src``, ``lib``, ``build``… — are never treated as mentions,
nor is a name used as a path fragment (``so3/usr/main.c``), nor a registry entry
whose tree has disappeared.

Selecting a corpus without moving the cwd is legitimate but asymmetric: the
model then reads about one tree and acts on another.  The harness warns it
explicitly with a *Working directory* note in the system prompt, telling it to
adapt the retrieved paths.  Prefer ``cd`` to ``--corpus`` when you intend to
edit.

Writable roots
**************

The launch directory is writable, and so is each registered corpus, mounted
at ``/workspaces/<name>``; which flags narrow or widen that is
:doc:`/harness/security_model`.

Indexing
********

.. code-block:: text

   /cd <dir>                 move the session's corpus to <dir>, inside the
                             launch directory
   /reindex                  rebuild the index for the current corpus
   /corpus                   list and manage the registry

``/cd`` moves the corpus, not the tools. Launched in ``opencn-ng``,
``/cd linux/linux`` makes the session's index, history, memories and rules
those of ``opencn-ng-linux``, and ``/reindex`` then rebuilds that tree; the
tools keep running in ``opencn-ng``, and retrieved paths come prefixed with
``linux/linux/`` so they open from there. The directory resolves as a launch
there would, except that a directory its corpus excludes is not part of it:
``/cd linux`` federates the corpora below ``linux/`` rather than falling back
on ``opencn-ng``. A target outside the launch directory is refused; ``/cd``
alone returns to it.

Indexing is not optional in practice. See :ref:`Retrieval <retrieval>` for what
it measurably buys, and :ref:`Troubleshooting <troubleshooting>` for the
symptoms of a missing index.

.. seealso::

   :ref:`Retrieval <retrieval>` · :ref:`Configuration reference
   <configuration>`
