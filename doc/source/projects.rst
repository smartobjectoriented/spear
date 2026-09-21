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
   * - ``build_commands``, ``test_commands``, ``lint_commands``, ``acceptance_commands``
     - the commands the agent runs to check its own work

.. important::

   What a corpus *does* is declared, not inferred from its ``kind``. A label
   that silently selected an indexer, a prompt and three other behaviours was
   the source of corpora that behaved differently for no visible reason.

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
