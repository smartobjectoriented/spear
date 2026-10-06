.. _implementation_mode:

Implementation mode
###################

A request to change, build or inspect a working tree — with no standard
engaged — is an **IMPLEMENTATION** request. It runs on the **coding core**, a
standalone agentic loop, behind **SpearHost**, the control plane that decides
what each of its calls may do and records what each one did. Requests that
name no subject at all (**GENERAL**) take the same path.

.. code-block:: text

   request ──► coding core ──► SpearHost ──► sandbox / workspace
                   ▲               │
                   └── results ◄───┴──► canonical execution evidence
                                               │
                                               ▼
                                implementation evidence (the verdict)

The split is the point. The coding core decides *what to try next*; it does not
decide whether it may, and it does not decide whether it succeeded. SpearHost
holds the first, and the evidence plane (:ref:`evidence_verdicts`) holds the
second.

The coding core
***************

The coding core is a conventional tool-calling loop: the model reads, searches,
edits and runs commands until it answers. It sees exactly six tools:

.. list-table::
   :header-rows: 1
   :widths: 22 30 48

   * - Tool
     - Arguments
     - What it does
   * - ``read_file``
     - ``path``, ``offset`` (1), ``limit`` (2000)
     - a numbered window of a file; a binary file is described, not dumped
   * - ``search_files``
     - ``pattern``, ``target`` (``content`` or ``files``), ``path``,
       ``file_glob``, ``limit``, ``offset``, ``output_mode``, ``context``
     - search file contents or file names, ripgrep-backed
   * - ``patch``
     - ``path``, ``old_string``, ``new_string``, ``replace_all``
     - replace an exact (or whitespace-tolerant) occurrence; ambiguity is
       refused rather than guessed
   * - ``write_file``
     - ``path``, ``content``
     - create or overwrite a file
   * - ``delete_file``
     - ``path``, ``reason``
     - delete one file or one symbolic link — the link itself, never what it
       points at
   * - ``terminal``
     - ``command``, ``timeout`` (180 s, at most 600 s), ``workdir``
     - run a shell command in the sandbox; output is capped at 50 000
       characters

Nothing else is offered on an implementation turn: no standard tools, no
retrieval tool, no web tool, no memory tool. A tool the model cannot see is a
tool it cannot misuse, and every schema left out is context the model keeps.

The loop ends when the model answers without calling a tool, when the round or
tool budget runs out (``--max-tool-rounds``, ``--max-commands``), or when the
conversation reaches half of the context window — the loop does not compact; it
stops and asks for a summary instead. The core sets its own output reservation
and sends no sampling parameters of its own, so ``--temp`` and ``--max-tokens``
apply to the other paths, not to this one.

.. note::

   The coding core talks to an **OpenAI-compatible** endpoint. The Anthropic
   backend serves the normative and general paths; it does not yet implement
   the message interface the coding core uses.

The coding core's request is built from the project's own rules, memories and
skills (:ref:`projects`), not from the retrieval corpus: an implementation turn
reads the tree it is changing, through its tools.

SpearHost — the execution boundary
**********************************

Every call the coding core makes crosses SpearHost, whichever tool made it:

``authorize``
   the session's permission mode (``--safe``, ``--ask``, ``--auto``), read-only
   and advisory turns, the write scope of the request, and the command policy.

``resolve``
   every path — read, write, delete, or a command's ``workdir`` — must resolve
   inside the workspace. Symbolic links are followed before the check, so a
   link cannot point a write outside it.

``write`` / ``delete``
   the generated-file guard, the snapshot-tree guard, the confirmation the mode
   requires, and the checkpoint that makes ``/undo`` possible.

``run_command``
   the command policy (what may run, what it may write, whether it may reach
   the network) and the sandbox.

``after_tool``
   the audit record and the canonical evidence of the call.

A refusal comes back to the model as a one-line reason in the coding core's own
vocabulary (``refused: …``), so the model is pointed at a tool it actually has.

Scope: what a turn may change
=============================

**Workspace containment.** The workspace is the launch directory, plus — unless
``--single-root`` is given — the registered corpora, each at
``/workspaces/<name>``. A path that resolves outside it is refused, for reading
as for writing.

**Write scope.** A request that names a target confines writes to that target
and what genuinely belongs to it; a sibling file of the same family is not
written because it resembles the one asked for.

**Terminal writes are held to the same scope.** A shell command's destinations
— a redirection, the target of ``cp``, ``mv`` or ``install``, an in-place
``sed`` — are judged exactly as a ``write_file`` to the same path would be. A
change the file tools would refuse cannot be made through the terminal instead.

**Protected and generated paths.** Files a build produces (``/generated/``,
``/build/tmp/``, or a file whose head says *DO NOT MODIFY* or
*auto-generated*) are refused with a pointer to their source. Snapshot and
third-party copies (``.back``, ``.pristine``, ``.0``, vendored ``u-boot``,
``atf``, ``qemu`` trees) are never written.

**Advisory and read-only turns.** A turn that asks only to be told something,
or is told not to change anything, keeps its reading tools and loses every
write — including the terminal's ability to write.

Terminal working directory
==========================

Where a ``terminal`` call runs is decided in this order:

#. an explicit ``workdir`` argument — resolved, when relative, from the
   directory the previous command finished in;
#. otherwise the directory the terminal session last recorded (a ``cd`` in an
   earlier command is remembered, as in an interactive shell);
#. otherwise the workspace root.

A ``workdir`` outside the workspace is refused before anything runs. Commands
run in the Bubblewrap sandbox (:ref:`sandbox`); network access follows the
session's mode — granted in ``--ask`` and ``--auto``, never in ``--safe``, and
removed everywhere by ``--no-network`` (:ref:`network`).

Canonical execution evidence
============================

Each call is recorded as structured evidence from the tool's own result —
which files it changed, which command it ran, how that command ended, whether
it was refused or timed out — never from the text the model was shown. The
verdict at the end of the turn (:ref:`evidence_verdicts`) is computed from
this record alone.

Example
*******

A synthetic project; the request names a target and how it will be proved.

.. code-block:: text

   > Fix the off-by-one in ring_next() in src/ring.c, then build it.

     Read     src/ring.c
     Update   src/ring.c
     Terminal make

   ring_next() wrapped at size + 1; it now wraps at size. `make` succeeds.

The turn changed ``src/ring.c`` and ran a build *after* the change, so its
implementation evidence is ``VERIFIED``. Had the turn edited the file again
after ``make``, the same answer would be reported ``UNVERIFIED``: the build
proved a tree that no longer exists. :ref:`evidence_verdicts` explains why.

.. seealso::

   :ref:`evidence_verdicts` · :ref:`security_model` · :ref:`tool_harness`
