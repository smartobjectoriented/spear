.. _security_model:

==============
Security model
==============

.. figure:: /img/SPEAR-Security.drawio.png
   :width: 100%
   :alt: Modes, capabilities and the authorization pipeline

   Capability matrix and the authorization pipeline.

Execution modes
===============

Role filtering happens before these modes are evaluated.  Planning, Explorer
and Reviewer receive registry-derived read-only tool views, no memory-write or
checkpoint capability, and ``SAFE`` command execution.  Registry metadata is
not an authorization substitute: every selected command and path still passes
through the capability, workspace, ``CommandRunner`` and Bubblewrap checks
described below.

``SAFE`` — ``spear-chat`` (no flag, or ``--safe``)
   The default.  Read-only work.  Nothing mutates, nothing reaches the
   network.  A write is refused outright; it is not proposed.

``ASK`` — ``--ask`` (aliases ``--confirm``, ``--no-bypass``)
   Mutation and network are possible, each behind an explicit confirmation.

``AUTO`` — ``--auto`` (aliases ``-y``, ``--yolo``, ``--bypass-permissions``)
   Mutation and network without confirmation.  ``--no-network`` takes the
   network away from any mode, the web tools included.

The flags pass straight through ``spear-chat`` to ``rag_chat.py``, and the
startup banner names the active mode plus the flags that reach the other two —
that line is the only place most users will ever read them.

.. _control_plane:

The control plane in front of the coding core
=============================================

The :term:`coding core` (:doc:`/reasoning/implementation`) never touches the host
itself: every read, write, deletion and command it asks for is a call on
:term:`SpearHost`, and SpearHost applies the same rules this page describes to all of
them, whichever tool asked.

**Workspace containment.** Every path — and every command's working directory
— is resolved, symbolic links included, and must land inside the workspace.
A link cannot carry a write outside it, and ``delete_file`` removes a link, not
what it points at.

**Source and write scope.** A request that names its target confines writes to
that target and what belongs to it: a sibling of the same family (another
board, another driver) is refused unless the request widened to the whole
family, and the coding core has no argument that lets a write past it. The
same judgement is applied to what a shell command writes — a redirection, the
destination of ``cp``, ``mv``, ``install``, ``ln`` or ``rsync``, the ``of=``
of ``dd``, the operands of ``rm``, ``touch``, ``mkdir``, ``tee``,
``truncate`` or ``sed -i``, and any path an inline interpreter program names —
with relative operands read from the directory the command runs in, a ``$VAR``
read as the value the command assigned it, ``x>>file`` read without its
spaces, and ``cp src dir/`` judged on ``dir/src`` (``request_scope.py``). **A terminal command cannot make a change the file
tools would refuse.**

**Project scope.** A command that searches outside the project (``find`` or
a recursive ``grep`` rooted elsewhere, ``locate``), or names a path of the
home directory outside it, is refused unless the operator named that path:
another checkout is another project, not a source of this one's build.

**Generated and protected trees.** Build outputs (``/generated/``,
``/build/tmp/``, files that say *DO NOT MODIFY*, *DO NOT EDIT*,
*auto-generated* or ``@generated`` in a comment within their first ten lines)
and snapshot copies (``.back`` and ``.pristine`` trees, a ``.0`` tree beside
its live one, never a directory because of its name) are never written; the
refusal points at the source to change instead. The same words in prose do not
count (Linux's ``init/Kconfig`` uses *auto-generated* in a help text), nor do
they in the ``+`` lines of a patch, and a saved Kconfig configuration
(``defconfig``, ``*_defconfig``) is a source even under its *Automatically
generated file; DO NOT EDIT.* header. The rule belongs to the target, not to
the tool (``target_policy.py``): ``write_file``, ``patch``, ``delete_file``
and every shell write listed above ask the same question, about the path as
written and about its ``realpath``, so a link or a ``..`` does not change the
answer.

The parser is not the boundary: a program may write what no word of its
command names. So for a command granted workspace write, the sandbox binds
back read-only the protected files the command can reach — those its words
name, those named in the files it runs or feeds (a diff given to ``patch``, a
script, the tests a runner discovers, a here-document) and every file refused
earlier in the session — and, unless the command builds or tests, the
generated and snapshot trees themselves (``write_protection.py``). A build
(``make``, ``cmake``, ``ninja``, ``bitbake``, a project's ``build.sh``,
``configure`` or a layer's ``updiff.sh``, a compiler, a test runner) keeps
writing its output, which is how generated files are meant to change; only
the files its own words name stay protected. The authority is the build's
alone: every other stage of the command must write nothing, and no
redirection or ``tee`` may write into a generated tree.

A launcher — ``timeout``, ``nice``, ``env``, ``nohup``, ``exec``,
``command``, ``time``, ``bash -c`` or ``sh -c`` — is judged by what it runs,
read through its options and, for a shell, through its script: a wrapped
build stays a build, and a wrapped write stays a write, refused the same.
A launcher this cannot read (an option it does not know, ``env -S``, words
after a ``-c`` script, a script the outer shell expands, a program that is
an expansion) is refused.

**Git metadata.** A repository's own state — a ``.git`` directory, a ``.git``
file (a worktree's or a submodule's) and the git directory it names, or any
directory holding git's ``HEAD`` beside its ``objects`` — is never edited by a
tool: the coding core's file tools, the legacy ones (``edit_file``,
``append_file`` and ``fetch_url``'s ``save_as`` included) and every shell
write are refused it (``target_policy.py``). Since a terminal command
may run a program whose writes no parser sees — a script, ``patch`` fed on
stdin, a Makefile rule — the sandbox also binds every ``.git`` and the git
directories they name read-only, on top of the writable tree, for every
command granted workspace write (``write_protection.py``, :doc:`sandbox`).
``git status``, ``log``, ``diff`` and ``show`` keep working.

Git itself keeps its records. ``add``, ``checkout``, ``restore``,
``switch``, ``stash``, ``clean``, ``fetch`` and ``pull`` find writable the
git directories of the repository they run in — followed through the
command's ``cd`` and ``-C`` steps, and through a ``.git`` file to the git
directory it names and that one's common directory, all canonically — and
no other (``write_protection.git_dirs``). Only when the command is git's
alone and run as itself (``CommandPolicy.git_writes``): every other stage
reads; nothing writes a file, substitutes, defines a function or sources a
script; no ``-c``, ``--git-dir``, ``--work-tree`` or interactive form
(``-p``, ``-i``, ``-e``); no ``GIT_*``, ``HOME``, ``PATH``, ``LD_*`` or
editor in the command or exported by the session; no directory outside the
roots or inside git's own state; no writable directory ahead of the system's
``git`` on ``PATH`` that holds a ``git`` or could before a second git
command runs. Otherwise the repository stays read-only and the git command
fails there. What git changes in the worktree is judged like any other
write: a protected file it names and the generated trees stay read-only
under it. ``reset``, ``commit``, ``config`` and the rest keep no records.

**Read-only and advisory turns.** A turn that only asks to be told something,
or is told not to change anything, keeps its reading tools and runs its
commands without workspace write — so the prohibition cannot be routed around
through the terminal.

**Network.** A command reaches the network only if the session's mode grants
it (below), through the sandbox's own network stack.

**What the model is told.** Only a ``denied`` result becomes a refusal: it
reaches the coding core as a one-line ``refused: …`` reason
(``control_plane.py``).  A command the operator declines in ``ASK`` comes back
``cancelled`` and is read as an interrupted command; one that cannot be
confined — the sandbox, ``prlimit`` or the cgroup mechanism unavailable — comes
back ``failed`` with its reason (:ref:`fail_closed`).  A turn that keeps
repeating a refused operation is stopped (:ref:`refusal_breaker`).

**External capabilities.** ``spear-capability`` and ``spear-knowledge`` are
host commands that SpearHost answers itself, never through a shell; a
capability runs only through the gateway's scope, schema and read/write checks,
and its MCP provider runs on the host, outside the Bubblewrap sandbox
(:doc:`/using/capabilities`, :doc:`/using/knowledge`).

**Audit and evidence.** Every call is audited, every mutation checkpointed for
``/undo``, and every call recorded as canonical evidence — what it changed,
what it ran, how that ended — from which the turn's verdict is computed
(:doc:`/reasoning/evidence`).

Capabilities
============

.. code-block:: python

   class Capability(StrEnum):
       FILESYSTEM_READ   = "filesystem:read"
       HOST_READ         = "host:read"
       WORKSPACE_WRITE   = "workspace:write"
       SHELL_COMPLEX     = "shell:complex"
       NETWORK           = "network"
       GPU               = "gpu"
       SSH               = "ssh"
       REMOTE_WRITE      = "remote:write"
       CONTAINER_RUNTIME = "container-runtime"
       SECRETS           = "secrets"

The default policy:

.. list-table::
   :header-rows: 1
   :widths: 40 20 20 20

   * - Capability
     - SAFE
     - ASK
     - AUTO
   * - ``filesystem:read``
     - yes
     - yes
     - yes
   * - ``host:read``
     - yes
     - **confirmed**
     - yes
   * - ``workspace:write``
     - no
     - yes
     - yes
   * - ``shell:complex``
     - **yes**
     - yes
     - yes
   * - ``network``
     - no
     - **confirmed**
     - **yes**

Why reading outside the workspace is a grant, not a hole
--------------------------------------------------------

``host:read`` lets a read-only command name an absolute path that no declared
root contains.  Refusing them would make whole questions unanswerable — "read
the notes in ``/srv/notes``" — while buying no containment, because looking at
a file is not changing it.

The grant is read-only *by construction*: the vetted paths are exposed
read-only (``--ro-bind-try``, so a file gone in between reports ``ENOENT``)
and bound **before** the workspace mounts, so a declared tree
nested under one of them keeps the write access the workspace gives it.  A
command that writes there meets ``EROFS``, which is the truthful error.

Three things stay refused, and each is checked against both the literal
spelling and the resolved one, so a symlink is not a way round:

* **credential stores** — ``.ssh``, ``.gnupg``, ``.aws``, ``.docker``,
  ``.kube``, ``.netrc``, ``.pgpass``, ``.git-credentials``, ``id_*``,
  ``shadow``, ``gshadow``, ``sudoers``.  The model is served over the network: a file read here is a
  file sent there.
* **the sandbox's own mount points** (``/proc``, ``/dev``, ``/usr``,
  ``/etc``, the sandbox ``$HOME`` and ``/tmp``, ``/workspace``) and any
  ancestor of one — binding ``/home`` would land on top of the sandbox's
  ``$HOME``.
* **relative ``..`` traversal** — the classifier cannot know which directory a
  shell stage resolved against, so the same string may denote two files.  The
  refusal says to name the path absolutely instead.

``host:read`` is listed as sensitive, so ``ASK`` confirms each command that
leaves the declared trees, and every such read is recorded in the audit trail
even though it is read-only.

Why SAFE may run a pipeline
---------------------------

``SAFE`` grants ``shell:complex`` because what makes ``SAFE`` safe is the
**mount**, not the classifier: without ``workspace:write`` every root is
bind-mounted read-only, so a pipeline physically cannot write.  Refusing
``find … | head`` while allowing ``find … -type f`` would buy nothing, and an
assistant that cannot pipe cannot search a tree.

Writing is still refused, twice over: a redirection or a writing binary
declares ``workspace:write``, which ``SAFE`` does not grant, and the read-only
mount would refuse it regardless.  ``tee`` is listed among the mutating
binaries for exactly that reason — it writes a file with no redirection
operator, so the ``>`` heuristic never sees it — while a redirection to
``/dev/null`` demands no write access, since it writes nothing.

Why AUTO has the network, and how to take it away
-------------------------------------------------

``AUTO`` is ``ASK`` without the prompt, so it cannot grant *less* than
``ASK``: network access, whose consequences leave the machine and cannot be
reviewed afterwards from the workspace diff, is confirmed in ``ASK`` and
granted in ``AUTO``.

``--no-network`` is the way to an offline session, and it means the whole
session: ``CapabilityPolicy.without_network()`` drops the capability from every
mode, and ``rag_chat`` also withholds ``search_internet`` and ``fetch_url``.
Those two reach the internet from the chat process, outside the capability
policy entirely, so a flag that only emptied the execution modes would have
read as offline without being it.

Reading a page needs no capability at all: ``fetch_url`` is a native tool, so
it works in ``SAFE``.  Its boundary is written in ``web_fetch.py`` instead —
http/https only, public addresses re-checked at every redirect, a byte cap
enforced while streaming.  Saving one to disk is a mutation and goes through
the ordinary write authorization, so ``SAFE`` refuses it.

Declared but not implemented
----------------------------

``gpu``, ``ssh``, ``container-runtime`` and ``secrets`` are named in the enum
but have no implementation.  ``build_argv()`` raises before execution:

.. code-block:: python

   unsupported = profile.capabilities & self._UNIMPLEMENTED_PROFILE_CAPABILITIES
   if unsupported:
       raise ValueError(f"capability/profile not implemented: {names}")

Naming them without implementing them is the point: a future profile that
requests one fails loudly at the boundary instead of silently receiving a
sandbox that does not actually grant it.

Command classification
======================

``CommandPolicy.classify()`` maps a command to one of the classes below
(``classify_script()`` judges a coding-core ``terminal`` command, which always
runs as a bash script, as that script); ``CommandPolicy.authorize()`` then
checks the class and its required capabilities against the mode:

``READ_ONLY``
   Inspection only.  Available in every mode.

``WORKSPACE_MUTATING``
   Writes inside the workspace.  Needs ``workspace:write``.

``SHELL_COMPLEX``
   Needs shell syntax — pipes, redirections, substitutions.  Needs
   ``shell:complex``, and is executed as an explicit ``bash -lc <script>``
   argv inside the sandbox (``/bin/sh`` where there is no bash), behind a
   ``set -o pipefail`` prelude so a failed stage is not masked by the last
   one.  The :term:`harness` never uses ``shell=True``.

``DANGEROUS``
   Refused.

Two rules about redirections are worth stating, because getting them wrong is
expensive in both directions:

* ``<`` and ``>`` introduce a **file**, not a command.  Read as a command
  stage, ``/dev/null`` would trip the "argv[0] contains a slash" rule and
  refuse ``ls 2>/dev/null`` — the most common idiom in shell — as a
  *dangerous shell stage*; an assistant that cannot suppress stderr cannot
  search a tree.
* The redirection **target is still checked, as a path**.  An absolute target,
  or one escaping the workspace, stays dangerous; ``/dev/null``,
  ``/dev/stdout`` and ``/dev/stderr`` are the documented exceptions, since they
  discard rather than write.

The workspace boundary
======================

``Workspace.resolve()`` is the single gate for every filesystem tool path.

Roots
-----

The boundary is a **set of roots**, not a single directory.

*Primary root* — the launch directory.  Relative paths resolve there and
nowhere else, so one string never denotes two files depending on the root list.

*Extra roots* — the corpora registered in ``projects.json``, and the build
tree that encloses the launch directory when there is one (recognised by its
``env.sh`` and ``scripts/build.sh``).  They are declared so a file in another
tree can be edited, or a component built from its umbrella, without
relaunching; ``--single-root`` drops them and restores the
launch-directory-only boundary.  Because widening
the write boundary must never be silent, the startup banner names the count.

Each root is bind-mounted in the sandbox at its own host path, so an
absolute path means the same file inside and outside (:doc:`sandbox`); where
that path is unsafe to mirror, or with ``SPEAR_SANDBOX_IDENTITY_MOUNT=0``, the
primary falls back to ``/workspace`` and the others to ``/workspaces/<name>``.
Every root has the same access as the primary (read only in ``SAFE``,
read-write otherwise).  Roots are canonicalised,
deduplicated, and never nested — a nested root would be mounted twice and make
a path's label ambiguous.  A registry entry whose tree has disappeared is
dropped rather than fatal.

**One addressing scheme.**  A mount path — the host path itself, or the
``/workspace`` and ``/workspaces/<name>/…`` names — is accepted by the shell
*and* by the file tools, which act on the host: the mount path is translated
back to its host root before containment is checked.  Two schemes
that silently disagree would be a guaranteed source of wrong paths.  Host
absolute paths into the launch directory remain gated by
``--allow-absolute-paths``; declaring extra roots does not widen what the
launch directory accepts.

Audit labels qualify a secondary root — ``so3:usr/src/ping.c`` rather than a
bare relative path that would read like a file of the launch directory.

The command classifier follows the same rule: an argument or redirection target
under a mount is not an escape, while ``/etc/passwd``, ``/workspace-evil`` and
anything traversing out of a mount (``/workspace/../etc``) stay dangerous.

.. code-block:: python

   candidate, from_mount = self._from_mount_path(Path(value).expanduser())
   resolved = (candidate if candidate.is_absolute() else self.root / candidate).resolve(strict=False)
   containing = self._containing_root(resolved)      # None if under no root

*Symlink escape.*
   ``Path.resolve()`` follows every existing symlink component, **including
   the parent of a file that does not yet exist**.  Checking containment on
   the resolved path is what stops a write to ``workspace/link-to-etc/passwd``
   — from every root, not only the primary one.

Audit
=====

``AuditLogger.record_mutation()`` appends a metadata-only record for mutating
attempts.  What is deliberately **not** recorded: file contents, command
output, environment values, bus addresses, cgroup paths, unit names.

Two redactions happen before anything is written:

.. code-block:: python

   _SECRET_ASSIGNMENT = re.compile(r"(?i)\b(api[_-]?key|token|password|secret)\s*=\s*[^\s]+")

and leading ``KEY=value`` environment assignments are stripped from the argv
before the executable summary is derived, so ``FOO=secret make`` is summarised
as ``make``, not as something containing ``secret``.

.. _fail_closed:

The fail-closed rule in practice
================================

.. admonition:: There is no degraded mode
   :class: important

   Every one of these returns a ``failed`` ``ToolResult`` and spawns nothing:

   * ``bwrap`` absent or not executable;
   * ``bwrap`` refused by the kernel at preflight;
   * ``prlimit`` absent while resource limits are active;
   * ``systemd-run`` or ``systemctl`` absent while cgroup limits are active
     (unless an outer container runtime owns the cgroup,
     :doc:`resource_control`);
   * the systemd user bus unavailable;
   * a required cgroup controller not delegated;
   * ``slirp4netns`` absent, or unable to attach through pinned namespaces;
   * a previous network containment failure (poisoning).

   None of them falls back to a less confined execution.

The tests assert this negatively — ``popen.assert_not_called()`` — because
"the command did not run" is the property that matters, and it is easy to
satisfy accidentally with a test that only checks the returned status.
