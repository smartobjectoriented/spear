.. _security_model:

==============
Security model
==============

.. figure:: /img/spear_security.svg
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

``SAFE`` — ``spear-chat`` (no flag)
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
root contains.  Refusing them made whole questions unanswerable — "read the
notes in ``/opt/llm/claude``" died on ``path argument may escape the
workspace``, a refusal the model could not act on — while buying no
containment, because looking at a file is not changing it.

The grant is read-only *by construction*: the vetted paths are exposed
``--ro-bind`` and bound **before** the workspace mounts, so a declared tree
nested under one of them keeps the write access the workspace gives it.  A
command that writes there meets ``EROFS``, which is the truthful error.

Three things stay refused, and each is checked against both the literal
spelling and the resolved one, so a symlink is not a way round:

* **credential stores** — ``.ssh``, ``.gnupg``, ``.aws``, ``.docker``,
  ``.kube``, ``.netrc``, ``.git-credentials``, ``id_*``, ``shadow``,
  ``sudoers``.  The model is served over the network: a file read here is a
  file sent there.
* **the sandbox's own mount points** and any ancestor of one — binding
  ``/home`` would land on top of the sandbox's ``$HOME``.
* **relative ``..`` traversal** — the classifier cannot know which directory a
  shell stage resolved against, so the same string may denote two files.  The
  refusal now says to name the path absolutely instead.

``host:read`` is listed as sensitive, so ``ASK`` confirms each command that
leaves the declared trees, and every such read is recorded in the audit trail
even though it is read-only.

Why SAFE may run a pipeline
---------------------------

``SAFE`` grants ``shell:complex`` because what makes ``SAFE`` safe is the
**mount**, not the classifier: without ``workspace:write`` every root is
bind-mounted read-only, so a pipeline physically cannot write.  Refusing
``find … | head`` bought nothing — ``find … -type f`` was allowed, adding
``| head`` was not — and an assistant that cannot pipe cannot search a tree.

Writing is still refused, twice over: a redirection or a writing binary
declares ``workspace:write``, which ``SAFE`` does not grant, and the read-only
mount would refuse it regardless.  ``tee`` is listed among the mutating
binaries for exactly that reason — it writes a file with no redirection
operator, so the ``>`` heuristic never sees it — while a redirection to
``/dev/null`` no longer demands write access, since it writes nothing.

Why AUTO has the network, and how to take it away
-------------------------------------------------

``AUTO`` used to withhold it while ``ASK`` granted it, on the argument that
network access is the one capability whose consequences leave the machine and
cannot be reviewed afterwards from the workspace diff.  The argument is sound
and the placement was not: ``AUTO`` is ``ASK`` without the prompt, so it cannot
grant *less* than ``ASK``.  The result was that ``-y`` — reached for precisely
to stop being blocked — was the one mode that still refused ``curl``, while the
banner advertised it as the permissive one.

``--no-network`` is the way to an offline session, and it now means the whole
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

``CommandPolicy.assess()`` maps an argv to one of:

``READ_ONLY``
   Inspection only.  Available in every mode.

``WORKSPACE_MUTATING``
   Writes inside the workspace.  Needs ``workspace:write``.

``SHELL_COMPLEX``
   Needs shell syntax — pipes, redirections, substitutions.  Needs
   ``shell:complex``, and is executed as an explicit ``/bin/sh -lc <script>``
   argv inside the sandbox.  The harness never uses ``shell=True``.

``DANGEROUS``
   Refused.

Two rules about redirections are worth stating, because getting them wrong is
expensive in both directions:

* ``<`` and ``>`` introduce a **file**, not a command.  Treating the token
  after them as a command stage made ``/dev/null`` trip the "argv[0] contains a
  slash" rule, so every command carrying ``2>/dev/null`` — the most common
  idiom in shell — was refused as a *dangerous shell stage*, ``ls 2>/dev/null``
  included.  An assistant that cannot suppress stderr cannot search a tree, and
  a real session was spent discovering that.
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

*Extra roots* — the corpora registered in ``projects.json``.  They are declared
so a file in another tree can be edited without relaunching; ``--single-root``
drops them and restores the launch-directory-only boundary.  Because widening
the write boundary must never be silent, the startup banner names the count.

Each root is bind-mounted in the sandbox: the primary at ``/workspace``, the
others at ``/workspaces/<name>``, with the same access as the primary (read
only in ``SAFE``, read-write otherwise).  Roots are canonicalised,
deduplicated, and never nested — a nested root would be mounted twice and make
a path's label ambiguous.  A registry entry whose tree has disappeared is
dropped rather than fatal.

**One addressing scheme.**  ``/workspaces/<name>/…`` is accepted by ``bash``
*and* by ``edit_file`` / ``write_file``, which act on the host: the mount path
is translated back to its host root before containment is checked.  Two schemes
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

The fail-closed rule in practice
================================

.. admonition:: There is no degraded mode
   :class: important

   Every one of these returns a ``failed`` ``ToolResult`` and spawns nothing:

   * ``bwrap`` absent or not executable;
   * ``bwrap`` refused by the kernel at preflight;
   * ``prlimit`` absent while resource limits are active;
   * ``systemd-run`` or ``systemctl`` absent while cgroup limits are active;
   * the systemd user bus unavailable;
   * a required cgroup controller not delegated;
   * ``slirp4netns`` absent, or unable to attach through pinned namespaces;
   * a previous network containment failure (poisoning).

   None of them falls back to a less confined execution.

The tests assert this negatively — ``popen.assert_not_called()`` — because
"the command did not run" is the property that matters, and it is easy to
satisfy accidentally with a test that only checks the returned status.
