"""Small, dependency-free safety primitives for SPEAR tool execution.

This module deliberately does not import ``rag_chat``.  It can be tested in
isolation and integrated into the existing harness incrementally.
"""

from __future__ import annotations

import atexit
import fcntl
import json
import os
import re
import select
import shlex
import shutil
import signal
import stat
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import AbstractSet, Callable, Literal, Mapping, Sequence


class ExecutionMode(StrEnum):
    SAFE = "safe"
    ASK = "ask"
    AUTO = "auto"


class CommandClassification(StrEnum):
    READ_ONLY = "read_only"
    WORKSPACE_MUTATING = "workspace_mutating"
    SHELL_COMPLEX = "shell_complex"
    DANGEROUS = "dangerous"


class Capability(StrEnum):
    FILESYSTEM_READ = "filesystem:read"
    #: Read a tree the workspace does not contain.  Looking at a file is not
    #: changing it, and refusing every path outside the declared roots made
    #: whole questions unanswerable -- "read the notes in /opt/llm/claude" died
    #: on a refusal the model could not act on.  The grant is read-only by
    #: construction: those paths are bind-mounted ``--ro-bind``, so the write
    #: boundary is exactly where it was.
    HOST_READ = "host:read"
    WORKSPACE_WRITE = "workspace:write"
    SHELL_COMPLEX = "shell:complex"
    NETWORK = "network"
    GPU = "gpu"
    SSH = "ssh"
    REMOTE_WRITE = "remote:write"
    CONTAINER_RUNTIME = "container-runtime"
    SECRETS = "secrets"


class NetworkBackend(StrEnum):
    CLOSED = "closed"
    SLIRP4NETNS = "slirp4netns"


@dataclass(frozen=True)
class CapabilityPolicy:
    """Immutable per-mode capability boundary."""

    safe: frozenset[Capability]
    ask: frozenset[Capability]
    auto: frozenset[Capability]

    def for_mode(self, mode: ExecutionMode) -> frozenset[Capability]:
        if mode == ExecutionMode.SAFE:
            return self.safe

        if mode == ExecutionMode.ASK:
            return self.ask

        return self.auto

    def without_network(self) -> "CapabilityPolicy":
        """The same policy with the network taken out of every mode that has
        it. It used to take it out of ASK only, which was enough while ASK was
        the one mode that granted it — the day AUTO did too, `--no-network`
        would have become a flag that quietly did nothing in the very mode
        where it matters most.
        """
        return CapabilityPolicy(self.safe - {Capability.NETWORK},
                                self.ask - {Capability.NETWORK},
                                self.auto - {Capability.NETWORK})


DEFAULT_CAPABILITY_POLICY = CapabilityPolicy(
    # SAFE grants shell:complex because what makes SAFE safe is the MOUNT, not
    # the classifier: without workspace:write every root is bind-mounted
    # read-only, so a pipeline physically cannot write. Refusing
    # `find … | head` bought nothing and cost the assistant its ability to
    # search a tree — `find … -type f` worked, adding `| head` did not.
    # host:read is granted in every mode -- SAFE included, where reading is the
    # entire point -- and is listed as sensitive, so ASK still confirms each
    # command that leaves the declared trees.
    safe=frozenset({
        Capability.FILESYSTEM_READ,
        Capability.HOST_READ,
        Capability.SHELL_COMPLEX,
    }),
    ask=frozenset({
        Capability.FILESYSTEM_READ,
        Capability.HOST_READ,
        Capability.WORKSPACE_WRITE,
        Capability.SHELL_COMPLEX,
        Capability.NETWORK,
    }),
    # AUTO is ASK without the prompt, so it cannot grant LESS than ASK. It did:
    # network was in ask and not in auto, so `-y` — reached for precisely to
    # stop being blocked — kept refusing `curl`, and the banner ("--auto to
    # stop asking") told the user it was the permissive one. Unlike SAFE's
    # omission, which is commented and asserted, nothing defended this one.
    # `--no-network` is the way to an offline AUTO, and it now works there.
    auto=frozenset({
        Capability.FILESYSTEM_READ,
        Capability.HOST_READ,
        Capability.WORKSPACE_WRITE,
        Capability.SHELL_COMPLEX,
        Capability.NETWORK,
    }),
)


# ``NS_GET_USERNS`` (``_IO(0xb7, 0x1)``) returns a descriptor for the user
# namespace that *owns* a namespace descriptor.  Deriving the owner from the
# netns object is what makes slirp attachment independent of timing: bwrap
# replaces its child's ``/proc/<pid>/ns/user`` a few milliseconds after
# reporting the info-fd JSON, but the netns owner never changes.

NS_GET_USERNS = 0xB701

# ``NS_GET_NSTYPE`` (``_IO(0xb7, 0x3)``) reports which kind of namespace a
# descriptor refers to; used as a cheap defence-in-depth check before the
# handles are handed to the network helper.

NS_GET_NSTYPE = 0xB703
CLONE_NEWNET = 0x40000000
CLONE_NEWUSER = 0x10000000


class SandboxAvailability(StrEnum):
    """Observed Bubblewrap state, populated by :meth:`BubblewrapSandbox.preflight`."""

    UNKNOWN = "unknown"
    AVAILABLE = "available"
    ABSENT = "absent"
    INEXECUTABLE = "inexecutable"
    REFUSED = "refused"


class CgroupAvailability(StrEnum):
    """Observed state of the transient-scope resource-control mechanism."""

    UNKNOWN = "unknown"
    AVAILABLE = "available"
    SYSTEMD_RUN_ABSENT = "systemd_run_absent"
    DELEGATED = "delegated"
    SYSTEMCTL_ABSENT = "systemctl_absent"
    USER_BUS_UNAVAILABLE = "user_bus_unavailable"
    CONTROLLERS_UNAVAILABLE = "controllers_unavailable"
    REFUSED = "refused"


ResultStatus = Literal[
    "ok", "denied", "cancelled", "invalid_path", "not_found", "timeout", "failed"
]


@dataclass(frozen=True)
class ToolResult:
    """Canonical internal representation for a tool operation result."""

    status: ResultStatus
    summary: str
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    changed_paths: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_legacy_text(self) -> str:
        """Render the pre-Phase-1 string protocol at the UI/model boundary."""

        if self.status == "ok":
            prefix = "OK"
        elif self.status == "cancelled":
            return "CANCELLED"
        else:
            prefix = "ERROR"

        pieces = [f"{prefix}: {self.summary}"]

        if self.stdout:
            pieces.append(self.stdout)

        if self.stderr:
            pieces.append(self.stderr)

        if self.exit_code not in (None, 0):
            pieces.append(f"(exit {self.exit_code})")

        return "\n".join(pieces)


class PathPolicyError(ValueError):
    pass


class PathNotFoundError(PathPolicyError):
    pass


# Shell used for the SHELL_COMPLEX form, and why it is not simply /bin/sh:
# `set -o pipefail`. Without it a pipeline reports the exit status of its LAST
# stage only, so `make 2>&1 | tail -20` -- the shape a model reaches for to
# truncate a long build log -- exits 0 on a failed build. The failure is then
# invisible to the harness AND to the model, which reads exit 0 and reports
# success. bash has pipefail; dash, the /bin/sh of Debian-family servers, does
# not, so the shell is chosen rather than assumed.

SHELL_BINARY = "/bin/bash" if os.access("/bin/bash", os.X_OK) else "/bin/sh"

# Probed in a subshell before being set for real: `set` is a special builtin,
# and a non-interactive dash EXITS on `set -o pipefail` instead of ignoring it.
# Probing in a subshell means an unsupporting shell loses pipefail rather than
# losing the command. Kept on one line so error messages that quote a line
# number still agree with the command the model wrote.

PIPEFAIL_PRELUDE = "if (set -o pipefail) 2>/dev/null; then set -o pipefail; fi; "


def shell_argv(command: str, login: bool = True) -> "list[str]":
    """argv for running `command` in a shell, with pipeline failures visible."""
    return [SHELL_BINARY, "-lc" if login else "-c", PIPEFAIL_PRELUDE + command]


def effective_mount_root(root: "Path | str",
                         spec: "SandboxSpec | None" = None) -> str:
    """Where a workspace rooted at ``root`` is exposed inside the sandbox.

    The prompt must tell the model the same path the sandbox actually binds,
    so both go through this one function. SPEAR_SANDBOX_IDENTITY_MOUNT=0
    restores the historical /workspace mount for debugging.
    """
    spec = spec or SandboxSpec()

    if os.environ.get("SPEAR_SANDBOX_IDENTITY_MOUNT", "1") == "0":
        return spec.workspace_mount

    return identity_mount_for(root, (spec.home, spec.tmpdir)) or spec.workspace_mount


@dataclass(frozen=True)
class Workspace:
    """A canonical root that all filesystem tool paths must stay within."""

    root: Path
    allow_absolute_paths: bool = False
    #: Additional writable trees, declared by the caller.  The launch directory
    #: stays the *primary* root: relative paths resolve there and nowhere else,
    #: so one string never denotes two files.  A secondary root is reached by
    #: absolute path only — which is also what makes the intent explicit in the
    #: audit trail.
    extra_roots: tuple[Path, ...] = ()

    @classmethod
    def from_path(cls, path: str | Path, *, allow_absolute_paths: bool = False,
                  extra_roots: Sequence[str | Path] = ()) -> "Workspace":
        root = Path(path).expanduser().resolve(strict=True)

        if not root.is_dir():
            raise PathPolicyError(f"workspace is not a directory: {root}")

        return cls(root=root, allow_absolute_paths=allow_absolute_paths,
                   extra_roots=cls._canonical_extra_roots(root, extra_roots))

    @staticmethod
    def _canonical_extra_roots(root: Path,
                               candidates: Sequence[str | Path]) -> tuple[Path, ...]:
        """Canonical, deduplicated, non-nested, existing directories.

        A root nested in another would be mounted twice and make a path's label
        ambiguous.  A root that no longer exists is dropped rather than fatal:
        a stale entry in a corpus registry must not stop the assistant from
        starting.
        """
        kept: list[Path] = []

        for candidate in candidates:
            try:
                resolved = Path(candidate).expanduser().resolve(strict=True)
            except OSError:
                continue

            if not resolved.is_dir():
                continue

            if resolved == root or resolved.is_relative_to(root):
                continue

            if any(resolved.is_relative_to(seen) for seen in kept):
                continue

            kept = [seen for seen in kept if not seen.is_relative_to(resolved)]
            kept.append(resolved)

        return tuple(kept)

    @property
    def roots(self) -> tuple[Path, ...]:
        """Every writable tree, primary first."""
        return (self.root,) + self.extra_roots

    def boundary_hint(self, *, readable_elsewhere: bool = True) -> str:
        """Name the trees a refusal leaves available, and the way out of it.

        A refusal that only states the rule is a dead end: the model re-ran the
        same denied command, then narrated "let me try a relative path" and
        issued the same absolute one. What it lacked was the two facts this
        sentence carries -- which trees it may write, and that looking at a
        file elsewhere is a different, permitted thing.
        """
        shown = [str(root) for root in self.roots[:3]]
        more = len(self.roots) - len(shown)
        listed = ", ".join(shown) + (f" (+{more} more)" if more > 0 else "")
        hint = f" Writable trees: {listed}."

        if readable_elsewhere:
            hint += (" Anywhere else is READ-ONLY and reachable from bash by "
                     "absolute path (cat/sed/ls/grep), not from the file tools."
                     " To make a tree writable, declare it as a corpus.")

        return hint

    def mount_map(self, primary_mount: str = "/workspace",
                  identity: bool = False) -> dict[str, str]:
        """Host root → path it is exposed at inside the sandbox.

        The model cannot use a secondary tree from bash without this: on the
        host it is /home/…/so3, inside the sandbox it is /workspaces/so3.
        Names are made unique because two roots may share a basename.
        """

        if identity:
            # Every root keeps its own path; nothing has to be renamed, so the
            # uniquifying below is unnecessary and the map is the identity.

            out = {str(self.root): primary_mount}

            for extra in self.extra_roots:
                mount = effective_mount_root(extra)
                out[str(extra)] = (mount if mount != SandboxSpec().workspace_mount
                                   else f"{primary_mount}s/{extra.name}")

            return out

        mounts = {str(self.root): primary_mount}
        used: set[str] = set()

        for extra in self.extra_roots:
            name, suffix = extra.name, 2

            while name in used:
                name = f"{extra.name}-{suffix}"
                suffix += 1

            used.add(name)
            mounts[str(extra)] = f"{primary_mount}s/{name}"

        return mounts

    def _from_mount_path(self, candidate: Path) -> tuple[Path, bool]:
        """Translate a sandbox mount path back to its host root.

        bash sees a secondary root at /workspaces/<name>; the file tools act on
        the host. Accepting the mount path everywhere gives the model a single
        addressing scheme instead of two that silently disagree. Translation is
        done BEFORE containment, so `.. ` out of a mount is still caught.
        """

        if not candidate.is_absolute():
            return candidate, False

        # Both namings are accepted: the /workspace literals, and the mounts
        # actually in force. Under the identity mount the latter are the host
        # paths themselves — which the prompt now tells the model to use, so
        # refusing them here would put bash and edit_file back in disagreement.

        effective = effective_mount_root(self.root)
        pairs = list(self.mount_map().items())

        if effective != SandboxSpec().workspace_mount:
            # Both maps are keyed by host root, so they are chained as pairs:
            # merging the dicts would drop one naming for the other.

            pairs += list(self.mount_map(effective, identity=True).items())

        for host, mount in pairs:
            mount_path = Path(mount)

            if candidate == mount_path:
                return Path(host), True

            if candidate.is_relative_to(mount_path):
                return Path(host) / candidate.relative_to(mount_path), True

        return candidate, False

    def _containing_root(self, resolved: Path) -> Path | None:
        for candidate in self.roots:
            if resolved.is_relative_to(candidate):
                return candidate

        return None

    def resolve(self, raw_path: str | Path, *, must_exist: bool = False) -> Path:
        """Resolve a user path and reject traversal, symlink, or root escapes.

        ``Path.resolve(strict=False)`` follows every existing symlink component,
        including the parent of a target file that does not yet exist.  Checking
        the resulting canonical path before opening it prevents symlink escapes.
        """
        value = str(raw_path).strip()

        if not value:
            raise PathPolicyError("path is empty")

        candidate, from_mount = self._from_mount_path(Path(value).expanduser())
        resolved = (candidate if candidate.is_absolute() else self.root / candidate).resolve(
            strict=False
        )

        if candidate.is_absolute():
            # An absolute path is how a secondary root is reached at all.  Into
            # the PRIMARY root it stays gated by the legacy flag, so declaring
            # extra roots never silently widens what the launch directory
            # accepts.

            containing = self._containing_root(resolved)

            if containing is None:
                raise PathPolicyError(
                    "path escapes the workspace." + self.boundary_hint())

            if (containing == self.root and not self.allow_absolute_paths
                    and not from_mount):

                # The legacy flag gates HOST absolute paths into the launch
                # directory. A /workspace/... path is the sandbox's own naming,
                # which the harness itself told the model to use.

                raise PathPolicyError(
                    "absolute paths are not allowed for the launch directory — "
                    f"pass it relative to {self.root}, e.g. "
                    f"'{resolved.relative_to(self.root).as_posix() or '.'}'")
        elif not resolved.is_relative_to(self.root):
            # Relative paths resolve against the primary root only.

            raise PathPolicyError(
                "path escapes the workspace." + self.boundary_hint())

        if must_exist and not resolved.exists():
            raise PathNotFoundError(f"path does not exist: {value}")

        return resolved

    def relative(self, path: str | Path) -> str:
        # Audit callers may hold an already-resolved absolute Path even when
        # user-supplied absolute paths are disabled.  This helper only formats
        # a checked workspace-relative label; it never grants path access.

        candidate = Path(path).expanduser()
        resolved = (candidate if candidate.is_absolute() else self.root / candidate).resolve(
            strict=False
        )
        containing = self._containing_root(resolved)

        if containing is None:
            raise PathPolicyError("path escapes the workspace")

        label = resolved.relative_to(containing).as_posix() or "."

        # Qualify secondary roots so a label can never be mistaken for a file
        # of the launch directory — in the audit trail above all.

        return label if containing == self.root else f"{containing.name}:{label}"


@dataclass(frozen=True)
class CommandAssessment:
    command: str
    argv: tuple[str, ...]
    classification: CommandClassification
    reason: str = ""
    required_capabilities: frozenset[Capability] = frozenset()
    #: Existing host paths the command names that no declared root contains.
    #: The sandbox exposes exactly these, read-only, at their own paths; the
    #: list is what the audit trail records, so an outside read is never silent.
    host_read_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class AuthorizationResult:
    """Authorization outcome and the capabilities actually granted."""

    result: ToolResult | None
    granted_capabilities: frozenset[Capability] = frozenset()

    @property
    def allowed(self) -> bool:
        return self.result is None


class CommandPolicy:
    """Conservative command classifier used before execution.

    It intentionally does not claim to sandbox programs.  ``AUTO`` only permits
    simple argv commands started with the workspace as their cwd; a future
    sandbox runner can replace that execution boundary without changing callers.
    """

    READ_ONLY_BINS = frozenset({
        "cat", "ls", "rg", "grep", "egrep", "fgrep", "find", "head", "tail",
        "wc", "stat", "file", "pwd", "tree", "du", "df", "which", "type",
        "readlink", "basename", "dirname", "md5sum", "sha256sum", "diff", "sort",
        "uniq", "cut", "sed",
    })

    # `sed -i` edits in place. It is refused rather than granted workspace
    # write: edit_file shows a diff and refuses a file not read this turn,
    # and routing edits through it is what keeps them visible and reviewable.
    # Matches -i, -i.bak, -ni and --in-place[=SUFFIX].

    _SED_IN_PLACE = re.compile(r"^(--in-place|-[a-hj-zA-Z]*i)")
    GIT_READ_ONLY = frozenset({
        "status", "log", "diff", "show", "branch", "ls-files", "blame", "describe",
        "rev-parse", "reflog",
    })

    # `tee` writes a file with no redirection operator at all, so the `>`
    # heuristic never sees it.

    WORKSPACE_MUTATING_BINS = frozenset({"make", "cmake", "ninja", "pytest", "tee",
                                         "gcc", "cc", "g++", "c++", "clang", "clang++"})

    # `python -m pytest` is the spelling a model reaches for, and the program
    # it runs is `pytest`, which is allowed on its own. Refusing the module
    # form cost the control-edit benchmark its first verification: the run was
    # denied, the model retried with something else, and the turn finished
    # with no test ever having run. Only the test runners are recognised, and
    # only as a bare binary name: `-c`, a script path, `-m` anything else, or
    # an interpreter named by path all run arbitrary code and stay refused.

    _PYTHON_BINS = frozenset({"python", "python3"})
    _PYTHON_TEST_MODULES = frozenset({"pytest", "unittest"})

    # Said FIRST, and said on every path that can refuse an interpreter. A run
    # tried `python3 -c` twice and never saw this: the quoted parentheses made
    # the command shell-complex, so it was refused by the pipeline classifier,
    # whose message named no alternative at all -- and the single-command
    # message that did name one buried it after the reason.

    _PYTHON_HINT = (
        "run tests with `python3 -m unittest discover -s tests` (or "
        "`python3 -m pytest` where pytest is installed). `python3 -c` and "
        "running a script directly are refused: the interpreter runs arbitrary "
        "code, and the two module forms above are the only ones that run here")
    DANGEROUS_BINS = frozenset({
        "sudo", "su", "doas", "rm", "mv", "dd", "mkfs", "mount", "umount", "chmod",
        "chown", "kill", "pkill", "shutdown", "reboot", "rsync", "python", "python3",
        "bash", "sh", "zsh", "fish", "env",
    })

    # Builtins carry no capability of their own: `cd` and `.` act through the
    # stage that follows them, and treating them as unknown programs would ask
    # for write on every pipeline that merely changes directory.

    SHELL_BUILTINS = frozenset({"cd", ".", "source", "export", "set", "unset",
                                "echo", "true", "false", "test", "["})
    NETWORK_BINS = frozenset({"curl", "wget"})
    SSH_BINS = frozenset({"ssh", "scp", "sftp"})
    CONTAINER_BINS = frozenset({"docker", "podman"})
    GPU_BINS = frozenset({"nvidia-smi"})
    GIT_NETWORK_WRITE = frozenset({"clone", "fetch", "pull"})
    GIT_NETWORK_ONLY = frozenset({"push"})
    _SHELL_SYNTAX = re.compile(r"[|&;<>`$()\n]")

    # Redirection targets that discard output rather than writing a file.

    _REDIRECT_SINKS = frozenset({"/dev/null", "/dev/stdout", "/dev/stderr"})

    # Mount points the sandbox builds for itself. An outside read may not be
    # granted at one of these, nor at any ancestor of one: the bind would be
    # applied over a mount the sandbox needs, and `ls /home` would cost the
    # command its own $HOME. Naming a subdirectory works and is what the
    # refusal asks for.

    _FIXED_SANDBOX_MOUNT_POINTS = ("/proc", "/dev", "/usr", "/bin", "/lib",
                                   "/lib64", "/etc")

    # Credential material stays refused even though reading is otherwise open:
    # the model is served over the network, so a file read here is a file sent
    # there. This is a short, honest list of the usual stores, not a filter.

    _SECRET_PATHS = re.compile(
        r"(^|/)(\.ssh|\.gnupg|\.aws|\.docker|\.kube|\.netrc|\.pgpass|"
        r"\.git-credentials|id_[a-z0-9]+|shadow|gshadow|sudoers)(/|$)"
    )

    @staticmethod
    def _glob_base(value: str) -> str:
        """The deepest directory of a pattern that the shell will expand.

        `grep -l x /opt/llm/claude/*.md` is the idiom a model reaches for
        first. The shell expands the pattern INSIDE the sandbox, so what has to
        be mounted is the directory holding the matches -- the literal
        `.../*.md` names nothing and would mount nothing.
        """
        kept: list[str] = []

        for part in value.split("/"):
            if any(char in part for char in "*?["):
                break

            kept.append(part)

        if len(kept) == len(value.split("/")):
            return value

        return "/".join(kept) or "/"

    def _outside_read_target(self, value: str) -> "tuple[str | None, str]":
        """Vet one absolute path that no declared root contains.

        Returns ``(path_to_bind, refusal)``: exactly one is meaningful. A path
        that does not exist binds nothing and is not refused -- the command
        reports ENOENT, which is the truthful answer and the one the model can
        act on, rather than a policy error about a file that was never there.

        The path is bound where the command SPELLS it, not where it resolves:
        /opt/llm/claude is a symlink, and mounting only its target would have
        the sandbox answer "No such file or directory" for the very path the
        user named. Both spellings are vetted, so a symlink is never a way
        round the checks below.
        """
        literal = os.path.normpath(self._glob_base(value))

        try:
            resolved = str(Path(literal).resolve(strict=False))
        except OSError:
            return None, "path cannot be resolved"

        for spelling in (literal, resolved):
            if self._SECRET_PATHS.search(spelling):
                return None, f"{literal} holds credentials and is never read."

        spec = SandboxSpec()

        for mount in (*self._FIXED_SANDBOX_MOUNT_POINTS, spec.home,
                      spec.tmpdir, spec.workspace_mount):
            for spelling in (literal, resolved):
                if spelling == mount or mount.startswith(spelling.rstrip("/") + "/"):
                    return None, (f"{literal} contains the sandbox's own mounts "
                                  f"and cannot be exposed — name a subdirectory.")

        if not Path(literal).exists():
            return None, ""

        return literal, ""

    def _outside_reads(self, operands: Sequence[str]) -> "tuple[list[str], str]":
        """Split path operands into outside reads to bind, and a first refusal."""
        binds: list[str] = []

        for operand in operands:
            if not operand.startswith("/") or self._inside_sandbox(operand):
                continue

            target, refusal = self._outside_read_target(operand)

            if refusal:
                return [], refusal

            # `/root/../elsewhere` can land back inside a declared tree; that
            # is an ordinary workspace path and must not be re-bound read-only.

            if (target is not None and target not in binds
                    and not self._inside_sandbox(target)):
                binds.append(target)

        return binds, ""

    def _inside_sandbox(self, value: str) -> bool:
        """True for a path under a workspace mount, primary or secondary.

        These are absolute only because that is how the sandbox names them; a
        blanket refusal of absolute arguments would make every secondary root
        unreachable from bash. Traversal is still caught by the `..` checks
        that surround every caller.

        The sandbox normally binds each tree at its own host path, so those
        paths are workspace paths too and must pass — otherwise every absolute
        path the model reads out of a Makefile is refused as an escape. The
        /workspace literals stay accepted for the legacy mount.
        """

        if ".." in value.split("/"):
            # "provably inside" must not be satisfiable by traversing out of a
            # mount: /workspace/../etc is not a workspace path.

            return False

        if value == "/workspace" or value.startswith(("/workspace/", "/workspaces/")):
            return True

        if value.startswith(SandboxSpec().tmpdir.rstrip("/") + "/"):
            return True     # the sandbox's own tmpfs: private and ephemeral

        return any(value == m or value.startswith(m.rstrip("/") + "/")
                   for m in self.mount_roots)

    SENSITIVE_CAPABILITIES = frozenset({
        # Leaving the declared trees is confirmable in ASK for the same reason
        # a network call is: it reaches something the user did not hand over.
        Capability.HOST_READ,
        Capability.NETWORK,
        Capability.REMOTE_WRITE,
        Capability.SSH,
        Capability.GPU,
        Capability.CONTAINER_RUNTIME,
        Capability.SECRETS,
    })

    def __init__(self, capability_policy: CapabilityPolicy = DEFAULT_CAPABILITY_POLICY,
                 mount_roots: "tuple[str, ...]" = ()):
        self.capability_policy = capability_policy

        # Paths the sandbox exposes the workspace at, when they are not the
        # /workspace literals. Set by bind_workspace() when a project is opened.

        self.mount_roots = tuple(mount_roots)

        # The workspace itself, kept only so a refusal can name its roots.

        self.workspace: "Workspace | None" = None

    def bind_workspace(self, workspace: "Workspace | None") -> None:
        """Teach the classifier where this workspace's trees are mounted.

        Called on every project switch: the acceptable absolute paths change
        with the workspace, and a stale set would either refuse legitimate
        paths or accept a previous project's.
        """

        if workspace is None:
            self.mount_roots = ()
            self.workspace = None

            return

        self.workspace = workspace
        mount = effective_mount_root(workspace.root)
        mounts = workspace.mount_map(
            mount, identity=mount != SandboxSpec().workspace_mount)
        self.mount_roots = tuple(sorted(set(mounts.values())))

    def boundary_hint(self) -> str:
        """The workspace's own hint, or a usable one before a project is bound."""
        workspace = getattr(self, "workspace", None)

        if workspace is not None:
            return workspace.boundary_hint()

        if not self.mount_roots:
            return ""

        return f" Writable trees: {', '.join(self.mount_roots[:3])}."

    def capability_hint(
        self, missing: AbstractSet[Capability], mode: ExecutionMode
    ) -> str:
        """Name what would grant `missing`, or say that nothing will.

        The same lesson as boundary_hint(): a refusal that only states the rule
        is a dead end. Told `network` was not allowed and nothing more, the
        model ran six more web searches and closed with "copy-paste these URLs
        into your browser" — it had found the right document and no sentence in
        the refusal said where reading one lives.

        Two facts get it moving again: which launch flag holds the capability
        (the USER's decision, not something to retry into), and — for the
        network — that reading a page does not need this capability at all,
        because fetch_url is a native tool and never reaches the shell.
        """
        grants = [candidate for candidate in
                  (ExecutionMode.ASK, ExecutionMode.AUTO, ExecutionMode.SAFE)
                  if candidate != mode
                  and missing <= self.available_capabilities(candidate)]

        if not grants:
            # Nothing to point at. This is what `--no-network` looks like from
            # here, and the web tools are gone in that session too, so naming
            # them would send the model after a tool it does not have.

            return (" No mode grants that combination here; say so rather "
                    "than trying another spelling of the same command.")

        flags = " or ".join(f"--{candidate.value}" for candidate in grants)
        hint = (f" Granted in {flags} mode, which is the user's call at "
                f"launch — say what you need it for instead of retrying.")

        if Capability.NETWORK in missing:
            hint += (" To READ a web page or a document, use the fetch_url "
                     "tool instead: it is a native tool, needs no shell "
                     "capability, and works in every mode.")

        return hint

    @staticmethod
    def _assessment(
        command: str,
        argv: Sequence[str],
        classification: CommandClassification,
        reason: str = "",
        capabilities: Sequence[Capability] = (),
        host_read_paths: Sequence[str] = (),
    ) -> CommandAssessment:
        return CommandAssessment(
            command, tuple(argv), classification, reason, frozenset(capabilities),
            tuple(host_read_paths),
        )

    def _is_workspace_program(self, value: str) -> bool:
        """True for a program that lives inside the sandbox.

        A model cannot verify its own work without running it. `make` is
        already permitted and a Makefile may run anything, so refusing the
        binary the model just compiled -- in the same sandbox, under the same
        confinement, with no network -- denies the feedback loop while buying
        no containment: the boundary is the sandbox, not this list.
        """

        if ".." in value.split("/"):
            return False

        if value.startswith("./"):
            # A subdirectory is fine. `..` is already refused above, and the
            # confinement is the sandbox: a program reachable by a relative
            # path inside it is inside a mounted tree by construction.
            # Refusing `./scripts/build.sh` refused the project's own entry
            # point -- exactly the command the model is meant to run.

            return True

        return value.startswith("/") and self._inside_sandbox(value)

    def _path_operands(self, argv: Sequence[str]) -> "list[str]":
        """The arguments that name files, for the escape check.

        Everything after argv[0] is a path for most commands. sed is the
        exception: its SCRIPT is an argument too, and an address like
        `/BEGIN/,/END/p` starts with a slash while naming no file at all.
        Checking it as a path refuses ordinary sed with "path argument may
        escape the workspace" -- a refusal the model cannot act on because the
        premise is wrong.
        """

        if Path(argv[0]).name != "sed":
            return list(argv[1:])

        operands: list[str] = []
        script_seen = expect_script = False

        for arg in argv[1:]:
            if expect_script:            # the -e argument: a script, not a path
                expect_script = False
                continue

            if arg in ("-e", "--expression"):
                expect_script = script_seen = True
                continue

            if arg.startswith("--expression="):
                script_seen = True
                continue

            if arg in ("-f", "--file"):
                # -f takes a script FILE, which IS a path: fall through so the
                # next operand is checked rather than taken for the script.

                script_seen = True
                continue

            if arg.startswith("-") and arg != "-":
                continue                 # any other option

            if not script_seen:
                script_seen = True       # the bare script argument
                continue

            operands.append(arg)

        return operands

    def _simple_capabilities(
        self, argv: Sequence[str], *, stdin_from_pipeline: bool = False
    ) -> frozenset[Capability]:
        if not argv:
            return frozenset()

        binary = Path(argv[0]).name

        # Same capabilities as the `pytest` it runs: a test suite reads the
        # tree and writes what it builds and caches inside it.

        if self._is_module_test_run(argv):
            return frozenset({Capability.FILESYSTEM_READ, Capability.WORKSPACE_WRITE})

        if binary == "wget":
            # wget writes its downloaded payload to the cwd by default.

            return frozenset({Capability.NETWORK, Capability.WORKSPACE_WRITE})

        if binary == "curl":
            output_flags = {"-o", "--output", "-O", "--remote-name"}
            writes_output = (
                any(arg in output_flags or arg.startswith("--output=") for arg in argv[1:])
            )
            capabilities = {Capability.NETWORK}

            if writes_output:
                capabilities.add(Capability.WORKSPACE_WRITE)

            if self._curl_remote_write(argv):
                capabilities.add(Capability.REMOTE_WRITE)

            return frozenset(capabilities)

        if binary in self.NETWORK_BINS:
            return frozenset({Capability.NETWORK})

        if binary == "scp":
            operands = [arg for arg in argv[1:] if not arg.startswith("-")]

            # This deliberately handles only the unambiguous common forms.
            # Options with operands or multiple remote endpoints fall back to
            # the conservative read+write result.

            if len(operands) >= 2:
                sources, destination = operands[:-1], operands[-1]
                source_is_remote = any(":" in source for source in sources)
                destination_is_remote = ":" in destination
                capabilities = {Capability.SSH, Capability.NETWORK}

                if destination_is_remote and not source_is_remote:
                    capabilities.add(Capability.FILESYSTEM_READ)
                    capabilities.add(Capability.REMOTE_WRITE)

                    return frozenset(capabilities)

                if source_is_remote and not destination_is_remote:
                    capabilities.add(Capability.WORKSPACE_WRITE)
                    return frozenset(capabilities)

            return frozenset({
                Capability.FILESYSTEM_READ, Capability.WORKSPACE_WRITE,
                Capability.SSH, Capability.NETWORK, Capability.REMOTE_WRITE,
            })

        if binary in self.SSH_BINS:
            return frozenset({Capability.SSH, Capability.NETWORK, Capability.REMOTE_WRITE})

        if binary in self.CONTAINER_BINS:
            return frozenset({Capability.CONTAINER_RUNTIME})

        if binary in self.GPU_BINS:
            return frozenset({Capability.GPU})

        if binary in self.WORKSPACE_MUTATING_BINS:
            return frozenset({Capability.FILESYSTEM_READ, Capability.WORKSPACE_WRITE})

        if binary == "git":
            subcommand = argv[1] if len(argv) > 1 else ""

            if subcommand in self.GIT_NETWORK_WRITE:
                return frozenset({
                    Capability.FILESYSTEM_READ, Capability.WORKSPACE_WRITE, Capability.NETWORK,
                })

            if subcommand in self.GIT_NETWORK_ONLY:
                return frozenset({
                    Capability.FILESYSTEM_READ, Capability.NETWORK, Capability.REMOTE_WRITE,
                })

            if subcommand in self.GIT_READ_ONLY:
                return frozenset({Capability.FILESYSTEM_READ})

            return frozenset()

        if binary in self.READ_ONLY_BINS:
            # A bare filter in a pipeline consumes stdin rather than opening a
            # workspace path (for example ``curl URL | cat``).

            if stdin_from_pipeline and len(argv) == 1 and binary in {
                "cat", "head", "tail", "wc", "sort", "uniq", "cut",
            }:
                return frozenset()

            return frozenset({Capability.FILESYSTEM_READ})

        return frozenset()

    @staticmethod
    def _curl_remote_write(argv: Sequence[str]) -> bool:
        """Recognize common explicit Curl request bodies/methods, not all syntax."""
        mutating_methods = {"POST", "PUT", "PATCH", "DELETE"}
        body_flags = {"-d", "--data", "--data-raw", "--data-binary", "-F", "--form"}

        for index, arg in enumerate(argv[1:], start=1):
            if arg in body_flags or arg.startswith(("--data=", "--data-raw=", "--data-binary=", "--form=")):
                return True

            if arg.startswith("-d") and arg != "-d":
                return True

            if arg.startswith("-F") and arg != "-F":
                return True

            if arg in {"-X", "--request"} and index + 1 < len(argv):
                if argv[index + 1].upper() in mutating_methods:
                    return True

            if arg.startswith("-X") and arg[2:].upper() in mutating_methods:
                return True

            if arg.startswith("--request=") and arg.split("=", 1)[1].upper() in mutating_methods:
                return True

        return False

    @classmethod
    def _is_module_test_run(cls, argv: Sequence[str]) -> bool:
        """True for `python -m pytest` / `python -m unittest` and nothing else.

        Interpreter options are deliberately not parsed: ``-m`` must be the
        first argument, so `python -X importtime -m pytest` is refused rather
        than reasoned about.
        """

        if not argv or "/" in argv[0] or argv[0] not in cls._PYTHON_BINS:
            return False

        return len(argv) > 2 and argv[1] == "-m" and argv[2] in cls._PYTHON_TEST_MODULES

    def _is_known_binary(self, binary: str) -> bool:
        """True when the allowlists say what this program does.

        Asked instead of "did _simple_capabilities return nothing", because an
        empty result is also the honest answer for a bare filter consuming
        stdin -- `… | head` needs no workspace at all, and treating it as
        unknown asked for write on every read-only pipeline, which SAFE then
        refused.
        """
        return (binary in self.READ_ONLY_BINS
                or binary in self.WORKSPACE_MUTATING_BINS
                or binary in self.NETWORK_BINS
                or binary in self.SSH_BINS
                or binary in self.CONTAINER_BINS
                or binary in self.GPU_BINS
                or binary in self.DANGEROUS_BINS
                or binary in self.SHELL_BUILTINS
                or binary == "git")

    def _sensitive_classification(
        self, argv: Sequence[str]
    ) -> CommandClassification:
        binary = Path(argv[0]).name

        if binary in {"wget"}:
            return CommandClassification.WORKSPACE_MUTATING

        if binary == "curl" and Capability.WORKSPACE_WRITE in self._simple_capabilities(argv):
            return CommandClassification.WORKSPACE_MUTATING

        if binary == "scp" and Capability.WORKSPACE_WRITE in self._simple_capabilities(argv):
            return CommandClassification.WORKSPACE_MUTATING

        return CommandClassification.READ_ONLY

    @classmethod
    def _writes_via_redirection(cls, command: str) -> bool:
        """True when a redirection actually creates or extends a file.

        `2>/dev/null` is a redirection that writes nothing, and demanding
        workspace:write for it made every read-only pipeline carrying the idiom
        unavailable in SAFE — the mode where searching a tree is the whole
        point.
        """

        for match in re.finditer(r"(?<![=])\d*>{1,2}\s*(\S*)", command):
            if match.group(1) not in cls._REDIRECT_SINKS:
                return True

        return False

    _HEREDOC_RE = re.compile(r"<<-?\s*([\'\"]?)([A-Za-z_][\w-]*)\1")

    # A here-document that feeds one of these really is code and must still be
    # scanned, including through a pipe: `cat <<EOF | bash`.

    _HEREDOC_INTERPRETERS = frozenset({
        "sh", "bash", "dash", "zsh", "ksh", "python", "python3", "perl",
        "ruby", "node", "eval",
    })

    @classmethod
    def _strip_heredoc_bodies(cls, command: str) -> str:
        """Drop here-document BODIES before any stage or path analysis.

        A here-document body is stdin data, not arguments.  Leaving it in is
        what made ``cat > doc/source/ls.rst <<'EOF'`` refuse with "/ contains
        the sandbox's own mounts and cannot be exposed": the chapter being
        written contained the example line ``/ % ls``, and shlex handed that
        bare ``/`` over as a path operand of ``cat``.  The refusal named a path
        the request never contained, which is the kind nobody can act on — the
        model spent six turns rediscovering the situation and finished by
        proposing to paste the file by hand.

        Bodies that feed an interpreter are kept, because there the body is
        commands and dropping it would hide them from the danger scan.
        """
        lines = command.split("\n")
        kept: list[str] = []
        index = 0

        while index < len(lines):
            line = lines[index]
            kept.append(line)
            index += 1
            matches = cls._HEREDOC_RE.findall(line)

            if not matches or cls._mentions_interpreter(line):
                continue

            for _, delimiter in matches:
                while index < len(lines) and lines[index].strip() != delimiter:
                    index += 1

                if index < len(lines):
                    kept.append(lines[index])
                    index += 1

        return "\n".join(kept)

    @classmethod
    def _mentions_interpreter(cls, line: str) -> bool:
        try:
            tokens = shlex.split(line, posix=True)
        except ValueError:
            return True          # unparseable: assume the worst and keep the body

        return any(Path(token).name in cls._HEREDOC_INTERPRETERS
                   for token in tokens)

    @staticmethod
    def _shell_stages(command: str) -> "list[str]":
        """Split on shell operators, but NOT inside quotes.

        ``re.split`` ignored quoting, so any quoted argument containing
        ``( ) ; | < >`` was torn into fragments and each fragment analysed as a
        command. A sed script rewriting the line ``basic file utilities (``ls``
        supports ``-l``; ``rm`` supports ``-r`` / ``-f``)`` split on its own
        parentheses and semicolon, leaving a fragment that began with ``/`` --
        refused as "'/' is not allowlisted, so the pipeline that contains it
        cannot run", about a path the command never named. The same flaw makes
        ``grep 'foo(bar)' file`` -- an everyday idiom -- look like a pipeline.

        Returns text and separators interleaved, the shape re.split produced.
        """
        parts: list[str] = []
        current: list[str] = []
        quote = ""
        index, length = 0, len(command)

        while index < length:
            char = command[index]

            if quote:
                current.append(char)

                if char == quote:
                    quote = ""

                index += 1

                continue

            if char in "'\"":
                quote = char
                current.append(char)
                index += 1

                continue

            if char == "\\" and index + 1 < length:
                current.append(char)
                current.append(command[index + 1])
                index += 2

                continue

            if command[index:index + 2] in ("||", "&&"):
                parts.append("".join(current))
                parts.append(command[index:index + 2])
                current = []
                index += 2

                continue

            if char in "|;&()<>":
                parts.append("".join(current))
                parts.append(char)
                current = []
                index += 1

                continue

            current.append(char)
            index += 1

        parts.append("".join(current))

        return parts

    def _shell_capabilities(
        self, command: str
    ) -> "tuple[frozenset[Capability], str, list[str], str]":
        """Best-effort stage analysis; this intentionally is not a shell parser.

        Returns ``(capabilities, danger_reason, outside_paths, refusal)``;
        ``danger_reason`` is empty when no stage is dangerous.
        """

        # filesystem:read is unconditional. A pipeline exists to act on the
        # workspace, and without this capability the sandbox mounts an EMPTY
        # directory instead of the trees -- so `cd /path/that/exists` failed
        # with "No such file or directory", blaming the path rather than the
        # missing mount. There is no useful pipeline that needs no workspace.

        capabilities: set[Capability] = {Capability.SHELL_COMPLEX,
                                         Capability.FILESYSTEM_READ}
        outside: list[str] = []
        refusal = ""

        # A shell output redirection mutates the cwd even when its producer
        # (for example ``printf``) is otherwise unknown to the allowlist.
        # The raw-command check is deliberately conservative: a false
        # positive merely asks for workspace write; a false negative would
        # give the command a private /workspace and report a misleading
        # success instead of persisting the requested file.

        if self._writes_via_redirection(command):
            capabilities.update({
                Capability.FILESYSTEM_READ,
                Capability.WORKSPACE_WRITE,
            })

        danger = ""
        stages = self._shell_stages(command)
        stdin_from_pipeline = False
        redirect_target = False
        redirect_reads = False

        for stage in stages:
            token = stage.strip()

            if not token:
                continue

            if token in {"|", "||", "&&", ";", "&", "(", ")", "<", ">"}:
                stdin_from_pipeline = token == "|"

                # `<` and `>` introduce a FILE, not a command. Analysing the
                # next token as a command stage made `2>/dev/null` hit the
                # "argv[0] contains /" rule, so the single most common idiom in
                # shell was refused as a dangerous stage — `ls 2>/dev/null`
                # included. The target is still checked, as a path, below.

                redirect_target = token in {"<", ">"}
                redirect_reads = token == "<"

                continue

            if redirect_target:
                redirect_target = False
                target = token.split()[0]

                # The path check the command check was accidentally providing:
                # an OUTPUT redirection may not escape the workspace. The /dev
                # sinks are the documented exception — they discard, they do not
                # write. `<` is the other direction: it reads, so it is treated
                # like any other outside read below.

                escapes = (target not in self._REDIRECT_SINKS
                           and not self._inside_sandbox(target)
                           and (target.startswith("/") or target.startswith("../")
                                or "/../" in target or target == ".."))

                if escapes:
                    if redirect_reads and target.startswith("/"):
                        found, denied = self._outside_reads([target])
                        refusal = refusal or denied
                        outside.extend(p for p in found if p not in outside)
                    else:
                        danger = danger or (
                            f"a redirection would write outside the workspace: "
                            f"'{target}'.")

                # Anything after the target on the same fragment is a real
                # command continuation (`> out.txt && make`), so fall through
                # to the stage analysis with the target removed.

                token = token[len(target):].strip()

                if not token:
                    continue

            try:
                argv = tuple(shlex.split(token, posix=True))
            except ValueError:
                continue

            if not argv:
                continue

            binary = Path(argv[0]).name

            # The in-place sed rule lives in the single-command classifier, so
            # a multi-line `sed -i '5a\...'` -- which carries shell syntax and
            # therefore lands HERE -- slipped past it and ran. It then failed
            # with "couldn't open temporary file: Read-only file system",
            # because sed writes its temp file beside the target and a
            # read-only-classified command gets the tree read-only. The model
            # read that as "the documentation is not writable" and gave up on a
            # tree it could in fact edit. Same rule, both paths.

            if binary == "sed" and any(self._SED_IN_PLACE.match(arg)
                                       for arg in argv[1:]):
                danger = danger or (
                    "in-place sed edits outside the checkpoint, so it cannot "
                    "be rolled back: use edit_file. To insert a line, pass the "
                    "existing surrounding line(s) as old_text and those same "
                    "lines plus the new one as new_text.")

            if not self._is_module_test_run(argv) and (
                    binary in self.DANGEROUS_BINS or (
                        "/" in argv[0] and not self._is_workspace_program(argv[0]))):
                danger = danger or (
                    f"{self._PYTHON_HINT} — so this command cannot run."
                    if binary in self._PYTHON_BINS else
                    f"'{argv[0]}' is not allowlisted, so the pipeline that "
                    f"contains it cannot run.")

            stage_caps = self._simple_capabilities(
                argv, stdin_from_pipeline=stdin_from_pipeline)

            if not self._is_known_binary(binary):
                # An unrecognised program: assume it writes. This is the same
                # conservatism the redirection check applies, and for the same
                # reason -- a false positive merely asks for workspace write,
                # while a false negative hands the command a read-only tree and
                # lets it fail deep inside a build. Every Infrabase entry point
                # (build.sh, deploy.sh, st.sh, updiff.sh) lands here.

                stage_caps = frozenset({Capability.FILESYSTEM_READ,
                                        Capability.WORKSPACE_WRITE})

            capabilities.update(stage_caps)

            # A stage naming a tree the workspace does not contain used to be
            # neither refused nor mounted: the command ran and reported "No
            # such file or directory" about a file that plainly exists, which
            # is the worst of the three outcomes. Expose it, read-only.

            found, denied = self._outside_reads(self._path_operands(argv))
            refusal = refusal or denied
            outside.extend(path for path in found if path not in outside)
            stdin_from_pipeline = False

        if outside:
            capabilities.add(Capability.HOST_READ)

        return frozenset(capabilities), danger, outside, refusal

    def classify(self, command: str) -> CommandAssessment:
        if not command or not command.strip():
            return self._assessment(command, (), CommandClassification.DANGEROUS, "empty command")

        if "\x00" in command:
            return self._assessment(command, (), CommandClassification.DANGEROUS, "NUL byte")

        command = self._strip_heredoc_bodies(command)

        if self._SHELL_SYNTAX.search(command):
            capabilities, danger, outside, refusal = self._shell_capabilities(command)

            if danger:
                return self._assessment(command, (), CommandClassification.DANGEROUS,
                                        danger + self.boundary_hint())

            if refusal:
                return self._assessment(command, (), CommandClassification.DANGEROUS,
                                        refusal + self.boundary_hint())

            return self._assessment(command, (), CommandClassification.SHELL_COMPLEX,
                                    "shell syntax", capabilities,
                                    host_read_paths=outside)

        try:
            argv = tuple(shlex.split(command, posix=True))
        except ValueError as exc:
            return self._assessment(
                command, (), CommandClassification.SHELL_COMPLEX, str(exc),
                (Capability.SHELL_COMPLEX,),
            )

        if not argv:
            return self._assessment(command, (), CommandClassification.DANGEROUS, "empty argv")

        operands = self._path_operands(argv)

        # A RELATIVE `..` still escapes with no way to check it: the classifier
        # does not know the cwd a shell stage may have moved to, so the same
        # string can denote two different files. An absolute path is checkable,
        # which is exactly what the refusal now asks for.

        traversal = next((arg for arg in operands
                          if not arg.startswith("/")
                          and (arg == ".." or "/../" in arg
                               or arg.startswith("../"))), None)

        if traversal is not None:
            return self._assessment(
                command, argv, CommandClassification.DANGEROUS,
                f"relative path leaves the workspace: '{traversal}' — name it by "
                f"absolute path instead." + self.boundary_hint())

        host_reads, refusal = self._outside_reads(operands)

        if refusal:
            return self._assessment(command, argv, CommandClassification.DANGEROUS,
                                    refusal + self.boundary_hint())

        return self._with_host_reads(self._classify_argv(command, argv), host_reads)

    def _with_host_reads(self, assessment: CommandAssessment,
                         host_reads: Sequence[str]) -> CommandAssessment:
        """Attach vetted outside paths to an assessment that may run.

        No classification is downgraded here and none is refused: the paths are
        exposed ``--ro-bind``, so what `make -C /elsewhere` meets is EROFS --
        the truthful, actionable error -- and the write boundary is still the
        mount rather than a list of binaries this had to keep in sync.
        """

        if not host_reads or assessment.classification == CommandClassification.DANGEROUS:
            return assessment

        return replace(
            assessment,
            required_capabilities=(assessment.required_capabilities
                                   | {Capability.HOST_READ}),
            host_read_paths=tuple(host_reads),
        )

    # Removing or renaming a file is not something any tool can do, and
    # pointing at edit_file/write_file for these actively misleads: a model
    # told to "use edit_file to change files" spent fifteen calls trying rm,
    # `bash -c rm`, python3 -c os.remove, and an empty write_file, then emptied
    # the file with edit_file -- leaving a zero-byte document that still warned
    # "isn't included in any toctree". Say what is actually true.

    _REMOVAL_BINS = frozenset({"rm", "rmdir", "unlink", "shred", "mv", "rename"})
    _NO_DELETE_HINT = (
        " — no tool can delete or rename a file. Nothing here can do it: say "
        "which file should go and why, and leave it to the operator. Emptying "
        "it instead leaves a file behind.")

    def _classify_argv(self, command: str, argv: Sequence[str]) -> CommandAssessment:
        binary = Path(argv[0]).name

        if self._is_module_test_run(argv):
            return self._assessment(command, argv, CommandClassification.WORKSPACE_MUTATING,
                                    capabilities=self._simple_capabilities(argv))

        if binary in self.DANGEROUS_BINS or (
                "/" in argv[0] and not self._is_workspace_program(argv[0])):
            if binary in self._PYTHON_BINS:
                # The way out comes before the refusal: a model reads the
                # first clause and acts on it.

                return self._assessment(
                    command, argv, CommandClassification.DANGEROUS,
                    f"{self._PYTHON_HINT} — so '{argv[0]}' as written is refused")

            return self._assessment(
                command, argv, CommandClassification.DANGEROUS,
                f"unapproved executable: '{argv[0]}'"
                + (self._NO_DELETE_HINT if binary in self._REMOVAL_BINS else
                   " — only a program built inside a declared tree may be run "
                   "by path" if "/" in argv[0] else
                   " — use edit_file/write_file to change files, and make/"
                   "pytest to build and test"))

        if "/" in argv[0]:
            # Running what was just built: the test half of the loop.

            return self._assessment(
                command, argv, CommandClassification.WORKSPACE_MUTATING,
                capabilities=(Capability.FILESYSTEM_READ, Capability.WORKSPACE_WRITE))

        if binary == "git":
            if len(argv) > 1 and argv[1] in self.GIT_READ_ONLY:
                return self._assessment(command, argv, CommandClassification.READ_ONLY,
                                        capabilities=self._simple_capabilities(argv))

            if len(argv) > 1 and argv[1] in self.GIT_NETWORK_ONLY:
                return self._assessment(command, argv, CommandClassification.READ_ONLY,
                                        capabilities=self._simple_capabilities(argv))

            if len(argv) > 1 and argv[1] in self.GIT_NETWORK_WRITE:
                return self._assessment(command, argv, CommandClassification.WORKSPACE_MUTATING,
                                        capabilities=self._simple_capabilities(argv))

            return self._assessment(command, argv, CommandClassification.DANGEROUS,
                                    "non-read-only git command")

        if binary in self.NETWORK_BINS | self.SSH_BINS | self.CONTAINER_BINS | self.GPU_BINS:
            return self._assessment(command, argv, self._sensitive_classification(argv),
                                    capabilities=self._simple_capabilities(argv))

        if binary in self.READ_ONLY_BINS:
            if binary == "find" and any(option in argv for option in
                                        ("-exec", "-execdir", "-delete", "-fprint", "-fprintf")):
                return self._assessment(command, argv, CommandClassification.DANGEROUS,
                                        "mutating find action")

            if binary == "sed" and any(self._SED_IN_PLACE.match(arg) for arg in argv[1:]):
                # Naming the alternative is not enough: a model told to "use
                # edit_file" while edit_file was refusing its whitespace-only
                # edit had no exit and repeated the pair nine times. Say how.

                return self._assessment(
                    command, argv, CommandClassification.DANGEROUS,
                    "in-place sed edits outside the checkpoint, so it cannot be "
                    "rolled back: use edit_file. To insert a line, pass the "
                    "existing surrounding line(s) as old_text and those same "
                    "lines plus the new one as new_text.")

            return self._assessment(command, argv, CommandClassification.READ_ONLY,
                                    capabilities=self._simple_capabilities(argv))

        if binary in self.WORKSPACE_MUTATING_BINS:
            return self._assessment(command, argv, CommandClassification.WORKSPACE_MUTATING,
                                    capabilities=self._simple_capabilities(argv))

        if binary in self._REMOVAL_BINS:
            return self._assessment(
                command, argv, CommandClassification.DANGEROUS,
                f"'{binary}' is not allowlisted" + self._NO_DELETE_HINT)

        return self._assessment(
            command, argv, CommandClassification.DANGEROUS,
            f"build and test with make, cmake, ninja, gcc, pytest or "
            f"`python3 -m unittest discover -s tests`; read with cat, ls, "
            f"grep, find, sed -n, head, tail, wc, stat, diff and read-only "
            f"git; change files with edit_file/write_file. "
            f"'{binary}' is none of those, so it is not allowlisted.")

    def available_capabilities(self, mode: ExecutionMode) -> frozenset[Capability]:
        return self.capability_policy.for_mode(mode)

    def authorize(
        self,
        assessment: CommandAssessment,
        mode: ExecutionMode,
        approve: Callable[[str], bool] | None = None,
    ) -> AuthorizationResult:
        """Return explicit granted capabilities or a legacy-compatible terminal result."""
        kind = assessment.classification

        if kind == CommandClassification.DANGEROUS:
            return AuthorizationResult(ToolResult("denied", f"command denied: {assessment.reason}"))

        if mode == ExecutionMode.SAFE and kind == CommandClassification.WORKSPACE_MUTATING:
            # Classification-based gate, kept only for the class that exists to
            # mutate. SHELL_COMPLEX now falls through to the capability check
            # below: a read-only pipeline needs no write capability, and SAFE
            # mounts every root read-only anyway, so refusing `find … | head`
            # bought nothing.

            return AuthorizationResult(
                ToolResult("denied", f"{kind.value} commands are disabled in safe mode")
            )

        unavailable = assessment.required_capabilities - self.available_capabilities(mode)

        if unavailable:
            names = ", ".join(sorted(capability.value for capability in unavailable))
            hint = ""

            if Capability.HOST_READ in unavailable:
                hint = self.boundary_hint()

            hint += self.capability_hint(unavailable, mode)

            return AuthorizationResult(
                ToolResult("denied",
                           f"required capabilities are not allowed: {names}{hint}")
            )

        if mode == ExecutionMode.SAFE:
            # Whatever passed the capability check in SAFE needs no prompt:
            # SAFE grants neither workspace:write nor network, so there is
            # nothing sensitive left to confirm.

            return AuthorizationResult(None, assessment.required_capabilities)

        if mode == ExecutionMode.AUTO:
            if kind in (
                CommandClassification.READ_ONLY,
                CommandClassification.WORKSPACE_MUTATING,
                CommandClassification.SHELL_COMPLEX,
            ):
                return AuthorizationResult(None, assessment.required_capabilities)

        # ASK mode: a sensitive capability is confirmable even when its
        # command classification is read-only (for example curl GET).

        requires_confirmation = (
            kind != CommandClassification.READ_ONLY
            or bool(assessment.required_capabilities & self.SENSITIVE_CAPABILITIES)
        )

        if not requires_confirmation:
            return AuthorizationResult(None, assessment.required_capabilities)

        if approve is not None and approve(f"Run: {assessment.command} ?"):
            return AuthorizationResult(None, assessment.required_capabilities)

        return AuthorizationResult(ToolResult("cancelled", "command was not approved"))


class ToolPolicy:
    """Authorization policy for filesystem and persistence mutations."""

    def __init__(
        self,
        mode: ExecutionMode,
        approve: Callable[[str], bool] | None = None,
    ):
        self.mode = mode
        self.approve = approve

    def authorize_mutation(self, description: str) -> ToolResult | None:
        """Return a terminal result when blocked; ``None`` means execute."""

        if self.mode == ExecutionMode.SAFE:
            return ToolResult("denied", "mutations are disabled in safe mode")

        if self.mode == ExecutionMode.AUTO:
            return None

        if self.approve is not None and self.approve(description):
            return None

        return ToolResult("cancelled", "mutation was not approved")


@dataclass(frozen=True)
class ExecutionProfile:
    """Pure execution contract derived from already-granted capabilities."""

    capabilities: frozenset[Capability]
    workspace_read: bool
    workspace_write: bool
    shell_complex: bool
    network: bool = False
    gpu: bool = False
    ssh: bool = False
    container_runtime: bool = False
    secrets_allowed: bool = False
    #: Host paths to expose read-only, at their own path, in addition to the
    #: workspace. Empty unless host:read was granted.
    host_read_paths: tuple[str, ...] = ()

    @classmethod
    def from_capabilities(cls, capabilities: Sequence[Capability],
                          host_read_paths: Sequence[str] = ()) -> "ExecutionProfile":
        granted = frozenset(capabilities)
        workspace_read = Capability.FILESYSTEM_READ in granted
        workspace_write = Capability.WORKSPACE_WRITE in granted
        network = Capability.NETWORK in granted
        ssh = Capability.SSH in granted

        if workspace_write and not workspace_read:
            raise ValueError("workspace:write requires filesystem:read")

        if ssh and not network:
            raise ValueError("ssh requires network")

        if Capability.REMOTE_WRITE in granted and not network:
            raise ValueError("remote:write requires network")

        if host_read_paths and Capability.HOST_READ not in granted:
            raise ValueError("outside paths require host:read")

        return cls(
            capabilities=granted,
            workspace_read=workspace_read,
            workspace_write=workspace_write,
            shell_complex=Capability.SHELL_COMPLEX in granted,
            network=network,
            gpu=Capability.GPU in granted,
            ssh=ssh,
            container_runtime=Capability.CONTAINER_RUNTIME in granted,
            secrets_allowed=Capability.SECRETS in granted,
            host_read_paths=tuple(host_read_paths),
        )

    def audit_metadata(self) -> Mapping[str, bool]:
        return {
            "host_read": bool(self.host_read_paths),
            "workspace_read": self.workspace_read,
            "workspace_write": self.workspace_write,
            "shell_complex": self.shell_complex,
            "network": self.network,
            "gpu": self.gpu,
            "ssh": self.ssh,
            "container_runtime": self.container_runtime,
            "secrets_allowed": self.secrets_allowed,
        }


@dataclass(frozen=True)
class ResourceLimits:
    """Resource contract for a sandboxed workload, independent of capabilities.

    Active limits are applied by ``prlimit`` *inside* Bubblewrap.  Keeping the
    hard and soft values equal prevents an unprivileged child from merely
    raising its own soft limit again.
    """

    nofile: int | None = None
    core_bytes: int | None = None
    cpu_seconds: int | None = None
    file_size_bytes: int | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("nofile", self.nofile),
            ("core_bytes", self.core_bytes),
            ("cpu_seconds", self.cpu_seconds),
            ("file_size_bytes", self.file_size_bytes),
        ):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be a non-negative integer or None")

        if self.nofile == 0:
            raise ValueError("nofile must be greater than zero")

    @property
    def active(self) -> bool:
        return any(
            value is not None
            for value in (self.nofile, self.core_bytes, self.cpu_seconds, self.file_size_bytes)
        )

    def prlimit_options(self) -> list[str]:
        """Return util-linux ``prlimit`` options in a stable, testable order."""
        options: list[str] = []

        for option, value in (
            ("nofile", self.nofile),
            ("core", self.core_bytes),
            ("cpu", self.cpu_seconds),
            ("fsize", self.file_size_bytes),
        ):
            if value is not None:
                options.append(f"--{option}={value}:{value}")

        return options


# Default only disables core dumps and bounds descriptor fan-out.  CPU and
# file-size limits remain opt-in: wall timeout already exists, and FSIZE is not
# a workspace quota and can break legitimate build/link workloads.

DEFAULT_RESOURCE_LIMITS = ResourceLimits(nofile=4096, core_bytes=0)


@dataclass(frozen=True)
class CgroupLimits:
    """Kernel-enforced resource contract for a whole sandboxed process tree.

    Orthogonal to :class:`Capability`, :class:`ExecutionProfile` and
    :class:`ResourceLimits`.  ``ResourceLimits`` are per-process rlimits applied
    by ``prlimit`` *inside* Bubblewrap; these limits are applied by the kernel to
    bwrap, the sandboxed command and every descendant at once, through a
    transient systemd user scope.

    Deliberately restricted to the three controllers delegated to a regular
    user session (``memory``, ``pids``, ``cpu``).  ``io``/``cpuset`` need root
    and are out of scope; ``RLIMIT_NPROC``/``RLIMIT_AS`` are not substitutes and
    are never used.

    Swap is *not* implied by ``memory_max_bytes``: coupling them is a policy
    decision, so the mechanism keeps both knobs explicit and independently
    testable.
    """

    memory_max_bytes: int | None = None
    memory_swap_max_bytes: int | None = None
    tasks_max: int | None = None
    cpu_quota_percent: int | None = None

    def __post_init__(self) -> None:
        for name, value, minimum in (
            ("memory_max_bytes", self.memory_max_bytes, 1),
            ("memory_swap_max_bytes", self.memory_swap_max_bytes, 0),
            ("tasks_max", self.tasks_max, 1),
            ("cpu_quota_percent", self.cpu_quota_percent, 1),
        ):
            if value is None:
                continue

            # ``type(value) is not int`` also rejects ``bool``, which would
            # otherwise silently become 0/1 through the int subclass.

            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be None or an integer >= {minimum}")

    @property
    def active(self) -> bool:
        return any(
            value is not None
            for value in (
                self.memory_max_bytes,
                self.memory_swap_max_bytes,
                self.tasks_max,
                self.cpu_quota_percent,
            )
        )

    @property
    def required_controllers(self) -> frozenset[str]:
        """Only the controllers the active fields actually need."""
        controllers: set[str] = set()

        if self.memory_max_bytes is not None or self.memory_swap_max_bytes is not None:
            controllers.add("memory")

        if self.tasks_max is not None:
            controllers.add("pids")

        if self.cpu_quota_percent is not None:
            controllers.add("cpu")

        return frozenset(controllers)

    def systemd_properties(self) -> list[str]:
        """Return ``Name=value`` unit properties in a stable, testable order."""
        properties: list[str] = []

        for name, value in (
            ("MemoryMax", self.memory_max_bytes),
            ("MemorySwapMax", self.memory_swap_max_bytes),
            ("TasksMax", self.tasks_max),
        ):
            if value is not None:
                properties.append(f"{name}={value}")

        if self.cpu_quota_percent is not None:
            properties.append(f"CPUQuota={self.cpu_quota_percent}%")

        return properties


# Production contract, calibrated in STEP 1d-3d against real workloads on this
# host (62 GiB RAM, 22 CPUs).  The measured peaks over read-only work, Python
# test runs, shell pipelines, git and parallel builds were 518 MiB of memory
# and 47 tasks, both reached by ``make -j22``; the values below keep a 4.1x and
# 5.4x margin on those.  ``CPUQuota=800%`` was the only point of the grid that
# left every workload except a massively parallel build entirely un-throttled
# (1.77x on ``make -j22``, versus 3.14x at 400% and 6.36x at 200%).
#
# Swap is pinned to zero deliberately: a memory cap that a workload can escape
# into swap is not a cap, and the measured swap delta under a runaway
# allocation is 0 kB.
#
# These are host-relative in nature — see the portability discussion in
# doc/source/resource_control.rst.  ``tasks_max`` and ``cpu_quota_percent``
# would both need revisiting on a many-core machine.

DEFAULT_CGROUP_LIMITS = CgroupLimits(
    memory_max_bytes=2 * 1024 * 1024 * 1024,
    memory_swap_max_bytes=0,
    tasks_max=256,
    cpu_quota_percent=800,
)


class SystemdScopeRunner:
    """Wrap an argv in a transient systemd user scope.

    Deliberately ignorant of command policy, capabilities, execution modes and
    the model layer: it receives an argv plus a :class:`CgroupLimits` and
    produces a structured argv.  One scope is created per sandboxed command.

    ``--collect`` makes systemd release the unit as soon as it finishes, so
    failed or OOM-killed scopes never accumulate and no ``reset-failed`` sweep
    is needed.
    """

    UNIT_PREFIX = "edgem-tool-"

    def __init__(
        self,
        *,
        systemd_run_binary: str = "/usr/bin/systemd-run",
        systemctl_binary: str = "/usr/bin/systemctl",
        runtime_dir: str | Path | None = None,
        terminate_timeout_seconds: float = 5.0,
    ):
        self.systemd_run_binary = systemd_run_binary
        self.systemctl_binary = systemctl_binary
        self.terminate_timeout_seconds = terminate_timeout_seconds
        self._runtime_dir = Path(runtime_dir) if runtime_dir is not None else None
        self.availability = CgroupAvailability.UNKNOWN

    @staticmethod
    def unit_name() -> str:
        """Unique unit name built only from a UUID hex; never from caller data."""
        return f"{SystemdScopeRunner.UNIT_PREFIX}{uuid.uuid4().hex}.scope"

    def runtime_dir(self) -> Path:
        if self._runtime_dir is not None:
            return self._runtime_dir

        configured = os.environ.get("XDG_RUNTIME_DIR")

        return Path(configured) if configured else Path(f"/run/user/{os.getuid()}")

    def supervisor_env(self) -> dict[str, str]:
        """Minimal environment letting systemd-run reach the user manager.

        This is the *supervisor's* environment, never the sandboxed command's:
        Bubblewrap's ``--clearenv`` remains the boundary, so nothing here can
        leak into the command.
        """
        return {"XDG_RUNTIME_DIR": str(self.runtime_dir())}

    def _delegated_controllers(self) -> frozenset[str]:
        """Controllers delegated to this user's systemd manager, from /proc."""

        try:
            relative = Path("/proc/self/cgroup").read_text().strip().split(":")[-1]
        except OSError:
            return frozenset()

        target = f"user@{os.getuid()}.service"
        parts = relative.strip("/").split("/")

        if target not in parts:
            return frozenset()

        prefix = parts[: parts.index(target) + 1]
        controllers = Path("/sys/fs/cgroup", *prefix, "cgroup.controllers")

        try:
            return frozenset(controllers.read_text().split())
        except OSError:
            return frozenset()

    @staticmethod
    def _binary_missing(binary: str) -> bool:
        if os.path.sep in binary:
            candidate = Path(binary)
            return not candidate.is_file() or not os.access(candidate, os.X_OK)

        return shutil.which(binary) is None

    def availability_for(self, limits: CgroupLimits) -> ToolResult | None:
        """Return a terminal failure, or ``None`` when the limits can be applied.

        Fail-closed by construction: there is no degraded mode.  An unusable
        mechanism with an active contract is an error, never plain Bubblewrap.
        """

        if not limits.active:
            return None

        # Delegation, not degradation. Inside a container there is no systemd
        # to make a scope with, but the limits are already enforced one level
        # up -- docker/spear-docker.sh passes --memory, --pids-limit and --cpus
        # that mirror DEFAULT_CGROUP_LIMITS, and sets this variable to say so.
        # The distinction matters: this is not "run unconfined because the
        # mechanism is missing", it is "the contract is honoured by the runtime
        # that owns the cgroup". A launcher that sets it without passing the
        # flags is lying, and nothing here can catch that -- which is why the
        # flags and the variable live on the same line of the same script.

        if self.delegated():
            self.availability = CgroupAvailability.DELEGATED
            return None

        if self._binary_missing(self.systemd_run_binary):
            self.availability = CgroupAvailability.SYSTEMD_RUN_ABSENT
            return ToolResult("failed", "resource control unavailable: systemd-run is absent")

        if self._binary_missing(self.systemctl_binary):
            self.availability = CgroupAvailability.SYSTEMCTL_ABSENT
            return ToolResult("failed", "resource control unavailable: systemctl is absent")

        runtime_dir = self.runtime_dir()

        if not runtime_dir.is_dir():
            self.availability = CgroupAvailability.USER_BUS_UNAVAILABLE
            return ToolResult(
                "failed", "resource control unavailable: XDG_RUNTIME_DIR is invalid"
            )

        # Existence and type only.  The socket is never connected to nor read,
        # and enabling linger is an administrator decision, never taken here.

        try:
            bus_mode = os.stat(runtime_dir / "bus").st_mode
        except OSError:
            self.availability = CgroupAvailability.USER_BUS_UNAVAILABLE
            return ToolResult("failed", "resource control unavailable: user bus is absent")

        if not stat.S_ISSOCK(bus_mode):
            self.availability = CgroupAvailability.USER_BUS_UNAVAILABLE
            return ToolResult("failed", "resource control unavailable: user bus is not a socket")

        missing = limits.required_controllers - self._delegated_controllers()

        if missing:
            self.availability = CgroupAvailability.CONTROLLERS_UNAVAILABLE
            names = ", ".join(sorted(missing))

            return ToolResult(
                "failed", f"resource control unavailable: cgroup controllers not delegated: {names}"
            )

        self.availability = CgroupAvailability.AVAILABLE

        return None

    @staticmethod
    def delegated() -> bool:
        """True when an outer runtime owns the cgroup — see availability_for."""
        return os.environ.get("SPEAR_RESOURCE_CONTROL") == "delegated"

    def wrap(self, argv: Sequence[str], limits: CgroupLimits, *, unit: str) -> list[str]:
        """Return ``argv`` unchanged, or wrapped in a transient scope."""

        if not argv:
            raise ValueError("scoped argv is empty")

        if not limits.active or self.delegated():
            # Delegated: the cgroup belongs to an outer runtime, and there is
            # no systemd here to make a scope with. availability_for() already
            # allows this; wrapping anyway executed a systemd-run that does not
            # exist, and the sandbox reported itself unavailable for EVERY
            # command -- the failure this delegation existed to prevent.

            return list(argv)

        wrapped = [
            self.systemd_run_binary,
            "--user",
            "--scope",
            "--quiet",
            "--collect",
            f"--unit={unit}",
        ]

        for prop in limits.systemd_properties():
            wrapped.extend(["-p", prop])

        wrapped.append("--")
        wrapped.extend(argv)

        return wrapped

    def terminate(self, unit: str) -> bool:
        """Kill every process of ``unit`` and let systemd collect it.

        Targets the unit by its unique generated name, so there is no PID-reuse
        race and no numeric PID tree walking.  The call is bounded: a hanging
        or failing systemctl can never block the supervisor.
        """

        try:
            completed = subprocess.run(
                [
                    self.systemctl_binary, "--user", "kill",
                    "--kill-whom=all", "--signal=KILL", unit,
                ],
                capture_output=True,
                text=True,
                timeout=self.terminate_timeout_seconds,
                shell=False,
            )
        except (subprocess.TimeoutExpired, OSError):
            return False

        return completed.returncode == 0


# Paths the sandbox builds its own runtime from. A workspace mounted at, or
# containing, one of these would either be shadowed by it or make it writable,
# so such a workspace keeps the neutral /workspace mount instead.

RUNTIME_MOUNTS = ("/usr", "/bin", "/lib", "/lib64", "/etc", "/proc", "/dev",
                  "/sys", "/run", "/boot")


def identity_mount_for(root: "Path | str",
                       reserved: "tuple[str, ...]" = ()) -> str | None:
    """Where to mount `root` so that its path is the SAME inside and outside.

    Tools that record absolute paths — CMake caches, generated Makefiles,
    compile_commands.json, object files — break when the tree answers to a
    different name inside the sandbox. A build configured on the host then
    fails in the sandbox with "cd /home/.../build: No such file or directory"
    and a CMakeCache pointing at a directory that no longer exists. Binding the
    tree at its own path removes that entire class of failure rather than
    papering over each instance.

    Returns None when the identity path is unsafe, and the caller falls back to
    the neutral mount.
    """
    r = os.path.realpath(str(root))

    if r == "/":
        return None                        # would shadow the whole runtime

    for m in RUNTIME_MOUNTS:
        # At or under a runtime mount: the bind would layer on top of it, and
        # under /usr it would make part of the read-only runtime writable.

        if r == m or r.startswith(m + "/"):
            return None

        # An ancestor of one: mounting here hides the runtime beneath it.

        if m.startswith(r.rstrip("/") + "/"):
            return None

    for res in reserved:
        # The sandbox home and tmpdir are created before this bind; a mount at
        # or above them would hide them and leave HOME/TMPDIR dangling.

        if r == res or res.startswith(r.rstrip("/") + "/"):
            return None

    return r


@dataclass(frozen=True)
class SandboxSpec:
    """Fixed, intentionally small filesystem and environment contract."""

    workspace_mount: str = "/workspace"
    home: str = "/home/sandbox"
    tmpdir: str = "/tmp"

    # /usr/local/bin FIRST, and it is not an afterthought: ib.md tells the
    # operator to symlink each cross-toolchain's whole bin/ there, so leaving
    # it off PATH meant `make so3` died on "aarch64-none-elf-gcc: No such file
    # or directory" while the binary sat in the sandbox all along, mounted and
    # unreachable. It costs no new mount -- /usr is already bound read-only.

    path: str = "/usr/local/bin:/usr/bin:/bin"

    # bitbake refuses to start without a UTF-8 locale, and --clearenv leaves
    # none. ib.md lists it among the host prerequisites for exactly this
    # reason; the failure is "Please make sure locale 'en_US.UTF-8' is
    # available", which reads like a host problem rather than a sandbox one.

    locale: str = "en_US.UTF-8"

    # World-readable identity data a build resolves uids and gids through.

    identity_files: tuple = ("/etc/passwd", "/etc/group")

    # Directory of symlinks that /usr/bin/cc and friends resolve through.

    alternatives: str = "/etc/alternatives"

    # Where cross-toolchains are unpacked. Bound read-only when present; the
    # /usr/local/bin symlinks that name them are useless without their target.

    toolchain_dirs: tuple = ("/opt/toolchains",)


class BubblewrapSandbox:
    """Run a pre-classified argv in a confined Bubblewrap environment.

    This class has no command-policy knowledge: callers supply an argv that
    they have already classified and authorized.  The host workspace is the
    sole writable host mount.  The runtime mount set is deliberately based on
    Ubuntu's merged-/usr layout: programs and their shared libraries reside
    under ``/usr`` and ``/bin``, ``/lib`` and ``/lib64`` are links into it.
    """

    def __init__(
        self,
        *,
        binary: str = "bwrap",
        slirp_binary: str = "slirp4netns",
        prlimit_binary: str = "/usr/bin/prlimit",
        timeout_seconds: int = 45,
        max_output_chars: int = 10_000,
        network_ready_timeout_seconds: int = 5,
        spec: SandboxSpec | None = None,
        scope_runner: SystemdScopeRunner | None = None,
    ):
        self.binary = binary
        self.slirp_binary = slirp_binary
        self.prlimit_binary = prlimit_binary
        self.timeout_seconds = timeout_seconds
        self.max_output_chars = max_output_chars
        self.network_ready_timeout_seconds = network_ready_timeout_seconds
        self.spec = spec or SandboxSpec()
        self.scope_runner = scope_runner or SystemdScopeRunner()
        self.availability = SandboxAvailability.UNKNOWN
        self.network_containment_failed = False

        # ``None`` until the helper's option set has been probed once.

        self._slirp_pinned_namespaces: bool | None = None

        # Created on first use: one /tmp for the whole session (_tmpdir_argv).

        self._session_tmpdir: str | None = None

    _UNIMPLEMENTED_PROFILE_CAPABILITIES = frozenset({
        Capability.GPU,
        Capability.SSH,
        Capability.CONTAINER_RUNTIME,
        Capability.SECRETS,
    })

    def _validate_profile(
        self,
        profile: ExecutionProfile | None,
        backend: NetworkBackend,
    ) -> None:
        if profile is None:
            if backend != NetworkBackend.CLOSED:
                raise ValueError("slirp4netns backend requires a network execution profile")

            return

        unsupported = profile.capabilities & self._UNIMPLEMENTED_PROFILE_CAPABILITIES

        if unsupported:
            names = ", ".join(sorted(capability.value for capability in unsupported))
            raise ValueError(f"capability/profile not implemented: {names}")

        if profile.network and backend != NetworkBackend.SLIRP4NETNS:
            raise ValueError("network execution profile requires slirp4netns backend")

        if not profile.network and backend != NetworkBackend.CLOSED:
            raise ValueError("slirp4netns backend requires network capability")

    def ensure_available(self, workspace: Workspace) -> ToolResult:
        """Preflight once and retain both success and terminal failures."""

        if self.availability == SandboxAvailability.AVAILABLE:
            return ToolResult("ok", "bubblewrap sandbox available")

        if self.availability in (
            SandboxAvailability.ABSENT,
            SandboxAvailability.INEXECUTABLE,
            SandboxAvailability.REFUSED,
        ):
            return ToolResult("failed", "bubblewrap sandbox unavailable")

        result = self.preflight(workspace)

        if result.ok:
            self.availability = SandboxAvailability.AVAILABLE
        elif self.availability == SandboxAvailability.UNKNOWN:
            self.availability = SandboxAvailability.REFUSED

        return result

    def _tmpdir_argv(self) -> "list[str]":
        """Give the session ONE /tmp instead of a fresh one per command.

        A tmpfs is created and destroyed with each bwrap invocation, so
        anything written to /tmp is gone before the next command runs. That
        breaks the ordinary way of checking work — compile to /tmp, then run
        it — since the binary vanishes between the two calls, and it forces
        every such check into a single unwieldy command line.

        A per-session directory is bound instead. It is created once, lives
        outside the workspace so it never pollutes the tree, is unique per
        sandbox instance so two sessions cannot see each other, and is removed
        when the process exits. SPEAR_SANDBOX_EPHEMERAL_TMP=1 restores the
        per-command tmpfs.
        """

        if os.environ.get("SPEAR_SANDBOX_EPHEMERAL_TMP") == "1":
            return ["--tmpfs", self.spec.tmpdir]

        if self._session_tmpdir is None:
            try:
                self._session_tmpdir = tempfile.mkdtemp(prefix="edgem-sandbox-")
            except OSError:
                return ["--tmpfs", self.spec.tmpdir]   # degrade, never fail

            atexit.register(shutil.rmtree, self._session_tmpdir,
                            ignore_errors=True)

        return ["--bind", self._session_tmpdir, self.spec.tmpdir]

    def mount_root(
        self, workspace: Workspace, profile: ExecutionProfile | None = None
    ) -> str:
        """The path the primary workspace answers to inside the sandbox.

        Defaults to the tree's own host path, so a build tree configured
        outside keeps working inside: CMake caches, generated Makefiles and
        compile_commands.json all embed absolute paths and break when the tree
        is renamed to /workspace. Falls back to the neutral mount when the
        identity path is unsafe (see identity_mount_for) or when
        SPEAR_SANDBOX_IDENTITY_MOUNT=0 turns the behaviour off.
        """

        if profile is not None and not profile.workspace_read and not profile.workspace_write:
            return self.spec.workspace_mount     # empty --dir, nothing to name

        return effective_mount_root(workspace.root, self.spec)

    def _workspace_mount_argv(
        self, workspace: Workspace, profile: ExecutionProfile | None
    ) -> list[str]:
        """Materialize exactly the workspace access granted by ``profile``.

        ``None`` is retained solely for legacy direct callers and preserves the
        historical read/write bind.  Functional command paths pass an explicit
        profile through ``CommandRunner``.
        """

        if profile is not None and not profile.workspace_read and not profile.workspace_write:
            # No read access at all: an empty directory, and no secondary tree
            # may leak in either.

            return ["--dir", self.spec.workspace_mount]

        flag = "--bind" if (profile is None or profile.workspace_write) else "--ro-bind"
        root_mount = self.mount_root(workspace, profile)
        identity = root_mount != self.spec.workspace_mount
        mounts = workspace.mount_map(root_mount, identity=identity)
        argv: list[str] = []

        # Primary first so the argv of a single-root workspace is unchanged.

        argv.extend([flag, str(workspace.root.resolve(strict=True)), root_mount])

        for extra in workspace.extra_roots:
            argv.extend([flag, str(extra), mounts[str(extra)]])

        return argv

    def _host_read_argv(self, profile: "ExecutionProfile | None") -> list[str]:
        """Expose the vetted outside paths, read-only, at their own path.

        ``--ro-bind-try`` rather than ``--ro-bind``: the path was checked when
        the command was classified, and a file that disappears in between must
        make the command report ENOENT, not make bwrap fail to start.
        """

        if profile is None or not profile.host_read_paths:
            return []

        argv: list[str] = []

        for path in profile.host_read_paths:
            argv.extend(["--ro-bind-try", path, path])

        return argv

    def build_argv(
        self,
        workspace: Workspace,
        command_argv: Sequence[str],
        profile: ExecutionProfile | None = None,
        backend: NetworkBackend = NetworkBackend.CLOSED,
        *,
        resource_limits: ResourceLimits | None = None,
        network_files: tuple[Path, Path, Path | None] | None = None,
        info_fd: int | None = None,
        block_fd: int | None = None,
        sync_fd: int | None = None,
    ) -> list[str]:
        """Build a structured Bubblewrap argv; never invoke a host shell."""

        if not command_argv:
            raise ValueError("sandbox command argv is empty")

        self._validate_profile(profile, backend)
        argv = [
            self.binary,
            "--die-with-parent",
            "--new-session",
            "--unshare-user",
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-uts",
            "--unshare-net",
            "--clearenv",
            "--setenv", "PATH", self.spec.path,
            "--setenv", "HOME", self.spec.home,
            "--setenv", "TMPDIR", self.spec.tmpdir,
            "--setenv", "LANG", self.spec.locale,
            "--setenv", "LC_ALL", self.spec.locale,
            "--ro-bind", "/usr", "/usr",
            "--symlink", "usr/bin", "/bin",
            "--symlink", "usr/lib", "/lib",
            "--symlink", "usr/lib64", "/lib64",
            "--proc", "/proc",
            "--dev", "/dev",
            *self._tmpdir_argv(),
            "--dir", "/home",
            "--dir", self.spec.home,
            # A tmpfs rather than a plain --dir: it is a real mount point, so
            # it can be remounted read-only once everything is in place (see
            # the --remount-ro below).  A writable /etc would weaken the
            # "nothing outside /workspace is writable" property.
            "--tmpfs", "/etc",
        ]

        # ``cc``, ``awk``, ``editor`` and friends are symlinks into
        # /etc/alternatives.  Without it they dangle and a plain ``make``
        # fails with "cc: No such file or directory" while ``gcc`` works,
        # which is a confusing way to lose the C toolchain.  The directory
        # holds no data of its own: every entry is a symlink into /usr, which
        # is already visible read-only.  Nothing else from /etc is exposed.

        if Path(self.spec.alternatives).is_dir():
            argv.extend(["--ro-bind", self.spec.alternatives, "/etc/alternatives"])

        # Identity files, read-only. /etc is a tmpfs so that nothing outside the
        # workspace is writable, but build tools resolve uids through these:
        # bitbake's is_local_uid() opens /etc/passwd and dies with FileNotFound
        # if it is absent, which reads as a corrupt host rather than a hidden
        # file. They carry no secret -- shadow and gshadow stay out, and the
        # command policy refuses them by name.

        for identity in self.spec.identity_files:
            if Path(identity).is_file():
                argv.extend(["--ro-bind", identity, identity])

        # The cross-toolchains. ib.md tells the operator to symlink each one's
        # bin/ into /usr/local/bin, and /usr is bound -- but the symlinks point
        # into /opt/toolchains, which was not, so every one of them dangled and
        # `make so3` died on "aarch64-none-elf-gcc: No such file or directory"
        # with the compiler sitting right there. Read-only: a toolchain is
        # something the build reads, never something it writes.

        for extra in self.spec.toolchain_dirs:
            if Path(extra).is_dir():
                argv.extend(["--ro-bind", extra, extra])

        argv.extend(self._host_read_argv(profile))

        # After the outside reads on purpose: a declared tree nested under one
        # of them must keep the access the workspace grants it, and in bwrap
        # the later bind is the one in force.

        argv.extend(self._workspace_mount_argv(workspace, profile))
        argv.extend(["--chdir", self.mount_root(workspace, profile)])

        if network_files is not None:
            resolv_conf, nsswitch_conf, ca_bundle = network_files

            # /etc itself is already created above.

            argv.extend([
                "--ro-bind", str(resolv_conf), "/etc/resolv.conf",
                "--ro-bind", str(nsswitch_conf), "/etc/nsswitch.conf",
            ])

            if ca_bundle is not None:
                argv.extend([
                    "--dir", "/etc/ssl",
                    "--dir", "/etc/ssl/certs",
                    "--ro-bind", str(ca_bundle), "/etc/ssl/certs/ca-certificates.crt",
                ])

        # Everything that had to be placed under /etc is placed; seal it.  This
        # must stay after the network files, which are bound into it above.

        argv.extend(["--remount-ro", "/etc"])

        if info_fd is not None:
            argv.extend(["--info-fd", str(info_fd)])

        if block_fd is not None:
            argv.extend(["--block-fd", str(block_fd)])

        if sync_fd is not None:
            argv.extend(["--sync-fd", str(sync_fd)])

        if resource_limits is not None and resource_limits.active:
            argv.extend([
                self.prlimit_binary,
                *resource_limits.prlimit_options(),
                "--",
            ])

        argv.extend(command_argv)

        return argv

    def _binary_status(self) -> ToolResult | None:
        """Return a terminal availability result, or ``None`` if runnable."""

        if os.path.sep in self.binary:
            candidate = Path(self.binary)

            if not candidate.exists():
                self.availability = SandboxAvailability.ABSENT
                return ToolResult("failed", "bubblewrap sandbox unavailable: bwrap is absent")

            if not candidate.is_file() or not os.access(candidate, os.X_OK):
                self.availability = SandboxAvailability.INEXECUTABLE
                return ToolResult("failed", "bubblewrap sandbox unavailable: bwrap is not executable")

            return None

        if shutil.which(self.binary, path=self.spec.path) is None:
            self.availability = SandboxAvailability.ABSENT
            return ToolResult("failed", "bubblewrap sandbox unavailable: bwrap is absent")

        return None

    def _slirp_status(self) -> ToolResult | None:
        if os.path.sep in self.slirp_binary:
            candidate = Path(self.slirp_binary)

            if not candidate.exists():
                return ToolResult("failed", "slirp4netns network backend unavailable: helper is absent")

            if not candidate.is_file() or not os.access(candidate, os.X_OK):
                return ToolResult("failed", "slirp4netns network backend unavailable: helper is not executable")

            return None

        if shutil.which(self.slirp_binary, path=self.spec.path) is None:
            return ToolResult("failed", "slirp4netns network backend unavailable: helper is absent")

        return None

    def _slirp_pinned_namespace_status(self) -> ToolResult | None:
        """Fail closed unless the helper can attach through pinned namespaces.

        Probed once, outside the sandbox setup, never between the info-fd read
        and the helper spawn.  There is deliberately no fallback to the
        PID-based attachment: that form is race-prone by construction.
        """
        unsupported = ToolResult(
            "failed",
            "slirp4netns network backend unavailable: "
            "pinned namespace attachment requires --netns-type and --userns-path",
        )

        if self._slirp_pinned_namespaces is True:
            return None

        if self._slirp_pinned_namespaces is False:
            return unsupported

        try:
            completed = subprocess.run(
                [self.slirp_binary, "--help"],
                capture_output=True, text=True, timeout=5, shell=False, env={},
            )
        except (OSError, subprocess.TimeoutExpired):
            self._slirp_pinned_namespaces = False
            return unsupported

        options = (completed.stdout or "") + (completed.stderr or "")
        self._slirp_pinned_namespaces = (
            "--netns-type" in options and "--userns-path" in options
        )

        return None if self._slirp_pinned_namespaces else unsupported

    def _prlimit_status(self, resource_limits: ResourceLimits | None) -> ToolResult | None:
        """Fail closed rather than silently dropping an active resource contract."""

        if resource_limits is None or not resource_limits.active:
            return None

        if os.path.sep in self.prlimit_binary:
            candidate = Path(self.prlimit_binary)

            if not candidate.exists():
                return ToolResult("failed", "resource limits unavailable: prlimit is absent")

            if not candidate.is_file() or not os.access(candidate, os.X_OK):
                return ToolResult("failed", "resource limits unavailable: prlimit is not executable")

            return None

        if shutil.which(self.prlimit_binary, path=self.spec.path) is None:
            return ToolResult("failed", "resource limits unavailable: prlimit is absent")

        return None

    @staticmethod
    def _pidfd_status() -> ToolResult | None:
        if not callable(getattr(os, "pidfd_open", None)):
            return ToolResult("failed", "slirp4netns network backend unavailable: pidfd_open is required")

        if not callable(getattr(signal, "pidfd_send_signal", None)):
            return ToolResult(
                "failed", "slirp4netns network backend unavailable: pidfd_send_signal is required"
            )

        return None

    def _result_from_completed(self, completed: subprocess.CompletedProcess[str]) -> ToolResult:
        stdout = (completed.stdout or "")[: self.max_output_chars]
        stderr = (completed.stderr or "")[: self.max_output_chars]

        if completed.returncode == 0:
            return ToolResult("ok", "bubblewrap sandbox command completed", stdout, stderr)

        return ToolResult(
            "failed",
            "bubblewrap sandbox command failed",
            stdout=stdout,
            stderr=stderr,
            exit_code=completed.returncode,
        )

    def run(
        self,
        workspace: Workspace,
        command_argv: Sequence[str],
        profile: ExecutionProfile | None = None,
        backend: NetworkBackend = NetworkBackend.CLOSED,
        resource_limits: ResourceLimits | None = None,
        cgroup_limits: CgroupLimits | None = None,
        cancellation: object | None = None,
    ) -> ToolResult:
        """Execute an argv in Bubblewrap and contain ordinary execution errors.

        ``cgroup_limits`` defaults to *no contract*, not to
        ``DEFAULT_CGROUP_LIMITS``: the sandbox is a mechanism, and the
        production contract belongs to :class:`CommandRunner`, which passes it
        explicitly.  Keeping it that way also means preflight and direct
        callers do not require a systemd user bus.
        """
        cgroup_limits = cgroup_limits if cgroup_limits is not None else CgroupLimits()
        unavailable = self._binary_status()

        if unavailable is not None:
            return unavailable

        limits_unavailable = self._prlimit_status(resource_limits)

        if limits_unavailable is not None:
            return limits_unavailable

        # Fail closed before anything is spawned: an active resource contract
        # that cannot be honoured never degrades to unlimited execution.

        cgroup_unavailable = self.scope_runner.availability_for(cgroup_limits)

        if cgroup_unavailable is not None:
            return cgroup_unavailable

        if backend == NetworkBackend.SLIRP4NETNS:
            if self.network_containment_failed:
                return ToolResult(
                    "failed", "slirp4netns network backend unavailable: containment failure"
                )

            slirp_unavailable = self._slirp_status()

            if slirp_unavailable is not None:
                return slirp_unavailable

            pinned_unavailable = self._slirp_pinned_namespace_status()

            if pinned_unavailable is not None:
                return pinned_unavailable

            pidfd_unavailable = self._pidfd_status()

            if pidfd_unavailable is not None:
                return pidfd_unavailable

            try:
                self._validate_profile(profile, backend)
            except ValueError as exc:
                return ToolResult("failed", str(exc))

            return self._run_with_slirp(
                workspace, command_argv, profile, resource_limits, cgroup_limits
            )

        try:
            bwrap_argv = self.build_argv(
                workspace,
                command_argv,
                profile=profile,
                backend=backend,
                resource_limits=resource_limits,
            )
        except ValueError as exc:
            return ToolResult("failed", f"could not build bubblewrap sandbox command: {exc}")

        # The unit name is generated before the spawn and retained for the whole
        # execution: cleanup never has to discover, glob or guess a scope.

        unit = self.scope_runner.unit_name() if cgroup_limits.active else None
        argv = (
            bwrap_argv if unit is None
            else self.scope_runner.wrap(bwrap_argv, cgroup_limits, unit=unit)
        )
        env = {} if unit is None else self.scope_runner.supervisor_env()

        try:
            process = subprocess.Popen(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=False,
                env=env,
            )

            if cancellation is None:
                try:
                    stdout, stderr = process.communicate(timeout=self.timeout_seconds)
                    timed_out = False
                except subprocess.TimeoutExpired:
                    timed_out = True
            else:
                deadline = time.monotonic() + self.timeout_seconds

                while True:
                    if bool(getattr(cancellation, "is_cancelled", False)):
                        if unit is not None:
                            self.scope_runner.terminate(unit)

                        self._stop_process(process, reap_output=False)

                        return ToolResult("cancelled", "bubblewrap sandbox command cancelled")

                    remaining = deadline - time.monotonic()

                    if remaining <= 0:
                        stdout = stderr = None
                        timed_out = True

                        break

                    try:
                        stdout, stderr = process.communicate(timeout=min(0.1, remaining))
                        timed_out = False

                        break
                    except subprocess.TimeoutExpired:
                        continue

            if timed_out:
                # Ask systemd to kill the whole scope first: it owns the unit and
                # reaches descendants that the Popen handle alone cannot.  Every
                # wait here is bounded, so a stuck tree cannot block the caller.

                if unit is not None:
                    self.scope_runner.terminate(unit)

                # ``reap_output=False``: never wait for EOF here.  A descendant
                # can still hold the inherited pipes, and every wait on this
                # path must stay bounded.

                self._stop_process(process, reap_output=False)

                return ToolResult(
                    "timeout", f"bubblewrap sandbox timed out after {self.timeout_seconds}s"
                )

            completed = subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)
        except OSError as exc:
            self.availability = SandboxAvailability.INEXECUTABLE
            return ToolResult("failed", f"bubblewrap sandbox unavailable: {exc}")

        return self._result_from_completed(completed)

    @staticmethod
    def _safe_close(fd: int | None) -> None:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass

    @staticmethod
    def _stop_process(
        process: subprocess.Popen[str] | None, *, reap_output: bool = True
    ) -> None:
        if process is None:
            return

        try:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=1)

        if not reap_output:
            # bwrap can have already exited while its blocked child still owns
            # the inherited stdout/stderr pipe. Do not wait for EOF before the
            # pidfd protocol has proved that child dead.

            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    stream.close()

            return

        # ``Popen`` owns pipe file objects even after the process has exited.
        # Reap them on normal error/lifecycle paths to avoid leaked descriptors.

        try:
            process.communicate(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()

    @staticmethod
    def _stop_namespace_child(pidfd: int | None) -> bool:
        """Terminate the authenticated bwrap child without a PID-reuse race."""

        if pidfd is None:
            return True

        try:
            signal.pidfd_send_signal(pidfd, signal.SIGKILL)
        except ProcessLookupError:
            # The child already exited, which is equivalent to a successful
            # stop for the purpose of releasing the blocker.

            return True
        except OSError:
            # Retry once through the same pidfd.  There is deliberately no
            # ``os.kill(pid)`` fallback: a PID may have been reused.

            try:
                signal.pidfd_send_signal(pidfd, signal.SIGKILL)
            except ProcessLookupError:
                return True
            except OSError:
                return False

        return True

    @staticmethod
    def _wait_pidfd_exit(pidfd: int | None, timeout_seconds: float) -> bool:
        """Observe child termination through its pidfd, never its numeric PID."""

        if pidfd is None:
            return False

        readable, _, _ = select.select([pidfd], [], [], timeout_seconds)

        return bool(readable)

    @staticmethod
    def _wait_fd(fd: int, timeout_seconds: int) -> bytes | None:
        readable, _, _ = select.select([fd], [], [], timeout_seconds)

        if not readable:
            return None

        return os.read(fd, 8192)

    @staticmethod
    def _wait_json_fd(fd: int, timeout_seconds: int) -> bytes | None:
        """Read Bubblewrap's multi-write info JSON without assuming pipe atomicity."""
        deadline = time.monotonic() + timeout_seconds
        data = bytearray()

        while time.monotonic() < deadline:
            remaining = max(0, deadline - time.monotonic())
            readable, _, _ = select.select([fd], [], [], remaining)

            if not readable:
                return None

            chunk = os.read(fd, 8192)

            if not chunk:
                return bytes(data) if data else None

            data.extend(chunk)

            try:
                json.loads(data.decode("utf-8"))
                return bytes(data)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue

        return None

    def _network_files(self, directory: Path) -> tuple[Path, Path, Path | None]:
        """Create the minimal resolver configuration for slirp's DNS proxy."""
        resolv_conf = directory / "resolv.conf"
        nsswitch_conf = directory / "nsswitch.conf"
        resolv_conf.write_text("nameserver 10.0.2.3\n", encoding="utf-8")
        nsswitch_conf.write_text("hosts: files dns\n", encoding="utf-8")
        ca_bundle = Path("/etc/ssl/certs/ca-certificates.crt")

        return resolv_conf, nsswitch_conf, ca_bundle if ca_bundle.is_file() else None

    def _run_with_slirp(
        self,
        workspace: Workspace,
        command_argv: Sequence[str],
        profile: ExecutionProfile | None,
        resource_limits: ResourceLimits | None,
        cgroup_limits: CgroupLimits | None = None,
    ) -> ToolResult:
        """Coordinate bwrap's private netns with slirp's ready/exit protocol."""
        cgroup_limits = cgroup_limits if cgroup_limits is not None else CgroupLimits()
        bwrap_process: subprocess.Popen[str] | None = None
        slirp_process: subprocess.Popen[str] | None = None
        sandbox_child_pid: int | None = None
        sandbox_child_pidfd: int | None = None

        # Pinned namespace identity handed to slirp4netns.  Distinct in role
        # from the pidfd: these pin *namespaces*, the pidfd pins the *process*.

        namespace_net_fd: int | None = None
        namespace_user_fd: int | None = None
        info_read = info_write = block_read = release_write = None
        ready_read = ready_write = exit_read = exit_write = None
        command_released = False
        child_termination_verified = False

        # Retained for the whole execution so cleanup targets exactly the scope
        # this call created.  ``None`` when the resource contract is inactive.

        scope_unit: str | None = None

        try:
            with tempfile.TemporaryDirectory(prefix="edgem-slirp-") as temporary:
                network_files = self._network_files(Path(temporary))
                info_read, info_write = os.pipe()
                block_read, release_write = os.pipe()
                bwrap_argv = self.build_argv(
                    workspace,
                    command_argv,
                    profile=profile,
                    backend=NetworkBackend.SLIRP4NETNS,
                    network_files=network_files,
                    info_fd=info_write,
                    block_fd=block_read,
                    sync_fd=release_write,
                    resource_limits=resource_limits,
                )

                # Only bwrap and its descendants enter the scope.  slirp4netns
                # is spawned separately below and stays outside, so an OOM or
                # task-limit hit inside the sandbox never kills the helper.

                if cgroup_limits.active:
                    scope_unit = self.scope_runner.unit_name()
                    scoped_argv = self.scope_runner.wrap(
                        bwrap_argv, cgroup_limits, unit=scope_unit
                    )
                    scope_env = self.scope_runner.supervisor_env()
                else:
                    scoped_argv = bwrap_argv
                    scope_env = {}

                bwrap_process = subprocess.Popen(
                    scoped_argv,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    shell=False,
                    pass_fds=(info_write, block_read, release_write),
                    env=scope_env,
                )
                self._safe_close(info_write)
                info_write = None
                self._safe_close(block_read)
                block_read = None
                info_data = self._wait_json_fd(info_read, self.network_ready_timeout_seconds)

                if not info_data:
                    stdout, stderr = bwrap_process.communicate(timeout=1)
                    return ToolResult(
                        "failed", "network sandbox setup failed before slirp4netns",
                        stdout=(stdout or "")[: self.max_output_chars],
                        stderr=(stderr or "")[: self.max_output_chars],
                        exit_code=bwrap_process.returncode,
                    )

                try:
                    info = json.loads(info_data.decode("utf-8"))

                    if not isinstance(info, dict):
                        raise ValueError("namespace info is not an object")

                    child_pid = info.get("child-pid")

                    if type(child_pid) is not int or child_pid <= 0:
                        raise ValueError("namespace info child-pid is invalid")

                    reported_netns = info.get("net-namespace")

                    if type(reported_netns) is not int or reported_netns <= 0:
                        raise ValueError("namespace info net-namespace is invalid")

                    sandbox_child_pid = child_pid

                    # Pin the namespace identity *first*, before the pidfd and
                    # before any other work.  ``/proc/<pid>/ns/user`` is only
                    # correct for a few milliseconds: bwrap moves its child into
                    # a second, nested user namespace right after reporting the
                    # info-fd JSON, and a helper that joins that nested one has
                    # no authority over the netns.  The netns inode is stable and
                    # is cross-checked against what bwrap reported, and the
                    # owning user namespace is derived from the netns object
                    # itself, so the attachment no longer depends on timing.

                    namespace_net_fd = os.open(
                        f"/proc/{child_pid}/ns/net", os.O_RDONLY | os.O_CLOEXEC
                    )

                    if os.stat(namespace_net_fd).st_ino != reported_netns:
                        raise ValueError("pinned network namespace does not match namespace info")

                    if fcntl.ioctl(namespace_net_fd, NS_GET_NSTYPE) != CLONE_NEWNET:
                        raise ValueError("pinned namespace is not a network namespace")

                    namespace_user_fd = fcntl.ioctl(namespace_net_fd, NS_GET_USERNS)

                    if fcntl.ioctl(namespace_user_fd, NS_GET_NSTYPE) != CLONE_NEWUSER:
                        raise ValueError("owning namespace is not a user namespace")

                    sandbox_child_pidfd = os.pidfd_open(sandbox_child_pid)
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError, OSError) as exc:
                    return ToolResult("failed", f"network sandbox returned invalid namespace info: {exc}")

                ready_read, ready_write = os.pipe()
                exit_read, exit_write = os.pipe()
                slirp_process = subprocess.Popen(
                    [
                        self.slirp_binary,
                        "--configure",
                        "--disable-host-loopback",
                        "--netns-type=path",
                        f"--userns-path=/proc/self/fd/{namespace_user_fd}",
                        "--ready-fd", str(ready_write),
                        "--exit-fd", str(exit_read),
                        f"/proc/self/fd/{namespace_net_fd}",
                        "tap0",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    shell=False,
                    pass_fds=(ready_write, exit_read, namespace_net_fd, namespace_user_fd),
                    env={},
                )
                self._safe_close(ready_write)
                ready_write = None
                self._safe_close(exit_read)
                exit_read = None
                ready_data = self._wait_fd(ready_read, self.network_ready_timeout_seconds)

                if not ready_data:
                    self._stop_process(slirp_process)
                    slirp_stdout, slirp_stderr = slirp_process.communicate()

                    return ToolResult(
                        "failed", "slirp4netns failed to become ready",
                        stdout=(slirp_stdout or "")[: self.max_output_chars],
                        stderr=(slirp_stderr or "")[: self.max_output_chars],
                    )

                # slirp has entered the namespaces; the handles are no longer
                # needed and must not outlive the attachment.

                self._safe_close(namespace_net_fd)
                namespace_net_fd = None
                self._safe_close(namespace_user_fd)
                namespace_user_fd = None

                # This is the only release primitive: Bubblewrap keeps its
                # own sync-fd writer, so closing our copy cannot manufacture
                # an EOF release on the block-fd reader.

                os.write(release_write, b"x")
                command_released = True
                self._safe_close(release_write)
                release_write = None

                try:
                    stdout, stderr = bwrap_process.communicate(timeout=self.timeout_seconds)
                except subprocess.TimeoutExpired:
                    # systemd owns the scope and reaches the whole tree; the
                    # pidfd sequence below still authenticates the namespace
                    # child.  Both are bounded.

                    if scope_unit is not None:
                        self.scope_runner.terminate(scope_unit)

                    self._stop_process(bwrap_process, reap_output=False)

                    return ToolResult("timeout", f"bubblewrap sandbox timed out after {self.timeout_seconds}s")

                completed = subprocess.CompletedProcess(bwrap_argv, bwrap_process.returncode, stdout, stderr)

                return self._result_from_completed(completed)

        # The coordinator must not leak an in-flight namespace/helper if a
        # Python-level failure occurs while reading control data.

        except Exception as exc:
            return ToolResult("failed", f"network sandbox setup failed: {exc}")
        finally:
            if not command_released and sandbox_child_pidfd is not None:
                # First request a PID-reuse-safe stop, then stop bwrap's
                # supervisor and observe its child. ``--sync-fd`` keeps the
                # pipe writer anchored inside bwrap, so closing our parent
                # writer below cannot release the blocked command.

                self._stop_namespace_child(sandbox_child_pidfd)

                if scope_unit is not None:
                    self.scope_runner.terminate(scope_unit)

                self._stop_process(bwrap_process, reap_output=False)
                self._stop_process(slirp_process)
                child_termination_verified = self._wait_pidfd_exit(
                    sandbox_child_pidfd, self.network_ready_timeout_seconds
                )

                if not child_termination_verified:
                    self.network_containment_failed = True
            else:
                self._stop_namespace_child(sandbox_child_pidfd)
                self._stop_process(bwrap_process, reap_output=False)
                self._stop_process(slirp_process)

            self._safe_close(sandbox_child_pidfd)
            self._safe_close(namespace_net_fd)
            self._safe_close(namespace_user_fd)
            self._safe_close(info_read)
            self._safe_close(info_write)
            self._safe_close(block_read)
            self._safe_close(release_write)
            self._safe_close(ready_read)
            self._safe_close(ready_write)
            self._safe_close(exit_read)

            # Closing exit-fd asks slirp to terminate before forced cleanup.

            self._safe_close(exit_write)

            if not command_released and sandbox_child_pidfd is not None and not child_termination_verified:
                return ToolResult(
                    "failed", "network sandbox cleanup could not verify child termination"
                )

    def preflight(self, workspace: Workspace) -> ToolResult:
        """Run a real, minimal sandbox rather than probing Bubblewrap's version."""
        unavailable = self._binary_status()

        if unavailable is not None:
            return unavailable

        # The mount is wherever mount_root() puts it — usually the tree's own
        # host path — so the check asks about that, not a hardcoded literal.

        mount = self.mount_root(workspace)
        result = self.run(
            workspace,
            [
                "/bin/sh", "-lc",
                f"test \"$PWD\" = {shlex.quote(mount)} && test -w {shlex.quote(mount)} "
                f"&& test \"$HOME\" = {shlex.quote(self.spec.home)} "
                f"&& test \"$TMPDIR\" = {shlex.quote(self.spec.tmpdir)}",
            ],
        )

        if result.ok:
            self.availability = SandboxAvailability.AVAILABLE
            return ToolResult(
                "ok",
                "bubblewrap sandbox preflight completed",
                stdout=result.stdout,
                stderr=result.stderr,
            )

        self.availability = SandboxAvailability.REFUSED

        return ToolResult(
            "failed",
            "bubblewrap sandbox refused by kernel or runtime",
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.exit_code,
        )

    def preflight_network(self, workspace: Workspace) -> ToolResult:
        """Exercise the complete pidfd + slirp path before a NETWORK use."""
        result = self.run(
            workspace,
            ["/bin/true"],
            profile=ExecutionProfile.from_capabilities({Capability.NETWORK}),
            backend=NetworkBackend.SLIRP4NETNS,
        )

        if result.ok:
            return ToolResult("ok", "slirp4netns network sandbox preflight completed")

        return ToolResult(
            result.status,
            f"slirp4netns network sandbox preflight failed: {result.summary}",
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.exit_code,
        )


class CommandRunner:
    """Execution boundary.  Simple commands always use ``shell=False``."""

    def __init__(
        self,
        *,
        timeout_seconds: int = 45,
        max_output_chars: int = 10_000,
        sandbox: BubblewrapSandbox | None = None,
        resource_limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
        cgroup_limits: CgroupLimits = DEFAULT_CGROUP_LIMITS,
    ):
        self.timeout_seconds = timeout_seconds
        self.max_output_chars = max_output_chars

        # Integration into ``rag_chat.run_cmd`` is deliberately deferred to
        # Phase 1b-3.  Keeping the reference here avoids another API change.

        self.sandbox = sandbox
        self.resource_limits = resource_limits

        # Resource contracts belong to the execution layer, never to the chat
        # layer: rag_chat knows nothing about MemoryMax/TasksMax/CPUQuota.

        self.cgroup_limits = cgroup_limits
        self._network_preflight: ToolResult | None = None
        self._network_preflight_sandbox: BubblewrapSandbox | None = None
        self._network_preflight_workspace: Path | None = None

    def ensure_sandbox(
        self, workspace: Workspace, profile: ExecutionProfile | None = None
    ) -> ToolResult:
        if self.sandbox is None:
            return ToolResult("failed", "bubblewrap sandbox unavailable")

        if profile is not None and profile.network:
            if getattr(self.sandbox, "network_containment_failed", False) is True:
                return ToolResult(
                    "failed", "slirp4netns network backend unavailable: containment failure"
                )

            canonical_workspace = workspace.root.resolve(strict=True)

            if (
                self._network_preflight_sandbox is not self.sandbox
                or self._network_preflight_workspace != canonical_workspace
            ):
                self._network_preflight = None
                self._network_preflight_sandbox = self.sandbox
                self._network_preflight_workspace = canonical_workspace

            if self._network_preflight is None:
                self._network_preflight = self.sandbox.preflight_network(workspace)

            return self._network_preflight

        return self.sandbox.ensure_available(workspace)

    def run_sandboxed(
        self,
        workspace: Workspace,
        argv: Sequence[str],
        profile: ExecutionProfile | None = None,
        *,
        availability: ToolResult | None = None,
        cancellation: object | None = None,
    ) -> ToolResult:
        """Execute a pre-classified argv through the configured sandbox."""

        if bool(getattr(cancellation, "is_cancelled", False)):
            return ToolResult("cancelled", "command cancelled before execution")

        availability = availability or self.ensure_sandbox(workspace, profile)

        if not availability.ok:
            return availability

        assert self.sandbox is not None

        if profile is not None and profile.network:
            kwargs = dict(
                profile=profile, backend=NetworkBackend.SLIRP4NETNS,
                resource_limits=self.resource_limits,
                cgroup_limits=self.cgroup_limits,
            )
        else:
            kwargs = dict(
                profile=profile, resource_limits=self.resource_limits,
                cgroup_limits=self.cgroup_limits,
            )

        if cancellation is not None:
            kwargs["cancellation"] = cancellation

        return self.sandbox.run(workspace, argv, **kwargs)

    def run_simple(self, assessment: CommandAssessment, workspace: Workspace) -> ToolResult:
        if not assessment.argv:
            return ToolResult("failed", "command has no argv")

        try:
            completed = subprocess.run(
                list(assessment.argv),
                cwd=workspace.root,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                shell=False,
            )
        except subprocess.TimeoutExpired:
            return ToolResult("timeout", f"timed out after {self.timeout_seconds}s")
        except OSError as exc:
            return ToolResult("failed", f"could not run command: {exc}")

        stdout = (completed.stdout or "")[: self.max_output_chars]
        stderr = (completed.stderr or "")[: self.max_output_chars]

        if completed.returncode == 0:
            return ToolResult("ok", "command completed", stdout=stdout, stderr=stderr)

        return ToolResult(
            "failed", "command failed", stdout=stdout, stderr=stderr, exit_code=completed.returncode
        )

    def run_complex_approved(self, command: str, workspace: Workspace) -> ToolResult:
        """ASK-only compatibility path; callers must authorize it first.

        This deliberately uses an explicit shell argv rather than ``shell=True``.
        It is not sandboxed and must never be used by safe or auto mode.
        """

        try:
            completed = subprocess.run(
                shell_argv(command, login=False),
                cwd=workspace.root,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                shell=False,
            )
        except subprocess.TimeoutExpired:
            return ToolResult("timeout", f"timed out after {self.timeout_seconds}s")
        except OSError as exc:
            return ToolResult("failed", f"could not run shell: {exc}")

        stdout = (completed.stdout or "")[: self.max_output_chars]
        stderr = (completed.stderr or "")[: self.max_output_chars]

        if completed.returncode == 0:
            return ToolResult("ok", "command completed", stdout=stdout, stderr=stderr)

        return ToolResult(
            "failed", "command failed", stdout=stdout, stderr=stderr, exit_code=completed.returncode
        )


_SECRET_ASSIGNMENT = re.compile(r"(?i)\b(api[_-]?key|token|password|secret)\s*=\s*[^\s]+")


class AuditLogger:
    """Append-only, metadata-only audit records for mutating tool attempts."""

    def __init__(self, log_path: str | Path):
        self.log_path = Path(log_path)

    @staticmethod
    def _command_summary(command: str | None) -> str | None:
        if not command:
            return None

        redacted = _SECRET_ASSIGNMENT.sub(r"\1=[REDACTED]", command)

        try:
            argv = shlex.split(redacted)
        except ValueError:
            return "[complex command]"

        # Never retain environment values in audit metadata; discard leading
        # KEY=value assignments before deriving the executable summary.

        while argv and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", argv[0]):
            argv.pop(0)

        return f"{Path(argv[0]).name} ({max(0, len(argv) - 1)} args)" if argv else None

    def record_mutation(
        self,
        *,
        action: str,
        mode: ExecutionMode,
        workspace: Workspace,
        paths: Sequence[str | Path] = (),
        command: str | None = None,
        approved: bool | None,
        result: ToolResult,
        sandboxed: bool | None = None,
        required_capabilities: Sequence[Capability] = (),
        granted_capabilities: Sequence[Capability] = (),
        execution_profile: Mapping[str, bool] | None = None,
    ) -> None:
        """Record an attempted mutation without content, output, or environment."""
        relative_paths: list[str] = []

        for path in paths:
            try:
                relative_paths.append(workspace.relative(path))
            except PathPolicyError:
                relative_paths.append("[rejected path]")

        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "mode": mode.value,
            "workspace": ".",
            "paths": relative_paths,
            "command": self._command_summary(command),
            "approved": approved,
            "status": result.status,
            "summary": result.summary[:500],
            "sandboxed": sandboxed,
            "required_capabilities": sorted(capability.value for capability in required_capabilities),
            "granted_capabilities": sorted(capability.value for capability in granted_capabilities),
            "execution_profile": dict(execution_profile) if execution_profile else None,
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
