"""Command classification and authorization.

``CommandPolicy`` decides what a shell command needs and whether the current
mode grants it; ``ToolPolicy`` does the same for filesystem and persistence
mutations. Neither runs anything.
"""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass, replace
from pathlib import Path
from typing import AbstractSet, Callable, Sequence

from harness import control_overrides, target_policy
from harness.tool_primitives import (
    DEFAULT_CAPABILITY_POLICY,
    Capability,
    CapabilityPolicy,
    CommandClassification,
    ExecutionMode,
    ToolResult,
)
from harness.workspace import SandboxSpec, Workspace, effective_mount_root


# A redirection that duplicates a descriptor: `>&2`, the `>&1` of `2>&1`.

_FD_DUPLICATE = re.compile(r"[<>]&(?:\d+|-)(?![\w./])")


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
        "uniq", "cut", "sed", "tr", "printf", "nl", "tac", "rev", "column",
        "realpath", "nproc", "uname", "date", "id",
    })

    # `sed -i` edits in place. It is refused rather than granted workspace
    # write: edit_file shows a diff and refuses a file not read this turn,
    # and routing edits through it is what keeps them visible and reviewable.
    # Matches -i, -i.bak, -ni and --in-place[=SUFFIX].

    _SED_IN_PLACE = re.compile(r"^(--in-place|-[a-hj-zA-Z]*i)")
    _SHELL_EDITORS = frozenset({"ed", "red", "ex", "vi", "vim", "nvim", "view", "vimdiff",
                                "emacs", "nano", "pico", "joe", "mcedit"})
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
        "dash", "ash", "ksh", "mksh", "csh", "tcsh", "busybox", "pypy", "pypy3",
        "perl", "ruby", "node", "nodejs", "php", "lua", "tclsh", "Rscript",
    })

    # Every interpreter is one program whatever its version: refused as
    # `python3`, `python3.12` ran the same script unjudged.

    _VERSIONED_INTERPRETER = re.compile(
        r"(?:python|pypy|perl|ruby|lua|php|tclsh|node)\d[\d.]*")

    # An awk program that starts a command runs what this cannot read.

    _AWK_BINS = frozenset({"awk", "gawk", "mawk", "nawk"})
    _AWK_RUNS = re.compile(r"\bsystem\s*\(|\|\s*\"|\"\s*\|\s*getline|\|&")

    # find's actions: -exec and -ok run a command, judged as that command;
    # the others write or delete themselves.

    _FIND_RUNS = frozenset({"-exec", "-execdir", "-ok", "-okdir"})
    _FIND_WRITES = frozenset({"-delete", "-fprint", "-fprint0", "-fprintf", "-fls"})

    # Builtins carry no capability of their own: `cd` and `.` act through the
    # stage that follows them, and treating them as unknown programs would ask
    # for write on every pipeline that merely changes directory.

    SHELL_BUILTINS = frozenset({"cd", ".", "source", "export", "set", "unset",
                                "echo", "true", "false", "test", "[", "read"})
    NETWORK_BINS = frozenset({"curl", "wget"})
    SSH_BINS = frozenset({"ssh", "scp", "sftp"})
    CONTAINER_BINS = frozenset({"docker", "podman"})
    GPU_BINS = frozenset({"nvidia-smi"})
    GIT_NETWORK_WRITE = frozenset({"clone", "fetch", "pull"})

    # The git commands that keep their own repository's records as they work:
    # the index, the stash, HEAD. What they change in the worktree is judged
    # like any other write.

    GIT_LOCAL_WRITE = frozenset({"add", "restore", "checkout", "switch", "stash", "clean"})
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
            subcommand = (self.git_command(argv) or ("", ()))[0]

            if subcommand in self.GIT_LOCAL_WRITE:
                return frozenset({Capability.FILESYSTEM_READ, Capability.WORKSPACE_WRITE})

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

            # Anything else changes the index, the history or the tree. Inside
            # a script this answered "no capability at all", so `ls; git commit`
            # passed in SAFE while `git commit` alone was refused.

            return frozenset({Capability.FILESYSTEM_READ,
                              Capability.WORKSPACE_WRITE})

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
                or self._runs_code(binary)
                or binary in self.SHELL_BUILTINS
                or binary == "git")

    def _runs_code(self, binary: str) -> bool:
        """Whether `binary` is refused as what it runs: a privileged, removing
        or interpreting program."""
        return binary in self.DANGEROUS_BINS or bool(self._VERSIONED_INTERPRETER.fullmatch(binary))

    @classmethod
    def _syntax_check(cls, argv: Sequence[str]) -> bool:
        """`bash -n script`: the shell reads the script and runs none of it."""
        options = [arg for arg in argv[1:] if arg.startswith("-")]
        operands = [arg for arg in argv[1:] if not arg.startswith("-")]

        return (Path(argv[0]).name in cls._SCRIPT_SHELLS and len(operands) == 1
                and argv[1].startswith("-") and len(options) == len(argv) - 2
                and all(re.fullmatch(r"-[eunx]+", option) for option in options)
                and any("n" in option for option in options))

    def _stage_danger(self, argv: Sequence[str], depth: int) -> str:
        """Why one stage may not run whatever wraps it, or "".

        The rules that judge what a program does rather than what it names:
        an interpreter, an awk program that starts a command, and find's
        actions. The same answer for a command alone and inside a script.
        """
        binary = Path(argv[0]).name

        if self._syntax_check(argv):
            return ""

        if not self._is_module_test_run(argv) and (
                self._runs_code(binary)
                or ("/" in argv[0] and not self._is_workspace_program(argv[0]))):
            if binary in self._PYTHON_BINS or self._VERSIONED_INTERPRETER.fullmatch(binary):
                return f"{self._PYTHON_HINT} — so this command cannot run."

            if binary in self._SCRIPT_SHELLS:
                return (f"'{binary}' running a script file runs commands this cannot "
                        f"judge as written, so it is not allowlisted: run them as "
                        f"commands.")

            return (f"'{argv[0]}' is not allowlisted, so the pipeline that "
                    f"contains it cannot run.")

        if binary in self._AWK_BINS and any(self._AWK_RUNS.search(arg) for arg in argv[1:]):
            return "an awk program that starts a command runs what this cannot read."

        if binary == "find":
            if any(arg in self._FIND_WRITES for arg in argv[1:]):
                return "mutating find action"

            for index, arg in enumerate(argv):
                if arg not in self._FIND_RUNS:
                    continue

                inner = []

                for word in argv[index + 1:]:
                    if word in (";", "+"):
                        break

                    if word != "{}":
                        inner.append(word)

                if not inner:
                    return f"find {arg} without a command"

                if depth >= self._MAX_UNWRAP:
                    return "commands nested this deep cannot be read."

                found = self._stage_danger(inner, depth + 1)

                if found:
                    return found

        return ""

    # Session cleanup (provenance.py): a command that only removes named
    # objects may run when every one of them was created in this session.
    # The shape is read here, the objects are judged where the command's
    # directory is known. Anything this cannot read literally -- a variable,
    # a glob, a substitution, another program -- leaves the command refused.

    _REMOVERS = frozenset({"rm", "rmdir", "unlink"})
    _REMOVE_OPTION = re.compile(r"-[rRfvd]+|--(?:recursive|force|verbose|dir)")
    _BESIDE_CLEANUP = frozenset({"ls", "echo", "true", "false", "test", "[", "pwd", "stat",
                                 "cat", "head", "tail", "wc", "du"})
    _FIND_VALUED = frozenset({"-name", "-iname", "-path", "-ipath", "-type", "-maxdepth",
                              "-mindepth"})
    _UNREAD = re.compile(r"[$`*?\[\]{}~]")

    def cleanup_plan(self, command: str) -> "list[tuple[str, tuple[str, ...], bool]] | None":
        """[(kind, words, recursive)] when `command` only removes named objects.

        Kinds: "cd" (words: the directory) and "remove" (words: the objects;
        recursive for -r and for find's -delete and -exec rm, whose start
        points are removed in full). None for anything else.
        """
        command = re.sub(r"(?:^|\s)\d*(?:>>?\s*/dev/null|>&\d)(?=\s|$|[;&|])", " ", command or "")

        if not command.strip() or re.search(r"[$`<>]", command):
            return None

        plan = []

        for argv in self.stage_argvs(command):
            name = argv[0]

            if name == "cd" and len(argv) == 2 and not self._UNREAD.search(argv[1]):
                plan.append(("cd", (argv[1],), False))
            elif name in self._REMOVERS:
                options = [arg for arg in argv[1:] if arg.startswith("-") and arg != "--"]
                operands = [arg for arg in argv[1:] if not arg.startswith("-")]

                if (not operands or not all(self._REMOVE_OPTION.fullmatch(arg) for arg in options)
                        or any(self._UNREAD.search(arg) for arg in operands)):
                    return None

                plan.append(("remove", tuple(operands),
                             any(re.search(r"[rR]|recursive", arg) for arg in options)))
            elif name == "find":
                found = self._find_removal(argv)

                if found is None:
                    return None

                if found:
                    plan.append(("remove", found, True))
            elif name not in self._BESIDE_CLEANUP:
                return None

        return plan if any(kind == "remove" for kind, _, _ in plan) else None

    def _find_removal(self, argv: Sequence[str]) -> "tuple[str, ...] | None":
        """find's start points when it deletes, () when it only lists, None
        when its expression is not one read here."""
        index = 1

        while index < len(argv) and not argv[index].startswith(("-", "(", "!")):
            index += 1

        starts, expression = tuple(argv[1:index]), list(argv[index:])
        removes = False

        if any(self._UNREAD.search(start) for start in starts):
            return None

        while expression:
            word = expression.pop(0)

            if word in self._FIND_VALUED and expression:
                expression.pop(0)
            elif word in ("-delete",):
                removes = True
            elif word in ("-print", "-depth", "-xdev"):
                continue
            elif word == "-exec" and expression and expression[0] in self._REMOVERS:
                command = []

                while expression and expression[0] not in (";", "+"):
                    command.append(expression.pop(0))

                if not expression or not all(self._REMOVE_OPTION.fullmatch(arg) or arg == "{}"
                                             for arg in command[1:]):
                    return None

                expression.pop(0)
                removes = True
            else:
                return None

        if removes and not starts:
            return None

        return starts if removes else ()

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

    # A literal assignment: `F=x`, `F='x'`. A value built from a substitution
    # is not known before the command runs.

    _LITERAL_ASSIGNMENT = re.compile(
        r"(?:^|[\s;&|(])([A-Za-z_]\w*)=(?:'([^']*)'|\"([^\"$`]*)\"|([^\s;&|()<>'\"$`]*))")

    @classmethod
    def _writes_via_redirection(cls, command: str) -> bool:
        """True when a redirection actually creates or extends a file.

        `2>/dev/null` is a redirection that writes nothing, and demanding
        workspace:write for it made every read-only pipeline carrying the idiom
        unavailable in SAFE — the mode where searching a tree is the whole
        point. Nor do `2>&1` and `>&2`, which only duplicate a descriptor: read
        as writes, they handed `git apply x 2>&1` a writable tree.

        A destination held in a variable is the value the command assigned it,
        when that value is literal. Otherwise nothing says what it is, and it
        earns no write: the tree stays read-only and the write fails there.
        """
        return any(not target.startswith("$") for target in cls._redirect_targets(command))

    @classmethod
    def _redirect_targets(cls, command: str) -> "list[str]":
        """The files the redirections of `command` write, sinks set aside.

        A variable destination is its literal value; one with no literal
        value is returned as written, starting with `$`.
        """
        values = {match.group(1): next(group for group in match.groups()[1:] if group is not None)
                  for match in cls._LITERAL_ASSIGNMENT.finditer(command)}
        found = []

        for match in re.finditer(r"(?<![=])\d*>{1,2}\|?(&?)\s*([^\s;|&()<>]*)", command):
            duplicate, target = match.groups()

            if duplicate and re.fullmatch(r"\d+|-", target):
                continue

            variable = re.match(r"\$(?:\{(\w+)\}|(\w+))", target)

            if variable:
                name = variable.group(1) or variable.group(2)

                if name in values:
                    target = values[name] + target[variable.end():]

            if target not in cls._REDIRECT_SINKS:
                found.append(target)

        return found

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
        commands and dropping it would hide them from the danger scan. A
        dropped body takes its closing delimiter with it: on a line of its
        own, the word would read as a program.
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

        A process substitution `<(cmd)` is a stage of its own, opened by "<("
        and closed by ")<": what follows it is more of the command around it,
        its operands (`diff <(git show HEAD:a.c) a.c`), not a program.

        Returns text and separators interleaved, the shape re.split produced.
        """
        parts: list[str] = []
        current: list[str] = []
        quote = ""
        index, length = 0, len(command)
        opened: list[bool] = []

        while index < length:
            char = command[index]

            # Inside double quotes a backslash still escapes: `"a \" ; b"` is
            # one word. Read as the closing quote, `\"` let the `;` after it
            # split a `python3 -c "..."` program into stages, none of which
            # was the interpreter.

            if quote:
                current.append(char)

                if quote == '"' and char == "\\" and index + 1 < length:
                    current.append(command[index + 1])
                    index += 2

                    continue

                if char == quote:
                    quote = ""

                index += 1

                continue

            # A `#` that starts a word starts a comment, to the end of the
            # line: its apostrophes open no quote.

            if char == "#" and (not current or current[-1] in " \t"):
                end = command.find("\n", index)
                index = length if end < 0 else end

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

            # `2>&1` and `>&2` duplicate a descriptor and name no file: they
            # stay inside the word, or the `1` reads as a program.

            duplicate = _FD_DUPLICATE.match(command, index)

            if duplicate:
                current.append(duplicate.group(0))
                index = duplicate.end()

                continue

            # A newline ends a command as `;` does: `cat > x <<EOF ... EOF`
            # then `python3 x` on the next line is two commands.

            if char in "<>" and command[index + 1:index + 2] == "(":
                parts.append("".join(current))
                parts.append(command[index:index + 2])
                opened.append(True)
                current = []
                index += 2

                continue

            if char in "()":
                substitution = char == ")" and bool(opened) and opened.pop()

                if char == "(":
                    opened.append(False)

                parts.append("".join(current))
                parts.append(")<" if substitution else char)
                current = []
                index += 1

                continue

            if char in "|;&<>\n":
                parts.append("".join(current))
                parts.append(char)
                current = []
                index += 1

                continue

            current.append(char)
            index += 1

        parts.append("".join(current))

        return parts

    @staticmethod
    def _substitutions(command: str) -> "list[str] | None":
        """The commands substituted where the stages cannot see them.

        A `$(...)` outside quotes splits into stages of its own; one inside
        double quotes, and any backquoted one, does not, and ran unjudged:
        `echo "$(rm -rf src)"`. Their bodies, or None when one is not closed.
        """
        bodies: list[str] = []
        quote = ""
        index, length = 0, len(command)

        while index < length:
            char = command[index]

            if quote == "'":
                quote = "" if char == "'" else quote
                index += 1

                continue

            if char == "\\":
                index += 2

                continue

            if char == "`":
                end = index + 1

                while end < length and command[end] != "`":
                    end += 2 if command[end] == "\\" else 1

                if end >= length:
                    return None

                bodies.append(command[index + 1:end].replace("\\`", "`"))
                index = end + 1

                continue

            if quote == '"' and command.startswith("$(", index) \
                    and not command.startswith("$((", index):
                depth, end, inner = 1, index + 2, ""

                while end < length and depth:
                    if inner:
                        inner = "" if command[end] == inner else inner
                    elif command[end] == "\\":
                        end += 1
                    elif command[end] in "'\"":
                        inner = command[end]
                    elif command[end] == "(":
                        depth += 1
                    elif command[end] == ")":
                        depth -= 1

                    end += 1

                if depth:
                    return None

                bodies.append(command[index + 2:end - 1])
                index = end

                continue

            if char in "'\"" and not (quote == '"' and char == "'"):
                quote = "" if quote == char else char

            index += 1

        return bodies

    # Reserved words that introduce a command inside a compound one, and the
    # ones that only close it.

    _SHELL_OPENERS = frozenset({"do", "then", "else", "elif", "if", "while",
                                "until", "!"})
    _SHELL_CLOSERS = frozenset({"done", "fi"})

    @classmethod
    def _past_shell_keywords(cls, argv: "tuple[str, ...]") -> "tuple[str, ...]":
        """The program a stage really runs, or () when it runs none.

        A loop or a conditional is split into stages like any other command,
        and each stage then began with a reserved word that no allowlist knows:
        `for f in a b; do grep x $f; done` asked for workspace:write three
        times over. The word is dropped and what follows is classified as the
        command it is, so a loop is exactly as privileged as its body.

        `for NAME in WORDS` runs nothing itself; a command substitution among
        the words is its own stage. `command -v` only looks a name up, and
        `command X` runs X.
        """

        while argv:
            head = argv[0]

            if head == "for" or head in cls._SHELL_CLOSERS:
                return ()

            if head == "command":
                if len(argv) > 1 and argv[1] in ("-v", "-V"):
                    return ()

                argv = argv[1:]
            elif head in cls._SHELL_OPENERS or cls._ASSIGNMENT.match(head):
                argv = argv[1:]
            else:
                break

        return argv

    def _shell_capabilities(
        self, command: str, depth: int = 0
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

        substituted = self._substitutions(command)

        if substituted is None:
            danger = "a command substitution is not closed, so what it runs cannot be read."
            substituted = []

        for body in substituted:
            if depth >= self._MAX_UNWRAP:
                danger = danger or "substitutions nested this deep cannot be read."
                break

            found, inner_danger, inner_outside, inner_refusal = self._shell_capabilities(
                self._strip_heredoc_bodies(body), depth + 1)
            capabilities.update(found)
            danger = danger or inner_danger
            refusal = refusal or inner_refusal
            outside.extend(path for path in inner_outside if path not in outside)

        # `2>&1` names no file and no program. Left in, the stage splitter
        # cuts it at `>` and `&` and reads the trailing `1` as a command,
        # which no allowlist knows and which therefore asked for write.

        stages = self._shell_stages(re.sub(r"\d*>&(\d+|-)", " ", command))
        stdin_from_pipeline = False
        redirect_target = False
        redirect_reads = False
        operands_only = False

        for stage in stages:
            token = stage.strip()

            if not token:
                continue

            if token in self._SEPARATORS:
                stdin_from_pipeline = token == "|"
                operands_only = token == ")<"

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

                # A relative `..` is judged as the traversal rule judges any
                # operand: where the command stands is not known here, so
                # neither is where it leads. Called a write, an input `<
                # ../x.patch` read to the model as a refusal of something it
                # never asked for.

                if escapes:
                    if redirect_reads and target.startswith("/"):
                        found, denied = self._outside_reads([target])
                        refusal = refusal or denied
                        outside.extend(p for p in found if p not in outside)
                    elif not target.startswith("/"):
                        danger = danger or (
                            f"relative path leaves the workspace: '{target}' — "
                            f"name it by absolute path instead.")
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

            # A stage whose quoting does not close cannot be read, and what
            # cannot be read does not run.

            try:
                argv = tuple(shlex.split(token, posix=True))
            except ValueError:
                danger = danger or (f"its quoting cannot be read ({token[:40]!r}): "
                                    f"close every quote, or write the text with write_file.")
                continue

            # After a process substitution, the words are the operands of the
            # command around it: checked as paths, never run.

            if operands_only:
                operands_only = False
                found, denied = self._outside_reads(list(argv))
                refusal = refusal or denied
                outside.extend(path for path in found if path not in outside)
                continue

            # `VAR=value cmd` runs cmd: the assignments are its environment,
            # not the program. Read as argv[0], `BASE_DIR=/x/y ./post_image.sh`
            # was refused as a program named by an outside path.

            argv = self._past_shell_keywords(argv)

            if not argv:
                continue

            argv, script, unreadable = self._peeled(argv, token)

            if unreadable:
                danger = danger or (f"{unreadable}: what it would run cannot be read here, so "
                                    f"the pipeline that contains it cannot run.")
                continue

            if script is not None:
                if depth >= self._MAX_UNWRAP:
                    danger = danger or "shells nested this deep cannot be read."
                    continue

                inner, inner_danger, inner_outside, inner_refusal = self._shell_capabilities(
                    self._strip_heredoc_bodies(script), depth + 1)
                capabilities.update(inner)
                danger = danger or inner_danger
                refusal = refusal or inner_refusal
                outside.extend(path for path in inner_outside if path not in outside)
                stdin_from_pipeline = False
                continue

            if not argv:
                continue

            # In a script handed to a shell, a program that is an expansion
            # runs what it expands to, unseen.

            if depth and argv[0].startswith("$"):
                danger = danger or (f"'{argv[0]}' runs a program this cannot read, so the "
                                    f"pipeline that contains it cannot run.")
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

            # An editor run from the shell edits in place as `sed -i` does:
            # `printf '1a\nX\n.\nw\n' | ed -s f` wrote what sed -i was refused.

            if binary in self._SHELL_EDITORS:
                danger = danger or (
                    f"'{binary}' edits files in place, outside the checkpoint, so its "
                    f"change cannot be rolled back: use edit_file.")

            danger = danger or self._stage_danger(argv, depth)

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

    _STAGE_KEYWORDS = frozenset({"do", "then", "else", "elif", "!", "{"})

    _SEPARATORS = frozenset({"|", "||", "&&", ";", "&", "(", ")", "<", ">", "<(", ">(", ")<"})

    # Launchers run the command they are handed, and that command is what is
    # judged: wrapped, it keeps the authority it has alone and gains none.
    # A shell's -c hands over a script, judged as a script; xargs runs its
    # command with operands read at run time. A launcher whose options are
    # not read here runs something unseen, and is refused.

    _LAUNCHER_OPTIONS = {
        "timeout": ({"--preserve-status", "--foreground", "-v", "--verbose"},
                    ("-s", "--signal", "-k", "--kill-after")),
        "nice": (set(), ("-n", "--adjustment")),
        "env": ({"-i", "--ignore-environment"}, ("-u", "--unset")),
        "nohup": (set(), ()),
        "exec": ({"-c", "-l", "-cl", "-lc"}, ("-a",)),
        "command": ({"-p"}, ()),
        "time": ({"-p"}, ()),
        "xargs": ({"-0", "--null", "-r", "--no-run-if-empty", "-t", "--verbose", "-x",
                   "--exit", "-o", "--open-tty"},
                  ("-n", "--max-args", "-L", "--max-lines", "-P", "--max-procs", "-s",
                   "--max-chars", "-d", "--delimiter", "-E", "-a", "--arg-file", "-I",
                   "--replace")),
    }
    _LAUNCHERS = frozenset(_LAUNCHER_OPTIONS)
    _SCRIPT_SHELLS = frozenset({"bash", "sh", "dash", "ash", "ksh", "mksh", "zsh"})
    _MAX_UNWRAP = 4
    _DURATION = re.compile(r"\d+(?:\.\d+)?[smhd]?")
    _DUPLICATION = re.compile(r"\d*[<>]&(?:\d+|-)")

    def _launched(self, argv: Sequence[str], raw: str = "") -> "tuple[str, object]":
        """What one stage runs: ("argv", argv), ("script", text), or ("", why).

        A stage that launches nothing comes back unchanged, as does a launcher
        given nothing to run (`env` alone prints the environment).
        """
        name, rest = argv[0], list(argv[1:])

        if name in self._SCRIPT_SHELLS:
            return self._shell_script(argv, raw)

        if name not in self._LAUNCHERS:
            return "argv", tuple(argv)

        flags, valued = self._LAUNCHER_OPTIONS[name]

        while rest and rest[0].startswith("-") and rest[0] != "-":
            option = rest.pop(0)

            if option == "--":
                break

            if name == "command" and option in ("-v", "-V"):
                return "argv", tuple(argv)

            if option in valued and rest:
                rest.pop(0)
            elif not (option in flags
                      or any(option.startswith(long + "=") for long in valued if long[1] == "-")
                      or any(option.startswith(short) for short in valued if short[1] != "-")
                      or (name == "nice" and re.fullmatch(r"-\d+", option))):
                return "", f"{name} {option}"

        if name == "timeout":
            if not rest or not self._DURATION.fullmatch(rest[0]):
                return "", "timeout without a duration"

            rest.pop(0)

        while name == "env" and rest and self._ASSIGNMENT.match(rest[0]):
            rest.pop(0)

        return "argv", tuple(rest) if rest else tuple(argv)

    def _shell_script(self, argv: Sequence[str], raw: str = "") -> "tuple[str, object]":
        """The script of `bash -c SCRIPT`; any other use of the shell unchanged.

        Options are the inert ones only: -e, -u, -x, -o pipefail. Words after
        the script would become its $0, $1..., which the script then runs
        unseen, so they are refused. So is an expansion the outer shell makes
        inside the script: `bash -c "$(cat x)"` runs what x holds. In single
        quotes (`raw`, the stage as written, shows them) the script reaches
        the inner shell as it reads here.
        """
        rest = list(argv[1:])
        script = None

        while rest:
            option = rest.pop(0)

            if option in ("-o", "+o") and rest and rest[0] in ("pipefail", "errexit", "nounset",
                                                                "xtrace"):
                rest.pop(0)
                continue

            if not re.fullmatch(r"-[euxc]+", option):
                return "argv", tuple(argv)

            if "c" in option:
                if not rest:
                    return "", f"{argv[0]} -c without a script"

                script = rest.pop(0)
                break

        if script is None:
            return "argv", tuple(argv)

        if rest:
            return "", f"{argv[0]} -c with arguments for its script"

        if re.search(r"[$`]", script) and f"'{script}'" not in raw:
            return "", (f"{argv[0]} -c with a script the outer shell expands (single quotes "
                        f"pass it as written)")

        return "script", script

    def _peeled(self, argv: Sequence[str], raw: str = "") -> "tuple[tuple[str, ...], str | None, str]":
        """(argv, script, unreadable) of one stage, its launchers taken off.

        A descriptor duplication (`2>&1`) stays in its stage's words and is
        no argument of the command.
        """
        argv = tuple(word for index, word in enumerate(argv)
                     if index == 0 or not self._DUPLICATION.fullmatch(word))

        for _ in range(self._MAX_UNWRAP):
            kind, value = self._launched(argv, raw)

            if not kind:
                return argv, None, value

            if kind == "script":
                return argv, value, ""

            if value == argv:
                break

            argv = value

            while argv and self._ASSIGNMENT.match(argv[0]):
                argv = argv[1:]

            if not argv:
                break

        return argv, None, ""

    def unwrapped(self, argv: Sequence[str], depth: int = 0,
                  raw: str = "") -> "list[tuple[str, ...]]":
        """The argvs one stage really runs, launchers taken off.

        An unreadable launcher is kept as it is, so that whatever asks about
        builds or git finds no build and no git in it.
        """
        peeled, script, unreadable = self._peeled(argv, raw)

        if unreadable:
            return [tuple(argv)]

        if script is not None:
            return self.stage_argvs(script, depth + 1) if depth < self._MAX_UNWRAP else [tuple(argv)]

        return [peeled] if peeled else []

    def stage_argvs(self, command: str, depth: int = 0) -> "list[tuple[str, ...]]":
        """The argv each stage of `command` runs, environment prefixes and
        launchers set aside."""
        found = []
        redirect_target = False
        operands_only = False

        for part in self._shell_stages(self._strip_heredoc_bodies(command or "")):
            if part.strip() in self._SEPARATORS:
                redirect_target = part.strip() in {"<", ">"}
                operands_only = part.strip() == ")<"
                continue

            if operands_only:
                operands_only = False
                continue

            try:
                argv = tuple(shlex.split(part, posix=True))
            except ValueError:
                continue

            # What follows `>` or `<` is its file; the rest of the fragment
            # is the command that continues.

            if redirect_target and argv:
                argv = argv[1:]
                redirect_target = False

            # A descriptor duplication left after a `$(...)` is no program.

            while argv and (self._ASSIGNMENT.match(argv[0])
                            or argv[0] in self._STAGE_KEYWORDS
                            or self._DUPLICATION.fullmatch(argv[0])):
                argv = argv[1:]

            if argv:
                found.extend(self.unwrapped(argv, depth, part))

        return found

    # What builds or tests: the build tools and test runners the policy lets
    # write, BitBake, and a project's own build drivers: its build script,
    # its configure step, and a BitBake layer's updiff.sh, which regenerates a
    # recipe's patch series from its unpacked tree.

    BUILD_BINS = (WORKSPACE_MUTATING_BINS - {"tee"}) | frozenset({
        "gmake", "meson", "scons", "ctest", "bitbake"})
    _BUILD_SCRIPT = re.compile(
        r"(?:^|[-_])build(?:[-_][\w-]*)?\.sh$|^configure$|^autogen\.sh$|^updiff\.sh$")

    # What may run beside a build without taking its authority: what writes
    # no file. `tee` qualifies for a file outside the generated trees.

    _BESIDE_A_BUILD = SHELL_BUILTINS | READ_ONLY_BINS | frozenset({
        "printf", "nproc", "sleep", "wait", "date", "uname"})
    _OUTPUT_PATH = re.compile(r"(?:^|/)(?:generated|build/tmp)(?:/|$)")

    def _writes_output(self, path: str) -> bool:
        return (path.startswith("$") or bool(self._OUTPUT_PATH.search(path))
                or target_policy.is_snapshot(path))

    def builds(self, command: str) -> bool:
        """Whether `command` builds or tests: its output is the build's to write.

        The authority is the build's alone. Every other stage must write
        nothing, and no redirection may write into a generated tree: a
        command beside a build, or a launcher around one, gains nothing from
        it (stage_argvs judges what a launcher runs).
        """
        found = False

        for argv in self.stage_argvs(command):
            name = Path(argv[0]).name

            if (name in self.BUILD_BINS or self._is_module_test_run(argv)
                    or self._BUILD_SCRIPT.search(name)):
                found = True
            elif argv[0] == "tee":
                if any(self._writes_output(arg) for arg in argv[1:] if not arg.startswith("-")):
                    return False
            elif argv[0] not in self._BESIDE_A_BUILD:
                return False

        return found and not any(self._writes_output(target)
                                 for target in self._redirect_targets(command))

    # Global options that leave git where it is and reading what it reads.

    _GIT_PLAIN_OPTIONS = frozenset({
        "--no-pager", "-P", "-p", "--paginate", "--no-optional-locks", "--literal-pathspecs",
        "--glob-pathspecs", "--noglob-pathspecs", "--icase-pathspecs", "--no-replace-objects",
        "--no-lazy-fetch", "--no-advice"})

    def git_command(self, argv: Sequence[str]) -> "tuple[str, tuple[str, ...]] | None":
        """(subcommand, the -C directories in order) of a git argv.

        None when a global option sends git somewhere else or runs something
        of its own: -c, --git-dir, --work-tree, --exec-path, --namespace...
        """
        directories: list[str] = []
        index = 1

        while index < len(argv) and argv[index].startswith("-"):
            option = argv[index]

            if option == "-C" and index + 1 < len(argv):
                directories.append(argv[index + 1])
                index += 2
            elif option in self._GIT_PLAIN_OPTIONS:
                index += 1
            else:
                return None

        return (argv[index] if index < len(argv) else ""), tuple(directories)

    def writes_git(self, command: str) -> bool:
        """Whether `command` runs a git command that writes a repository's records."""
        return any(argv[0] == "git"
                   and (self.git_command(argv) or ("", ()))[0] in (self.GIT_NETWORK_WRITE
                                                                  | self.GIT_LOCAL_WRITE)
                   for argv in self.stage_argvs(command))

    # What may run beside a git command that writes its records: what runs
    # no program of the command's choosing and writes no file.

    _BESIDE_GIT = (READ_ONLY_BINS - {"find"}) | frozenset({"echo", "printf", "true", "false",
                                                           "test", "[", "pwd"})

    # Interactive forms hand the change to an editor or a prompt.

    _GIT_INTERACTIVE = re.compile(r"^(?:-[a-zA-Z]*[pie][a-zA-Z]*|--(?:patch|interactive|edit)\b.*)$")

    # Words that steer git, or a program it runs, from the environment.

    _GIT_STEERING = re.compile(
        r"GIT_|(?:^|[^\w$])(?:HOME|PATH|XDG_\w+|LD_\w+|BASH_ENV|ENV|EDITOR|VISUAL|PAGER)=")

    def steers_git(self, command: str) -> bool:
        """Whether `command` steers git, or a program it runs, from the environment."""
        return bool(self._GIT_STEERING.search(command))

    def git_writes(self, command: str) -> "list[tuple[str, tuple[str, ...]]] | None":
        """Where the git commands that write their repository's records run.

        The steps in order: ("cd", directory) and ("git", its -C
        directories). None unless the command is git's alone -- every other
        stage reads, and nothing writes a file, substitutes, defines a
        function, sources a script or steers git (-c, --git-dir,
        --work-tree, a GIT_ variable, HOME, PATH, an editor). Only then may
        the repositories it names keep their records (write_protection.py).
        Clone writes no repository that exists, and adds no step.
        """
        if (self._GIT_STEERING.search(command) or re.search(r"[`]|\$\(|<<|\(\s*\)|\bfunction\b",
                                                             command)
                or self._writes_via_redirection(command)):
            return None

        steps: list[tuple[str, tuple[str, ...]]] = []

        for argv in self.stage_argvs(command):
            if argv[0] == "cd":
                operands = [word for word in argv[1:] if word not in ("-P", "-L", "--")]
                steps.append(("cd", tuple(operands[:1]) if len(operands) == 1 else ("",)))
                continue

            if argv[0] != "git":
                if argv[0] in self._BESIDE_GIT and not (argv[0] == "sed" and any(
                        self._SED_IN_PLACE.match(arg) for arg in argv[1:])):
                    continue

                return None

            parsed = self.git_command(argv)

            if parsed is None:
                return None

            subcommand, directories = parsed

            if subcommand in self.GIT_LOCAL_WRITE | {"fetch", "pull"}:
                if any(self._GIT_INTERACTIVE.match(word)
                       for word in argv[argv.index(subcommand) + 1:]):
                    return None

                steps.append(("git", directories))
            elif subcommand not in self.GIT_READ_ONLY | {"clone"}:
                return None

        return steps

    def classify(self, command: str) -> CommandAssessment:
        if not command or not command.strip():
            return self._assessment(command, (), CommandClassification.DANGEROUS, "empty command")

        if "\x00" in command:
            return self._assessment(command, (), CommandClassification.DANGEROUS, "NUL byte")

        # Read before the heredoc bodies go: a line written into a script or
        # a configuration file throws the same switch later.

        override = control_overrides.command_refusal(command)

        if override:
            return self._assessment(command, (), CommandClassification.DANGEROUS, override)

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
                command, (), CommandClassification.DANGEROUS,
                f"its quoting cannot be read ({exc}): close every quote, or write the "
                f"text with write_file.")

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

    _ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")

    def classify_script(self, command: str) -> CommandAssessment:
        """A command the coding core's terminal runs: always as a bash script.

        The simple-argv allowlist refused `cd dir`, `mkdir -p dir` or `bitbake
        x` while `cd dir && ls`, `true && mkdir -p dir` and `. ./env.sh &&
        bitbake x` ran -- the same words, judged by a stricter rule only
        because they had no shell syntax. The core runs every command through
        bash either way, so a refused simple command is judged again as the
        script it is. What the shell analysis itself refuses (rm, mv,
        inline interpreters, sed -i, writes outside the workspace) stays
        refused, with its own reason.
        """
        assessment = self.classify(command)

        if control_overrides.command_refusal(command or ""):
            return assessment

        if (assessment.classification != CommandClassification.DANGEROUS
                or not command or not command.strip() or "\x00" in command
                or self._SHELL_SYNTAX.search(self._strip_heredoc_bodies(command))):
            return assessment

        capabilities, danger, outside, refusal = self._shell_capabilities(
            self._strip_heredoc_bodies(command))

        if danger or refusal:
            return assessment

        return self._assessment(command, (), CommandClassification.SHELL_COMPLEX,
                                "session script", capabilities,
                                host_read_paths=outside)

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

        if self._syntax_check(argv):
            return self._assessment(command, argv, CommandClassification.READ_ONLY,
                                    capabilities=(Capability.FILESYSTEM_READ,))

        if binary in self._AWK_BINS and self._stage_danger(argv, 0):
            return self._assessment(command, argv, CommandClassification.DANGEROUS,
                                    self._stage_danger(argv, 0))

        if self._runs_code(binary) or (
                "/" in argv[0] and not self._is_workspace_program(argv[0])):
            if binary in self._PYTHON_BINS or self._VERSIONED_INTERPRETER.fullmatch(binary):
                # The way out comes before the refusal: a model reads the
                # first clause and acts on it.

                return self._assessment(
                    command, argv, CommandClassification.DANGEROUS,
                    f"{self._PYTHON_HINT} — so '{argv[0]}' as written is refused")

            if binary in self._SCRIPT_SHELLS:
                return self._assessment(command, argv, CommandClassification.DANGEROUS,
                                        self._stage_danger(argv, 0))

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
            subcommand = (self.git_command(argv) or ("", ()))[0]

            if subcommand in self.GIT_READ_ONLY:
                return self._assessment(command, argv, CommandClassification.READ_ONLY,
                                        capabilities=self._simple_capabilities(argv))

            if subcommand in self.GIT_NETWORK_ONLY:
                return self._assessment(command, argv, CommandClassification.READ_ONLY,
                                        capabilities=self._simple_capabilities(argv))

            if subcommand in self.GIT_NETWORK_WRITE | self.GIT_LOCAL_WRITE:
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
