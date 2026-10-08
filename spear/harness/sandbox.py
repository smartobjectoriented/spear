"""The Bubblewrap sandbox that runs authorized commands.

``BubblewrapSandbox`` confines an already-authorized argv: its mounts, its
optional slirp4netns network and the resource limits it applies.
"""

from __future__ import annotations

import atexit
import fcntl
import json
import os
import select
import shlex
import shutil
import signal
import subprocess
import tempfile
import time
from enum import StrEnum
from pathlib import Path
from typing import Sequence

from harness.resource_control import (
    CgroupLimits,
    ExecutionProfile,
    ResourceLimits,
    SystemdScopeRunner,
)
from harness.tool_primitives import (
    Capability,
    NetworkBackend,
    ToolResult,
    decode_command_output,
)
from harness.workspace import SandboxSpec, Workspace, effective_mount_root


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
                self._session_tmpdir = tempfile.mkdtemp(prefix="spear-sandbox-")
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
        stdout = decode_command_output(completed.stdout)[: self.max_output_chars]
        stderr = decode_command_output(completed.stderr)[: self.max_output_chars]

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
            with tempfile.TemporaryDirectory(prefix="spear-slirp-") as temporary:
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
                        stdout=decode_command_output(stdout)[: self.max_output_chars],
                        stderr=decode_command_output(stderr)[: self.max_output_chars],
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
