"""Resource control for sandboxed commands.

The execution profile derived from granted capabilities, the per-process
rlimits and cgroup limits a command runs under, and the transient systemd
user scope that enforces the cgroup ones.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import uuid
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Mapping, Sequence

from harness.tool_primitives import Capability, ToolResult


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

    UNIT_PREFIX = "spear-tool-"

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
        # up -- scripts/docker/spear-docker.sh passes --memory, --pids-limit and --cpus
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
