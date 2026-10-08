"""The command runner and the audit log of SPEAR tool execution.

``CommandRunner`` is the execution boundary that ties a workspace, the
sandbox and the resource limits together; ``AuditLogger`` records every
mutating tool attempt. Policy, workspace, sandbox and resource control live
in their own modules.

This module deliberately does not import ``rag_chat``.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from harness.command_policy import CommandAssessment
from harness.resource_control import (
    DEFAULT_CGROUP_LIMITS,
    DEFAULT_RESOURCE_LIMITS,
    CgroupLimits,
    ExecutionProfile,
    ResourceLimits,
)
from harness.sandbox import BubblewrapSandbox
from harness.tool_primitives import (
    Capability,
    ExecutionMode,
    NetworkBackend,
    PathPolicyError,
    ToolResult,
    decode_command_output,
    shell_argv,
)
from harness.workspace import Workspace


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
                timeout=self.timeout_seconds,
                shell=False,
            )
        except subprocess.TimeoutExpired:
            return ToolResult("timeout", f"timed out after {self.timeout_seconds}s")
        except OSError as exc:
            return ToolResult("failed", f"could not run command: {exc}")

        stdout = decode_command_output(completed.stdout)[: self.max_output_chars]
        stderr = decode_command_output(completed.stderr)[: self.max_output_chars]

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
                timeout=self.timeout_seconds,
                shell=False,
            )
        except subprocess.TimeoutExpired:
            return ToolResult("timeout", f"timed out after {self.timeout_seconds}s")
        except OSError as exc:
            return ToolResult("failed", f"could not run shell: {exc}")

        stdout = decode_command_output(completed.stdout)[: self.max_output_chars]
        stderr = decode_command_output(completed.stderr)[: self.max_output_chars]

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
