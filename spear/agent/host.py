"""What the agent core needs from the outside, and nothing more.

The core decides which tool to call and shapes what the model reads. It does
not decide whether a call is allowed, where a path may resolve, how a write
is authorised or audited, or how a command is sandboxed: those belong to the
host, which SPEAR implements with its control plane (control_plane.py). The
core can be run against any host that satisfies this protocol -- a test fake
included -- and never imports SPEAR's chat client.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol


@dataclass(frozen=True)
class CommandOutcome:
    """One command as the sandbox ran it.

    status: "ok" (ran, any exit code), "denied" (refused before running),
    "cancelled" (interrupted), or "error" (the boundary failed).
    """
    status: str
    output: str = ""
    exit_code: int | None = None
    summary: str = ""


@dataclass
class ToolRecord:
    """What the host is told after every call, for audit and evidence."""
    call_id: str
    name: str
    arguments: Mapping[str, object]
    result: str                       # exactly what the model reads
    ok: bool                          # the tool's own verdict
    refused: bool = False             # the host refused it before it ran
    changed_paths: tuple[str, ...] = ()
    read_paths: tuple[str, ...] = ()
    command: str | None = None
    exit_code: int | None = None
    timed_out: bool = False
    metadata: dict = field(default_factory=dict)


class Host(Protocol):
    #: The directory the model's relative paths and commands start from.
    workspace_root: str
    #: Optional: the locale the host's commands run under (LC_ALL), which
    #: orders directory listings the way the model's own `ls` would. A host
    #: without it leaves the core's process environment in charge.
    collation_locale: str | None

    def authorize(self, name: str, arguments: Mapping[str, object]) -> str | None:
        """None to allow the call, or the short refusal the model reads."""

    def resolve_read(self, path: str) -> tuple[str | None, str | None]:
        """(absolute path, None) inside the read boundary, or (None, refusal)."""

    def resolve_write(self, path: str) -> tuple[str | None, str | None]:
        """(absolute path, None) inside the write boundary, or (None, refusal)."""

    def write_file(self, path: str, content: str, *, action: str) -> str | None:
        """Write after every policy check; None on success, else the reason."""

    def delete_file(self, path: str, reason: str) -> str | None:
        """Delete after every policy check; None on success, else the reason."""

    def run_command(self, command: str, script: str, *, timeout: int,
                    output_chars: int) -> CommandOutcome:
        """Authorise `command` and run `script` (which wraps it) sandboxed."""

    def after_tool(self, record: ToolRecord) -> None:
        """Audit and evidence for one completed call."""
