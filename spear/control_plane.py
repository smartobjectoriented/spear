"""SPEAR's control plane for the agent core: the Host the core runs against.

The core (agent/) asks to read, write and run; this decides whether it may,
and records what happened. Every operation crosses the same boundary
whichever tool the model used:

  authorize      the router's hard-policy gates -- read-only and advisory
                 turns, target and sibling write scope, project and shell
                 scope, the write gate, the session's mode, arguments.
  resolve_*      the workspace policy: reads, writes and a command's workdir
                 resolve inside the workspace or are refused.
  write/delete   the generated-file and snapshot-tree guards, the mode's
                 confirmation (--safe, --ask, --auto), the checkpoint that
                 makes /undo possible, the audit trail.
  run_command    the command policy (allowlist, write scope, network) and
                 the sandbox; the policy judges the model's command, never
                 the script that carries the session around it.
  after_tool     the evidence and audit record of every call.

The SPEAR operations behind these are passed in as `ports`, so this module
has no dependency on the chat client, and the core has none on this one.
"""

from __future__ import annotations

from typing import Callable, Mapping

from agent.host import CommandOutcome, ToolRecord

#: How a refusal reads to the model: what was refused and why, in a line.
REFUSED = "refused: {}"


class SpearHost:
    def __init__(self, *, workspace_root: str, authorize: Callable,
                 resolve: Callable, write: Callable, delete: Callable,
                 run: Callable, record: Callable, collation_locale: str | None = None):
        self.workspace_root = workspace_root
        # The locale the sandboxed commands run under (LANG/LC_ALL), so a
        # listing the core makes itself is ordered the way the model's own
        # `ls` in that sandbox would order it.
        self.collation_locale = collation_locale
        self._authorize = authorize
        self._resolve = resolve
        self._write = write
        self._delete = delete
        self._run = run
        self._record = record

    @staticmethod
    def _short(message: str) -> str:
        """The refusal line, without SPEAR's internal framing."""
        text = (message or "").strip()

        for prefix in ("ERROR: ", "REFUSED: "):
            if text.startswith(prefix):
                text = text[len(prefix):]

        return REFUSED.format(text.split("\n", 1)[0])

    def authorize(self, name: str, arguments: Mapping[str, object]) -> str | None:
        refusal = self._authorize(name, arguments)
        return self._short(refusal) if refusal else None

    def _resolved(self, path: str, purpose: str):
        try:
            return self._resolve(path or ".", purpose), None
        except Exception as exc:          # the workspace policy's refusal
            return None, self._short(str(exc))

    def resolve_read(self, path: str):
        return self._resolved(path, "read")

    def resolve_write(self, path: str):
        return self._resolved(path, "write")

    def resolve_workdir(self, path: str):
        return self._resolved(path, "workdir")

    def write_file(self, path: str, content: str, *, action: str) -> str | None:
        failure = self._write(path, content, action)
        return self._short(failure) if failure else None

    def delete_file(self, path: str, reason: str) -> str | None:
        failure = self._delete(path, reason)
        return self._short(failure) if failure else None

    def run_command(self, command: str, script: str, *, timeout: int,
                    output_chars: int) -> CommandOutcome:
        outcome = self._run(command, script, timeout, output_chars)

        if outcome.status == "denied":
            return CommandOutcome("denied", "", -1, self._short(outcome.summary))

        return outcome

    def after_tool(self, record: ToolRecord) -> None:
        self._record(record)
