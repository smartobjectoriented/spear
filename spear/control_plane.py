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

import json
import os
import re
from typing import Callable, Mapping

from agent.host import CommandOutcome, ToolRecord
from agent.tools import usable_dir

#: How a refusal reads to the model: what was refused and why, in a line.
REFUSED = "refused: {}"

#: SPEAR's refusals were written for its legacy tools (edit_file with
#: old_text/new_text, a scope_reason argument, no deletion tool). The core
#: offers none of those, so a refusal that pointed at them would send the
#: model after a tool or an argument it does not have. Each pair is the legacy
#: wording and what the core's tools make of it.
CORE_VOCABULARY = (
    (re.compile(r" If this file genuinely has to change .*? call again with "
                r"scope_reason saying exactly why\."), ""),
    (re.compile(re.escape("make the change with edit_file, on the requested "
                          "target, or give scope_reason there.")),
     "make the change on the requested target."),
    (re.compile(re.escape("use edit_file. To insert a line, pass the existing "
                          "surrounding line(s) as old_text and those same lines "
                          "plus the new one as new_text.")),
     "use patch. To insert a line, pass the existing surrounding line(s) as "
     "old_string and those same lines plus the new one as new_string."),
    (re.compile(re.escape("edit_file/write_file")), "patch/write_file"),
    (re.compile(re.escape(" — no tool can delete or rename a file. Nothing here "
                          "can do it: say which file should go and why, and leave "
                          "it to the operator. Emptying it instead leaves a file "
                          "behind.")),
     " — delete a file with delete_file; nothing renames one (write the new "
     "path with write_file, then delete_file the old one)."),
    (re.compile(r"'(rm|rmdir|unlink|shred|mv|rename)' is not allowlisted, so the "
                r"pipeline that contains it cannot run\."),
     r"'\1' is not allowlisted, so the pipeline that contains it cannot run: "
     r"delete a file with delete_file; nothing renames one (write the new path "
     r"with write_file, then delete_file the old one)."),
)


def core_vocabulary(text: str) -> str:
    """A SPEAR refusal in the coding core's tool vocabulary."""
    for pattern, replacement in CORE_VOCABULARY:
        text = pattern.sub(replacement, text)

    return text


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
        # The core's terminal session as its results show it (agent/tools.py
        # terminal_from_outcome): `cwd`, where a command without workdir
        # starts, and `last`, where the last command finished. A command's
        # relative paths are judged from where it runs, not from the root.
        self._session = {"cwd": None, "last": None}
        # The path the delete_file call being authorised asked for, as written.
        self._deleting = ""

    @staticmethod
    def _short(message: str) -> str:
        """The refusal line, without SPEAR's internal framing."""
        text = (message or "").strip()

        for prefix in ("ERROR: ", "REFUSED: "):
            if text.startswith(prefix):
                text = text[len(prefix):]

        return REFUSED.format(core_vocabulary(text.split("\n", 1)[0]))

    def command_cwd(self, arguments: Mapping[str, object]) -> str:
        """Where a terminal call runs: its workdir, the session's, or the root."""
        workdir = str(arguments.get("workdir") or "")

        if workdir:
            start = usable_dir(self._session["last"] or self.workspace_root)
            return os.path.normpath(os.path.join(start, os.path.expanduser(workdir)))

        return self._session["cwd"] or self.workspace_root

    def authorize(self, name: str, arguments: Mapping[str, object]) -> str | None:
        # scope_reason is the legacy tools' way past a scope refusal. The
        # core's schemas do not offer it, so it opens nothing here.
        arguments = {key: value for key, value in arguments.items()
                     if key != "scope_reason"}
        cwd = self.command_cwd(arguments) if name == "terminal" else None

        if name == "delete_file":
            self._deleting = str(arguments.get("path") or "")

        refusal = self._authorize(name, arguments, cwd)
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
        # The core resolved the path, following a link to what it points at.
        # When the path asked for names a link, the link is what is deleted,
        # and the delete port judges it as one.
        requested = self._deleting
        failure = self._delete(requested if self._names_link(requested) else path, reason)
        return self._short(failure) if failure else None

    def _names_link(self, path: str) -> bool:
        if not path:
            return False

        candidate = path if os.path.isabs(path) else os.path.join(self.workspace_root, path)
        directory, refusal = self._resolved(os.path.dirname(candidate), "write")

        return not refusal and os.path.islink(os.path.join(directory, os.path.basename(candidate)))

    def run_command(self, command: str, script: str, *, timeout: int,
                    output_chars: int) -> CommandOutcome:
        outcome = self._run(command, script, timeout, output_chars)

        if outcome.status == "denied":
            return CommandOutcome("denied", "", -1, self._short(outcome.summary))

        return outcome

    def after_tool(self, record: ToolRecord) -> None:
        if record.name == "terminal" and not record.refused and not record.timed_out:
            self._follow(record)

        self._record(record)

    def _follow(self, record: ToolRecord) -> None:
        """Move the session mirror as the core moved its session."""
        try:
            result = json.loads(record.result)
        except ValueError:
            return

        workdir = str(record.arguments.get("workdir") or "")
        entered = not (record.exit_code == 126 and "cd: " in str(result.get("output", "")))
        observed = result.get("cwd") or (
            self.command_cwd(record.arguments) if entered else None)

        if observed:
            self._session["last"] = observed

            if not workdir:
                self._session["cwd"] = observed
