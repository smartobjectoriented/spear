"""Small, dependency-free safety primitives for SPEAR tool execution.

Execution modes, capabilities and the policy that grants them, the canonical
tool result, path-policy errors and the shell a complex command runs in.
Every other tool-execution module builds on these.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal


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


def decode_command_output(data) -> str:
    """A command's output as the model reads it: UTF-8, invalid bytes replaced,
    and line endings exactly as emitted.

    Read as bytes and decoded here, never through ``text=True``: Python's
    text mode applies universal-newline translation and turns every ``\r``
    and ``\r\n`` into ``\n`` -- a progress bar, a CRLF file or a binary
    dump would reach the model rewritten. (Hermes decodes the raw byte
    stream the same way: incremental UTF-8, ``errors="replace"``.)
    """
    if data is None:
        return ""

    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")

    return data


def shell_argv(command: str, login: bool = True) -> "list[str]":
    """argv for running `command` in a shell, with pipeline failures visible."""
    return [SHELL_BINARY, "-lc" if login else "-c", PIPEFAIL_PRELUDE + command]
