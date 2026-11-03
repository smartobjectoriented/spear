"""Switches that turn an integrity guard off, and who may throw them.

Infrabase's attach guard refuses to re-attach a tree whose fingerprint no
longer matches it; IB_FORCE_ATTACH=1 tells it to proceed anyway. The switch
is documented, which makes it the operator's, not the model's: a model that
had its deletion of the fingerprint refused set the variable on the next
call and got the effect the refusal had withheld.

A model may not set such a switch, by any spelling: a command's environment,
an export, a declaration, a bitbake configuration line a write would add. An
operator who wants it for a session names it in SPEAR_AUTHORIZED_OVERRIDES
before SPEAR starts, where no command of the session can reach.

Only the switches listed here are judged. Every other variable is the
model's to set, except one whose name a command builds at run time
(`export "$name=1"`): what it sets cannot be read, so it is refused.
"""

from __future__ import annotations

import os
import re

# Name -> the guard it turns off.

OVERRIDES = {
    "IB_FORCE_ATTACH": "the Infrabase attach guard",
}

_AUTHORIZED_ENV = "SPEAR_AUTHORIZED_OVERRIDES"

# A shell or bitbake assignment, with bitbake's operators and override
# suffixes: NAME=1, NAME ?= "1", NAME:pn-buildroot = "1".

_ASSIGNMENT = r"\b{name}(?::[\w-]+)*\s*(?:\?\?=|\?=|:=|\+=|=\+|\.=|=\.|=)"

# A builtin that sets or exports variables it names among its arguments.

_SETTERS = re.compile(r"(?:^|[\s;&|(`])(export|declare|typeset|readonly|local|read|env)\b"
                      r"([^;&|\n`)]*)")
_PRINTF_V = re.compile(r"\bprintf\s+(?:-\w+\s+)*-v\s*(\S+)")


def authorized() -> frozenset[str]:
    """The switches the operator granted this session."""
    names = os.environ.get(_AUTHORIZED_ENV, "")

    return frozenset(name.strip() for name in names.split(",") if name.strip())


def _guarded() -> dict[str, str]:
    granted = authorized()

    return {name: guard for name, guard in OVERRIDES.items() if name not in granted}


def _refusal(name: str, guard: str) -> str:
    return (f"{name} is a control-plane override: it turns off {guard}, and only the "
            f"operator may enable it. Nothing in this session can set it; if the guard "
            f"is wrong, say so and leave it to the operator")


def command_refusal(command: str) -> str:
    """Why `command` may not run because it sets an override, or "".

    The whole text is read, heredoc bodies included: a line written into a
    configuration file or a script is the same switch, thrown later.
    """
    command = command or ""

    for name, guard in _guarded().items():
        if not re.search(rf"\b{name}\b", command):
            continue

        if re.search(_ASSIGNMENT.format(name=name), command):
            return _refusal(name, guard)

        for match in _SETTERS.finditer(command):
            if re.search(rf"(?:^|\s)-?\w*\s*{name}\b", match.group(2)):
                return _refusal(name, guard)

        for match in _PRINTF_V.finditer(command):
            if match.group(1).strip("'\"") == name:
                return _refusal(name, guard)

    for match in _SETTERS.finditer(command):
        words = match.group(2).split()

        if any(re.match(r"""^['"]?[^=\s'"]*[$`]""", word) for word in words
               if not word.startswith("-")):
            return (f"'{match.group(1)}' sets a variable whose name is built at run "
                    f"time, so what it sets cannot be read here; name the variable "
                    f"literally")

    for match in _PRINTF_V.finditer(command):
        if re.search(r"[$`]", match.group(1)):
            return ("'printf -v' sets a variable whose name is built at run time, so "
                    "what it sets cannot be read here; name the variable literally")

    return ""


def content_refusal(old: str, new: str) -> str:
    """Why a write that turns `old` into `new` may not happen, or "".

    Only an assignment the write adds counts: a class that declares the
    default (IB_FORCE_ATTACH ??= "0") stays editable.
    """
    for name, guard in _guarded().items():
        pattern = re.compile(rf"^.*{_ASSIGNMENT.format(name=name)}.*$", re.MULTILINE)
        before = {line.strip() for line in pattern.findall(old or "")}

        if any(line.strip() not in before for line in pattern.findall(new or "")):
            return _refusal(name, guard)

    return ""
