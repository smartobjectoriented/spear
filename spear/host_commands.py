"""Commands SPEAR answers itself, from the coding core's terminal.

The core's surface is six tools and stays six. What a turn may reach beyond
them -- external capabilities, the workspace's recorded knowledge -- it reaches
with one command that the control plane answers and never hands to a shell.

A name is recognised lexically, without parsing the shell: where a shell
would run it -- the command's first word, or the first word after a control
operator, a newline, an opening parenthesis, a backquote or "$(". Anywhere
else (an argument to grep, a path such as ./spear-knowledge) it is ordinary
text and the command an ordinary one. Once recognised, the command is
answered -- a result or a refusal -- whatever is wrong with it: it must
stand alone on one line under a size bound, and its words are split with
punctuation kept apart, never expanded.
"""

from __future__ import annotations

import re
import shlex

NAMES = ("spear-capability", "spear-knowledge")

#: Longest command answered; arguments are at most one JSON object.
MAX_COMMAND_CHARS = 65_536

_PUNCTUATION = ";&|()<>"

_NAMED = {name: re.compile(r"(?:^|[;&|\n\r(`]|\$\()[ \t]*" + re.escape(name)
                           + r"(?![\w./-])") for name in NAMES}


def recognised(command) -> str | None:
    """The host command this terminal command addresses, or None."""
    text = str(command or "")
    found = [(match.start(), name) for name, pattern in _NAMED.items()
             for match in [pattern.search(text)] if match]

    return min(found)[1] if found else None


def words(command: str, name: str):
    """(the words of one plain `name` command, "") or ((), why it is not one)."""
    if len(command) > MAX_COMMAND_CHARS:
        return (), f"{name}: the command is longer than {MAX_COMMAND_CHARS} characters."

    if any(char in command for char in "\n\r\x00"):
        return (), f"{name} takes one line; run it on its own."

    lexer = shlex.shlex(command, posix=True, punctuation_chars=_PUNCTUATION)
    lexer.whitespace_split = True

    try:
        found = list(lexer)
    except ValueError as exc:
        return (), f"{name}: {exc}."

    if (not found or found[0] != name
            or any(word and set(word) <= set(_PUNCTUATION) for word in found)):
        return (), (f"{name} must be run on its own -- not in a pipeline, a list, a "
                    f"redirection or a substitution.")

    return found, ""


def session_outcome(code: int, text: str):
    """An answer in the session format the core's terminal reads: an exit
    code, the output, and no change of directory."""
    from agent.host import CommandOutcome
    from agent.tools import _MARK

    return CommandOutcome("ok", f"{_MARK}\nexit {code}\nsize {len(text)}\ncwd \n{_MARK}\n{text}",
                          code, "")
