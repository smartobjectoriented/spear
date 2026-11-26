"""File:line references in an answer, checked against the files they name.

A model that reads code through `sed -n 'a,bp'` sees no line numbers and
counts them itself, and one-off slips follow: `ipi_setup_sgi` cited at
smp.c:1230 when it is defined at 1231. Nothing downstream checked a
reference against the file.

The check is narrow on purpose. A reference is examined only when the
answer names the symbol it points at, beside it, and only files this turn
touched are opened. If the line holds the symbol, the reference stands; if
exactly one line within a few of it does, the reference is corrected and the
answer says so; anything else is left as written. A wrong correction would
be worse than none.
"""

from __future__ import annotations

import os
import re

#: How far a counted line number is allowed to have drifted.

_WINDOW = 3

#: `path/to/file.c:123`, or `:123` continuing the file named last -- the
#: bare form only after a space or a bracket, so that a bit range such as
#: [47:44] is never read as one.

_REFERENCE = re.compile(
    r"(?:(?P<path>(?:[\w.+-]+/)*[\w.+-]+\.[A-Za-z]{1,4})|(?<=[\s(]))"
    r":(?P<line>\d{1,6})(?![\d:\]])")

#: The symbol a reference is about: the identifier closest before it.

_SYMBOL = re.compile(r"`?([A-Za-z_][A-Za-z0-9_]{2,})`?(?:\(\))?")

_PATH_TOKEN = re.compile(r"(?:[\w.+-]+/)+[\w.+-]+\.[A-Za-z]{1,4}")

_KEYWORDS = {"the", "and", "for", "case", "line", "lines", "file", "at", "in",
             "see", "from", "with", "via", "then", "which", "calls", "call",
             "defined", "registered", "static", "void", "int", "return",
             "struct", "const", "unsigned"}


def _candidates(root, tool_log):
    """Files this turn touched, as absolute paths that exist."""

    found = set()

    for entry in tool_log or ():
        for token in _PATH_TOKEN.findall(str(entry)):
            path = token if os.path.isabs(token) else os.path.join(root or ".", token)

            if os.path.isfile(path):
                found.add(os.path.normpath(path))

    return found


def _resolve(cited, candidates):
    """The one touched file a cited path designates, or None."""

    cited = cited.lstrip("./")
    matches = [path for path in candidates
               if path.endswith("/" + cited) or os.path.basename(path) == cited]

    return matches[0] if len(matches) == 1 else None


def _symbol_before(text, start):
    """The identifier the reference at `start` is about, if one is beside it."""

    window = text[max(0, start - 80):start]
    names = [name for name in _SYMBOL.findall(window)
             if name.lower() not in _KEYWORDS and not name.isdigit()]

    return names[-1] if names else None


class _Lines:
    def __init__(self):
        self.files = {}

    def get(self, path):
        if path not in self.files:
            try:
                with open(path, encoding="utf-8", errors="replace") as handle:
                    self.files[path] = handle.read().splitlines()
            except OSError:
                self.files[path] = []

        return self.files[path]


def check(answer, root, tool_log):
    """The answer with drifted references corrected, and what was corrected."""

    candidates = _candidates(root, tool_log)

    if not answer or not candidates:
        return answer, []

    lines, corrections, out = _Lines(), [], []
    position, current = 0, None

    for match in _REFERENCE.finditer(answer):
        cited = match.group("path")
        path = _resolve(cited, candidates) if cited else current

        if cited:
            current = path

        if path is None:
            continue

        number = int(match.group("line"))
        symbol = _symbol_before(answer, match.start())
        content = lines.get(path)

        if not symbol or not content or not 0 < number <= len(content) + _WINDOW:
            continue

        def holds(n):
            return 0 < n <= len(content) and re.search(
                rf"\b{re.escape(symbol)}\b", content[n - 1])

        if holds(number):
            continue

        near = [n for n in range(number - _WINDOW, number + _WINDOW + 1) if holds(n)]

        if len(near) != 1:
            continue

        out.append(answer[position:match.start("line")])
        out.append(str(near[0]))
        position = match.end("line")
        corrections.append((cited or os.path.basename(path), number, near[0], symbol))

    if not corrections:
        return answer, []

    out.append(answer[position:])
    note = ("\n\nLine references corrected against the files: "
            + ", ".join(f"{name}:{old} → {new} ({symbol})"
                        for name, old, new, symbol in corrections) + ".")

    return "".join(out) + note, corrections
