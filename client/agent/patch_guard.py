"""Refuse the fuzzy matches that rewrite literal text instead of whitespace.

The Hermes matching chain (hermes/fuzzy_match.py, kept unmodified) falls back
to strategies that strip leading whitespace or accept lines that are only
similar. In a unified diff the first column is a marker, not indentation: a
'+' line sent without its '+', or a ' ' context line sent as an empty one,
matched anyway, and the replacement dropped or grafted the markers -- a
corrupt patch reported as a successful edit. The same similarity let a model's
diff-prefixed text match plain C lines and write the '+' into the source.

Two rules, applied only when the chain matched through a non-exact strategy:
diff content (hunk headers plus file headers) is matched exactly or not at
all, and no aligned line pair may differ by a leading '+' or '-'. Both refuse
the edit; nothing is repaired.
"""

from __future__ import annotations

import re

from .hermes import fuzzy_match

DIFF_HUNK = re.compile(r"^@@ -\d+(,\d+)? \+\d+(,\d+)? @@", re.MULTILINE)
DIFF_FILE = re.compile(r"^(---|\+\+\+) \S", re.MULTILINE)

DIFF_EXACT_ONLY = (
    "This file is a unified diff: old_string must match it exactly, first-column "
    "markers included (' ' for context, '+' for added, '-' for removed lines). "
    "Re-read the lines and copy them verbatim, or edit the source tree and "
    "regenerate the patch with the project's workflow.")

MARKER_DRIFT = (
    "old_string only resembles the file here: line {line} of old_string starts "
    "with {pattern!r} where the file's line starts with {content!r}. A leading "
    "'+' or '-' is text, not indentation; copy the lines exactly as the file "
    "has them.")


def is_unified_diff(content: str) -> bool:
    return bool(DIFF_HUNK.search(content) and DIFF_FILE.search(content))


def _lead(line: str) -> str:
    return line.lstrip(" \t")[:1]


def marker_drift(content: str, matches: list, old_string: str) -> str | None:
    """The refusal for a match whose lines differ by a leading +/- marker."""
    pattern_lines = old_string.split("\n")

    for start, end in matches:
        for number, (pattern, found) in enumerate(zip(pattern_lines, content[start:end].split("\n")), 1):
            a, b = _lead(pattern), _lead(found)

            if a != b and {a, b} & {"+", "-"}:
                return MARKER_DRIFT.format(line=number, pattern=a, content=b)
    return None


def find_and_replace(content: str, old_string: str, new_string: str,
                     replace_all: bool = False):
    """fuzzy_find_and_replace with the two literal-text rules on top."""
    new_content, count, strategy, error = fuzzy_match.fuzzy_find_and_replace(
        content, old_string, new_string, replace_all)

    if error or not count or strategy == "exact":
        return new_content, count, strategy, error

    if is_unified_diff(content):
        return content, 0, None, DIFF_EXACT_ONLY

    matches = getattr(fuzzy_match, f"_strategy_{strategy}")(content, old_string)
    refusal = marker_drift(content, matches, old_string)

    if refusal:
        return content, 0, None, refusal
    return new_content, count, strategy, error
