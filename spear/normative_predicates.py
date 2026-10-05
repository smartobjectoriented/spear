"""Deterministic predicates: a constraint decided from the final source alone.

The model's reading of a constraint is a candidate finding, never a verdict.
A constraint is decided only where nothing has to be interpreted: the packet
states the value, and the final source assigns the thing the constraint names
in a form whose value can be read off.

  EXACT_COUNT        "<subject> shall contain exactly N ..." against the
                     element count of a literal list, array or initializer
                     assigned to that subject.
  VALUE_EQUALS       "<subject> shall be V" against a literal assigned to it.
  CONDITIONAL_VALUE  "<subject> shall be V when <name> is set" against a
                     literal, or a conditional expression on that name.

Every site that assigns the subject must be readable and must agree. A
subject that names several things, a count that is not exact, a value whose
statement goes on past it, a condition that names no single symbol, an
expression where a literal should be -- each leaves the constraint undecided,
which is what None means here.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

EXACT_COUNT = "EXACT_COUNT"
VALUE_EQUALS = "VALUE_EQUALS"
CONDITIONAL_VALUE = "CONDITIONAL_VALUE"


@dataclass(frozen=True)
class SourceFact:
    """One place in the final source, as it reads, apart from what it means."""
    path: str
    line: int | None
    excerpt: str
    digest: str
    symbol: str = ""
    observed: str = ""

    def to_dict(self) -> dict:
        return {"path": self.path, "line": self.line, "excerpt": self.excerpt,
                "digest": self.digest, "symbol": self.symbol, "observed": self.observed}


def fact(path: str, line: int | None, excerpt: str, symbol: str = "",
         observed: str = "") -> SourceFact:
    text = " ".join(str(excerpt).split())[:200]

    return SourceFact(path, line, text, hashlib.sha256(text.encode()).hexdigest()[:16],
                      symbol, observed)


@dataclass(frozen=True)
class PredicateResult:
    predicate: str
    expected: str
    observed: str
    holds: bool
    facts: tuple[SourceFact, ...]


_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
            "seven": 7, "eight": 8, "nine": 9, "ten": 10}
_EXACT = re.compile(r"\b(?:exactly|one\s+and\s+only)\s+(one|two|three|four|five|six|"
                    r"seven|eight|nine|ten|\d+)\b", re.I)
_LITERAL = r"0x[0-9A-Fa-f]+|-?\d+|'[^'\n]{0,40}'|\"[^\"\n]{0,40}\""
_VALUE = re.compile(r"^(?P<subject>.+?)\b(?:shall|must)\s+(?:be\s+set\s+to|be\s+equal\s+to|"
                    r"equal|be)\s+(?P<value>" + _LITERAL + r")(?P<rest>.*)$", re.I)
_MODAL = re.compile(r"\b(?:shall|must)\b", re.I)
_CONDITION = re.compile(r"^(?:when|if|whenever)\s+(?P<name>.+?)\s+is\s+"
                        r"(?:set|true|enabled|non-?zero)$", re.I)

#: Words that say what kind of thing a subject is, not which one.
_GENERIC = frozenset("""a an the this that its their each every any all of in for to
    field fields value values entry entries element elements item items list array
    member members bit bits flag register parameter parameters setting number only
    one two three four five six seven eight nine ten rule permission requirement
    recommendation observation note""".split())

_ASSIGNMENT = re.compile(
    r"(?<!\w)(?P<target>[A-Za-z_]\w*(?:[ \t]*(?:\.|->)[ \t]*[A-Za-z_]\w*)*)"
    r"[ \t]*(?P<size>\[[^\]\n]*\])?[ \t]*(?<![=!<>+\-*/%&|^~])=(?!=)")
_END = re.compile(r"^[ \t]*(?:;|,|\)|\r?\n|$|#|//|/\*)")

#: What may follow a mention of the subject that only reads it.
_READ = re.compile(
    r"[ \t]*(?:[)\]};,]|:(?![^\n]*?(?<![=!<>])=(?!=))|\r?\n|$|==|!=|<=|>=|<(?![<=])|"
    r">(?![>=])|\+(?![+=])|-(?![-=>])|\*(?![*=])|/(?![/=])|%(?!=)|&&|\|\||\?|"
    r"(?:and|or|if|else|in|is|not)\b)")
_WRITES_BEFORE = re.compile(r"(?:(?<!&)&|\+\+|--|\b(?:for|as|global|nonlocal|del)"
                            r"[ \t]+(?:[\w, \t]*,)?)[ \t]*$")
#: Calls that read their argument and cannot change it.
_PURE = frozenset({"len", "sizeof", "ARRAY_SIZE", "countof", "_countof", "print"})
_KEYWORDS = frozenset({"if", "while", "for", "switch", "return", "and", "or", "not",
                       "elif", "in", "assert", "sizeof"})


def evaluate(constraint, files: dict[str, str]) -> PredicateResult | None:
    """The constraint decided from `files`, or None where it cannot be."""
    if constraint.modality != "SHALL" or not constraint.resolved:
        return None

    sentence = _sentence(constraint)

    if not sentence or re.search(r"\b(?:shall|must)\s+not\b", sentence, re.I):
        return None

    count = _EXACT.search(sentence)

    if count and not constraint.condition:
        modal = _MODAL.search(sentence)

        if modal is None or modal.start() > count.start():
            return None

        expected = _number(count.group(1))
        keys = _keys(sentence[:modal.start()])

        return _decide_count(expected, keys, files) if keys else None

    value = _VALUE.match(sentence)

    if not value:
        return None

    rest = value.group("rest").strip().rstrip(".").strip()

    if constraint.condition:
        rest = rest.replace(constraint.condition, "").strip()

    keys = _keys(value.group("subject"))

    if rest or not keys:
        return None

    expected = _literal(value.group("value"))

    if not constraint.condition:
        return _decide_value(expected, keys, files)

    condition = _CONDITION.match(constraint.condition.strip().rstrip("."))

    if not condition or re.search(r",|\bor\b|\band\b", condition.group("name")):
        return None

    return _decide_conditional(expected, _norm(condition.group("name")), keys, files)


# ------------------------------------------------------------- the packet side

def _sentence(constraint) -> str:
    """The provision's statement, without its printed label."""
    text = " ".join(constraint.requirement.split())

    if constraint.provision and text.startswith(constraint.provision):
        text = text[len(constraint.provision):].lstrip(" :.-")

    sentences = [part for part in re.split(r"(?<=\.)\s+(?=[A-Z])", text) if _MODAL.search(part)]

    return sentences[0].strip() if len(sentences) == 1 else ""


def _keys(subject: str) -> frozenset[str]:
    """The names a single-thing subject can be spelled as in code."""
    if re.search(r",|\bor\b|\band\b", subject):
        return frozenset()

    words = [word for word in re.findall(r"[A-Za-z][A-Za-z0-9_]*", subject)
             if word.lower() not in _GENERIC]
    keys = {_norm(word) for word in words}
    keys |= {_norm("".join(words[start:start + size]))
             for size in (2, 3) for start in range(len(words) - size + 1)}

    return frozenset(key for key in keys if len(key) >= 3)


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _number(word: str) -> int:
    return int(word) if word.isdigit() else _NUMBERS[word.lower()]


def _literal(text: str):
    text = text.strip()

    if text[:1] in "'\"":
        return text[1:-1]

    return int(text, 16) if text.lower().startswith("0x") else int(text)


# ------------------------------------------------------------- the source side

def _masked(path: str, text: str) -> str:
    """`text` with comments and the inside of string literals blanked, every
    offset and line kept: a name in a comment is not a use of it."""
    out, index, size = list(text), 0, len(text)
    c_like = not path.endswith(".py")

    def blank(start, end):
        for position in range(start, min(end, size)):
            if out[position] != "\n":
                out[position] = " "

    while index < size:
        char = text[index]
        line_start = index == 0 or text[index - 1] == "\n"

        if (c_like and text.startswith("//", index)) or (not c_like and char == "#") \
                or (c_like and char == "#" and line_start):
            end = text.find("\n", index)
            end = size if end < 0 else end
            blank(index, end)
            index = end
        elif c_like and text.startswith("/*", index):
            end = text.find("*/", index + 2)
            end = size if end < 0 else end + 2
            blank(index, end)
            index = end
        elif char in "'\"":
            quote = text[index:index + 3] if text[index:index + 3] in ("\'\'\'", '"""') else char
            end = index + len(quote)

            while end < size and not text.startswith(quote, end):
                if text[end] == "\\":
                    end += 1
                elif len(quote) == 1 and text[end] == "\n":
                    break

                end += 1

            blank(index + len(quote), end)
            index = end + len(quote)
        else:
            index += 1

    return "".join(out)


def _opener(masked: str, position: int) -> tuple[str, int]:
    """The innermost bracket open around `position`, and where it is."""
    depth = 0

    for index in range(position - 1, -1, -1):
        char = masked[index]

        if char in ")]}":
            depth += 1
        elif char in "([{":
            if depth == 0:
                return char, index

            depth -= 1

    return "", -1


_CONTROL = re.compile(r"\b(?:if|elif|else|for|while|do|switch|case|default|try|except|"
                      r"finally|with|match|lambda)\b|\?")


def _unconditional(path: str, masked: str, start: int) -> bool:
    """Whether the statement at `start` runs whenever its function does: not
    in a branch, a loop or a handler. A value assigned only on some paths is
    not the value."""
    if path.endswith(".py"):
        lines = masked[:start].split("\n")

        if _CONTROL.search(lines[-1]):
            return False

        indent = len(lines[-1]) - len(lines[-1].lstrip())

        for line in reversed(lines[:-1]):
            stripped = line.lstrip()

            if not stripped or len(line) - len(stripped) >= indent:
                continue

            if _CONTROL.match(stripped):
                return False

            indent = len(line) - len(stripped)

            if indent == 0:
                break

        return True

    head = masked[max(masked.rfind(mark, 0, start) for mark in ";{}") + 1:start]

    if _CONTROL.search(head):
        return False

    position = start

    while True:
        opener, at = _opener(masked, position)

        if not opener:
            return True

        if opener == "{":
            head = masked[max(masked.rfind(mark, 0, at) for mark in ";{}") + 1:at]

            if _CONTROL.search(head):
                return False

        position = at


def _callee(masked: str, opener: int) -> str | None:
    """The name called by the `(` at `opener`, or None when it is not a call."""
    head = masked[:opener].rstrip()
    match = re.search(r"([A-Za-z_]\w*)$", head)

    if match:
        return None if match.group(1) in _KEYWORDS else match.group(1)

    return "?" if head[-1:] in ")]" else None


def _sites(keys, files):
    """(path, line, line text, target, the text after its `=`, size) for every
    plain assignment to a name the subject spells -- or None when the subject
    is changed anywhere in a way a literal cannot show: a compound assignment,
    an increment, a method call, its address taken, a call argument, an
    unpacking."""
    found = []

    for path, text in files.items():
        masked = _masked(path, text)
        starts = set()

        for match in _ASSIGNMENT.finditer(masked):
            target = re.split(r"[ \t]*(?:\.|->)[ \t]*", match.group("target"))[-1]

            if _norm(target) not in keys:
                continue

            start = match.start("target") + match.group("target").rfind(target)
            head = masked[masked.rfind("\n", 0, start) + 1:match.start("target")]
            opener, _ = _opener(masked, start)

            if opener == "(" or re.search(r",", head.split("=")[-1] if "=" in head else head) \
                    or not _unconditional(path, masked, start):
                return None

            starts.add(start)
            line = text.count("\n", 0, start) + 1
            found.append((path, line, text.splitlines()[line - 1], target,
                          text[match.end():], match.group("size")))

        for match in re.finditer(r"[A-Za-z_]\w*", masked):
            if _norm(match.group(0)) not in keys or match.start() in starts:
                continue

            before = masked[masked.rfind("\n", 0, match.start()) + 1:match.start()]
            after = masked[match.end():]

            if _WRITES_BEFORE.search(before) or not _READ.match(after):
                return None

            follower = after.lstrip(" \t")[:1]
            opener, at = _opener(masked, match.start())

            if follower in ",)" and opener == "(" and _callee(masked, at) not in _PURE | {None}:
                return None

            if follower == "," and not opener and re.search(
                    r"(?<![=!<>+\-*/%&|^~])=(?!=)", after.split("\n")[0]):
                return None

    return found


def _elements(rhs: str) -> int | None:
    """The element count of the literal `rhs` starts with, if it is one."""
    body = rhs.lstrip()

    if not body or body[0] not in "[{(":
        return None

    closing, depth, quote, parts, current = {"[": "]", "{": "}", "(": ")"}, 0, "", [], ""

    for index, char in enumerate(body):
        if quote:
            current += char
            quote = "" if char == quote else quote
            continue

        if char in "'\"":
            quote = char
        elif char in "[{(":
            depth += 1

            if depth == 1:
                continue
        elif char in "]})":
            depth -= 1

            if depth == 0:
                if char != closing[body[0]] or not _END.match(body[index + 1:]):
                    return None

                parts.append(current)
                items = [part.strip() for part in parts]
                items = items[:-1] if items and not items[-1] else items

                if any(not item or item.startswith("*") or item == "..."
                       or re.search(r"\bfor\b", item) for item in items):
                    return None

                return len(items)
        elif char == "," and depth == 1:
            parts.append(current)
            current = ""
            continue

        current += char

    return None


def _scalar(rhs: str):
    match = re.match(r"\s*(" + _LITERAL + r")", rhs)

    if match and _END.match(rhs[match.end():]):
        return _literal(match.group(1))

    return None


def _ternary(rhs: str):
    """(name, value when it holds, value otherwise) for a conditional literal."""
    python = re.match(r"\s*(" + _LITERAL + r")\s+if\s+([A-Za-z_][\w.]*)\s+else\s+("
                      + _LITERAL + r")", rhs)
    c_like = re.match(r"\s*\(?\s*([A-Za-z_][\w.>-]*)\s*\)?\s*\?\s*(" + _LITERAL
                      + r")\s*:\s*(" + _LITERAL + r")", rhs)

    if python and _END.match(rhs[python.end():]):
        return python.group(2), _literal(python.group(1)), _literal(python.group(3))

    if c_like and _END.match(rhs[c_like.end():]):
        return c_like.group(1), _literal(c_like.group(2)), _literal(c_like.group(3))

    return None


def _decide_count(expected, keys, files):
    facts, counts = [], set()

    for path, line, text, target, rhs, size in _sites(keys, files) or ():
        # A declared size counts only with an initializer: `a[2] = 5` writes
        # one element, `int a[4] = {0}` declares four.
        inner = (size or "")[1:-1].strip()
        count = _elements(rhs)

        if count is None or (inner and not inner.isdigit()):
            return None

        count = int(inner) if inner else count

        counts.add(count)
        facts.append(fact(path, line, text, target, str(count)))

    if len(counts) != 1:
        return None

    observed = counts.pop()

    return PredicateResult(EXACT_COUNT, str(expected), str(observed),
                           observed == expected, tuple(facts))


def _decide_value(expected, keys, files):
    facts, values = [], set()

    for path, line, text, target, rhs, size in _sites(keys, files) or ():
        value = None if size else _scalar(rhs)

        if value is None:
            return None

        values.add(value)
        facts.append(fact(path, line, text, target, str(value)))

    if len(values) != 1:
        return None

    observed = values.pop()

    return PredicateResult(VALUE_EQUALS, str(expected), str(observed),
                           observed == expected, tuple(facts))


def _decide_conditional(expected, condition, keys, files):
    """The subject's value wherever the condition holds."""
    facts, values = [], set()

    for path, line, text, target, rhs, size in _sites(keys, files) or ():
        value, ternary = (None, None) if size else (_scalar(rhs), _ternary(rhs))

        if value is None and ternary is not None and _norm(ternary[0].split(".")[-1]
                                                           .split(">")[-1]) == condition:
            value = ternary[1]

        if value is None:
            return None

        values.add(value)
        facts.append(fact(path, line, text, target, str(value)))

    if len(values) != 1:
        return None

    observed = values.pop()

    return PredicateResult(CONDITIONAL_VALUE, str(expected), str(observed),
                           observed == expected, tuple(facts))
