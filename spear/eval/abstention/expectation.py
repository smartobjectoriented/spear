"""Scoring a question whose evidence settles one dimension and not another.

Two supported controls ask for a word number *and* a bit range. The evidence
gives the word numbers and no bit ranges, so the complete, correct, evidence-
disciplined answer states the words and says the bit ranges are not
established. The original oracle read any retraction phrase anywhere in an
answerable scenario's answer as over-abstention, and marked that answer wrong.
Worse, it marked the opposite answer right: "the standard contains no approved
structures, so there is no word number to report" carries no phrase from the
list and passed, while being false.

The repair is not to drop the phrases -- they are real over-abstention signals
elsewhere -- but to give them a subject. A denial is read as belonging to one
dimension, the first one it names, and it can only retract a fact stated in
that same dimension. A denial about bits leaves a word number standing; a
denial about the word number does not.

Nothing here knows a scenario by name. It reads what the fixture says the
evidence establishes and what it says the evidence leaves open.
"""

from __future__ import annotations

import re

import normative_dimensions
from corpus import BITS, OFFSET, WORD, WORD_COUNT

ANSWERED = "ANSWERED"
OVER_ABSTAINED = "OVER_ABSTAINED"
UNSUPPORTED_ASSIGNMENT = "UNSUPPORTED_ASSIGNMENT"

# How each dimension is spoken about. Shared with the final-answer guard, so
# the oracle and the guard cannot disagree about what a denial is denying.

_NOUNS = normative_dimensions.NOUNS

# Phrases that deny. Kept as the generic detector had them: the repair is the
# subject they are attached to, not the list itself.

_DENIAL = (r"does not (?:specify|state|establish|define|say|indicate|give)",
           r"do not (?:specify|state|establish|define|say|indicate|give)",
           r"doesn't (?:specify|state|establish|define|say|indicate|give)",
           r"not (?:specified|established|stated|defined|given|determined|"
           r"provided|available)",
           r"no (?:defined|stated|established|approved|specified)",
           r"there (?:is|are) no", r"cannot be determined", r"cannot determine",
           r"unresolved", r"unspecified", r"undefined", r"not enough "
           r"information", r"insufficient", r"unable to")
_DENIAL_RE = re.compile("|".join(_DENIAL), re.I)

# A denial names its subject soon after itself. "does not specify the bit
# ranges within those words" is about bits, not about words, because bits is
# the noun it reaches first.

_SUBJECT_WINDOW = normative_dimensions.SUBJECT_WINDOW

_EMPHASIS = re.compile(r"[*_`#]+")
_SEGMENT = re.compile(r"(?<=[.!?;])\s+|\n+")


def _plain(text):
    """The answer without its markdown, so matching reads the words only."""
    return _EMPHASIS.sub("", text or "")


def _segments(text):
    return [part.strip() for part in _SEGMENT.split(_plain(text)) if part.strip()]


def _value_pattern(dimension, value):
    """What stating this fact looks like, in the dimension it belongs to."""
    if dimension == WORD:
        return rf"\bword\s*(?:number\s*|index\s*)?{re.escape(value)}\b"

    if dimension == WORD_COUNT:
        spelled = {"1": "one", "2": "two", "3": "three", "4": "four",
                   "5": "five", "6": "six"}.get(value, value)

        return (rf"\b(?:{re.escape(value)}|{spelled})\s+(?:\d{{1,3}}-bit\s+)?"
                rf"words\b")

    if dimension == OFFSET:
        return rf"\b(?:octet|byte)\s*(?:offset\s*)?{re.escape(value)}\b"

    return rf"\bbits?\s*{re.escape(value)}\b"


def _names(segment, label):
    return re.search(re.escape(label), segment, re.I) is not None


_first_noun = normative_dimensions.first_noun
_dimension_of = normative_dimensions.dimension_of


def _denials(answer):
    """Every denial in the answer, with the dimension and segment it sits in."""
    found = []

    for segment in _segments(answer):
        for match in _DENIAL_RE.finditer(segment):
            found.append({"segment": segment, "dimension":
                          _dimension_of(segment, match.end())})

    return found


def states(answer, fact):
    """Does the answer carry this established fact, attached to its label?"""
    pattern = _value_pattern(fact.dimension, fact.value)

    for segment in _segments(answer):
        if _names(segment, fact.label) and re.search(pattern, segment, re.I):
            return True

    return False


def retracts(answer, fact):
    """Does the answer deny this fact, in the dimension the fact is about?"""
    for denial in _denials(answer):
        if (denial["dimension"] == fact.dimension
                and _names(denial["segment"], fact.label)):
            return True

    return False


def fills(answer, unresolved):
    """Does the answer put a value on a dimension the evidence leaves open?"""
    claim = {WORD: r"\bword\s*(?:number\s*|index\s*)?\d{1,3}\b",
             BITS: r"\bbits?\s*\[?\s*\d{1,2}\s*(?:\.\.|:|-|–|to)\s*\d{1,2}",
             OFFSET: r"\b(?:octet|byte)\s*(?:offset\s*)?\d{1,4}\b",
             WORD_COUNT: r"\b\d{1,3}\s+words\b"}[unresolved.dimension]

    for segment in _segments(answer):
        if not _names(segment, unresolved.label):
            continue

        for match in re.finditer(claim, segment, re.I):
            # A value inside a denial is the sentence saying it is absent.
            before = segment[max(0, match.start() - 80):match.start()]

            if not _DENIAL_RE.search(before):
                return True

    return False


def acknowledges(answer, unresolved):
    """Does the answer keep this dimension open where it discusses it?

    The denial does not have to name the field. A question that asks about two
    fields and nothing else is answered as a whole, and "the evidence does not
    give the bit ranges" is about both of them.
    """
    return any(denial["dimension"] == unresolved.dimension
               for denial in _denials(answer))


def discusses(answer, unresolved):
    """Does the answer raise this dimension at all?"""
    nouns = "|".join(_NOUNS[unresolved.dimension])

    return re.search(rf"\b(?:{nouns})\b", _plain(answer), re.I) is not None


def judge(scenario, answer):
    """The corrected classification, with the findings behind it."""
    findings = {"required_present": [], "required_missing": [],
                "required_retracted": [], "unresolved_preserved": [],
                "unresolved_filled": [], "unresolved_unacknowledged": []}

    for fact in scenario.required_supported_facts:
        key = f"{fact.label} -> {fact.dimension} {fact.value}"

        if retracts(answer, fact):
            findings["required_retracted"].append(key)
        elif states(answer, fact):
            findings["required_present"].append(key)
        else:
            findings["required_missing"].append(key)

    for unresolved in scenario.expected_unresolved_facts:
        key = f"{unresolved.label} -> {unresolved.dimension}"

        if fills(answer, unresolved):
            findings["unresolved_filled"].append(key)
        elif not discusses(answer, unresolved):
            # Not raised, so no boundary to hold. The question asked for it,
            # which is recorded, but silence invents nothing.
            findings["unresolved_unacknowledged"].append(key)
        elif acknowledges(answer, unresolved):
            findings["unresolved_preserved"].append(key)
        else:
            findings["unresolved_unacknowledged"].append(key)

    if findings["unresolved_filled"]:
        return UNSUPPORTED_ASSIGNMENT, findings

    if findings["required_missing"] or findings["required_retracted"]:
        return OVER_ABSTAINED, findings

    if findings["unresolved_unacknowledged"] and any(
            discusses(answer, item)
            for item in scenario.expected_unresolved_facts):
        return OVER_ABSTAINED, findings

    return ANSWERED, findings
