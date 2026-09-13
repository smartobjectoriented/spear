"""One deterministic evidence call when a bound question is about to be
answered with no evidence at all.

Two supported controls fail identically on three separate servers by answering
at round one with zero tool calls -- one of them in French, claiming the bound
standard cannot be found. The evidence is present and reachable in both cases,
so the model is not blocked; it simply does not look. Telling it to look has
already been measured as ineffective (FT3P), so the orchestration layer makes
the call itself, once, and hands the result back as ordinary evidence.

This is deliberately not part of the final-answer guard. The guard decides what
may leave; this decides that something was never fetched.
"""

from __future__ import annotations

import re

STANDARD_TOOLS = ("standard.search", "standard.fetch", "standard.get_structure",
                  "standard.cite")

# The identifier forms standard.get_structure resolves directly.
_STRUCTURE_ID = re.compile(r"\b(?:bfd|fld|pkg|vgr|bit)-[0-9a-f]{8,}\b")

# What is left of a vocabulary test. The BINDING is now decided per turn by
# standard_scope.engages(), so by the time this module runs, the harness has
# already established that the turn engages the standard — and this list can
# only disagree with that decision by being narrower.
#
# It did. Bound to NIST-RS274NGC, "in RS274NGC, which G codes are in modal
# group 1?" engaged the standard by name and matched not one word below, so no
# retrieval was forced and the answer came out of interp_array.cc: plausible,
# uncited, and from the implementation rather than the specification the
# session was bound to. Every standard has its own nouns — bits and octets for
# a wire format, modal groups and canonical functions for a machining
# language — and a list here can only ever hold one of them.
#
# `bound` carries the decision instead. The words survive only for callers that
# ask about a turn WITHOUT a binding in force.
_NORMATIVE = ("bit", "bits", "word", "octet", "offset", "field", "fields",
              "structure", "struct", "layout", "range", "width", "encoding",
              "enum", "reserved", "register", "packet", "record", "standard",
              "revision", "rule", "section", "normative")


def is_normative_turn(question, bound=True):
    """Does this turn ask for something the bound standard would settle?

    A bound turn does, by construction: standard_scope.engages() is what put
    the binding there. Filtering again here is how a specification question
    got answered from the source code.
    """
    if bound:
        return True

    lowered = (question or "").lower()

    return bool(lowered.strip()) and any(
        re.search(rf"\b{word}\b", lowered) for word in _NORMATIVE)


def used_standard_tools(calls):
    return any((call.get("tool") or "") in STANDARD_TOOLS for call in calls)


def should_bootstrap(question, calls, *, bound=True, already_fired=False,
                     answer=""):
    """Exactly the situation this policy exists for, and nothing wider."""
    if already_fired or not bound:
        return False

    if used_standard_tools(calls):
        return False

    if not (answer or "").strip():
        # An empty answer is a different fault; the model produced nothing to
        # stand behind, and inventing evidence for it would hide that.
        return False

    return is_normative_turn(question, bound=bound)


def route(question):
    """The one call to make. Deterministic, from the user's own words.

    An explicit structure identifier is handled by get_structure; anything else
    goes to search with the original query, unmodified. Nothing is rewritten,
    and no second query is invented.
    """
    found = _STRUCTURE_ID.search(question or "")

    if found:
        return "standard.get_structure", {"definition_id": found.group(0)}

    return "standard.search", {"query": (question or "").strip()}
