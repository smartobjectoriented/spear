"""One deterministic search when an empty structure registry ends the search.

The last failing supported control asks for two word numbers, calls
standard.get_structure twice, is told structure_count is zero, and concludes
that the standard defines no word positions. It does. Two numbered requirements
state them, one standard.search away. The model never looks, because an empty
approved-structure registry reads to it as an empty standard.

That is neither a zero-tool turn (FT3E already covers those) nor a gap in what
the guard can see (FT3F covers that): the evidence was never acquired. Telling
the tool payload to suggest a search has already been measured as ineffective
in FT3T, so the orchestration layer makes the one call itself, hands the result
back as ordinary evidence, and lets the model decide again.

One search, once per turn, never a fetch, and never a second attempt.
"""

from __future__ import annotations

import evidence_bootstrap

EMPTY_STRUCTURE_RECOVERY = "EMPTY_STRUCTURE_RECOVERY"
ZERO_TOOL_BOOTSTRAP = "ZERO_TOOL_BOOTSTRAP"

STRUCTURE_TOOL = "standard.get_structure"

# The two tools that reach normative text. A turn that used either of them
# after the empty result went looking on its own and needs nothing from here.

PROSE_TOOLS = ("standard.search", "standard.fetch")

# Calls this policy and FT3E injected. Only what the model asked for itself
# counts towards the trigger, so neither policy can recover the other's turn.

INJECTED = (EMPTY_STRUCTURE_RECOVERY, ZERO_TOOL_BOOTSTRAP, "BOOTSTRAP",
            "RECOVERY")


def is_empty_structure(payload):
    """Does this standard.get_structure result offer no usable structure?"""
    if not isinstance(payload, dict):
        return False

    if payload.get("error") == "STRUCTURE_NOT_FOUND":
        recovery = payload.get("recovery") or {}

        # A named structure that is merely misspelled is not an empty registry;
        # the payload still lists the identifiers that do exist.

        return not recovery.get("valid_definition_ids")

    if payload.get("error"):
        return False

    if payload.get("structure_count") == 0:
        return True

    return "structure_count" in payload and not payload.get("structures")


def _model_issued(call):
    return (call.get("origin") or "") not in INJECTED


def last_empty_structure(calls):
    """Index of the last empty structure result the model asked for itself."""
    found = None

    for index, call in enumerate(calls or ()):
        if ((call.get("tool") or "") == STRUCTURE_TOOL and _model_issued(call)
                and call.get("empty_structure")):
            found = index

    return found


def should_recover(question, calls, *, bound=True, already_fired=False,
                   answer=""):
    """Exactly the situation this policy exists for, and nothing wider."""
    if already_fired or not bound:
        return False

    if not (answer or "").strip():
        # An empty answer is a different fault, and the same one FT3E declines
        # to paper over.
        return False

    index = last_empty_structure(calls)

    if index is None:
        return False

    if any((call.get("tool") or "") in PROSE_TOOLS
           for call in (calls or ())[index + 1:]):
        return False

    return evidence_bootstrap.is_normative_turn(question, bound=bound)


def route(question):
    """The question as the user wrote it. Nothing rewrites it, and nothing
    invents an identifier for it."""
    return "standard.search", {"query": (question or "").strip()}
