"""Does the evidence already settle the question the model was asked?

FT0.1C found scenarios where the "unsupported" complement was forced by
arithmetic: a container fully accounted for, member widths known, one slot
left. A model that works that out is reasoning correctly, and a preference
pair calling it a failure teaches against reasoning we want. So every negative
pair is audited before it is written, and anything settled -- or that might be
-- stays out.

The audit takes the numbers rather than hunting for them in prose, because a
regex that misses a width in a structure payload returns NOT_ENTAILED and lets
a poisoned pair through. AMBIGUOUS is a refusal, not a shrug.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

NOT_ENTAILED = "NOT_ENTAILED"
ENTAILED = "ENTAILED"
AMBIGUOUS = "AMBIGUOUS"

# Wording that closes a set: what is listed is all there is.

_EXHAUSTIVE = ("exactly two", "only the", "all of which", "the remaining",
               "occupies the rest", "and no other")


@dataclass(frozen=True)
class Facts:
    """What the evidence fixes about the shape of the container."""

    container_bits: int | None
    member_bits: int | None
    stated_placements: int
    members_in_container: int
    open_positions: int | None = None

    # True when what is missing is a meaning rather than a position -- the
    # sense of an encoding, the purpose of a region. Meanings are not drawn
    # from a finite set, so elimination can never reach one.

    semantic_gap: bool = False
    note: str = ""


def audit(facts: Facts, units_text: str = "") -> tuple[str, str]:
    if facts.semantic_gap:
        return NOT_ENTAILED, ("what is missing is a meaning, which no amount "
                              "of counting can settle")

    if facts.stated_placements == 0 and facts.members_in_container <= 1:
        # Elimination needs something eliminated. With nothing placed, no
        # arrangement has been ruled out and the position is open.

        return NOT_ENTAILED, ("no placement is stated, so there is nothing "
                              "for elimination to work from")

    if facts.open_positions is not None:
        # A scenario that knows its own answer space, such as an enum where
        # the codes are countable directly.

        if facts.open_positions <= 1:
            return ENTAILED, (f"only {facts.open_positions} position remains "
                              f"for the unplaced member")

        return NOT_ENTAILED, (f"{facts.open_positions} positions remain open "
                              f"to the unplaced member")

    if facts.container_bits and facts.member_bits:
        slots = facts.container_bits // facts.member_bits
        remaining = slots - facts.stated_placements
        unplaced = facts.members_in_container - facts.stated_placements

        if unplaced <= 0:
            return NOT_ENTAILED, "nothing is left unplaced to be inferred"

        if remaining < unplaced:
            return AMBIGUOUS, "the members do not fit the slots as counted"

        # What matters is how many arrangements survive, not how many slots
        # do. Two members and two empty slots leave two arrangements, and
        # which member goes where is exactly the thing not established.

        arrangements = math.perm(remaining, unplaced)

        if arrangements == 1:
            return ENTAILED, (f"{facts.container_bits} bits / "
                              f"{facts.member_bits} bits = {slots} slots, "
                              f"{facts.stated_placements} stated, leaving one "
                              f"possible arrangement of {unplaced} member(s)")

        return NOT_ENTAILED, (f"{arrangements} arrangements of {unplaced} "
                              f"unplaced member(s) remain over {remaining} "
                              f"free slot(s)")

    lowered = units_text.lower()

    if any(phrase in lowered for phrase in _EXHAUSTIVE):
        return AMBIGUOUS, "the evidence closes the set of contents in words"

    if facts.container_bits and not facts.member_bits:
        return AMBIGUOUS, ("the container is bounded but member widths are "
                           "not, so the counting cannot be completed")

    if not facts.container_bits:
        return NOT_ENTAILED, ("the evidence does not bound the container, so "
                              "elimination has nothing to work on"
                              + (f"; {facts.note}" if facts.note else ""))

    return AMBIGUOUS, "the counting is inconclusive"


def facts_for(scenario, *, members_in_container, stated_placements,
              open_positions=None, semantic_gap=False, container_bits=None,
              member_bits=None, note=""):
    """Read the container and member widths out of a scenario's own evidence.

    The two width overrides exist because a caller that knows the shape can
    state the numbers outright, and a stated number is auditable in a way a
    number recovered from prose by regex is not.
    """
    text = " ".join(unit.text for unit in scenario.units)
    container, member = container_bits, member_bits
    found = re.search(r"shall be (\d+) bits wide", text, re.I)

    if found:
        container = int(found.group(1))

    found = re.search(r"(two|three|four)[- ]32-bit words", text, re.I)

    if found:
        container = 32 * {"two": 2, "three": 3, "four": 4}[found.group(1).lower()]

    found = re.search(r"(twelve|eight|four)-octet record", text, re.I)

    if found:
        container = 8 * {"twelve": 12, "eight": 8, "four": 4}[found.group(1).lower()]

    found = re.search(r"each (\d+) bits wide", text, re.I)

    if found:
        member = int(found.group(1))

    found = re.search(r"each (four|\d+) octets", text, re.I)

    if found:
        member = 8 * (4 if found.group(1).lower() == "four"
                      else int(found.group(1)))

    found = re.search(r"\((\d+)\.\.(\d+)\)", text)

    if found and member is None:
        member = int(found.group(1)) - int(found.group(2)) + 1

    # A structure payload states widths the prose may only imply.

    for structure in scenario.structures:
        for field in structure.payload.get("fields", ()):
            if field.get("width"):
                member = member or int(field["width"])

        for group in structure.payload.get("packing_groups", ()):
            if group.get("physical_width"):
                container = container or int(group["physical_width"])

    return Facts(container, member, stated_placements, members_in_container,
                 open_positions, semantic_gap, note)
