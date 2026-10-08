"""When a position nobody stated is nevertheless the only one there is.

FT3F drew the line at what the evidence says literally, and that is right for
H4: a label with no established width, in a word with sixteen spare bits, can
be put anywhere in them, and choosing is invention. It is wrong for a fully
pinned record. If a twelve-octet record has two four-octet members at octets 0
and 8, and a third member whose length is established as four, then octets 4..7
is not a guess -- it is the only interval that satisfies the constraints
already established, and refusing it would be refusing arithmetic.

So the rule is uniqueness, not plausibility, and it is deliberately hard to
satisfy. Every quantity it uses must already be in the ledger: the container's
extent, the candidate's own width, and the position *and* width of every
positioned member. If the constraints admit two placements, or none, or if any
input is missing, there is no proof. Nothing here prefers the first gap, the
lowest offset, or contiguous packing; those are conventions, and a convention
is what this exists to refuse.
"""

from __future__ import annotations

from dataclasses import dataclass, field

EXPLICIT = "EXPLICIT"
DETERMINISTIC_ENTAILMENT = "DETERMINISTIC_ENTAILMENT"
REJECTED = "REJECTED"

# Why a proof did not close. Reported rather than collapsed into a bare no, so
# a trace says which established fact was missing.

NO_CONTAINER_EXTENT = "NO_CONTAINER_EXTENT"
NO_CANDIDATE_WIDTH = "NO_CANDIDATE_WIDTH"
NO_MEMBER_EXTENT = "NO_MEMBER_EXTENT"
OVERLAPPING_MEMBERS = "OVERLAPPING_MEMBERS"
MEMBER_OUTSIDE_CONTAINER = "MEMBER_OUTSIDE_CONTAINER"
MEMBERS_DO_NOT_ACCOUNT_FOR_EXTENT = "MEMBERS_DO_NOT_ACCOUNT_FOR_EXTENT"
NO_PLACEMENT_FITS = "NO_PLACEMENT_FITS"
PLACEMENT_NOT_UNIQUE = "PLACEMENT_NOT_UNIQUE"


@dataclass(frozen=True)
class Member:
    """One component of a container, as far as the evidence establishes it."""

    label: str
    offset: int | None = None
    length: int | None = None


@dataclass(frozen=True)
class Proof:
    """Whether a placement follows, and everything that decided it."""

    entailed: bool
    low: int | None = None
    high: int | None = None
    reason: str = ""
    candidates: tuple[tuple[int, int], ...] = ()
    occupied: tuple[tuple[int, int], ...] = ()
    extent: int | None = None
    width: int | None = None
    accounted: bool = False

    @property
    def interval(self):
        return (self.low, self.high) if self.entailed else None


def _fails(reason, **over):
    return Proof(entailed=False, reason=reason, **over)


def unique_placement(candidate, members, extent):
    """The one interval `candidate` can occupy, or a proof that there is none.

    `members` are the other components the evidence names. Only those with both
    an offset and a length constrain anything; one with a length and no offset
    still has to be accounted for, because a container with room for it has
    room for the candidate somewhere else too.
    """
    if extent is None or extent <= 0:
        return _fails(NO_CONTAINER_EXTENT)

    if candidate.length is None or candidate.length <= 0:
        # H4 stops here: the label is real, the spare bits are real, and the
        # width that would make the placement forced was never established.
        return _fails(NO_CANDIDATE_WIDTH, extent=extent)

    occupied = []

    for member in members:
        if member.label == candidate.label:
            continue

        if member.offset is None:
            continue

        if member.length is None or member.length <= 0:
            return _fails(NO_MEMBER_EXTENT, extent=extent,
                          width=candidate.length)

        low, high = member.offset, member.offset + member.length - 1

        if low < 0 or high >= extent:
            return _fails(MEMBER_OUTSIDE_CONTAINER, extent=extent,
                          width=candidate.length)

        occupied.append((low, high))

    occupied.sort()

    for before, after in zip(occupied, occupied[1:]):
        if after[0] <= before[1]:
            # The established positions contradict each other, so nothing can
            # be derived from them. Fail closed rather than pick a reading.
            return _fails(OVERLAPPING_MEMBERS, extent=extent,
                          occupied=tuple(occupied), width=candidate.length)

    # Every component the evidence names must have a length, and together they
    # must account for the container exactly. Without that the record may hold
    # something nobody mentioned, and "the only free interval" is not a fact
    # about the record, only about the part of it that was described.

    lengths = [member.length for member in members
               if member.label != candidate.label]

    if any(length is None for length in lengths) or (
            sum(lengths) + candidate.length != extent):
        return _fails(MEMBERS_DO_NOT_ACCOUNT_FOR_EXTENT, extent=extent,
                      occupied=tuple(occupied), width=candidate.length)

    taken = set()

    for low, high in occupied:
        taken |= set(range(low, high + 1))

    candidates = []

    for start in range(0, extent - candidate.length + 1):
        window = set(range(start, start + candidate.length))

        if not (window & taken):
            candidates.append((start, start + candidate.length - 1))

    if not candidates:
        return _fails(NO_PLACEMENT_FITS, extent=extent,
                      occupied=tuple(occupied), width=candidate.length,
                      accounted=True)

    if len(candidates) > 1:
        return _fails(PLACEMENT_NOT_UNIQUE, extent=extent,
                      occupied=tuple(occupied), width=candidate.length,
                      candidates=tuple(candidates), accounted=True)

    low, high = candidates[0]

    return Proof(entailed=True, low=low, high=high,
                 reason=DETERMINISTIC_ENTAILMENT, candidates=tuple(candidates),
                 occupied=tuple(occupied), extent=extent,
                 width=candidate.length, accounted=True)
