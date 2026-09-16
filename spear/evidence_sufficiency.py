"""When retrieved evidence is enough to stand a conclusion on.

A benchmark that names ONE unit per question asks the wrong thing of a corpus
that segments well. It was adequate while an extraction packed several
provisions into a single block -- naming any of them named the block. Once
each provision is its own unit, two separate questions appear, and only a
person can answer them: which provision does this conclusion actually rest
on, and does it rest on one or on several together?

    [[A]]        A alone is sufficient
    [[A], [B]]   either A or B independently suffices
    [[A, B]]     A and B are required together

The identities are what the document PRINTS. A source id names a unit inside
one extraction and is a binding derived per store; it is never the gold, and
treating it as the gold is what made a re-extraction look like a reasoning
regression.

Nothing here decides sufficiency. It applies decisions that were made
elsewhere, by someone entitled to make them.
"""

from __future__ import annotations


def is_covered(sets, retrieved):
    """Is at least one acceptable set entirely present in this evidence?

    An empty list of sets means no evidence was named -- an absence case --
    and nothing can cover it. An empty SET inside the list would mean "no
    evidence required", which no reviewer has ever meant, so it is not
    treated as satisfied.
    """
    found = set(retrieved)

    return any(group and set(group) <= found for group in sets or ())


def first_covering_rank(sets, ranks):
    """The smallest k at which some acceptable set is wholly retrieved.

    `ranks` maps a printed identity to where it was found, or to None when it
    was not found at all. For a joint set the answer is the rank of its LAST
    member: both have to be there, so the set arrives when the later one
    does.
    """
    best = None

    for group in sets or ():
        if not group:
            continue

        places = [ranks.get(item) for item in group]

        if any(place is None for place in places):
            continue

        arrives = max(places)
        best = arrives if best is None else min(best, arrives)

    return best


def covered_at(sets, ranks, k):
    """Would a window of this size carry a complete set?"""
    arrives = first_covering_rank(sets, ranks)

    return arrives is not None and arrives <= k


def recall_at(cases, k):
    """The share of cases whose evidence a window of k would carry.

    `cases` is an iterable of (sets, ranks). A case that names no evidence is
    excluded rather than counted as a failure: nothing was asked of
    retrieval.
    """
    scored = [item for item in cases if item[0]]

    if not scored:
        return None

    return sum(1 for sets, ranks in scored if covered_at(sets, ranks, k)) / len(scored)


def mean_reciprocal_rank(cases):
    """Averaged over the rank at which each case's evidence becomes COMPLETE.

    Not the rank of the first useful hit: a conclusion resting on two
    provisions is not half-supported by one of them.
    """
    scored = [item for item in cases if item[0]]

    if not scored:
        return 0.0

    total = 0.0

    for sets, ranks in scored:
        arrives = first_covering_rank(sets, ranks)
        total += 1 / arrives if arrives else 0.0

    return total / len(scored)
