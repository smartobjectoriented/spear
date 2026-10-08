"""A shorthand in a question, and the identifiers a standard spells it with.

Standards name things in families. A document that defines a family of
related fields writes the members out -- one suffix per kind -- and a reader
asking about them writes the stem. The index sees two different words, because it sees
exactly what it was given: the stem is one term and each member is another,
and a stem that happens to appear on its own somewhere incidental is both rare
and therefore, to BM25, highly informative. So the handful of places that
mention the stem in passing outrank every clause that actually defines the
family, and a question phrased the way people phrase questions retrieves the
wrong part of the document.

The fix is to let the corpus answer for itself. A query term may bring in the
terms the BOUND STANDARD already contains that extend it by a short suffix --
nothing from a synonym list, nothing from another document, nothing invented.
Where there is no such family the query is unchanged, which is most of the
time.

The suffix bound is what separates a family from a coincidence. Members of a
technical family differ from their stem by a character or three, because that
is what distinguishes one member from the next. Ordinary words that merely
begin alike run on much further, and a stem that prefixes a great many terms
is not a stem at all but a common beginning, so it is left alone.
"""

from __future__ import annotations

import re

#: Words as the question wrote them, before the index casefolds them away.
_SURFACE = re.compile(r"[A-Za-z0-9]+(?:[._/-][A-Za-z0-9]+)*")

#: How much longer than the stem a family member may be. One or two characters
#: is the usual shape; three admits a hyphenated member without admitting an
#: unrelated longer word.
MAX_SUFFIX = 3

#: A stem shorter than this prefixes too much of any vocabulary to mean
#: anything, and one longer is already specific enough to find its own matches.
MIN_STEM, MAX_STEM = 3, 8

#: A stem with more members than this is a common beginning, not a family.
#: Measured against real standards vocabularies, where a single field family
#: runs to ten or thirteen members: the bound has to clear those while still
#: refusing a stem that prefixes half the dictionary.
MAX_FAMILY = 16

#: The most terms one query may gain, however many of its words qualify.
#: At least one whole family always fits, because half a family is worse than
#: none: it drops whichever member happens to sort last, which is arbitrary
#: and invisible.
MAX_ADDED = 16

#: A term this much of the corpus already carries is ordinary language here,
#: whatever it is elsewhere, and expanding it would widen rather than sharpen.
COMMON_SHARE = 0.05


def shorthands_in(query):
    """The words a question wrote as technical shorthand.

    Case is the signal, and it is the writer's own: someone asking about an
    subject in general writes it in lower case, and someone naming the field
    family writes it in capitals. Expanding on that mark alone keeps ordinary words
    out without a list of ordinary words -- which could not be had for a
    technical corpus anyway, since half its vocabulary is ordinary words used
    as terms of art.

    Requires more than one capital, so an ordinary word at the start of a
    sentence is not mistaken for a shorthand.
    """
    found = set()

    for match in _SURFACE.finditer(query or ""):
        word = match.group(0)

        if sum(1 for letter in word if letter.isupper()) > 1:
            found.add(word.casefold())

    return found


def family_for(stem, vocabulary):
    """The terms this corpus spells with that stem and a short suffix."""
    if not (MIN_STEM <= len(stem) <= MAX_STEM):
        return ()

    found = sorted(
        term for term in vocabulary
        if term != stem and term.startswith(stem)
        and 0 < len(term) - len(stem) <= MAX_SUFFIX)

    # A stem that begins half the vocabulary is a prefix, not a name.
    return tuple(found) if len(found) <= MAX_FAMILY else ()


def expand(terms, vocabulary, document_frequency, document_count, *, query=""):
    """Query terms, plus the corpus-native family terms they stand for.

    Returns `(terms, provenance)`. The original terms come first and unchanged:
    this adds, and never replaces or reorders, so a query that needed no help
    scores exactly as it did before.
    """
    added, provenance = [], {}

    if not vocabulary or document_count <= 0:
        return tuple(terms), provenance

    present = set(terms)
    shorthands = shorthands_in(query)

    if not shorthands:
        return tuple(terms), provenance

    # Rarest stem first. The budget is small, and a word this corpus almost
    # never uses is the one carrying the question: spending the allowance on a
    # common stem would crowd out the family that was actually being asked
    # about. Ties break alphabetically so the same query always expands the
    # same way.
    stems = sorted(dict.fromkeys(terms),
                   key=lambda term: (document_frequency.get(term, 0), term))

    for stem in stems:
        if stem not in shorthands:
            continue

        # Already ordinary here: expanding it widens the question instead of
        # sharpening it. Measured against this corpus, not against English.
        if document_frequency.get(stem, 0) > COMMON_SHARE * document_count:
            continue

        family = [term for term in family_for(stem, vocabulary)
                  if term not in present]

        if not family:
            continue

        # Whole family or none. Admitting part of one would answer the
        # question about some of its members and silently not the others.
        if len(added) + len(family) > MAX_ADDED:
            continue

        added.extend(family)
        present.update(family)
        provenance[stem] = tuple(family)

    return tuple(terms) + tuple(added), provenance
