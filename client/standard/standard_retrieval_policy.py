"""What a normative search ranks, and what is only kept.

A corpus that covers every span of a document exhaustively -- which is what
makes it verifiable -- also turns every isolated fragment into a document of
its own: a figure label, a lone number, a two-word bullet. Measured on one
bound standard, 59% of the units were four words or fewer with a median of
three, against 26% and nine for a corpus built from paragraph blocks.

That is not a cosmetic difference. BM25 normalises a document's score by its
length against the corpus average, so a tail that size dragged the average
from 42 words to 11 and pushed a 25-word table row from rank 4 to rank 8 --
off the first page of results, in a corpus whose text was strictly MORE
faithful than the one it was being compared against. The turn that needed
that row spent fourteen rounds looking for it and then withheld.

So a fragment is stored, is fetchable by its identity, and is not what a
normative question searches. Nothing is dropped: every span stays in the
corpus, stays verifiable and stays reachable.

The test is about what a unit CARRIES, never about how short it is on its
own. A four-word table row, a captioned figure, a heading, a formula and
anything bearing a printed provision label are evidence at any length.
"""

from __future__ import annotations

import re

#: A printed provision label -- "Rule 5.2-1", "Definition 3.5.1-7". The kinds
#: are the ones a specification-style document numbers its provisions with;
#: no document is assumed to use all of them.
_LABELLED = re.compile(
    r"\b(?:Rule|Recommendation|Permission|Observation|Definition|Requirement)"
    r"\s+\d+(?:\.\d+)*-\d+", re.I)

#: A unit short enough that, alone, it is layout rather than statement.
#: Deliberately small: the question is whether anything can be read OUT of
#: the unit, and five words is about where a clause stops being a fragment.
FRAGMENT_WORDS = 5

#: Content a document structures deliberately. Its pieces are meant to be
#: short -- a cell, a caption, a heading -- and shortness says nothing about
#: whether they answer a question.
STRUCTURED_CONTENT = frozenset({"TABLE", "TABLE_ROW", "TABLE_CELL", "CAPTION",
                                "HEADING", "FORMULA", "FIGURE"})


def carries_printed_identity(text):
    """Does the document give this text a name a citation could use?"""
    return bool(_LABELLED.search(text or ""))


def carries_sentence(text, *, minimum=FRAGMENT_WORDS):
    """Is there enough here to read a statement out of?"""
    return len((text or "").split()) >= minimum


def retrievable(text, *, content_type="", has_structure=False,
                minimum=FRAGMENT_WORDS):
    """Should a normative search rank this unit?

    Never a judgement about importance. A unit that fails this is still
    stored, still verifiable and still fetchable by identity; it is simply
    not something a question about what a document REQUIRES should surface
    ahead of the clauses that require it.
    """
    if has_structure:
        return True

    if str(content_type).upper() in STRUCTURED_CONTENT:
        return True

    if carries_printed_identity(text):
        return True

    return carries_sentence(text, minimum=minimum)
