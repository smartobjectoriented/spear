"""Which dimension a sentence is denying, when it denies something.

A question about a field can be about several independent things: which word
carries it, which bits inside that word, which octet it starts at, how many
words the record has. Evidence can settle one and leave another open, so a
denial has to be read as being about one of them -- "the standard does not
specify the bit ranges within those words" denies the bits and leaves the word
numbers standing, even though both nouns are in the sentence.

FT3G2 established this reading for the evaluation oracle and pinned it with
wording variants. The final-answer guard needed the same reading and did not
have it, which made it contradict correct answers. Rather than have two
implementations drift apart, the resolution lives here once and both import it.
"""

from __future__ import annotations

import re

WORD = "word"
BITS = "bits"
OFFSET = "offset"
WORD_COUNT = "word_count"

# How each dimension is spoken about. Order inside a dimension does not
# matter; what matters is which dimension is named earliest in the text.

NOUNS = {
    WORD: (r"word\s*(?:number|index|position)?", r"located", r"location",
           r"carried", r"placement", r"which word"),
    BITS: (r"bit\s*ranges?", r"bit\s*positions?", r"bits?", r"msb", r"lsb"),
    OFFSET: (r"octet\s*offsets?", r"octets?", r"offsets?", r"bytes?"),
    WORD_COUNT: (r"word\s*count", r"number\s+of\s+words", r"how\s+many\s+words"),
}

# A denial names its subject soon after itself.
SUBJECT_WINDOW = 60


def first_noun(text):
    """The dimension named earliest in `text`, or None if none is."""
    best = None

    for dimension, nouns in NOUNS.items():
        for noun in nouns:
            found = re.search(rf"\b(?:{noun})\b", text, re.I)

            if found and (best is None or found.start() < best[0]):
                best = (found.start(), dimension)

    return best[1] if best else None


def dimension_of(segment, at):
    """Which dimension the denial ending at `at` is talking about.

    English puts the subject on either side: "does not specify the bit ranges"
    names it after, "the bit ranges are not specified" before. What follows the
    denial wins, because that is where an object goes; failing that the clause
    is read from its head, which is the noun it opens with -- "the bit
    positions within each word are undefined" is about bits, not about words.
    """
    after = first_noun(segment[at:at + SUBJECT_WINDOW])

    return after if after else first_noun(segment[:at])
