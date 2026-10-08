"""A standard that does not exist, for tests about standards.

The harness is bound, in production, to documents somebody licensed. Their
clause text is theirs: it may be ingested into a user's own runtime state and
read from there, and it may not be copied into this repository. Identifiers
are a different matter -- "ANSI-VITA-49.2", "§8.4.1.1", "AckV" name a thing
without reproducing it, and a test about binding or about name plumbing is
entitled to use the real ones.

What a test must not do is carry the sentences. Several here did: the rule
that an Acknowledge packet sets exactly one of three bits was quoted verbatim
from the document, in a fixture, because that was the case being debugged at
the time. The case is worth keeping and the sentence is not ours.

So the normative CONTENT is synthetic. SYNTH-WIDGET-1 is invented, its clauses
are written here, and its shape is chosen to exercise exactly what the real
one did:

  * §4.2.1.1 carries sibling rules -2 and -3, so a citation of one must not be
    credited with the other, while §4.2.1 must cover both;
  * Rule 4.2.1.1-2 is an exactly-one-of-three constraint, which is what makes
    a contradicting implementation detectable;
  * Rule 4.2.1.1-3 says what happens when more than one is requested, so an
    answer can be right about one rule and silent about its sibling;
  * Observation 4.1.1-1 is explanatory rather than binding, which is the
    distinction an answer has to preserve.

The vocabulary -- Widget Reply, selectors SelA/SelB/SelC -- is deliberately
unlike any real protocol. A reader who mistakes it for a specification has
misread it, which is the point.
"""

from __future__ import annotations

#: A binding to something invented. Tests about the binding MECHANISM may use
#: a real identifier instead; tests about normative content use this.
SID, REV = "SYNTH-WIDGET-1", "2024-R2026"

BINDING = {"standard_id": SID, "revision": REV}

#: The clause the exactly-one constraint lives in, and its page.
SECTION, PAGE = "4.2.1.1", 42

#: The rule an implementation can satisfy or contradict.
RULE = ("Rule 4.2.1.1-2: A Widget Reply shall carry exactly one of the "
        "selectors SelA, SelB and SelC.")

#: Its sibling. Satisfying one says nothing about the other, which is what a
#: clause ledger has to get right.
SIBLING = ("Rule 4.2.1.1-3: When a Widget Request names more than one "
           "selector, a separate Widget Reply shall be produced for each.")

#: Explanatory, not binding: an answer that reports it as a requirement has
#: lost a distinction the harness is supposed to keep.
OBSERVATION = ("Observation 4.1.1-1: A Widget Request may name any "
               "combination of selectors; the constraint above is on the "
               "reply, not on the request.")

#: What an implementation artefact says instead -- and it is wrong. This is
#: the contradiction the precedence rule exists to settle.
CODE_COMMENT = "/* the standard allows SelA and SelB together */"

#: A claim repeating that comment as though it described the document.
REPEATED_COMMENT = ("The standard permits SelA to be combined with SelB, so "
                    "both selectors may be set in one Widget Reply.")

#: The same claim, grounded: it names the clause a turn retrieved.
GROUNDED_CLAIM = ("Rule 4.2.1.1-2 requires exactly one of SelA, SelB and SelC "
                  "in a Widget Reply.")

#: A feature the document defines and an implementation need not support.
#: Optional by the document's own words, so its absence is not a defect.
OPTIONAL_FEATURE = "the Widget Origin Tag"
OPTIONAL_CLAUSE = ("Rule 4.4.2-1: A Widget Reply may carry the Widget Origin "
                   "Tag. Where it is absent the remaining words move up.")


def fetch(section=SECTION, text=RULE, *, page=PAGE, neighbours=(SIBLING,),
          source_id="std-synthwidget000000000000000000001"):
    """What `standard.fetch` hands back, in the shape the real tool uses."""

    return {"citation": {"section": section, "page": page,
                         "source_id": source_id},
            "unit": {"section": section, "text": text},
            "neighbors": [{"section": section, "text": item}
                          for item in neighbours]}


def search(section="4.2", snippet="Rule 4.2-6: a reply names its request ...",
           heading=("4 Widget Exchanges", "4.2 The Widget Reply Prologue")):
    """What `standard.search` hands back."""

    return {"results": [{"section": section, "snippet": snippet,
                         "heading_path": list(heading)}]}
