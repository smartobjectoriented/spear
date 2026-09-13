"""Does THIS question engage the bound standard?

The binding used to attach to every prompt of every session. Once an operator
bound ANSI-VITA-49.2, each turn carried the "BOUND STANDARD … normative claims
must use retrieved canonical sources" system rule, the standard.* tools, and
the evidence policy that fires when such a turn is about to answer with nothing
retrieved.

Asked to download the RS274/NGC G-code specification, the assistant therefore
spent a whole task explaining that "the system is bound to ANSI-VITA-49.2, not
RS274/NGC" — correct, irrelevant, and repeated eleven times. The word that
dragged it there was "standard", in "inject it in our RAG as a standard": it
sits in the normative vocabulary, so the turn read as normative and the
orchestration fetched VITA evidence for a question about a PDF download.

The binding is decided per turn now, from two kinds of sign:

  - the question NAMES the bound standard. The terms are derived from the
    binding itself — its identifier, the parts of that identifier, its
    revision — and not from a list someone maintains: a subject's own name is
    not a guess about vocabulary.
  - the question asks the sort of thing this corpus settles: a structure
    identifier, or the concrete words of a data-format standard (bit, octet,
    offset, field, width, encoding…). The generic ones — standard, revision,
    rule, section — are deliberately absent. They are ordinary English, and
    they are exactly what bound a question about downloading a file.

Binding when in doubt is the safe direction: an unnecessary binding costs a
system rule and two tool schemas, a missing one costs grounding on a licensed
source. That is why the second sign is kept at all.

Checking the question ALONE is what the remaining two signs fix, and both are
about the turn's surroundings rather than its words.

A file the operator put in front of the model counts. "carry out the task in
doc/ack-task.md" names nothing; the file it points at said ANSI-VITA-49.2
2017-R2024 in its third paragraph, and the turn ran free. Only the standard's
NAME is looked for there — never the subject nouns, because any C file says
"bit" and "packet" and that would bind every turn after reading one.

And a session that has engaged the standard once stays engaged until /clear.
That began as a narrow rule — follow-up vocabulary, anaphora — and the narrow
rule kept being outrun. "can you validate the implementation ?" ran free and
declared the code "correct and compliant with ANSI/VITA 49.2-2017 (R2024)" on
the strength of twelve greps; it is not. Later a stray "/quit" reached the
model as a prompt, matched no follow-up word either, and the unbound turn that
followed went looking for the specification on the public web and edited five
source files against what it found. The words a follow-up can take are not
enumerable. The session is.

The asymmetry decides it, as it does everywhere else here: an unnecessary
binding costs a system rule and two tool schemas; a missing one has now cost a
false compliance verdict and a network fetch standing in for a licensed
source. /clear is how an operator says the subject changed, and it clears this
along with the history — so stickiness still follows a binding and cannot
create one, which is what keeps the RS274/NGC failure fixed.
"""

from __future__ import annotations

import re

# The identifier forms standard.get_structure resolves directly. Unambiguous:
# nothing else in a sentence looks like one.
_STRUCTURE_ID = re.compile(r"\b(?:bfd|fld|pkg|vgr|bit)-[0-9a-f]{8,}\b")

# The subject matter of a data-format standard. Compare evidence_bootstrap's
# wider list, which may stay wide because it only ever runs once a binding is
# already in force; this one DECIDES the binding, so "standard", "revision",
# "rule" and "section" are left out on purpose.
_SUBJECT = ("bit", "bits", "word", "words", "octet", "octets", "offset",
            "offsets", "field", "fields", "structure", "structures", "struct",
            "layout", "width", "encoding", "encodings", "enum", "reserved",
            "register", "registers", "packet", "packets", "record", "records",
            "normative", "conformance")



def identity_terms(binding):
    """Every way a user might name this standard, taken from the binding.

    ANSI-VITA-49.2 / 2017-R2024 yields ansi-vita-49.2, vita, 49.2, vita-49.2,
    vita 49.2, 2017-r2024, r2024 … Single short parts with no digit are
    dropped: they are initials, and matching them would bind on noise.
    """
    terms = set()

    for raw in (getattr(binding, "standard_id", ""),
                getattr(binding, "revision", "")):
        value = (raw or "").strip().lower()

        if not value:
            continue

        terms.add(value)
        parts = [part for part in re.split(r"[-_/\s]+", value) if part]
        terms.update(part for part in parts
                     if len(part) > 3 or any(c.isdigit() for c in part))

        # Adjacent pairs, so "vita 49.2" and "vita-49.2" both land even though
        # the identifier spells it with one separator.

        for first, second in zip(parts, parts[1:]):
            terms.add(f"{first}-{second}")
            terms.add(f"{first} {second}")

    return terms


def engages(binding, question, *, engaged_before=False, context=""):
    """Should this turn carry the bound standard?

    @param engaged_before  a previous turn of THIS session engaged the
                           standard, and /clear has not run since. The
                           session is then bound whatever this turn says.
    @param context         text the operator put in front of the model for
                           this turn without typing it -- a !read file. Only
                           the standard's own NAME counts there: "carry out
                           the task in doc/ack-task.md" engaged nothing, while
                           the file it named said "ANSI-VITA-49.2 2017-R2024"
                           in its third paragraph. The subject nouns are not
                           consulted in context on purpose: any C file says
                           "bit" and "packet", and that would bind every turn
                           after reading one.
    """
    if binding is None:
        return False

    text = (question or "").lower()

    if not text.strip():
        return False

    # Once a session has engaged the standard, it stays engaged until /clear.
    #
    # This started as a narrow rule -- follow-up vocabulary and anaphora --
    # and the narrow rule kept being outrun. A stray "/quit" reached the model
    # as a prompt, matched none of it, and the unbound turn that followed went
    # looking for the specification on the public web and edited five source
    # files against what it found there. The words a follow-up can take are
    # not enumerable; the session is.
    #
    # The asymmetry that decides it is the module's own: an unnecessary
    # binding costs a system rule and two tool schemas. A missing one has now
    # cost a false compliance verdict and a network fetch of a licensed
    # standard's substitute. /clear is the operator's way to say the subject
    # changed, and it clears this with the history.

    if engaged_before:
        return True

    if context:
        lowered = context.lower()

        for term in identity_terms(binding):
            if re.search(rf"(?<![\w.]){re.escape(term)}(?![\w])", lowered):
                return True

    if _STRUCTURE_ID.search(text):
        return True

    for term in identity_terms(binding):
        # Not \b: a term may end in a digit or a dot ("49.2"), where \b sits in
        # the wrong place and matches inside "149.25".
        if re.search(rf"(?<![\w.]){re.escape(term)}(?![\w])", text):
            return True

    return any(re.search(rf"\b{word}\b", text) for word in _SUBJECT)
