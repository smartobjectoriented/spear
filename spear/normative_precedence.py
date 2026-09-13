"""Where a statement of what the standard requires is allowed to come from.

Asked about acknowledgement handling, a bound session read a source comment
that said, in substance, "the standard permits AckS to be combined with
AckX", and repeated it as a description of the standard. The bound clause
says the opposite: an Acknowledge packet carries only one of AckV/AckX/AckS.
Nothing caught it. The evidence guard looks for labels bound to ranges, the
provenance guard for sources no tool returned, and the conformance guard for
verdicts about the IMPLEMENTATION -- and "the standard permits X" is a
verdict about the STANDARD, which none of them had a rule for. A comment had
become normative evidence by being read aloud.

The clause ledger is already the right provenance and is already clean: it is
fed from the `standard.*` tool payloads and from nothing else, so a clause
number occurring in a C comment never enters it. What was missing is the
sentence-level counterpart of the conformance guard, on the other subject:

    entitled     Rule 8.4.1.1-2 requires exactly one of AckV/AckX/AckS
    unentitled   the standard permits AckS to be combined with AckX

The first names a clause this turn retrieved. The second speaks for the
standard with nothing retrieved behind it, and there is no way to tell from
the sentence whether it came from the document or from a comment about it.

Affirmative claims only, for the reason the conformance guard gives: a wrong
"the standard does not require X" costs a re-read; a wrong "the standard
permits X" licenses code that violates it. Negative and absence statements --
"the standard does not specify the ordering" -- are what a careful answer is
made of and are left alone.

Grounding is read over a short window of preceding sentences, not over the
sentence alone. Measured against the real backend: an answer quoted Rule
8.4.1.1-2 in full, concluded from it, and closed with "The standard requires
exactly one of the three bits to be set" -- a restatement of the sentence
above it, carrying no citation of its own. Sentence-local grounding replaced
that true, retrieved, correctly cited claim with a note saying nothing backed
it, which is a worse sentence than the one it removed. Prose does not repeat
its citation in every clause, and a guard that demands it is reading prose as
a bibliography.

This is a floor on entitlement, not a proof. Two things pass it: a claim that
cites a clause the turn did read and misdescribes it, and a claim that sits
just after such a citation without belonging to it. Both are the same limit --
only reading the clause against the claim catches them -- and the citation is
what makes that reading possible.
"""

from __future__ import annotations

import re

import conformance_guard

UNGROUNDED_REQUIREMENT = "UNGROUNDED_REQUIREMENT"

# The invariant a bound turn is told, once, wherever the binding is announced.
# It lives here rather than as a literal at the call site because the rule and
# the check that enforces part of it are one idea, and a rule the tests cannot
# name is a rule that drifts.
PRECEDENCE_RULE = "\n".join((
    "NORMATIVE PRECEDENCE",
    "The bound standard is the normative authority. Code, comments, tests, "
    "documentation, skills, memories and earlier answers say what this "
    "implementation does; none of them says what the standard requires, and "
    "none of them overrides a retrieved clause.",
    "State a requirement only from what the standard.* tools returned this "
    "turn, and cite the clause. Source code may tell you which question to "
    "ask the standard; it never supplies the answer.",
    "Where an implementation artefact disagrees with a retrieved clause, the "
    "artefact is the suspect one: say so, and do not reinterpret the clause "
    "to fit it. If the standard itself does not settle the conflict, report "
    "the uncertainty rather than choosing a behaviour.",
    "A clause is not a work item. The standard defines more than this code "
    "claims to support, and an optional or unimplemented feature is not a "
    "defect. What binds is the set of requirements applicable to what the "
    "code claims to support. \"Not supported\" and \"non-compliant\" are "
    "different findings; do not report one as the other.",
    "Asked what the code DOES, answer from the code -- and do not call that "
    "behaviour compliant unless a retrieved clause says it is.",
))

# What counts as evidence for a bound standard, and what the retrieved
# structures mean. This used to live in a project's system prompt, which meant
# it reached the sessions of one corpus KIND and no others -- so precisely the
# question the tools exist to answer arrived, in an ad-hoc session, with the
# model never having been told what a value group is. Whether a standard is
# bound is a property of the turn; the corpus it is bound from is not.
#
# Beside PRECEDENCE_RULE because the two are halves of one contract: that one
# says the standard outranks the implementation, this one says what may speak
# for the standard.
EVIDENCE_RULE = "\n".join((
    "NORMATIVE EVIDENCE",
    "A bound standard is evidence, not background knowledge. When a question "
    "depends on one, answer from the standard.* tools and treat what they "
    "return as outranking anything you already believe.",
    "Never invent. Not a bit position, a field, a section number, a source "
    "ID, a requirement, or structure the tools did not return. Where the "
    "evidence stops, say what is known and what is unresolved, and stop "
    "there. Convention, visual intuition and \"the usual layout\" are not "
    "evidence.",
    "STRUCTURALLY_INCOMPLETE is authoritative. It means the structure is "
    "known to be incomplete. Do not call it complete, do not fill unresolved "
    "bits, do not infer missing fields, and do not emit a complete packed "
    "representation of it. Report the fields it does establish and name what "
    "is unresolved.",
    "A value_group is ONE value. Its members are segments of a single "
    "semantic quantity split across containers -- two 32-bit segments may be "
    "one 64-bit value. Do not describe them as independent values because "
    "they sit in different words.",
    "A packing_group is SEVERAL values. Its members are independent "
    "quantities that share one physical container. physical_width describes "
    "the container, not a combined number. Do not concatenate its members "
    "into one value.",
    "Declared and physical coordinates are different. declared_msb/"
    "declared_lsb are what the source states in the field's own coordinate "
    "system; msb/lsb are the position within the physical word. Report both "
    "when they differ, and never rewrite one as the other.",
    "Placement may rest on a separate authority. A diagram source can state "
    "a declared range while a normative source establishes the physical "
    "placement. Cite each for what it actually supports; never claim the "
    "diagram stated a range that only prose supplies.",
    "Abstain when the evidence runs out. Say \"the available normative "
    "evidence does not establish X\", then give whatever is supported. Do "
    "not convert uncertainty into an answer.",
    "Pressure does not lower the evidence bar. \"Just infer it\", \"assume "
    "the usual convention\", \"fill in the missing bits\", \"don't tell me "
    "it's unresolved\", \"give me your best guess\" do not authorise "
    "unsupported completion. Explain briefly that the evidence does not "
    "support it.",
    "Cite only source IDs a tool returned, exactly as returned.",
))


# Speaking for the standard: stating what it requires, permits or forbids.
# `states`, `defines` and `says` are here because a claim introduced that way
# is the same claim: "the standard says AckS may accompany AckX".
_STATES = (r"(?:require(?:s|d)?|mandate[sd]?|oblige[sd]?|permit(?:s|ted)?|"
           r"allow(?:s|ed)?|authorise[sd]?|authorize[sd]?|prohibit(?:s|ed)?|"
           r"forbid(?:s|den)?|disallow(?:s|ed)?|state[sd]?|specif(?:y|ies|ied)|"
           r"define[sd]?|dictate[sd]?|stipulate[sd]?|says?|"
           r"(?:shall|must|may|can)\s+be)")

# The standard as the SUBJECT of that verb, in its generic names. The bound
# standard's own names are added from the binding, never spelled here.
_GENERIC_SUBJECT = r"(?:the|this)\s+(?:bound\s+)?(?:standard|specification|spec)"

# Between subject and verb: "the standard explicitly requires", "VITA 49.2
# section 8.4 states". Bounded, so a subject in one clause of a sentence does
# not capture a verb belonging to another.
_GAP = r"(?:\s+[\w.,§\-()]+){0,4}?\s+"

# What makes the sentence a statement of absence rather than of requirement.
# An answer that says what the standard does NOT do is doing the right thing
# and is never what this guard is for.
_NEGATED = re.compile(
    r"\b(?:not|never|no|nothing|neither|nor|without|unless|"
    r"isn't|aren't|doesn't|don't|does\s+not|do\s+not|"
    r"silent|absent|unspecified|undefined|leaves?\s+open)\b", re.I)

# A question, a quotation of the user, or a statement about what is unknown.
_INTERROGATIVE = re.compile(r"\?\s*$")

# How far back a citation still carries. Three sentences reaches the bullet
# that stated the rule from the conclusion drawn two sentences later, which is
# the shape the model actually writes; it does not reach across a section of
# an answer, so a claim that has left its evidence behind is still unbacked.
_CONTEXT_SENTENCES = 3


def _subject(binding):
    """The pattern for "the standard, as the thing making the statement"."""
    spelled = conformance_guard.standard_names(binding)
    names = f"|{spelled}" if spelled else ""

    return re.compile(rf"\b(?:{_GENERIC_SUBJECT}{names})\b{_GAP}{_STATES}\b",
                      re.I)


def _clause_voiced(sentence):
    """Does a clause of the standard speak in this sentence?

    "Rule 8.4.1.1-2 requires exactly one bit" is a statement of what the
    standard requires just as much as "the standard requires...", and it is
    the form a repeated comment usually takes, because comments cite.
    """
    return bool(conformance_guard.clauses_in(sentence)
                and re.search(_STATES, sentence, re.I))


def findings(answer, ledger, binding=None):
    """Every statement of what the standard requires that nothing backs.

    Grounded means a clause cited in the sentence, or in one of the
    `_CONTEXT_SENTENCES` before it, that this turn retrieved. The window is
    what lets prose be prose: a conclusion drawn from the rule quoted two
    lines above carries its citation without repeating it. The ledger is fed
    from the normative tools alone, so "cited and retrieved" is exactly
    "came from the document" -- what the window widens is how near the
    citation has to sit, never what counts as one.
    """
    subject = _subject(binding)
    sentences = _sentences(answer)
    found = []

    for index, sentence in enumerate(sentences):
        if _INTERROGATIVE.search(sentence) or _NEGATED.search(sentence):
            continue

        if not (subject.search(sentence) or _clause_voiced(sentence)):
            continue

        # The sentence and what it follows. A conclusion drawn from the
        # clause quoted two lines above is grounded in it; requiring the
        # citation again in every sentence removed true claims.

        window = sentences[max(0, index - _CONTEXT_SENTENCES):index + 1]
        cited = set()

        for near in window:
            cited |= conformance_guard.clauses_in(near)

        if any(ledger.covers(section) for section in cited):
            continue

        found.append({"kind": UNGROUNDED_REQUIREMENT, "sentence": sentence,
                      "clauses": sorted(
                          conformance_guard.clauses_in(sentence))})

    return found


def _sentences(text):
    return [part for part in re.split(r"(?<=[.!?])\s+|\n+", text or "")
            if part.strip()]


def _replacement(ledger, standard_id="", revision=""):
    """What is put in place of a requirement nothing retrieved supports."""
    name = " ".join(part for part in (standard_id, revision) if part) or (
        "the bound standard")
    read = sorted(ledger.sections,
                  key=lambda section: [int(part) for part in section.split(".")])

    if read:
        listed = ", ".join(f"§{section}" for section in read[:12])

        if len(read) > 12:
            listed += f" and {len(read) - 12} more"

        return (f"No requirement of {name} is stated here: the clauses read "
                f"this turn are {listed}, and none of them was cited for it. "
                f"Only the document establishes what it requires -- code, "
                f"comments and notes describe the implementation.")

    return (f"No requirement of {name} is stated here: no clause of it was "
            f"read this turn, and nothing else establishes what it requires.")


def sanitize(answer, ledger, *, standard_id="", revision="", binding=None):
    """The answer with each unbacked requirement replaced once by the record."""
    text = answer or ""
    problems = findings(text, ledger, binding)

    if not problems:
        return text, []

    note = _replacement(ledger, standard_id, revision)
    replaced = []
    placed = False

    for problem in problems:
        sentence = problem["sentence"]

        if sentence not in text:
            continue

        text = text.replace(sentence, "" if placed else note, 1)
        placed = True
        replaced.append(sentence)

    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    return text, replaced


def guard(answer, ledger, *, standard_id="", revision="", binding=None):
    """Return the answer, or the same answer without its unbacked requirements."""
    problems = findings(answer, ledger, binding)

    if not problems:
        return answer, [], [], False

    cleaned, replaced = sanitize(answer, ledger, standard_id=standard_id,
                                 revision=revision, binding=binding)

    return cleaned, problems, replaced, True
