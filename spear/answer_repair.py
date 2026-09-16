"""One constrained attempt to say correctly what the guards refused.

Withholding is safe and it is not the goal. A turn that retrieved the right
provision, quoted it accurately, and then opened with the opposite conclusion
has everything needed for a correct answer; refusing it protects the reader
and teaches nothing. So when the guards reject an answer for a reason the
evidence itself can settle, the model is asked once more -- and the same
guards run again on what comes back.

Four rules, and they are what keep this from being a way around the gate:

  * NO NEW RETRIEVAL. The repair sees the provisions already retrieved and
    nothing else. A repair that could fetch would be a second turn wearing
    the first one's clothes.
  * ONE ATTEMPT. No loop. A second failure withholds, exactly as before.
  * THE SAME GUARDS. The repaired answer is validated by the identical
    pipeline, not a relaxed one.
  * ONLY FOR REPAIRABLE FINDINGS. A missing-evidence problem is not
    repairable by rewriting prose, and asking the model to try anyway is how
    an abstention becomes an invention.

The prompt carries the rejected conclusion, the exact ProvisionKeys the answer
may rest on with their own modality and text, and the findings themselves. It
does not carry the right answer: nothing here knows it.
"""

from __future__ import annotations

import re

import normative_claims

#: What a repair produced. "It passed the guards" is not the same fact as
#: "it answered", and conflating them accepted a repair that deleted a
#: correct conclusion: nothing asserted, nothing to object to.
REPAIR_ANSWERED = "REPAIR_ANSWERED"
REPAIR_WITHHELD = "REPAIR_WITHHELD"
REPAIR_FAILED = "REPAIR_FAILED"
#: The rewrite named the very identifier it was told was ungrounded. Caught
#: here so the trace says WHY the repair was refused; the guards would reject
#: the result either way.
REPAIR_UNGROUNDED = "REPAIR_UNGROUNDED"

#: Findings a rewrite can honestly fix, because the evidence is already in
#: hand and the defect is in what was said about it.
REPAIRABLE = frozenset({
    normative_claims.INCOHERENT_CONCLUSION,
    normative_claims.STRENGTHENED_MODALITY,
    normative_claims.UNGROUNDED_IDENTIFIER,
    normative_claims.AMBIGUOUS_CITATION,
    # A provision credited with the wrong force is a wording problem: the
    # evidence in hand already says which provision imposes what.
    normative_claims.MISATTRIBUTED_FORCE,
})

#: And ones it cannot. A bound nothing states is not a wording problem.
NOT_REPAIRABLE = frozenset({normative_claims.UNSUPPORTED_CARDINALITY})

#: Findings that bear on WHAT was concluded. A repair may legitimately change
#: the conclusion to settle one of these -- lower a modality, reattribute a
#: force, reverse an opening that contradicts its own evidence.
CONCLUSION_BEARING = frozenset({
    normative_claims.INCOHERENT_CONCLUSION,
    normative_claims.STRENGTHENED_MODALITY,
    normative_claims.MISATTRIBUTED_FORCE,
})

#: Findings about the wording AROUND a conclusion that still stands. An
#: unglossed name and an under-specified citation say nothing about whether
#: the answer was right, and a repair that drops the answer to settle one has
#: thrown away the part that was correct.
PRESENTATION_ONLY = frozenset({
    normative_claims.UNGROUNDED_IDENTIFIER,
    normative_claims.AMBIGUOUS_CITATION,
})

#: How a turn says it is not answering. Matched against the opening of a
#: draft, not the whole of it: an answer may well observe that some OTHER
#: question is unsettled further down.
_DECLINES = re.compile(
    r"\b(?:do(?:es)?\s+not\s+(?:settle|establish|specify|determine|answer|"
    r"support|provide|state)|cannot\s+be\s+(?:determined|established|"
    r"answered)|can(?:not|'t)\s+(?:be\s+)?(?:determined|established)|"
    r"is\s+not\s+(?:established|specified|determined|settled)|"
    r"insufficient\s+(?:evidence|information)|no\s+provision\s+(?:in\s+the\s+"
    r"(?:provided|retrieved)\s+evidence\s+)?(?:specifies|establishes|states))\b",
    re.I)

#: How much of a draft counts as its opening. Two sentences: a rewrite may
#: lead with a caveat before it declines -- one did, echoing a line of its
#: own instructions before abandoning the answer -- but reaching further
#: would read an answer's closing note about some OTHER unsettled question
#: as a refusal of the one asked.
_OPENING_SENTENCES = 2


def answers(text):
    """Does this draft assert something, or decline to?

    Deliberately small. The pipeline has no answer/withhold status of its own
    and inventing a large one here would be a second classifier to keep
    correct; this decides one question, and its tests stand alone.
    """
    body = (text or "").strip()

    if not body:
        return False

    opening = " ".join(
        re.split(r"(?<=[.!?])\s+", body)[:_OPENING_SENTENCES])

    return not _DECLINES.search(opening)


def _normalise(token):
    return re.sub(r"[-_\s]", "", str(token)).lower()


def repeats_flagged_identifier(repaired, problems):
    """Did the rewrite name a flagged identifier again, however spelled?

    Comparison is on the normalised form, because "Req-S" and "ReqS" are one
    name and a rewrite that merely re-hyphenates has not corrected anything.
    """
    flagged = {_normalise(problem["identifier"])
               for problem in problems
               if problem["kind"] == normative_claims.UNGROUNDED_IDENTIFIER
               and problem.get("identifier")}

    if not flagged:
        return None

    for token in normative_claims._IDENTIFIER.findall(repaired or ""):
        if _normalise(token) in flagged:
            return token

    return None


def outcome(original, repaired, problems):
    """What the repair did, as a fact about the two drafts.

    A repair that turns an answering draft into a non-answer has not fixed
    the findings; it has removed everything they could attach to. When every
    finding was repairable, that is a worse outcome than the original, and
    accepting it hid a correct conclusion behind a refusal.
    """
    if not repaired:
        return REPAIR_FAILED

    if repeats_flagged_identifier(repaired, problems):
        return REPAIR_UNGROUNDED

    if answers(repaired):
        return REPAIR_ANSWERED

    return REPAIR_WITHHELD if answers(original) else REPAIR_FAILED


def is_repairable(problems):
    """Only when every finding is one a rewrite can settle."""
    if not problems:
        return False

    kinds = {problem["kind"] for problem in problems}

    return bool(kinds) and kinds <= REPAIRABLE


def _provision_lines(evidence):
    lines = []

    for record in sorted(evidence.provisions.instances.values(),
                         key=lambda item: str(item.key)):
        if record.key.kind in ("ScopePreamble", "TableRow"):
            continue

        if record.declaration_status != "DECLARATION":
            continue        # a candidate may not ground an answer

        # The printed label, disambiguated only where the document reuses it,
        # so the model cites what a reader can find.
        reused = len(evidence.provisions.instances_for(record.key)) > 1
        cited = record.rendered_citation(disambiguate=reused)
        lines.append(f"  [{cited}]  modality: {record.modality}\n"
                     f"      {record.text[:400]}")

    return lines


def _finding_lines(problems):
    lines = []

    for problem in problems:
        kind = problem["kind"]

        if kind == normative_claims.UNGROUNDED_IDENTIFIER:
            lines.append(
                f"  - `{problem['identifier']}` is named in your answer and "
                "appears in no provision below. Do not use it again, and do "
                "not substitute a variant spelling, an expansion or a "
                "synonym for it. If it was not needed to answer the "
                "question, leave it out and keep the rest of the answer as "
                "it was. If the answer cannot be made without it, say that "
                "the evidence does not support the point rather than naming "
                "it anyway.")
        elif kind == normative_claims.STRENGTHENED_MODALITY:
            lines.append(f"  - Your conclusion states a {problem['claimed']}; the "
                         f"provision you cite establishes a {problem['supported']}. "
                         "Say what the provision says.")
        elif kind == normative_claims.INCOHERENT_CONCLUSION:
            lines.append("  - Your opening sentence contradicts the provision the "
                         "answer relies on. The opening is the answer.")
        elif kind == normative_claims.AMBIGUOUS_CITATION:
            lines.append(f"  - `{problem['reference']}` names several provisions "
                         f"({', '.join(problem['candidates'])}). Name the kind.")
        elif kind == normative_claims.MISATTRIBUTED_FORCE:
            lines.append(f"  - You credited {problem['reference']} with "
                         f"{problem['claimed']} force; its role is "
                         f"{problem['role'].lower()} and it supports "
                         f"{problem['supported']}. If another provision above "
                         "imposes the obligation, attribute it to that one.")

    return lines


def prompt(question, rejected, evidence, problems):
    """What the model is shown. It contains no answer, only constraints."""
    return "\n".join([
        "Your previous answer was rejected by deterministic validation.",
        "",
        f"QUESTION: {question}",
        "",
        "YOUR REJECTED ANSWER:",
        f"  {(rejected or '').strip()[:1200]}",
        "",
        "WHY IT WAS REJECTED:",
        *_finding_lines(problems),
        "",
        "THE ONLY EVIDENCE YOU MAY USE -- no other source exists for this turn,",
        "and you cannot retrieve more:",
        *_provision_lines(evidence),
        *_grounded_lines(evidence, problems),
        "",
        "",
        "Correct ONLY the problems listed above. Everything else in your",
        "answer that the provisions support must survive unchanged.",
        *_preservation_lines(problems),
        "Open with the conclusion those provisions support, at their own",
        "modality -- a permission is not a requirement and a recommendation",
        "is not a rule -- and cite each provision by kind and number.",
        "Do not strengthen a provision's force. Do not introduce evidence",
        "that is not above; you cannot retrieve more.",
        "",
        "Do NOT replace a supported answer with 'cannot be determined', 'the",
        "provisions do not settle the question', 'insufficient evidence' or",
        "anything equivalent in order to avoid the problems listed. Where the",
        "provisions above directly establish what was asked, state it.",
        "Say the question is unsettled only if, after the corrections above,",
        "the provisions genuinely do not answer it.",
    ])


def _grounded_lines(evidence, problems):
    """The identifiers the evidence contains, when grounding was the problem.

    Shown as a bound, not a menu: the failure this addresses is an answer
    reaching for a name it half-remembered, and the fix is to say which names
    exist -- not to invite more of them.
    """
    if not any(problem["kind"] == normative_claims.UNGROUNDED_IDENTIFIER
               for problem in problems):
        return []

    forms = evidence.identifier_forms()

    if not forms:
        return ["", "The evidence above names no field identifiers. Do not "
                    "introduce any."]

    return ["", "The ONLY identifiers the evidence above contains, spelled as "
                "it spells them:", "  " + ", ".join(forms[:40]),
            "Use no others, and add none of these merely to be more explicit."]


def _preservation_lines(problems):
    """Told outright when the conclusion itself was not what was wrong."""
    kinds = {problem["kind"] for problem in problems}

    if kinds and kinds <= PRESENTATION_ONLY:
        return ["Your conclusion was not rejected -- only the wording noted",
                "above was. Keep the conclusion and fix that wording.", ""]

    return [""]


def attempt(question, rejected, evidence, problems, *, ask):
    """One repair, or None.

    `ask` takes the prompt and returns the model's text. It is injected so
    this module never reaches a network or a tool of its own.
    """
    if not is_repairable(problems):
        return None

    try:
        repaired = ask(prompt(question, rejected, evidence, problems))
    except Exception:
        return None

    return (repaired or "").strip() or None
