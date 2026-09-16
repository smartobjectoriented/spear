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


def outcome(original, repaired, problems):
    """What the repair did, as a fact about the two drafts.

    A repair that turns an answering draft into a non-answer has not fixed
    the findings; it has removed everything they could attach to. When every
    finding was repairable, that is a worse outcome than the original, and
    accepting it hid a correct conclusion behind a refusal.
    """
    if not repaired:
        return REPAIR_FAILED

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
            lines.append(f"  - You named `{problem['identifier']}`. No retrieved "
                         "provision contains it. Remove it.")
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
