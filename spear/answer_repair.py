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

import normative_claims

#: Findings a rewrite can honestly fix, because the evidence is already in
#: hand and the defect is in what was said about it.
REPAIRABLE = frozenset({
    normative_claims.INCOHERENT_CONCLUSION,
    normative_claims.STRENGTHENED_MODALITY,
    normative_claims.UNGROUNDED_IDENTIFIER,
    normative_claims.AMBIGUOUS_CITATION,
})

#: And ones it cannot. A bound nothing states is not a wording problem.
NOT_REPAIRABLE = frozenset({normative_claims.UNSUPPORTED_CARDINALITY})


def is_repairable(problems):
    """Only when every finding is one a rewrite can settle."""
    if not problems:
        return False

    kinds = {problem["kind"] for problem in problems}

    return bool(kinds) and kinds <= REPAIRABLE


def _provision_lines(evidence):
    lines = []

    for key, record in sorted(evidence.provisions.records.items(),
                              key=lambda item: str(item[0])):
        if record.key.kind == "ScopePreamble":
            continue

        lines.append(f"  [{key}]  modality: {record.modality}\n"
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
        "Write the answer again, using only the provisions above.",
        "Open with the conclusion those provisions support, at their own",
        "modality: a permission is not a requirement and a recommendation is",
        "not a rule. Cite each provision by kind and number.",
        "If the provisions above do not settle the question, say exactly that.",
    ])


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
