"""Audit the preference set before anyone trains on it.

Every check here exists because the corresponding mistake makes the data worse
than no data: a pair whose sides differ in ten unrelated ways teaches ten
things, an eval prompt that also appears in training measures nothing, and a
chosen answer that refuses what the evidence establishes trains the failure we
are trying to avoid creating.
"""

from __future__ import annotations

import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

# These scripts are run directly (``python eval/abstention/<name>.py``), so the
# package root is not on the path. The sibling modules come first, then the
# harness modules two levels up.

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

from sft_dataset import SFTDatasetError, _validate_structure

# Identifier shapes the tools hand out. Anything cited must have been returned.

_IDS = re.compile(r"\b(?:std|bfd|bit|fld|pkg|vgr)-[0-9a-f]{8,}\b")

# Wording that means the model reached past its evidence.

_INFERENCE_TELLS = (
    "usual convention", "common convention", "normal layout", "by convention",
    "leftmost", "rightmost", "left-to-right", "by symmetry", "best guess",
    "most likely", "typically the", "standard practice",
)

# Wording that means a chosen answer gave up when it should not have.
# Ways an answer can mark a boundary. The list is long because honest
# gap-marking is written in English rather than in a formula, and a tell list
# that only knows one phrasing reports good answers as bad ones.

_ABSTENTION_TELLS = (
    "does not establish", "not established", "cannot determine", "unresolved",
    "does not support", "cannot confirm", "isn't established",
    "doesn't establish", "not supported by", "does not settle",
    "cannot be determined", "not determined by", "would be guessing",
    "cannot say", "not something the evidence", "evidence settles",
    "does not say", "not say which", "does not name", "not name which",
    "does not state", "not stated", "does not specify", "not specified",
    "no meaning is defined", "is not defined", "not defined in",
    "left open", "leaves open", "not going to pick", "nothing retrieved",
    "not settled", "does not choose", "not known to be", "undefined in",
    "not assigned by", "not something this standard", "outside the scope",
    "does not follow", "not follow from", "would be an inference",
    "not fix", "does not fix", "cannot give you", "not established by",
)

# A tell inside a clause that repudiates it is not the answer relying on it.
# Deliberately narrow. "would be" and "is not" appear in answers that lean on
# a convention while hedging about it, and treating those as repudiation lets
# the exact reasoning we are trying to catch through.

_REPUDIATION = ("rather than", "not something", "does not state",
                "does not say", "is not evidence", "not what this")
ANSWERING_CLASS = "ANSWER_WHEN_SUPPORTED"

# A definite placement: what a control's winning answer has to contain.

_DEFINITE = re.compile(r"bits?\s*\d+\s*\.\.\s*\d+|word\s*\d+|"
                       r"offset\s*\d+|=\s*\d+\b|upper half|lower half",
                       re.I)

# The behaviours a losing side is allowed to be an example of. A label outside
# this set means the pair was filed under something nobody defined.

FAILURE_CLASSES = frozenset({
    "UNSUPPORTED_COMPLEMENT_INFERENCE", "PRESSURE_OVERRIDE",
    "CANNOT_CONCLUDE_ABSENCE", "GUESS_THEN_ABSTAIN", ANSWERING_CLASS})

# A citation pointing into somebody's filesystem was invented, whatever else
# the answer got right. Such a sample is discarded rather than repaired: the
# fabrication is a property of how the answer was produced, not a typo in it.

_LOCAL_PATH = re.compile(
    r"file://|localhost[:/]|(?<![\w-])/(?:home|Users|mnt|var|opt)/")

# How many times the same behaviour was seen from the same server process.

MINIMUM_REPRODUCTIONS = 2

# Only a pair whose answer the evidence leaves open may be a negative. The
# other two verdicts are refusals: an entailed complement is a correct
# deduction, and an inconclusive audit is not evidence that it is not.

ENTAILMENT_OK = "NOT_ENTAILED"
ENTAILMENT_NA = "NOT_APPLICABLE"
REQUIRED_FIELDS = ("pair_id", "class", "semantic_family", "structural_shape",
                   "source_phase", "rejected_origin", "reproductions",
                   "entailment_checked", "entailment_result",
                   "tool_path_clean", "train_or_eval", "licensed_content")


class AuditError(RuntimeError):
    pass


def _text(messages):
    return " ".join(str(m.get("content") or "") for m in messages)


def _distinctive(messages):
    """The part of a prompt that actually differs between examples.

    Every prompt carries the same system contract, which is most of its
    characters. Comparing whole prompts makes every pair look like a near
    duplicate of every other, so similarity is measured on the question and the
    evidence alone.
    """
    return " ".join(str(m.get("content") or "") for m in messages
                    if m.get("role") != "system")


def _admits_a_gap(text: str) -> bool:
    """Does this answer mark something as not established?

    Whitespace is collapsed first: an answer wrapped for reading splits "does
    not establish" across a line break, and a phrase match that misses it
    reports the opposite of the truth. "no unresolved words" contains
    "unresolved" and means the opposite, so a tell preceded by a negation does
    not count either.
    """
    lowered = " ".join(text.lower().split())

    for tell in _ABSTENTION_TELLS:
        start = 0

        while True:
            at = lowered.find(tell, start)

            if at < 0:
                break

            before = lowered[max(0, at - 12):at]

            if not any(word in before for word in ("no ", "zero ", "without ",
                                                   "nothing ")):
                return True

            start = at + 1

    return False


def _near_duplicate(left: str, right: str) -> float:
    return SequenceMatcher(None, left, right).ratio()


def audit(records, *, eval_prompts=(), duplicate_threshold=0.92):
    """Return one problem string per failed check; empty means the set is clean."""
    problems, seen_pairs, prompts = [], {}, []

    for record in records:
        pid = record["pair_id"]
        prompt, chosen, rejected = (record["prompt"], record["chosen"],
                                    record["rejected"])
        tools = record.get("tools", [])

        # Shape: both sides must be a legal single-assistant continuation.

        for side, completion in (("chosen", chosen), ("rejected", rejected)):
            messages = list(prompt) + list(completion)

            try:
                _validate_structure(messages, tools, len(messages) - 1)
            except SFTDatasetError as exc:
                problems.append(f"{pid}: {side} is not a valid target ({exc})")

        chosen_text = _text(chosen)
        rejected_text = _text(rejected)
        prompt_text = _text(prompt)

        if not chosen_text.strip():
            problems.append(f"{pid}: chosen is empty")

        if not rejected_text.strip():
            problems.append(f"{pid}: rejected is empty")

        if chosen_text.strip() == rejected_text.strip():
            problems.append(f"{pid}: chosen and rejected are identical")

        missing = [key for key in REQUIRED_FIELDS if key not in record]

        if missing:
            problems.append(f"{pid}: missing metadata {missing}")

        if record.get("licensed_content"):
            problems.append(f"{pid}: marked as carrying licensed content")

        # The audit that stands between a correct deduction and a poisoned
        # pair. It has to have run, and it has to have come back clear.

        if not record.get("entailment_checked"):
            problems.append(f"{pid}: entailment was never audited")

        verdict = record.get("entailment_result")

        if record.get("class") == ANSWERING_CLASS:
            if verdict not in (ENTAILMENT_OK, ENTAILMENT_NA):
                problems.append(f"{pid}: control has entailment verdict {verdict!r}")
        elif verdict != ENTAILMENT_OK:
            problems.append(f"{pid}: negative pair is {verdict!r}, not "
                            f"{ENTAILMENT_OK}")

        # A label nobody defined makes the set unauditable later.

        label = record.get("class")

        if label not in FAILURE_CLASSES:
            problems.append(f"{pid}: unknown failure class {label!r}")

        # Provenance: a losing side that was written rather than observed is
        # allowed, but it has to say so, and only for the answering controls.

        origin = record.get("rejected_origin")
        observed = origin in ("sampled", "OBSERVED")

        if origin not in ("sampled", "authored", "authored_fallback",
                          "OBSERVED", "AUTHORED"):
            problems.append(f"{pid}: rejected_origin is {origin!r}")
        elif not observed and label != ANSWERING_CLASS:
            problems.append(f"{pid}: losing side for a {label} pair was authored "
                            f"rather than observed")

        # A failure that was seen once from a deterministic endpoint is an
        # anecdote; it has to have been seen again.

        if observed:
            seen = int(record.get("reproductions", 0))

            if seen < MINIMUM_REPRODUCTIONS:
                problems.append(f"{pid}: observed failure reproduced {seen} "
                                f"times, needs {MINIMUM_REPRODUCTIONS}")

        # A sample taken through a broken tool path is evidence about the
        # tools, not about discipline.

        if record.get("tool_path_clean") is not True:
            problems.append(f"{pid}: tool path was not recorded as clean")

        # Grounding: an identifier in either answer must appear in the evidence.

        returned = set(_IDS.findall(prompt_text))

        for side, text in (("chosen", chosen_text), ("rejected", rejected_text)):
            invented = sorted(set(_IDS.findall(text)) - returned)

            if invented:
                problems.append(f"{pid}: {side} cites identifiers the evidence "
                                f"never returned: {invented}")

            if _LOCAL_PATH.search(text):
                problems.append(f"{pid}: {side} cites a local filesystem path")

        # Polarity: the defect must sit on the losing side, not the winning one.

        if record["class"] == ANSWERING_CLASS:
            # Noting a genuinely open side point is fine; failing to answer the
            # question is not. What a control's winning answer owes is a
            # definite placement, so that is what gets checked.

            if _admits_a_gap(chosen_text) and not _DEFINITE.search(chosen_text):
                problems.append(f"{pid}: chosen abstains on a case the evidence "
                                f"supports")

            if not _admits_a_gap(rejected_text):
                problems.append(f"{pid}: rejected does not show the "
                                f"over-abstention it is supposed to lose for")
        else:
            if not _admits_a_gap(chosen_text):
                problems.append(f"{pid}: chosen never marks anything unresolved")

            flat_chosen = " ".join(chosen_text.lower().split())

            for tell in _INFERENCE_TELLS:
                at = flat_chosen.find(tell)

                if at < 0:
                    continue

                # Naming a convention in order to reject it is the opposite of
                # leaning on it, so look at the clause around the mention.

                around = flat_chosen[max(0, at - 90):at + 120]

                if any(word in around for word in _REPUDIATION):
                    continue

                problems.append(f"{pid}: chosen uses convention/layout reasoning")

                break

        # A chosen answer that only refuses is not the behaviour we want.

        if record["class"] != ANSWERING_CLASS and len(chosen_text.split()) < 25:
            problems.append(f"{pid}: chosen is too thin to be useful")

        # Duplicates and leakage.

        distinctive = _distinctive(prompt)
        key = (distinctive, chosen_text, rejected_text)

        if key in seen_pairs:
            problems.append(f"{pid}: duplicate of {seen_pairs[key]}")

        seen_pairs[key] = pid

        for other_id, other in prompts:
            if _near_duplicate(distinctive, other) >= duplicate_threshold:
                problems.append(f"{pid}: near-duplicate prompt of {other_id}")

        prompts.append((pid, distinctive))

        for held in eval_prompts:
            if _near_duplicate(distinctive, held) >= duplicate_threshold:
                problems.append(f"{pid}: prompt leaks a held-out evaluation case")

    return problems


def main(path, eval_path=None):
    records = [json.loads(line) for line in
               Path(path).read_text().splitlines() if line.strip()]
    held = ()

    if eval_path:
        held = tuple(json.loads(line)["prompt_text"] for line in
                     Path(eval_path).read_text().splitlines() if line.strip())

    problems = audit(records, eval_prompts=held)

    for problem in problems:
        print("PROBLEM:", problem)

    print(f"\n{len(records)} pairs audited, {len(problems)} problems")

    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
