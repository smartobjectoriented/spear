"""What a recorded turn has to survive before it becomes training material.

Recording is deliberately wide: a turn is kept whatever it did, because
filtering later is free and re-running a session is not. Promotion is the
opposite, and it is where the two decisions must not be confused. The local
trajectory file is the operator's own working record of their own machine;
this module governs the boundary the data crosses to become a corpus, which
is the point where a mistake stops being local.

Three things are checked, and they are checked in this order because the
first two are cheap and the third is the one that matters.

  1. THE VERDICT. Only a turn the project itself verified is something to
     imitate. `fail` and `unrated` are still stored -- a corpus of successes
     cannot teach what to stop doing, and "nothing judged this" is not
     "this was judged wrong" -- but neither becomes a positive candidate.

  2. THE CHANGE. A verified turn that wrote nothing was verified vacuously:
     the build gate only runs when a turn changed something, so a `pass`
     without a change is a recording error rather than an achievement.

  3. THE SECRETS. Tool results carry whatever the commands printed:
     paths, file contents, environment. TrainingRedactionPolicy already
     scrubs the export; here it runs BEFORE the episode is written, so the
     store never holds the unscrubbed text in the first place.

A private key is the one finding that refuses the row outright rather than
scrubbing it. Everything else redacts and is recorded: the counts travel with
the episode, so a reviewer can see that a sample was cleaned and of what,
instead of having to trust that it was.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import trajectory_episodes
from training_data import TrainingRedactionPolicy

# Findings that cannot be made safe by replacing them. A private key in a
# recorded result means the key was on that machine's disk and was printed;
# scrubbing the text does not undo that, and keeping the row buys nothing.
REFUSED_FINDINGS = frozenset({"private_key"})


def _report(categories, count):
    return {"categories": tuple(sorted(categories)), "redaction_count": count}


@dataclass(frozen=True)
class PromotionDecision:
    """Why a recorded row is, or is not, training material."""

    stored: bool
    positive: bool
    reasons: tuple[str, ...] = ()
    redactions: Mapping[str, int] | None = None

    def describe(self) -> str:
        return "; ".join(self.reasons) or ("promoted" if self.positive
                                           else "stored, not to imitate")


def _redact(row, policy):
    """The row with its text scrubbed, what was found, and how much.

    The policy reports {policy_version, redaction_count, categories} rather
    than a count per finding. Assuming the latter is what made the first
    version of this module unable to see a private key at all: the refusal
    below tested a set of names against a mapping that never held them.
    """
    cleaned, report = policy.redact(row)
    report = report or {}
    categories = report.get("categories") or ()

    return cleaned, tuple(str(name) for name in categories), int(
        report.get("redaction_count") or 0)


def decide(row, *, policy=None):
    """Whether this recorded row may be stored, and whether it may be imitated.

    Pure: it reads the row and nothing else. The verdict it reads was decided
    when the turn ran, by the project's own build and tests -- nothing here
    re-judges the work, only what may be done with the record of it.
    """
    policy = policy or TrainingRedactionPolicy.from_environment()

    if not isinstance(row, Mapping):
        return None, PromotionDecision(False, False, ("not a recorded row",))

    cleaned, found, count = _redact(row, policy)
    refused = sorted(REFUSED_FINDINGS.intersection(found))

    if refused:
        return None, PromotionDecision(
            False, False,
            tuple(f"refused: a {name.replace('_', ' ')} appears in this turn"
                  for name in refused), _report(found, count))

    verdict = str(cleaned.get("verdict") or "unrated")
    reasons: list[str] = []

    if verdict != "pass":
        reasons.append("the project did not verify this turn"
                       if verdict == "fail"
                       else "nothing judged this turn")

    # A pass is only meaningful on a turn that wrote: the build gate does not
    # run otherwise, so a pass without a change was never actually tested.
    elif not cleaned.get("steps"):
        reasons.append("verified but the turn ran no tool")

    positive = not reasons

    if found:
        reasons.append(f"redacted {count} value(s): " + ", ".join(sorted(found)))

    return cleaned, PromotionDecision(True, positive, tuple(reasons),
                                      _report(found, count))


def promote(row, *, policy=None, system="", session_id=None, index=0,
            tool_schemas=None):
    """The episode to store for this row, and the decision behind it.

    Returns (None, decision) when the row must not be stored at all. An
    episode that may be stored but not imitated keeps its own eligibility --
    the converter derives that from the verdict -- and carries the promotion
    reasons so a later reader is not left guessing why.
    """
    cleaned, decision = decide(row, policy=policy)

    if cleaned is None or not decision.stored:
        return None, decision

    episode = trajectory_episodes.episode_from_row(
        cleaned, system=system, session_id=session_id, index=index,
        tool_schemas=tool_schemas)

    metadata = dict(episode.training_metadata)
    metadata["promotion"] = {
        "positive": decision.positive,
        "reasons": list(decision.reasons),
        "redactions": dict(decision.redactions or {}),
    }

    # An episode the policy will not have imitated must not claim to be a
    # positive candidate, whatever the verdict said. The verdict is about the
    # code; this is about the record.
    if not decision.positive:
        metadata["eligibility"] = _demote(metadata.get("eligibility", ""))

    return type(episode)(**{**episode.to_dict(), "turns": episode.turns,
                            "training_metadata": metadata}), decision


def _demote(eligibility: str) -> str:
    from training_data import TrainingEligibility

    if eligibility == TrainingEligibility.POSITIVE_CANDIDATE.value:
        return TrainingEligibility.INCOMPLETE.value

    return eligibility
