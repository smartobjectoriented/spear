"""Whether a round of navigation added anything, and what to say when it stops.

F1 spends its whole round budget searching, fetching and re-reading structures
and then returns nothing at all. The hypothesis this module was written to test
is that the loop has no notion of "there is no more evidence" and simply runs
until it is cut off -- so the fix would be to notice the repetition and stop.

Progress is decided by comparing what a result establishes against what the
session has already seen: source ids, fetched units, resolved structures, the
normative facts a requirement states, the labels a structure settles or leaves
open, and citations. Wording, ordering and the route taken to the same evidence
are not progress. Nothing here asks a model anything.

The second half is what to say on the way out. An empty answer is the worst
possible ending for a question about a standard: it neither answers nor
records what was looked at. So the accumulated evidence is rendered into a
bounded conclusion -- what was found, what was asked for and remains open, and
an explicit statement that the retrieved evidence does not settle it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import evidence_fetch

EVIDENCE_TOOLS = ("standard.search", "standard.fetch", "standard.get_structure",
                  "standard.cite")

EXHAUSTED = "EVIDENCE_EXHAUSTION"
BUDGET = "ROUND_BUDGET"

# Three rounds that establish nothing new. One repeat is ordinary navigation --
# a refused identifier, a search rerun with different words -- and stopping
# there would cut off a session that was about to find something.

NO_PROGRESS_LIMIT = 3

_SOURCE = re.compile(r"\bstd-[0-9a-f]{8,}\b")
_STRUCTURE = re.compile(r"\b(?:bfd|fld|pkg|vgr)-[0-9a-f]{8,}\b")


def _fact_key(fact):
    return (f"fact:{fact.fact_class}:{fact.key}:{fact.offset}:{fact.length}:"
            f"{fact.word_index}:{fact.msb}:{fact.lsb}:{fact.octets}:"
            f"{fact.words}")


def evidence_signature(name, text, payload, ledger=None):
    """What this one result establishes, as strings two rounds can compare.

    Derived from the result, never from its presentation: the same unit reached
    by search and then by fetch signs the same, and a payload that differs only
    in ordering or wrapper metadata signs identically. The payload itself is
    never touched.
    """
    found = {f"source:{value}" for value in _SOURCE.findall(text or "")}

    if not isinstance(payload, dict) or payload.get("error"):
        return found

    found |= {f"structure:{value}" for value in _STRUCTURE.findall(text or "")}

    if name == "standard.fetch":
        unit = payload.get("unit") or {}

        if unit.get("source_id"):
            found.add(f"unit:{unit['source_id']}")

        for fact in evidence_fetch.unit_facts(payload):
            found.add(_fact_key(fact))

    if name == "standard.get_structure" and ledger is not None:
        before = set(ledger.established) | set(ledger.unresolved)
        ledger.observe(payload)
        found |= {f"established:{key}"
                  for key in set(ledger.established) - before}
        found |= {f"unresolved:{key}"
                  for key in set(ledger.unresolved) - before}

    citation = payload.get("citation") or {}

    for value in (payload.get("source_id"), citation.get("source_id")):
        if value:
            found.add(f"citation:{value}")

    return found


@dataclass
class ProgressTracker:
    """What the session has established, and how long since that last grew."""

    seen: set = field(default_factory=set)
    consecutive_no_progress: int = 0
    by_round: list = field(default_factory=list)
    evidence_rounds: int = 0

    def observe_round(self, round_index, results, ledger=None):
        """One round's results. Returns whether the round added evidence."""
        fresh = set()
        tools = []

        for name, text, payload in results:
            if name not in EVIDENCE_TOOLS:
                continue

            tools.append(name)
            fresh |= evidence_signature(name, text, payload, ledger) - self.seen

        if not tools:
            return True

        self.evidence_rounds += 1
        self.seen |= fresh

        if fresh:
            # Mandatory: any genuinely new evidence puts the counter back to
            # zero, so the policy can never stop a session that is still
            # finding normative material.
            self.consecutive_no_progress = 0
        else:
            self.consecutive_no_progress += 1

        self.by_round.append({"round": round_index, "tools": tools,
                              "new_evidence": sorted(fresh),
                              "new_count": len(fresh),
                              "progress": bool(fresh),
                              "consecutive_no_progress":
                                  self.consecutive_no_progress})

        return bool(fresh)

    def exhausted(self):
        return self.consecutive_no_progress >= NO_PROGRESS_LIMIT


def should_terminate(tracker, *, bound=True, answer="", calls=()):
    """Exactly the situation this policy exists for, and nothing wider."""
    if not bound or (answer or "").strip():
        return False

    if not any((call.get("tool") or "") in EVIDENCE_TOOLS for call in calls):
        return False

    return tracker.exhausted()


# --------------------------------------------------------------------------
# what to say when the loop stops without the model having said anything
# --------------------------------------------------------------------------

_ASKED = re.compile(r"\b(?:bit position|bit range|physical bit|bit)s?\b", re.I)


def _wanted(question):
    """The names the question asks about, as the question wrote them."""
    found = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\s+(?:and|"
                       r"fields?|field)\b", question or "")
    found += re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b",
                        question or "")
    ordered = []

    for name in found:
        if name not in ordered and len(name.split()) > 1:
            ordered.append(name)

    return ordered[:4]


def bounded_absence(question, ledger, tracker, *, reason=EXHAUSTED,
                    rounds=0, sources=()):
    """A grounded ending: what was read, what stayed open, and what follows.

    Deterministic, and built only from what the session actually retrieved. It
    states no position, no range and no width that the evidence did not, which
    is the whole reason it exists rather than a second model call.
    """
    lines = []
    asked = _wanted(question)

    if asked:
        lines.append("Requested: " + ", ".join(asked) + ".")
        lines.append("")

    if ledger.established:
        lines.append("Established by the normative evidence retrieved:")

        for item in sorted(ledger.established.values(),
                           key=lambda f: (f["word_index"] or 0,
                                          -(f["msb"] or 0)))[:12]:
            where = (f" of word {item['word_index']}"
                     if item["word_index"] is not None else "")
            lines.append(f"- {item['label']} — bits {item['msb']}..{item['lsb']}"
                         f"{where}")

        lines.append("")

    if ledger.unresolved:
        lines.append("Present in the evidence with no established position:")

        for label in sorted(ledger.unresolved.values()):
            lines.append(f"- {label}")

        lines.append("")

    # "Bit positions" is what this ending was written for, and it is wrong
    # for a question that never asked for any: "how should the ACK be
    # managed" ended with "does not establish the requested bit positions",
    # which reads as an answer to some other question. The question decides
    # the noun; the ledger decides whether "bit positions" was ever the
    # subject.

    positional = bool(asked) or bool(ledger.established) or bool(
        ledger.unresolved)
    outcome = ("does not establish the requested bit positions"
               if positional else
               "does not settle the question as asked")

    lines.append(f"The retrieved normative evidence {outcome}. "
                 + ("Navigation stopped because the last "
                    f"{tracker.consecutive_no_progress} rounds of retrieval "
                    f"returned no evidence that had not already been read"
                    if reason == EXHAUSTED else
                    f"Navigation stopped after {rounds} rounds of retrieval")
                 + f"; {len(tracker.seen)} distinct pieces of evidence were "
                   f"read in total.")
    lines.append("")
    lines.append("No position, range or width is inferred here beyond what the "
                 "evidence states. Nothing above is a guess.")

    citations = [value for value in sources if value.startswith("std-")][:8]

    if citations:
        lines.append("")
        lines.append("Sources read: " + ", ".join(citations))

    return "\n".join(lines)
