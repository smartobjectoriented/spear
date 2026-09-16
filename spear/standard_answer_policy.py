"""What a standard-bound turn may say, decided deterministically around the loop.

The model reaches the right evidence and then over-reads it. It places an
unresolved label in the spare bits, reads two diagram cells as the halves of one
word, treats an empty structure registry as an empty standard, answers without
looking, spends its whole round budget and says nothing, and cites a PDF path no
tool ever returned. Nine model-side interventions -- five adapters, serving
scale to 32x, a system-prompt rule, a tool-payload change, a dataset expansion
-- left the core failure exactly where it was.

So the boundary is drawn where it can be drawn deterministically: around the
tool loop, after the model has finished, out of its sight. This module owns that
lifecycle for one turn and nothing else owns it -- the interactive runtime and
the evaluation harness both drive this same object, because two copies of a
policy are two policies.

Order matters and is the validated one:

    tool loop -> zero-tool bootstrap -> empty-structure recovery
              -> evidence accumulation -> bounded rendering at the budget
              -> structural validation -> provenance validation -> answer

Nothing here is exposed to the model. It sees the same four normative tools it
always saw.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import conformance_guard
import conformance_mode
import diagram_geometry
import evidence_bootstrap
import evidence_progress
import evidence_recovery
import answer_repair
import normative_claims
import normative_precedence
import provenance_guard
from evidence_guard import EvidenceLedger, guard as evidence_guard_answer

STANDARD_TOOLS = evidence_bootstrap.STANDARD_TOOLS

MODEL = "MODEL"
ZERO_TOOL_BOOTSTRAP = evidence_recovery.ZERO_TOOL_BOOTSTRAP
EMPTY_STRUCTURE_RECOVERY = evidence_recovery.EMPTY_STRUCTURE_RECOVERY

# The same retrieval as the zero-tool bootstrap, made before the model has said
# anything rather than after it has finished.
#
# Bound to ANSI-VITA-49.2 and asked how ACK is managed, a session spent twelve
# bash rounds in the C sources and made not one normative call. The bootstrap
# fired at the boundary, as designed, and searched the standard once -- into a
# context already holding twelve grep outputs and a finished answer. The answer
# came back saying the implementation "correctly follows the VITA 49.2
# specification". It does not.
#
# A retrieval that arrives after the conclusion is a rebuttal, and the model
# treats it as one. The same call placed before the first round is a source:
# the greps then have to agree with the clause instead of filling its absence.
OPENING_RETRIEVAL = "OPENING_RETRIEVAL"

# How the turn ended, which decides whether a bounded conclusion is rendered.

STOPPED_BY_MODEL = "MODEL"
STOPPED_BY_BUDGET = evidence_progress.BUDGET
STOPPED_BY_EXHAUSTION = evidence_progress.EXHAUSTED


def canonical_arguments(arguments):
    """A stable key for one tool call's arguments, order-independent."""
    return json.dumps(arguments or {}, sort_keys=True, default=str)


@dataclass(frozen=True)
class Injection:
    """One deterministic tool call the policy wants made before answering."""

    tool: str
    arguments: dict
    origin: str


@dataclass
class StandardAnswerPolicy:
    """The deterministic lifecycle of one standard-bound turn.

    Constructed per turn. It holds every ledger the boundary needs and makes
    every decision the harness used to make inline, so the interactive runtime
    and the evaluation runner cannot drift apart.
    """

    #: the four normative tools, so a caller can filter without importing more
    STANDARD_TOOLS = STANDARD_TOOLS

    question: str
    standard_id: str = ""
    revision: str = ""

    #: The binding itself, so the guard can build the standard's own names
    #: from it rather than from a pattern that spells one out.
    binding: object = None

    evidence: EvidenceLedger = field(default_factory=EvidenceLedger)
    provenance: provenance_guard.ProvenanceLedger = field(
        default_factory=provenance_guard.ProvenanceLedger)
    clauses: conformance_guard.ClauseLedger = field(
        default_factory=conformance_guard.ClauseLedger)

    # The same retrievals, kept with their words, their modality and their
    # type. The clause ledger holds section numbers: it answers "was it
    # retrieved?" and cannot answer "does it agree?".
    claim_evidence: normative_claims.NormativeEvidence = field(
        default_factory=normative_claims.NormativeEvidence)
    code: conformance_mode.CodeReadLedger = field(
        default_factory=conformance_mode.CodeReadLedger)

    #: the question asks whether code conforms; decided once, from its words
    conformance_turn: bool = False
    progress: evidence_progress.ProgressTracker = field(
        default_factory=evidence_progress.ProgressTracker)

    calls: list = field(default_factory=list)
    round_results: list = field(default_factory=list)

    opening_fired: bool = False
    bootstrap_fired: bool = False
    recovery_fired: bool = False

    #: (tool, arguments) this policy has already issued, so a later stage does
    #: not spend a round re-asking a question it has the answer to. The
    #: bootstrap and the recovery route identically for a question carrying no
    #: structure identifier, which after the opening retrieval makes the second
    #: of them a verbatim repeat.
    issued: list = field(default_factory=list)
    suppressed: list = field(default_factory=list)
    exhaustion_fired: bool = False
    stopped_by: str | None = None

    pre_bootstrap_answer: str | None = None
    pre_recovery_answer: str | None = None
    raw_answer: str = ""
    evidence_guarded_answer: str = ""
    guard_fired: bool = False
    conformance_fired: bool = False
    conformance_findings: list = field(default_factory=list)
    replaced_verdicts: list = field(default_factory=list)
    provenance_fired: bool = False
    precedence_fired: bool = False
    claims_fired: bool = False
    repair_attempted: bool = False
    repair_accepted: bool = False
    #: REPAIR_ANSWERED / REPAIR_WITHHELD / REPAIR_FAILED, or "" when no
    #: repair was attempted. A refused repair says which way it failed.
    repair_outcome: str = ""
    #: Injected by whatever drives the loop. Absent means no repair is
    #: possible, and the answer is withheld exactly as before.
    repair_ask: object = None
    claim_findings: list = field(default_factory=list)
    precedence_findings: list = field(default_factory=list)
    replaced_requirements: list = field(default_factory=list)
    guard_violations: list = field(default_factory=list)
    removed_provenance: list = field(default_factory=list)
    rounds: int = 0

    # -- observation ------------------------------------------------------

    def observe_tool_result(self, name, text, *, origin=MODEL):
        """One tool result, exactly as the model was shown it.

        Every ledger reads from here and none of them writes anywhere: the
        payload is parsed, never modified, and a result that is not JSON or is
        a refusal contributes nothing but its provenance.
        """
        self.provenance.observe(text)

        try:
            payload = json.loads(text) if text else None
        except ValueError:
            payload = None

        empty = False

        # Which clauses the model was actually shown. A refusal carries none.

        if isinstance(payload, dict) and not payload.get("error"):
            self.clauses.observe(payload, text)
            self.claim_evidence.observe(payload)

        if isinstance(payload, dict) and not payload.get("error"):
            if name == "standard.get_structure":
                self.evidence.observe(payload)
            elif name == "standard.fetch":
                self.evidence.observe_fetch(payload)

                # The unit handed over is a flattened diagram row; the cells it
                # was printed as are recovered read-only from the store's own
                # layout artifact.

                unit = payload.get("unit")

                if isinstance(unit, dict) and self.standard_id and self.revision:
                    self.evidence.observe_cells(diagram_geometry.cells_for_unit(
                        unit, standard_id=self.standard_id,
                        revision=self.revision))

        if name == evidence_recovery.STRUCTURE_TOOL:
            empty = evidence_recovery.is_empty_structure(payload)

        self.calls.append({"tool": name, "origin": origin,
                           "empty_structure": empty})
        self.round_results.append((name, text, payload))

        return payload

    def observe_code_read(self, name, arguments=None, text=""):
        """Any tool call at all, for the one thing the record needs from the
        non-normative ones: which source files were put in front of the
        model. The evidence ledgers never see these."""
        self.code.observe(name, arguments, text)

    def close_round(self, round_index):
        """End of one round of retrieval: did it establish anything new?"""
        results, self.round_results = self.round_results, []

        if results:
            self.progress.observe_round(round_index, results)

        self.rounds = max(self.rounds, round_index)

    # -- decisions --------------------------------------------------------

    def opening(self):
        """The one retrieval to make BEFORE the model's first round.

        Unconditional on a bound turn, and that is the point: by construction
        the turn engages the standard, so the standard is what it should read
        first. There is no answer yet to judge and no call history to inspect,
        which is exactly why this can be decided without either.

        Fires once. The boundary bootstrap does not fire behind a successful
        one, because it is skipped as soon as any standard tool has been used
        -- the two are the same retrieval at two different moments, not two
        policies. It stays reachable for the case where this call could not be
        executed at all, which is the one situation where a turn can still
        arrive at the boundary having read nothing.
        """
        if self.opening_fired:
            return None

        self.opening_fired = True
        tool, arguments = evidence_bootstrap.route(self.question)

        return self._issue(Injection(tool, arguments, OPENING_RETRIEVAL))

    def _issue(self, injection):
        """Record an injection, or drop it as one already made this turn."""
        key = (injection.tool, canonical_arguments(injection.arguments))

        if key in self.issued:
            self.suppressed.append(injection.origin)

            return None

        self.issued.append(key)

        return injection

    def answer_boundary(self, answer):
        """The model is about to answer. One deterministic call, or nothing.

        The two policies answer different failures and cannot both fire for the
        same one: the bootstrap is for a turn that called nothing, the recovery
        for a turn that asked the structure registry, was told it holds
        nothing, and stopped there.
        """
        if evidence_bootstrap.should_bootstrap(
                self.question, self.calls, already_fired=self.bootstrap_fired,
                answer=answer):
            self.bootstrap_fired = True
            self.pre_bootstrap_answer = answer
            tool, arguments = evidence_bootstrap.route(self.question)

            return self._issue(Injection(tool, arguments, ZERO_TOOL_BOOTSTRAP))

        if evidence_recovery.should_recover(
                self.question, self.calls, already_fired=self.recovery_fired,
                answer=answer):
            self.recovery_fired = True
            self.pre_recovery_answer = answer
            tool, arguments = evidence_recovery.route(self.question)

            return self._issue(Injection(tool, arguments,
                                         EMPTY_STRUCTURE_RECOVERY))

        return None

    def should_stop_for_exhaustion(self, answer=""):
        """Has retrieval stopped establishing anything at all?"""
        if evidence_progress.should_terminate(self.progress, answer=answer,
                                              calls=self.calls):
            self.exhaustion_fired = True
            self.stopped_by = STOPPED_BY_EXHAUSTION

            return True

        return False

    # -- the boundary -----------------------------------------------------

    def finalize(self, answer, *, rounds=None, stopped_by=None):
        """The user-visible answer, and only what the evidence supports of it."""
        self.raw_answer = answer or ""

        if rounds is not None:
            self.rounds = rounds

        if stopped_by is not None and self.stopped_by is None:
            self.stopped_by = stopped_by

        text = self.raw_answer

        # A loop that ends without the model saying anything still has to say
        # what it read. An empty answer records neither.

        if not text.strip() and self.stopped_by in (STOPPED_BY_BUDGET,
                                                    STOPPED_BY_EXHAUSTION):
            text = evidence_progress.bounded_absence(
                self.question, self.evidence, self.progress,
                reason=self.stopped_by, rounds=self.rounds,
                sources=list(self.provenance.source_ids))

        guarded, violations, replaced = evidence_guard_answer(text,
                                                              self.evidence)
        self.evidence_guarded_answer = guarded
        self.guard_fired = replaced
        self.guard_violations = list(violations)

        # A compliance verdict the turn is not entitled to. Between the two
        # guards on purpose: it reads the evidence-guarded text, and what it
        # writes in place of a verdict is built from the clause ledger, so
        # the provenance guard behind it has nothing to find there.

        guarded, findings, replaced, conformance_fired = conformance_guard.guard(
            guarded, self.clauses, standard_id=self.standard_id,
            revision=self.revision, binding=self.binding)
        self.conformance_fired = conformance_fired
        self.conformance_findings = list(findings)
        self.replaced_verdicts = list(replaced)

        # ...and the same question on the other subject: a statement of what
        # the STANDARD requires, backed by a clause this turn retrieved.
        # Beside the conformance guard because they are one idea seen from
        # two sides -- what the code does about a rule, and what the rule
        # says -- and before the provenance guard for its reason: what it
        # writes is built from the ledger, so nothing downstream finds a
        # source in it.

        guarded, requirements, replaced_requirements, precedence_fired = (
            normative_precedence.guard(
                guarded, self.clauses, standard_id=self.standard_id,
                revision=self.revision, binding=self.binding))
        self.precedence_fired = precedence_fired
        self.precedence_findings = list(requirements)
        self.replaced_requirements = list(replaced_requirements)

        # What the answer CLAIMS, against what the evidence SAYS. The guards
        # above establish that a supporting clause was retrieved; none of them
        # asks whether it agrees, and three distinct failures walked through
        # them on that. Before the provenance guard, because what this writes
        # is built from the ledger and must not then be stripped of sources.

        before_repair = guarded
        guarded, claim_problems, claims_fired = normative_claims.guard(
            guarded, self.claim_evidence, question=self.question)

        # One constrained rewrite, when the evidence in hand can settle what
        # was wrong with the prose. No new retrieval, no loop, and the same
        # guard again on what comes back -- a repair that fails is withheld
        # exactly as the original was.

        if (claims_fired and self.repair_ask is not None
                and answer_repair.is_repairable(claim_problems)):
            self.repair_attempted = True
            repaired = answer_repair.attempt(
                self.question, before_repair, self.claim_evidence,
                claim_problems, ask=self.repair_ask)

            # A repair that stops answering has not settled the findings; it
            # has removed what they attached to. Passing the guards is not
            # the test -- a draft that asserts nothing passes every one of
            # them, and accepting that hid a correct conclusion behind a
            # refusal for the sake of an unglossed field name.
            self.repair_outcome = answer_repair.outcome(
                before_repair, repaired, claim_problems)

            if repaired and self.repair_outcome == answer_repair.REPAIR_ANSWERED:
                checked, retry_problems, retry_fired = normative_claims.guard(
                    repaired, self.claim_evidence, question=self.question)

                if not retry_fired:
                    guarded, claim_problems, claims_fired = checked, [], False
                    self.repair_accepted = True
                else:
                    claim_problems = list(retry_problems)

        self.claims_fired = claims_fired
        self.claim_findings = list(claim_problems)

        cleaned, problems, removed, fired = provenance_guard.guard(
            guarded, self.provenance)
        self.provenance_fired = fired
        self.removed_provenance = list(removed)
        self._problems = list(problems)

        # Last, and only on a conformance turn: the record of what was and
        # was not looked at. After the provenance guard because the record
        # is built from ledgers, not from the answer, and must not be
        # sanitised as if the model had written it.

        # Three guards in series, each entitled to remove what it cannot
        # support -- and between them they can remove all of it. A master
        # turn that had read fourteen clauses and five files rendered as a
        # blank body under an assessment record: nothing said the answer had
        # been withheld rather than never written, and the run was scored as
        # if the model had said nothing. What was removed is not recoverable,
        # but the fact of the removal is, and the reader is owed it.

        if not cleaned.strip() and self.raw_answer.strip():
            cleaned = self._withheld_note()

        if self.conformance_turn:
            cleaned = conformance_mode.render(
                cleaned, self.clauses, self.code,
                standard_id=self.standard_id, revision=self.revision)

        return cleaned

    def _withheld_note(self):
        """Why the answer is empty when the model was not."""
        fired = [name for name, did in (
            ("the evidence guard (claims not supported by what was "
             "retrieved)", self.guard_fired),
            ("the conformance guard (a verdict the turn was not entitled "
             "to)", self.conformance_fired),
            ("normative precedence (a requirement stated with no clause "
             "behind it)", self.precedence_fired),
            ("the provenance guard (sources that cannot be traced)",
             self.provenance_fired),
        ) if did]
        why = "; ".join(fired) or "the answer guards"

        return (
            "The model produced an answer this turn and none of it survived "
            f"the guards: {why}. Nothing above is withheld silently -- the "
            "record below states what was read. Ask again more narrowly, or "
            "ask for the clauses and files themselves."
        )

    # -- diagnosis --------------------------------------------------------

    def trace(self):
        """Everything a diagnosis needs and nothing the user sees."""
        problems = getattr(self, "_problems", [])

        return {
            "standard_id": self.standard_id, "revision": self.revision,
            "rounds": self.rounds, "stopped_by": self.stopped_by,
            "model_tool_calls": sum(1 for call in self.calls
                                    if call["origin"] == MODEL),
            "opening_retrieval_calls": sum(
                1 for call in self.calls
                if call["origin"] == OPENING_RETRIEVAL),
            "opening_retrieval_triggered": self.opening_fired,
            "suppressed_injections": list(self.suppressed),
            "zero_tool_bootstrap_calls": sum(
                1 for call in self.calls
                if call["origin"] == ZERO_TOOL_BOOTSTRAP),
            "empty_structure_recovery_calls": sum(
                1 for call in self.calls
                if call["origin"] == EMPTY_STRUCTURE_RECOVERY),
            "bootstrap_triggered": self.bootstrap_fired,
            "empty_structure_recovery_triggered": self.recovery_fired,
            "exhaustion_triggered": self.exhaustion_fired,
            "pre_bootstrap_answer": self.pre_bootstrap_answer,
            "pre_recovery_answer": self.pre_recovery_answer,
            "guard_replaced": self.guard_fired,
            "guard_violations": [{"kind": item.get("kind"),
                                  "label": item.get("label"),
                                  "detail": item.get("detail"),
                                  "reason": item.get("reason")}
                                 for item in self.guard_violations],
            "conformance_guard_triggered": self.conformance_fired,
            "unentitled_verdicts": [{"kind": item["kind"],
                                     "clauses": item["clauses"],
                                     "sentence": item["sentence"]}
                                    for item in self.conformance_findings],
            "clauses_read": sorted(self.clauses.sections),
            "conformance_turn": self.conformance_turn,
            "code_read": sorted(self.code.files),
            "code_tool_calls": dict(self.code.tools),
            "normative_precedence_triggered": self.precedence_fired,
            "ungrounded_requirements": [{"kind": item["kind"],
                                         "clauses": item["clauses"],
                                         "sentence": item["sentence"]}
                                        for item in self.precedence_findings],
            "provenance_guard_triggered": self.provenance_fired,
            "fabricated_source_ids": sorted(
                {item["value"] for item in problems
                 if item["kind"] == provenance_guard.FABRICATED_SOURCE_ID}),
            "fabricated_urls": sorted(
                {item["value"] for item in problems
                 if item["kind"] == provenance_guard.FABRICATED_URL}),
            "fabricated_paths": sorted(
                {item["value"] for item in problems
                 if item["kind"] == provenance_guard.FABRICATED_PATH}),
            "removed_spans": self.removed_provenance,

            # What the answer CLAIMED against what the evidence SAYS, and
            # whether the one constrained rewrite was tried. A battery that
            # cannot see these reports that a turn withheld without being
            # able to say which check withheld it.
            "claims_guard_triggered": self.claims_fired,
            "claim_findings": [{"kind": item.get("kind"),
                                "identifier": item.get("identifier"),
                                "asserted": item.get("asserted"),
                                "reference": item.get("reference"),
                                "sentence": item.get("sentence")}
                               for item in self.claim_findings],
            "repair_attempted": self.repair_attempted,
            "repair_accepted": self.repair_accepted,
            "repair_outcome": self.repair_outcome,
            "diagram_cells": [item["label"]
                              for item in self.evidence.cell_fields.values()],
            "evidence_progress_by_round": self.progress.by_round,
            "consecutive_no_progress": self.progress.consecutive_no_progress,
            "raw_answer": self.raw_answer,
            "evidence_guarded_answer": self.evidence_guarded_answer,
        }


def policy_for(binding, question):
    """A policy for a turn bound to a standard, or None for every other turn.

    Activation is the session's own binding and nothing else -- never the
    user's wording. A turn with no binding is not a standard-bound turn and is
    left exactly as it was.
    """
    if not binding:
        return None

    identity = dict(binding)

    return StandardAnswerPolicy(
        question=question or "",
        standard_id=str(identity.get("standard_id") or ""),
        revision=str(identity.get("revision") or ""),
        binding=identity,
        conformance_turn=conformance_mode.is_conformance_turn(question))
