"""A change that has to satisfy a bound standard: three passes, not one loop.

  1. The normative runtime, read-only, finds the provisions that govern the
     change. Its cited provisions become the constraint packet
     (normative_constraints.build); nothing from its prose enters a
     constraint.
  2. The frozen coding core makes the change from a brief: the objective and
     the packet. It sees the accepted coding tools only -- no standard tools,
     no corpus -- and runs inside the same SpearHost envelope as any
     implementation turn.
  3. A tool-less check reads the FINAL source against the same packet. What
     it says is a candidate finding; a constraint is SATISFIED or VIOLATED only
     where a deterministic predicate decides it from the final source
     (normative_constraints.adjudicate). An established violation of a
     requirement earns one repair brief to the same core, and one re-check;
     a model's finding alone earns none.

The verdict keeps two dimensions apart: the implementation evidence (the
Phase-8.5 final-state verdict over every core call) and the normative status
of the packet. Neither is allowed to stand in for the other, and the answer is
qualified wherever it claims compliance the check did not establish.

Generic over standards: nothing here knows one document from another.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field, replace

from normative import normative_constraints as nc
from context.context_selection import MIXED_IMPLEMENTATION, MIXED_POSTCHECK, MIXED_PREPASS
from normative.normative_evidence import ProjectCheckProvider, SourcePredicateProvider
from runtime.agent_roles import AgentRole
from models.model_backend import ConversationMessage, TextBlock
from runtime.tracing import EventStatus, EventType

PREPASS = (
    "Do not change anything yet, and do not look at any code: answer from the "
    "bound standard alone. Before this change is made, identify the "
    "provisions of the standard that govern it, and state exactly what each "
    "one requires: its force (shall, should or may), any condition it is "
    "stated under, any count, and any identifiers or values it names. Cite "
    "every provision by its printed label.\n\nChange requested: {objective}")

REPAIR = (
    "The final source contradicts these constraints, as read from the source "
    "itself. Change only what is needed to satisfy them, as stated; leave "
    "everything else as it is.\n\n{items}\n\nConstraint set {set_id} "
    "({standard} {revision}) is otherwise unchanged.")

VERIFIER_MAX_TOKENS = 8192
MAX_FILE_BYTES = 200_000


@dataclass
class MixedRecord:
    """What the orchestration did, for the audit trail and the report."""
    packet: nc.NormativeConstraintSet | None = None
    prepass: str = ""                               # ANSWERED, WITHHELD or FAILED
    statuses: tuple = ()
    checks: list = field(default_factory=list)     # every post-check, in order
    implementation: str = "NO_CHANGE"
    normative: str = nc.NOT_DEMONSTRATED
    verdict: str = ""
    repaired: bool = False
    tokens: dict = field(default_factory=dict)
    excluded: int = 0                               # established as not applicable
    report: dict = field(default_factory=dict)
    check_runs: list = field(default_factory=list)  # normative_evidence.ConstraintCheckEvidence


# How the normative pass ended, which an empty packet does not say by itself.
# ANSWERED with nothing cited is a valid empty packet; WITHHELD is an answer
# the evidence guards would not let through; FAILED is a pass that produced
# no answer at all. All three give the core no constraint, and none of them
# may be reported as the others.
PREPASS_ANSWERED, PREPASS_WITHHELD, PREPASS_FAILED = "ANSWERED", "WITHHELD", "FAILED"

_GUARDS = ("guard_fired", "claims_fired", "precedence_fired", "conformance_fired")


def prepass_outcome(normative, prepass, packet):
    """(outcome, reason) of the normative pass; the reason is "" when the
    packet's own coverage reason already says it."""
    from runtime.agent_context import RuntimeTerminalReason
    from runtime.agent_finalization import FAILED

    window = getattr(normative, "finalization", None)
    broken = (RuntimeTerminalReason.MODEL_FAILURE, RuntimeTerminalReason.RUNTIME_FAILURE,
              RuntimeTerminalReason.INVALID_TURN, RuntimeTerminalReason.INTERRUPTED)

    if window is not None and window.state == FAILED:
        return PREPASS_FAILED, f"the normative pass did not complete: {window.reason}"

    if not (prepass.final_response or "").strip() or prepass.terminal_reason in broken:
        return PREPASS_FAILED, ("the normative pass did not complete: it ended "
                                f"{getattr(prepass.terminal_reason, 'value', prepass.terminal_reason)}")

    policy = getattr(normative, "standard_policy", None)

    if not packet.constraints and any(getattr(policy, flag, False) for flag in _GUARDS):
        return PREPASS_WITHHELD, ("the normative pass's answer was withheld: the "
                                  "retrieved evidence does not support it")

    return PREPASS_ANSWERED, ""


class _Metered:
    """The backend, counting the prompt tokens each phase spends."""

    def __init__(self, backend, tokens: dict):
        self._backend, self._tokens, self.phase = backend, tokens, "prepass"

    def __getattr__(self, name):
        return getattr(self._backend, name)

    def _count(self, turn):
        usage = getattr(turn, "usage", None) or {}
        prompt = int(usage.get("prompt_tokens", 0) or 0)
        bucket = self._tokens.setdefault(self.phase, {"requests": 0, "first": 0,
                                                      "max": 0, "total": 0})
        bucket["requests"] += 1
        bucket["first"] = bucket["first"] or prompt
        bucket["max"] = max(bucket["max"], prompt)
        bucket["total"] += prompt
        return turn

    def complete(self, *args, **kwargs):
        return self._count(self._backend.complete(*args, **kwargs))

    def complete_messages(self, *args, **kwargs):
        return self._count(self._backend.complete_messages(*args, **kwargs))


def applies(controller, request, context, view) -> bool:
    """Whether this turn is MIXED: bound, about both sides, and asking for a change."""
    from context import answer_scope
    from runtime.agent_notes import wants_write

    if context.standard_binding is None or not request.coding_core or view.read_only:
        return False

    scope = answer_scope.of(request.objective, standard_bound=True,
                            prior=answer_scope.prior_scope(context.conversation,
                                                           standard_bound=True))

    return scope == answer_scope.MIXED and wants_write(context, request.objective)


class MixedOrchestrator:
    def __init__(self, controller):
        self.controller = controller

    # ------------------------------------------------------------- the passes

    def run(self, request, context):
        from standard.standard_schema import StandardBinding
        from runtime.task_controller import TaskStatus

        binding = StandardBinding.from_dict(context.standard_binding)
        root = os.path.realpath(request.project_scope[0])
        turn = list(context.conversation)
        record = MixedRecord()
        metered = _Metered(context.backend, record.tokens)
        context.backend = metered
        context.mixed_record = record

        # 1. The normative pass: the established runtime, read-only.
        self._event(context, EventType.NORMATIVE_PREPASS_STARTED,
                    {"standard_id": binding.standard_id, "revision": binding.revision})
        question = PREPASS.format(objective=request.objective)
        normative = self._normative_context(context, request, question, turn)
        prepass = self.controller.runtime.run(normative, defer_completion=True)
        record.tokens["prepass_status"] = prepass.terminal_status
        policy = getattr(normative, "standard_policy", None)
        ledger = getattr(getattr(policy, "claim_evidence", None), "provisions", None)
        records = dict(getattr(ledger, "records", None) or {})
        records.update(self._cited_sources(binding, prepass.final_response or "", records))
        record.packet = self._packet(context, request, binding, records,
                                     prepass.final_response or "")
        record.prepass, why = prepass_outcome(normative, prepass, record.packet)

        if why and not record.packet.constraints:
            record.packet = replace(record.packet, coverage_reason=why)

        if record.prepass == PREPASS_FAILED:
            self._event(context, EventType.NORMATIVE_PREPASS_FAILED, {
                "reason": why, "terminal_reason": str(getattr(
                    prepass.terminal_reason, "value", prepass.terminal_reason)),
                "finalization_retries": getattr(
                    getattr(normative, "finalization", None), "retries", 0)})

        brief = nc.brief(record.packet)
        record.tokens["packet_chars"] = len(brief)
        self._event(context, EventType.NORMATIVE_CONSTRAINT_SET_CREATED, {
            "set_id": record.packet.set_id, "constraints": len(record.packet.constraints),
            "prepass": record.prepass,
            "provisions": [item.provision for item in record.packet.constraints],
            "instance_ids": [item.instance_id for item in record.packet.constraints],
            "unresolved": [item.constraint_id for item in record.packet.constraints
                           if not item.resolved],
            "coverage_complete": record.packet.coverage_complete,
            "coverage_reason": record.packet.coverage_reason})

        # 2. The frozen coding core, from the brief alone.
        metered.phase = "implementation"
        context.core_evidence = None
        result = self._implement(context, request, turn, brief, history=())
        before = nc.source_fingerprint(root)

        # 3. The check of the final source.
        metered.phase = "postcheck"
        record.statuses, checked = self._check(context, record.packet, root)
        record.checks.append({"statuses": record.statuses, "fingerprint": checked})

        # A failing bound check that its binding makes decisive, or a predicate,
        # is a contradiction the source itself shows; nothing else is.
        violated = nc.repairable(record.packet, record.statuses)

        # One repair, for an established violation of a requirement only: a
        # model's finding, an ambiguity or a MAY is reported, never acted on.
        if violated:
            record.repaired = True
            self._event(context, EventType.REPAIR_STARTED, {
                "set_id": record.packet.set_id,
                "violated": [item.constraint_id for item in violated]})
            metered.phase = "repair"
            items = "\n".join(
                f"{item.constraint_id} [{record.packet.get(item.constraint_id).provision}, "
                f"{record.packet.get(item.constraint_id).modality}] "
                f"{record.packet.get(item.constraint_id).requirement}\n   Contradiction: "
                f"{item.reason}" for item in violated)
            repair = REPAIR.format(items=items, set_id=record.packet.set_id,
                                   standard=binding.standard_id, revision=binding.revision)
            result = self._implement(context, request, turn, repair,
                                     history=(("user", brief),
                                              ("assistant", result.final_response or "")))
            metered.phase = "postcheck-after-repair"
            record.statuses, checked = self._check(context, record.packet, root)
            record.checks.append({"statuses": record.statuses, "fingerprint": checked})

        # The verdict, on the final source only: a check of an earlier tree
        # is stale and is not carried forward.
        verdict = getattr(context, "core_verdict", None)
        record.implementation = getattr(verdict, "state", "NO_CHANGE")
        record.normative = nc.normative_status(record.packet, record.statuses)

        if checked != nc.source_fingerprint(root):
            record.normative = nc.NOT_DEMONSTRATED

        record.verdict = nc.composite(record.implementation, record.normative)
        record.report = nc.coverage_report(record.packet, record.statuses, record.excluded)
        self._event(context, EventType.FINAL_MIXED_VERDICT, {
            "set_id": record.packet.set_id, "verdict": record.verdict,
            "implementation": record.implementation, "normative": record.normative,
            "repaired": record.repaired, "source_changed_by_check": before != checked,
            "report": record.report})

        result.final_response = self.render(record, binding,
                                            result.final_response or "")
        context.backend = metered._backend
        result = self.controller.runtime.complete(context, result)

        return self.controller._result(
            request, TaskStatus.COMPLETED if not result.completion_deferred
            else TaskStatus.BLOCKED, result, warnings=(f"mixed:{record.verdict}",))

    def _packet(self, context, request, binding, records, answer):
        """The constraint packet: the cited provisions, closed over the
        structure the store records, each with its applicability decided from
        deterministic facts -- the request's words, the project's declarations
        and the checks it binds."""
        from normative import normative_coverage as cov
        from normative import normative_evidence as ne

        universe = self._universe(binding, records)
        retrieved = tuple(cov.instance_of(item) for item in records.values())
        cited = nc.cited(records, answer)
        known = {cov.instance_of(item) for item in cited}
        cited += [item for item in cov.quoted(answer, universe)
                  if cov.instance_of(item) not in known]
        coverage = cov.close(cited, universe, retrieved=retrieved)
        self._event(context, EventType.CONSTRAINT_COVERAGE_EXPANDED, {
            "retrieved": len(retrieved),
            "cited": [item.instance for item in coverage.cited],
            "added": [{"instance": item.instance, "reason": item.coverage_reason,
                       "relation": item.source_relation, "originating": item.originating,
                       "parent": item.parent} for item in coverage.added],
            "ungrouped": list(coverage.ungrouped), "complete": coverage.complete,
            "incomplete_reason": coverage.incomplete_reason})

        spec = getattr(context, "normative_project", None) or {}
        bindings, declarations, problems = ne.load(
            spec, standard_id=binding.standard_id, revision=binding.revision,
            authority=getattr(context, "normative_authority", "") or "projects.json")
        declared, bound = {}, {}

        for item in declarations:
            found = universe.resolve(item.provision)
            problems += ([] if found else [f"normative_applicability: {item.provision} names "
                                           f"no single provision of the bound revision"])

            for target in found:
                declared.setdefault(cov.instance_of(target), []).append(
                    (item.status, item.reason, item.authority))

        self.bindings, self.bound_instances = bindings, {}

        for item in bindings:
            instances = set()

            for identity in item.provisions:
                found = universe.resolve(identity)
                problems += ([] if found else [f"normative_checks {item.binding_id}: {identity} "
                                               f"names no single provision of the bound revision"])
                instances |= {cov.instance_of(target) for target in found}

            self.bound_instances[item.binding_id] = instances

            for instance in instances:
                bound.setdefault(instance, set()).add(item.binding_id)

            self._event(context, EventType.NORMATIVE_CHECK_BINDING_LOADED, {
                "binding_id": item.binding_id, "provisions": list(item.provisions),
                "instances": sorted(instances), "command": item.command,
                "semantics": item.semantics, "authority": item.authority,
                "enabled": item.enabled})

        if problems:
            self._event(context, EventType.NORMATIVE_CHECK_BINDING_LOADED,
                        {"problems": problems[:20]})

        decisions = {}

        for entry in coverage.entries:
            decision = cov.decide(entry, request.objective, declared, bound)
            decisions[entry.instance] = decision
            self._event(context, EventType.CONSTRAINT_APPLICABILITY_DECIDED, {
                "instance": entry.instance, "provision": str(entry.record.key),
                "origin": entry.origin, "applicability": decision.status,
                "basis": decision.basis, "advisory": decision.advisory})

        context.mixed_record.excluded = sum(1 for item in decisions.values()
                                            if item.status == cov.NOT_APPLICABLE)

        return nc.from_coverage(coverage, decisions, standard_id=binding.standard_id,
                                revision=binding.revision, objective=request.objective)

    def _universe(self, binding, records):
        """Every provision record of the bound revision, read once per store.
        Without a readable store, the records the pass itself observed."""
        from normative import normative_coverage as cov
        from normative import provision_identity

        store = getattr(self.controller, "standard_store", None)
        cache = getattr(self.controller, "_provision_universe", None) or {}
        key = (binding.standard_id, binding.revision)

        if key not in cache:
            try:
                units = [unit.to_dict() if hasattr(unit, "to_dict") else unit
                         for unit in store.load_units(binding.standard_id, binding.revision)]
                cache[key] = cov.Universe(provision_identity.records_for_units(units), units)
            except Exception:                       # noqa: BLE001
                return cov.Universe(list(records.values()), [])

            try:
                self.controller._provision_universe = cache
            except AttributeError:
                pass

        return cache[key]

    def _cited_sources(self, binding, answer, records):
        """Provision records for the sources the answer cites that the pass's
        ledger never recorded, read from the bound store itself.

        The answer's citations name their sources exactly; the ledger holds
        only the units it observed in certain shapes. A real pass cited seven
        sources the ledger had no record of, and the packet came out empty.
        """
        import re

        from normative import provision_identity

        store = getattr(self.controller, "standard_store", None)
        known = {str(getattr(item, "source_id", "") or "") for item in records.values()}
        found = {}

        if store is None:
            return found

        for source_id in dict.fromkeys(re.findall(r"std-[0-9a-f]{32}", answer)):
            if source_id in known:
                continue

            try:
                unit = store.resolve_source(binding.standard_id, binding.revision, source_id)
            except Exception:                       # noqa: BLE001
                continue

            payload = unit.to_dict() if hasattr(unit, "to_dict") else unit

            for item in provision_identity.records_for_units([payload]):
                found[str(item.key)] = item

        return found

    def _normative_context(self, context, request, question, turn):
        """The normative pass's own context: the same session, its own working
        state, and the standard's tools as the only tools that run.

        Its own working state, because the pass ends as a turn ends -- done or
        failed -- and the implementation that follows must start from a live
        one. Its own executor, because the legacy loop runs a call the model
        writes as text whether or not the view offered it, and this pass reads
        the standard and nothing else.
        """
        import copy

        from runtime.working_state import WorkingState

        normative = copy.copy(context)
        normative.working_state = WorkingState.start(
            context.task_id, question, max_model_rounds=context.max_model_rounds,
            max_tool_actions=context.max_tool_actions)
        normative.conversation = turn[:-1] + [ConversationMessage("user", (TextBlock(question),))]
        normative.work_phase = None
        normative.capability_gateway = normative.knowledge_door = None
        self._view(normative, request, question, coding=False)
        prepass = (getattr(context, "phase_contexts", None) or {}).get(MIXED_PREPASS)

        if prepass and prepass.get("context_items"):
            # The pre-pass reads the standard: it is given the normative
            # selection, with no coding rule, memory, procedure or code corpus.

            normative.context_items = prepass["context_items"]
            normative.system_prompt = "".join(item.content for item in prepass["context_items"])

        self._toolset(normative, MIXED_PREPASS, normative.tools, coding=False)
        offered = {item.name for item in normative.tools}
        execute = context.tool_executor

        def standard_only(agent, call_id, name, arguments, cache):
            if name not in offered:
                return _refusal(call_id, name)

            return execute(agent, call_id, name, arguments, cache)

        normative.tool_executor = standard_only

        return normative

    def _view(self, context, request, objective, *, coding):
        view = self.controller.tool_exposure_policy.select(
            self.controller.registry, AgentRole.MAIN, objective=objective,
            web_enabled=False, memory_write_enabled=False,
            standard_bound=not coding, toolset="coding" if coding else None,
            retrieval_available=False)
        definitions = view.definitions

        if not coding:
            # The normative pass reads the standard and nothing else.

            definitions = tuple(item for item in definitions
                                if item.name.startswith("standard."))

        context.tools = definitions
        context.execution_core = "coding" if coding else "legacy"
        context.read_only = not coding
        context.advisory = not coding

        return view

    def _implement(self, context, request, turn, text, *, history):
        self._event(context, EventType.IMPLEMENTATION_STARTED,
                    {"brief_chars": len(text), "repair": bool(history)})
        self._view(context, request, request.objective, coding=True)
        implementation = (getattr(context, "phase_contexts", None) or {}).get(
            MIXED_IMPLEMENTATION)

        if implementation is not None:
            context.coding_context = implementation["coding_context"]

        # Only the implementation may reach the external capabilities its
        # workspace admits; the passes on either side of it never do.
        context.capability_gateway = (implementation or {}).get("gateway")
        context.knowledge_door = (implementation or {}).get("knowledge_door")
        self._toolset(context, MIXED_IMPLEMENTATION, context.tools, coding=True)
        self._event(context, EventType.CONTEXT_SELECTED, {
            "phase": MIXED_IMPLEMENTATION, "id": "normative:constraint-set",
            "type": "NORMATIVE_CONSTRAINT_SET",
            "source": getattr(getattr(context, "mixed_record", None), "packet", None)
            and context.mixed_record.packet.set_id,
            "reason": "the compact constraint packet, as the brief carries it",
            "chars": len(text), "workspace": getattr(getattr(context, "workspace_context",
                                                             None), "workspace_id", "")})

        # The coding core never runs under the legacy write gate: SpearHost's
        # envelope is its boundary, as for any implementation turn.
        context.work_phase = None
        context.conversation = turn[:-1] + [
            ConversationMessage(role, (TextBlock(content),)) for role, content in history
        ] + [ConversationMessage("user", (TextBlock(text),))]
        result = self.controller.runtime.run(context, defer_completion=True)
        verdict = getattr(context, "core_verdict", None)
        self._event(context, EventType.IMPLEMENTATION_FINISHED, {
            "terminal_reason": result.terminal_reason.value,
            "implementation": getattr(verdict, "state", None),
            "changed": list(getattr(verdict, "changed", ()) or ())[:20]})

        return result

    def _toolset(self, context, phase, definitions, *, coding):
        """Name the family this pass is offered, hold it to the tool contract,
        and record it -- for the audit trail, never for the model."""
        from harness import tool_selection

        gateway = getattr(context, "capability_gateway", None)
        selection = tool_selection.DeterministicToolSelector().select(
            phase, [item.name for item in definitions], coding=coding,
            external=getattr(gateway, "providers", ()) if gateway and gateway.items else ())
        tool_selection.check_contract(definitions, self.controller.registry, coding=coding)
        self._event(context, EventType.TOOLSET_SELECTED, selection.to_dict())

    # --------------------------------------------------------------- checking

    def _check(self, context, packet, root):
        """Every constraint's status against the final files, and the source
        fingerprint they were read at."""
        context.capability_gateway = context.knowledge_door = None
        self._toolset(context, MIXED_POSTCHECK, (), coding=False)
        self._event(context, EventType.NORMATIVE_POSTCHECK_STARTED,
                    {"set_id": packet.set_id, "constraints": len(packet.constraints)})
        fingerprint = nc.source_fingerprint(root)
        files = final_files(root)

        if not packet.constraints:
            return (), fingerprint

        answer = ""

        try:
            turn = context.backend.complete_messages(
                nc.verifier_messages(packet, files), [], max_tokens=VERIFIER_MAX_TOKENS)
            answer = getattr(turn, "text", "") or ""
        except Exception as exc:                    # noqa: BLE001
            answer = json.dumps({"constraints": [], "error": str(exc)[:200]})

        candidates = nc.read_candidates(packet, answer, files)
        candidates = {key: self._advised(context, packet, item, files)
                      for key, item in candidates.items()}
        evidence = SourcePredicateProvider().evidence(packet, files, fingerprint)
        checks = ProjectCheckProvider(
            getattr(self, "bindings", ()),
            lambda item: [constraint for constraint in packet.constraints
                          if constraint.instance_id in self.bound_instances.get(
                              item.binding_id, ())],
            getattr(context, "normative_check_runner", None),
            lambda: nc.source_fingerprint(root),
            lambda name, metadata: self._event(context, EventType(name), metadata))
        evidence += checks.evidence(packet, files, fingerprint)
        context.mixed_record.check_runs.extend(checks.runs)
        statuses = nc.adjudicate(packet, candidates, files, evidence=evidence,
                                 epoch=fingerprint)
        by_constraint = {}

        for item in evidence:
            by_constraint.setdefault(item.constraint_id, []).append(item)

        for item in statuses:
            candidate = item.candidate or nc.CandidateFinding(item.constraint_id)
            constraint = packet.get(item.constraint_id)

            if item.status == nc.EVIDENCE_CONFLICT:
                self._event(context, EventType.NORMATIVE_EVIDENCE_CONFLICT, {
                    "set_id": packet.set_id, "constraint": item.constraint_id,
                    "authority": item.authority, "reason": item.reason[:400]})

            self._event(context, EventType.NORMATIVE_CONSTRAINT_STATUS, {
                "set_id": packet.set_id, "constraint": item.constraint_id,
                "provision": constraint.provision, "instance_id": constraint.instance_id,
                "applicability": constraint.applicability,
                "status": item.status, "authority": item.authority,
                "candidate": candidate.candidate, "advisory": candidate.advisory,
                "interpretation": candidate.interpretation[:300],
                "candidate_facts": [fact.to_dict() for fact in candidate.facts][:5],
                "ungrounded_quotes": candidate.ungrounded,
                "evidence": [{"provider": found.provider, "kind": found.kind,
                              "status": found.status, "decisive": found.decisive,
                              "fresh": found.source_epoch == fingerprint,
                              "reason": found.reason[:200]}
                             for found in by_constraint.get(item.constraint_id, [])]})

        return statuses, fingerprint

    def _advised(self, context, packet, candidate, files):
        """A second reading of a possible violation, kept as advice.

        It decides nothing: on real protocol code the same model confirmed a
        false finding -- an error bit counted as an acknowledgement subtype --
        as readily as a true one.
        """
        from dataclasses import replace

        if candidate.candidate != nc.POSSIBLE_VIOLATION or not candidate.facts:
            return candidate

        claim = candidate.interpretation + "".join(
            f"\n{fact.path}:{fact.line}: {fact.excerpt}" for fact in candidate.facts)

        try:
            turn = context.backend.complete_messages(
                nc.confirmation_messages(packet.get(candidate.constraint_id), claim, files),
                [], max_tokens=VERIFIER_MAX_TOKENS)
            verdict = str(nc._json_of(getattr(turn, "text", "")).get("verdict") or "").upper()
        except Exception:                           # noqa: BLE001
            verdict = ""

        return replace(candidate, advisory=verdict if verdict in ("CONFIRMED", "REFUTED")
                       else "NO_ANSWER")

    # ---------------------------------------------------------------- answer

    @staticmethod
    def render(record, binding, answer) -> str:
        packet, report = record.packet, record.report or nc.coverage_report(
            record.packet, record.statuses, record.excluded)
        lines = [f"**MIXED VERDICT: {record.verdict}** — {binding.standard_id} "
                 f"{binding.revision}, constraint set {packet.set_id} "
                 f"({len(packet.constraints)} constraint(s); compliance is judged "
                 f"against these only).",
                 f"Coverage: {report['coverage']} — {report['cited']} cited, "
                 f"{report['closure_added']} added by the document's structure, "
                 f"{report['not_applicable_excluded']} excluded as not applicable"
                 + (f" ({packet.coverage_reason})" if packet.coverage_reason else "") + ".",
                 f"Required: {report['required']} — applicable {report['required_applicable']} "
                 f"(satisfied {report['satisfied']}, violated {report['violated']}, not "
                 f"demonstrated {report['not_demonstrated']}), applicability unresolved "
                 f"{report['unresolved_applicability']}, evidence conflicts "
                 f"{report['conflicts']}."]
        by_id = {item.constraint_id: item for item in record.statuses}

        for item in packet.constraints:
            status = by_id.get(item.constraint_id)
            where = ", ".join(f"{path}:{line}" for path, line, _ in
                              (status.evidence if status else ())[:2])
            origin = (f"; added: {item.source_relation}" if item.origin == "CLOSURE" else "")
            lines.append(
                f"- {item.constraint_id} {item.provision} ({item.modality}"
                + (f", {item.condition}" if item.condition else "")
                + f"; applicability {item.applicability}{origin}): "
                + (status.status if status else nc.NOT_DEMONSTRATED)
                + (f" — {status.reason}" if status and status.reason else "")
                + (f" [{where}]" if where else ""))

        lines.append(f"Implementation evidence: {record.implementation}. Normative: "
                     f"{record.normative}." + (" One repair cycle ran." if record.repaired else ""))

        return "\n".join(lines) + "\n\n" + nc.qualify(answer, record.normative)

    # ----------------------------------------------------------------- audit

    @staticmethod
    def _event(context, event_type, metadata):
        context.trace.emit(event_type, context.task_id, session_id=context.session_id,
                           status=EventStatus.OK, metadata=metadata)
        context.observer.notice(event_type.value, metadata)


def _refusal(call_id: str, name: str):
    """A call the normative pass does not run, as the model reads it."""
    from harness.tool_router import ToolResultEnvelope, ToolResultStatus
    from runtime.tracing import new_action_id

    text = (f"ERROR: '{name}' is not available here: this pass answers from the "
            f"bound standard alone, with the standard's own tools.")

    return ToolResultEnvelope(call_id, new_action_id(name), name, False,
                              ToolResultStatus.DENIED, text, text, "other", 0.0,
                              len(text), error_category="not_offered", error_summary=text)


def final_files(root: str) -> dict[str, str]:
    """The final text of every file the change touched: tracked changes and
    new files, as the working tree holds them now."""
    names = []

    for argv in (["git", "-C", root, "diff", "--name-only", "HEAD"],
                 ["git", "-C", root, "ls-files", "-o", "--exclude-standard"]):
        try:
            done = subprocess.run(argv, capture_output=True, text=True, timeout=60)
            names += [line for line in done.stdout.splitlines() if line.strip()]
        except (OSError, subprocess.SubprocessError):
            pass

    from evidence.completion import generated_path

    files = {}

    for name in dict.fromkeys(name for name in names if not generated_path(name)):
        path = os.path.join(root, name)

        try:
            if os.path.isfile(path) and os.path.getsize(path) <= MAX_FILE_BYTES:
                with open(path, encoding="utf-8") as handle:
                    files[name] = handle.read()
        except (OSError, UnicodeDecodeError):
            continue

    return files
