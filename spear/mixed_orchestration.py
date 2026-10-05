"""A change that has to satisfy a bound standard: three passes, not one loop.

  1. The normative runtime, read-only, finds the provisions that govern the
     change. Its cited provisions become the constraint packet
     (normative_constraints.build); nothing from its prose enters a
     constraint.
  2. The frozen coding core makes the change from a brief: the objective and
     the packet. It sees the accepted coding tools only -- no standard tools,
     no corpus -- and runs inside the same SpearHost envelope as any
     implementation turn.
  3. A tool-less check reads the FINAL source against the same packet. Its
     statuses must quote lines the final tree contains. A violation earns one
     repair brief to the same core, and one re-check; an ambiguity earns none.

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
from dataclasses import dataclass, field

import normative_constraints as nc
from agent_roles import AgentRole
from model_backend import ConversationMessage, TextBlock
from tracing import EventStatus, EventType

PREPASS = (
    "Do not change anything yet, and do not look at any code: answer from the "
    "bound standard alone. Before this change is made, identify the "
    "provisions of the standard that govern it, and state exactly what each "
    "one requires: its force (shall, should or may), any condition it is "
    "stated under, any count, and any identifiers or values it names. Cite "
    "every provision by its printed label.\n\nChange requested: {objective}")

REPAIR = (
    "The normative check of the final source found these constraints "
    "violated. Change only what is needed to satisfy them, as stated; leave "
    "everything else as it is.\n\n{items}\n\nConstraint set {set_id} "
    "({standard} {revision}) is otherwise unchanged.")

VERIFIER_MAX_TOKENS = 8192
MAX_FILE_BYTES = 200_000


@dataclass
class MixedRecord:
    """What the orchestration did, for the audit trail and the report."""
    packet: nc.NormativeConstraintSet | None = None
    statuses: tuple = ()
    checks: list = field(default_factory=list)     # every post-check, in order
    implementation: str = "NO_CHANGE"
    normative: str = nc.NOT_DEMONSTRATED
    verdict: str = ""
    repaired: bool = False
    tokens: dict = field(default_factory=dict)


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
    import answer_scope
    from agent_runtime import wants_write

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
        from standard_schema import StandardBinding
        from task_controller import TaskStatus

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
        record.packet = nc.build(records, prepass.final_response or "",
                                 standard_id=binding.standard_id, revision=binding.revision,
                                 objective=request.objective)
        brief = nc.brief(record.packet)
        record.tokens["packet_chars"] = len(brief)
        self._event(context, EventType.NORMATIVE_CONSTRAINT_SET_CREATED, {
            "set_id": record.packet.set_id, "constraints": len(record.packet.constraints),
            "provisions": [item.provision for item in record.packet.constraints],
            "instance_ids": [item.instance_id for item in record.packet.constraints],
            "unresolved": [item.constraint_id for item in record.packet.constraints
                           if not item.resolved]})

        # 2. The frozen coding core, from the brief alone.
        metered.phase = "implementation"
        context.core_evidence = None
        result = self._implement(context, request, turn, brief, history=())
        before = nc.source_fingerprint(root)

        # 3. The check of the final source.
        metered.phase = "postcheck"
        record.statuses, checked = self._check(context, record.packet, root)
        record.checks.append({"statuses": record.statuses, "fingerprint": checked})

        violated = [item for item in record.statuses if item.status == nc.VIOLATED]
        ambiguous = any(item.status == nc.AMBIGUOUS for item in record.statuses)

        # One repair, for a concrete violation only: an ambiguity is reported,
        # never guessed at.
        if violated and not ambiguous:
            record.repaired = True
            self._event(context, EventType.REPAIR_STARTED, {
                "set_id": record.packet.set_id,
                "violated": [item.constraint_id for item in violated]})
            metered.phase = "repair"
            items = "\n".join(
                f"{item.constraint_id} [{record.packet.get(item.constraint_id).provision}, "
                f"{record.packet.get(item.constraint_id).modality}] "
                f"{record.packet.get(item.constraint_id).requirement}\n   Found: {item.reason}"
                for item in violated)
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
        self._event(context, EventType.FINAL_MIXED_VERDICT, {
            "set_id": record.packet.set_id, "verdict": record.verdict,
            "implementation": record.implementation, "normative": record.normative,
            "repaired": record.repaired, "source_changed_by_check": before != checked})

        result.final_response = self.render(record, binding,
                                            result.final_response or "")
        context.backend = metered._backend
        result = self.controller.runtime.complete(context, result)

        return self.controller._result(
            request, TaskStatus.COMPLETED if not result.completion_deferred
            else TaskStatus.BLOCKED, result, warnings=(f"mixed:{record.verdict}",))

    def _cited_sources(self, binding, answer, records):
        """Provision records for the sources the answer cites that the pass's
        ledger never recorded, read from the bound store itself.

        The answer's citations name their sources exactly; the ledger holds
        only the units it observed in certain shapes. A real pass cited seven
        sources the ledger had no record of, and the packet came out empty.
        """
        import re

        import provision_identity

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

        from working_state import WorkingState

        normative = copy.copy(context)
        normative.working_state = WorkingState.start(
            context.task_id, question, max_model_rounds=context.max_model_rounds,
            max_tool_actions=context.max_tool_actions)
        normative.conversation = turn[:-1] + [ConversationMessage("user", (TextBlock(question),))]
        normative.work_phase = None
        self._view(normative, request, question, coding=False)
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

    # --------------------------------------------------------------- checking

    def _check(self, context, packet, root):
        """Every constraint's status against the final files, and the source
        fingerprint they were read at."""
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

        statuses = tuple(self._confirmed(context, packet, item, files)
                         for item in nc.read_verdicts(packet, answer, files))

        for item in statuses:
            self._event(context, EventType.NORMATIVE_CONSTRAINT_STATUS, {
                "set_id": packet.set_id, "constraint": item.constraint_id,
                "provision": packet.get(item.constraint_id).provision,
                "status": item.status,
                "evidence": [f"{path}:{line}" for path, line, _ in item.evidence][:5]})

        return statuses, fingerprint

    def _confirmed(self, context, packet, status, files):
        """A violation stands only when an independent reading confirms it.

        On real protocol code the check's violations were wrong as often as
        right -- an error bit counted as an acknowledgement subtype, five
        samples out of five. A violation drives a repair; one that a second
        reading does not confirm is not demonstrated, and drives nothing.
        """
        if status.status != nc.VIOLATED:
            return status

        claim = status.reason + "".join(f"\n{path}:{line}: {text}"
                                        for path, line, text in status.evidence)
        verdict = ""

        try:
            turn = context.backend.complete_messages(
                nc.confirmation_messages(packet.get(status.constraint_id), claim, files),
                [], max_tokens=VERIFIER_MAX_TOKENS)
            verdict = str(nc._json_of(getattr(turn, "text", "")).get("verdict") or "").upper()
        except Exception:                           # noqa: BLE001
            verdict = ""

        if verdict == "CONFIRMED":
            return status

        return nc.ConstraintStatus(status.constraint_id, nc.NOT_DEMONSTRATED,
                                   f"a violation was claimed ({status.reason[:160]}) but an "
                                   f"independent reading did not confirm it", status.evidence)

    # ---------------------------------------------------------------- answer

    @staticmethod
    def render(record, binding, answer) -> str:
        packet = record.packet
        lines = [f"**MIXED VERDICT: {record.verdict}** — {binding.standard_id} "
                 f"{binding.revision}, constraint set {packet.set_id} "
                 f"({len(packet.constraints)} constraint(s); compliance is judged "
                 f"against these only)."]
        by_id = {item.constraint_id: item for item in record.statuses}

        for item in packet.constraints:
            status = by_id.get(item.constraint_id)
            where = ", ".join(f"{path}:{line}" for path, line, _ in
                              (status.evidence if status else ())[:2])
            lines.append(
                f"- {item.constraint_id} {item.provision} ({item.modality}"
                + (f", {item.condition}" if item.condition else "") + f"): "
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
    from tool_router import ToolResultEnvelope, ToolResultStatus
    from tracing import new_action_id

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

    files = {}

    for name in dict.fromkeys(names):
        path = os.path.join(root, name)

        try:
            if os.path.isfile(path) and os.path.getsize(path) <= MAX_FILE_BYTES:
                with open(path, encoding="utf-8") as handle:
                    files[name] = handle.read()
        except (OSError, UnicodeDecodeError):
            continue

    return files
