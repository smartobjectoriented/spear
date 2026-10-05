"""Authoritative evidence about a constraint, from providers that are not the model.

  SourcePredicateProvider  a deterministic predicate over the packet's own
                           fields and the final source (normative_predicates).
  ProjectCheckProvider     a project check explicitly bound to a provision.

A binding is the project's own statement that THIS check is evidence for THIS
provision, made where the project states how it is built and tested
(projects.json, `normative_checks`), against the provision's stable identity:
its ProvisionInstanceId, or a printed key that names exactly one instance in
the bound revision. A passing ordinary test proves nothing normative; a model
suggesting a binding makes none.

  {"normative_checks": [{
      "id": "ack-subtype",
      "standard": "ANSI-VITA-49.2", "revision": "2017-R2024",
      "provisions": ["Rule 8.4.1.1-2@9f4e826caf39"],
      "command": "ctest --test-dir build -R ack_subtype",
      "evidence": {"kind": "test", "success": "exit_zero",
                   "semantics": "PASS_AND_FAIL_DECISIVE"},
      "enabled": true}],
   "normative_applicability": [{
      "provision": "Rule 8.4.1.1-3@371c5d2e09fd",
      "applicability": "NOT_APPLICABLE", "reason": "..."}]}

A check runs through the caller's confined runner, on the final source, and
counts only for the source epoch it ran on. A pass establishes SATISFIED for
the provisions bound to it and nothing else. A failure establishes VIOLATED
only under PASS_AND_FAIL_DECISIVE; otherwise it establishes nothing. A check
that could not run, or broke, establishes nothing either.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

PASS_ESTABLISHES_SATISFIED = "PASS_ESTABLISHES_SATISFIED"
PASS_AND_FAIL_DECISIVE = "PASS_AND_FAIL_DECISIVE"
SEMANTICS = (PASS_ESTABLISHES_SATISFIED, PASS_AND_FAIL_DECISIVE)

PASSED, FAILED, NOT_RUN, ERROR = "PASSED", "FAILED", "NOT_RUN", "ERROR"


@dataclass(frozen=True)
class ConstraintCheckBinding:
    binding_id: str
    standard_id: str
    revision: str
    provisions: tuple[str, ...]
    check_kind: str                     # test | check
    command: str
    semantics: str = PASS_ESTABLISHES_SATISFIED
    success: str = "exit_zero"
    requires_final_epoch: bool = True
    scope: str = "workspace"
    authority: str = ""
    enabled: bool = True


@dataclass(frozen=True)
class ApplicabilityDeclaration:
    provision: str
    status: str                         # APPLICABLE | NOT_APPLICABLE
    reason: str
    authority: str


@dataclass(frozen=True)
class ConstraintCheckEvidence:
    """One run of one bound check, as it happened."""
    binding_id: str
    constraint_ids: tuple[str, ...]
    source_epoch: str
    command: str
    status: str                         # PASSED | FAILED | NOT_RUN | ERROR
    exit_code: int | None
    output_sha256: str
    output_tail: str = ""


@dataclass(frozen=True)
class Evidence:
    """What one provider establishes about one constraint."""
    constraint_id: str
    provider: str                       # predicate | check
    kind: str                           # the predicate type, or the binding id
    status: str                         # SATISFIED | VIOLATED | NOT_DEMONSTRATED
    decisive: bool
    source_epoch: str
    reason: str
    facts: tuple = ()                   # (path, line, excerpt)

    @property
    def authority(self) -> str:
        return f"{self.provider}:{self.kind}"


def load(spec, *, standard_id: str, revision: str, authority: str):
    """(bindings, declarations, problems) a project spec states for this
    standard. An entry that is malformed or for another standard is not a
    binding; each is reported, none is guessed at."""
    bindings, declarations, problems = [], [], []

    for index, entry in enumerate((spec or {}).get("normative_checks") or ()):
        where = f"normative_checks[{index}]"

        if not isinstance(entry, dict):
            problems.append(f"{where}: not an object")
            continue

        evidence = entry.get("evidence") or {}
        provisions = tuple(str(item) for item in entry.get("provisions") or () if str(item).strip())
        semantics = str(evidence.get("semantics") or PASS_ESTABLISHES_SATISFIED).upper()
        command = str(entry.get("command") or "").strip()
        binding_id = str(entry.get("id") or "").strip()

        if not binding_id or not command or not provisions:
            problems.append(f"{where}: needs an id, a command and provisions")
            continue

        if semantics not in SEMANTICS or str(evidence.get("success") or "exit_zero") != "exit_zero":
            problems.append(f"{where} ({binding_id}): unsupported evidence semantics")
            continue

        if str(entry.get("standard") or "") != standard_id or \
                str(entry.get("revision") or revision) != revision:
            problems.append(f"{where} ({binding_id}): bound to another standard or revision")
            continue

        bindings.append(ConstraintCheckBinding(
            binding_id, standard_id, revision, provisions,
            str(evidence.get("kind") or "test"), command, semantics,
            requires_final_epoch=True, scope=str(entry.get("scope") or "workspace"),
            authority=authority, enabled=bool(entry.get("enabled", True))))

    for index, entry in enumerate((spec or {}).get("normative_applicability") or ()):
        where = f"normative_applicability[{index}]"
        status = str((entry or {}).get("applicability") or "").upper() if isinstance(entry, dict) else ""

        if status not in ("APPLICABLE", "NOT_APPLICABLE") or not entry.get("provision") \
                or not str(entry.get("reason") or "").strip():
            problems.append(f"{where}: needs a provision, an applicability and a reason")
            continue

        if str(entry.get("standard") or standard_id) != standard_id or \
                str(entry.get("revision") or revision) != revision:
            problems.append(f"{where}: bound to another standard or revision")
            continue

        declarations.append(ApplicabilityDeclaration(str(entry["provision"]), status,
                                                     str(entry["reason"]), authority))

    return bindings, declarations, problems


class SourcePredicateProvider:
    """A deterministic predicate's decision of a constraint, from the final source."""

    name = "predicate"

    def evidence(self, packet, files, epoch):
        import normative_predicates

        found = []

        for constraint in packet.constraints:
            if not constraint.resolved:
                continue

            result = normative_predicates.evaluate(constraint, files)

            if result is None:
                continue

            where = result.facts[0]
            found.append(Evidence(
                constraint.constraint_id, self.name, result.predicate,
                "SATISFIED" if result.holds else "VIOLATED", True, epoch,
                f"{result.predicate}: the provision states {result.expected}, the final "
                f"source gives {result.observed} ({where.path}:{where.line} "
                f"`{where.excerpt[:80]}`)",
                tuple((item.path, item.line, item.excerpt) for item in result.facts)))

        return found


class ProjectCheckProvider:
    """The project checks bound to the packet's provisions, run once each.

    `bound(binding)` names the packet constraints a binding covers; `runner`
    is the caller's confined execution of one command, (status, exit code,
    output); `fingerprint` reads the source epoch; `notice(event, metadata)`
    audits. A check that changes the source it was run on proves nothing
    about the final source.
    """

    name = "check"

    def __init__(self, bindings, bound, runner, fingerprint, notice=lambda *a: None):
        self.bindings, self.bound = bindings, bound
        self.runner, self.fingerprint, self.notice = runner, fingerprint, notice
        self.runs: list[ConstraintCheckEvidence] = []

    def evidence(self, packet, files, epoch):
        found = []

        for binding in self.bindings:
            constraints = self.bound(binding)

            if not binding.enabled or not constraints:
                continue

            ids = tuple(item.constraint_id for item in constraints)
            self.notice("normative_check_started", {
                "binding_id": binding.binding_id, "constraints": list(ids),
                "command": binding.command, "authority": binding.authority})
            before = self.fingerprint()

            if self.runner is None:
                status, code, output = NOT_RUN, None, "no confined runner for project checks"
            else:
                try:
                    status, code, output = self.runner(binding.command)
                except Exception as exc:            # noqa: BLE001
                    status, code, output = ERROR, None, f"{type(exc).__name__}: {exc}"

            after = self.fingerprint()
            ran_on = before if before == after else f"changed-by-check:{after}"
            run = ConstraintCheckEvidence(
                binding.binding_id, ids, ran_on, binding.command, status, code,
                hashlib.sha256(str(output).encode()).hexdigest(), str(output)[-600:])
            self.runs.append(run)
            self.notice("normative_check_finished", {
                "binding_id": binding.binding_id, "constraints": list(ids),
                "status": status, "exit_code": code, "output_sha256": run.output_sha256,
                "source_epoch_matches": ran_on == epoch})

            for constraint in constraints:
                found.append(self._evidence(binding, constraint, run))

        return found

    def _evidence(self, binding, constraint, run):
        head = f"project check {binding.binding_id} (`{binding.command[:80]}`)"

        if run.status == PASSED:
            return Evidence(constraint.constraint_id, self.name, binding.binding_id,
                            "SATISFIED", True, run.source_epoch, f"{head} passed")

        if run.status == FAILED and binding.semantics == PASS_AND_FAIL_DECISIVE:
            return Evidence(constraint.constraint_id, self.name, binding.binding_id,
                            "VIOLATED", True, run.source_epoch,
                            f"{head} failed (exit {run.exit_code}); its binding makes a "
                            f"failure decisive")

        why = {FAILED: f"failed (exit {run.exit_code}), and its binding does not make a "
                       f"failure decisive",
               ERROR: "could not complete (infrastructure error)",
               NOT_RUN: "did not run"}.get(run.status, run.status.lower())

        return Evidence(constraint.constraint_id, self.name, binding.binding_id,
                        "NOT_DEMONSTRATED", False, run.source_epoch, f"{head} {why}")
