"""What a bound standard asks of a change, carried to the coding core and
checked against what it delivered.

A MIXED turn -- a change that has to satisfy a standard -- is not one loop
doing two jobs. The normative runtime finds the provisions; this module turns
what they say into a small packet the coding core can work from, and later
judges the final source against the same packet:

  build             the constraints, one per provision the normative pass
                    cited, each taken from the provision's own text: its
                    force (from the provision record, never from prose),
                    the condition it is stated under, and the counts,
                    values and identifiers it names, verbatim. Nothing is
                    inferred. A provision the store could not settle is
                    carried as unresolved, never strengthened.
  brief             the packet as the coding core reads it.
  verifier_messages the final source and the packet, for a tool-less check.
  read_candidates   that check's answer, as candidate findings: the source
                    facts it quotes that the final tree holds, and its
                    reading of them, kept apart.
  adjudicate        each constraint's authoritative status: SATISFIED or
                    VIOLATED only where a deterministic predicate decides it
                    from the final source; otherwise NOT DEMONSTRATED.
  normative_status  the packet's verdict from its required constraints.
  composite         that verdict beside the implementation evidence.
  qualify           the answer, with every compliance claim the verdict does
                    not support marked where it stands.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field, replace

from normative.normative_force import PERMISSION_FORCE, RECOMMENDATION_FORCE, REQUIREMENT_FORCE
from normative.requirement_set import _family_of, _trigger_of, citations_in

SATISFIED = "SATISFIED"
VIOLATED = "VIOLATED"
NOT_DEMONSTRATED = "NOT_DEMONSTRATED"
NOT_APPLICABLE = "NOT_APPLICABLE"
AMBIGUOUS = "AMBIGUOUS"
EVIDENCE_CONFLICT = "EVIDENCE_CONFLICT"
STATUSES = (SATISFIED, VIOLATED, NOT_DEMONSTRATED, NOT_APPLICABLE, AMBIGUOUS, EVIDENCE_CONFLICT)

_MODALITY = {REQUIREMENT_FORCE: "SHALL", RECOMMENDATION_FORCE: "SHOULD", PERMISSION_FORCE: "MAY"}
_CARRY_LIMIT = 12
_STATEMENT_CHARS = 400

#: Counts as a provision states them, verbatim: "exactly four", "one and only
#: one", "at least 2", "no more than eight entries".
_CARDINALITY = re.compile(
    r"\b(?:exactly|only|at\s+least|at\s+most|no\s+more\s+than|no\s+fewer\s+than|"
    r"not\s+more\s+than|not\s+less\s+than|up\s+to|one\s+and\s+only|a\s+single|"
    r"one\s+or\s+more|zero\s+or\s+more)\s+(?:one|two|three|four|five|six|seven|"
    r"eight|nine|ten|\d+)\b(?:\s+(?:of\b|entries|entry|elements?|items?|bits?|"
    r"words?|bytes?|packets?|fields?|values?|times?))?", re.I)

#: Values and positions a provision names, verbatim.
_VALUES = re.compile(
    r"\bbits?\s+\d+(?:\s*(?:-|to|through|and)\s*\d+)?\b"
    r"|\b(?:shall|must|should|may)\s+(?:be\s+)?(?:set\s+to|equal(?:\s+to)?|be)\s+"
    r"(?:0x[0-9A-Fa-f]+|\d+|'[^']{1,40}'|\"[^\"]{1,40}\")"
    r"|\b0x[0-9A-Fa-f]+\b", re.I)

#: Identifiers a provision names: CamelCase or upper-case words with a capital
#: inside them, which is how documents spell field and register names.
_IDENTIFIER = re.compile(r"\b[A-Z][a-z0-9]*[A-Z][A-Za-z0-9_]*\b|\b[A-Z]{2,}[0-9_]*[A-Z0-9]\b")
_NOT_IDENTIFIERS = frozenset({"SHALL", "MUST", "SHOULD", "MAY", "NOT", "AND", "OR", "THE"})


@dataclass(frozen=True)
class Constraint:
    """One provision, as a change has to honour it."""
    constraint_id: str
    provision: str                  # the printed key, e.g. "Rule 8.4.1.1-2"
    instance_id: str                # the provision's unique identity in the store
    source_id: str
    page: int | None
    modality: str                   # SHALL | SHOULD | MAY
    requirement: str                # the provision's own words, capped
    condition: str = ""             # the clause it is stated under, verbatim
    cardinality: tuple[str, ...] = ()
    values: tuple[str, ...] = ()
    identifiers: tuple[str, ...] = ()
    resolved: bool = True
    unresolved_reason: str = ""
    origin: str = "CITED"           # CITED | CLOSURE (normative_coverage)
    coverage_reason: str = ""
    source_relation: str = ""
    originating: str = ""
    applicability: str = "UNRESOLVED"
    applicability_basis: str = ""
    applicability_advisory: str = ""

    @property
    def implication(self) -> str:
        """What the modality, condition and count mean for an implementation."""
        if not self.resolved:
            return ("Unresolved in the standard as stored: do not guess an "
                    "implementation for it.")

        what = {
            "SHALL": "Required: the implementation must do what the provision states.",
            "SHOULD": "Recommended: follow it unless there is a stated reason not to; "
                      "it is not a hard requirement.",
            "MAY": "Permitted, not required: the implementation must not make it "
                   "mandatory, and must still work when it is absent or not done.",
        }[self.modality]

        if self.condition:
            what += f" It applies only under its condition ({self.condition}); do not apply it unconditionally."

        if self.cardinality:
            what += f" Keep the count exactly as stated: {'; '.join(self.cardinality)}."

        return what

    def to_dict(self) -> dict:
        return {"id": self.constraint_id, "provision": self.provision,
                "instance_id": self.instance_id, "source_id": self.source_id,
                "page": self.page, "modality": self.modality,
                "requirement": self.requirement, "condition": self.condition,
                "cardinality": list(self.cardinality), "values": list(self.values),
                "identifiers": list(self.identifiers), "resolved": self.resolved,
                "unresolved_reason": self.unresolved_reason,
                "implication": self.implication, "origin": self.origin,
                "coverage_reason": self.coverage_reason,
                "source_relation": self.source_relation, "originating": self.originating,
                "applicability": self.applicability,
                "applicability_basis": self.applicability_basis,
                "applicability_advisory": self.applicability_advisory}


@dataclass(frozen=True)
class NormativeConstraintSet:
    set_id: str
    standard_id: str
    revision: str
    objective: str
    constraints: tuple[Constraint, ...] = ()
    coverage_complete: bool = False
    coverage_reason: str = ""
    retrieved: int = 0

    def get(self, constraint_id: str) -> Constraint | None:
        return next((item for item in self.constraints
                     if item.constraint_id == constraint_id), None)

    def to_dict(self) -> dict:
        return {"set_id": self.set_id, "standard_id": self.standard_id,
                "revision": self.revision, "objective": self.objective,
                "coverage_complete": self.coverage_complete,
                "coverage_reason": self.coverage_reason, "retrieved": self.retrieved,
                "constraints": [item.to_dict() for item in self.constraints]}


def _verbatim(pattern, text, limit=6) -> tuple[str, ...]:
    found = []

    for match in pattern.finditer(text):
        phrase = " ".join(match.group(0).split())

        if phrase.upper() not in _NOT_IDENTIFIERS and phrase not in found:
            found.append(phrase)

    return tuple(found[:limit])


def constraint_from(record, constraint_id: str) -> Constraint | None:
    """The constraint a provision record states, or None for an informative one."""
    try:
        force = record.effective_force
    except Exception:
        force = None

    text = " ".join((getattr(record, "text", "") or "").split())

    if force is None or force not in _MODALITY or not text:
        return None

    status = str(getattr(getattr(record, "declaration_status", ""), "value",
                         getattr(record, "declaration_status", "")) or "").upper()
    unresolved = ""

    if getattr(record, "needs_review", False):
        unresolved = "the store marks this provision as needing review"
    elif status == "CANDIDATE":
        unresolved = "the store could not establish this as a declared provision"

    try:
        instance = str(record.instance_id)
    except Exception:
        instance = ""

    members = _family_of(text)

    return Constraint(
        constraint_id=constraint_id, provision=str(getattr(record, "key", "")),
        instance_id=instance, source_id=str(getattr(record, "source_id", "") or ""),
        page=getattr(record, "page", None), modality=_MODALITY[force],
        requirement=text[:_STATEMENT_CHARS], condition=_trigger_of(text),
        cardinality=_verbatim(_CARDINALITY, text), values=_verbatim(_VALUES, text),
        identifiers=tuple(dict.fromkeys(members + _verbatim(_IDENTIFIER, text, 8)))[:8],
        resolved=not unresolved, unresolved_reason=unresolved)


def _provision_order(record):
    """Document order, as the printed keys give it: 4.2.1-2 before 4.2.1-10."""
    key = str(getattr(record, "key", ""))
    return [int(part) for part in re.findall(r"\d+", key)] or [10**9], key


def cited(records, answer: str) -> list:
    """The records the normative pass's answer cites, in document order.

    Cited by printed label ("Rule 8.4.1.1-2") or by the source the normative
    runtime's citations name ("source std-650fe01d..."). Only those: a pass
    whose answer cites nothing identifiable cites nothing, and never the
    provisions retrieval happened to return.
    """
    named = citations_in(answer)
    sources = {match.lower() for match in re.findall(r"std-[0-9a-f]{12,}", answer or "")}
    chosen = []

    for key, record in records.items():
        label = str(key).lower()
        bare = label.split()[-1] if label else ""
        source = str(getattr(record, "source_id", "") or "").lower()

        if (label in named or bare in named
                or any(source and (source.startswith(item) or item.startswith(source))
                       for item in sources)):
            chosen.append(record)

    return sorted(chosen, key=_provision_order)


def build(records, answer: str, *, standard_id: str, revision: str,
          objective: str, limit: int = _CARRY_LIMIT) -> NormativeConstraintSet:
    """The packet of the cited provisions alone, without structural closure:
    the coverage is what the answer happened to cite, and says so."""
    from normative import normative_coverage as cov

    coverage = cov.Coverage((), tuple(cov.CoverageEntry(record, cov.CITED,
                                                        "cited by the normative pass")
                                      for record in cited(records, answer)), False,
                            "no structural closure was run")

    return from_coverage(coverage, {}, standard_id=standard_id, revision=revision,
                         objective=objective, limit=limit)


def from_coverage(coverage, decisions, *, standard_id: str, revision: str,
                  objective: str, limit: int = _CARRY_LIMIT) -> NormativeConstraintSet:
    """The packet: every normative provision of the coverage set that is not
    established as inapplicable, with how it came to be there and whether it
    applies. `decisions` maps an instance id to its Applicability; one not
    given is UNRESOLVED."""
    from normative import normative_coverage as cov

    complete, reasons = coverage.complete, [coverage.incomplete_reason] if \
        coverage.incomplete_reason else []
    constraints = []
    active = []

    for entry in coverage.entries:
        decision = decisions.get(entry.instance) or cov.Applicability(
            cov.UNRESOLVED, "not assessed")

        if decision.status != cov.NOT_APPLICABLE:
            active.append((entry, decision))

    if len(active) > limit:
        complete = False
        reasons.append(f"{len(active)} provisions apply or may apply; the packet carries "
                       f"{limit}")

    for entry, decision in active[:limit]:
        item = constraint_from(entry.record, f"C{len(constraints) + 1}")

        if item is not None:
            constraints.append(replace(
                item, origin=entry.origin, coverage_reason=entry.coverage_reason,
                source_relation=entry.source_relation, originating=entry.originating,
                applicability=decision.status, applicability_basis=decision.basis,
                applicability_advisory=decision.advisory))

    digest = hashlib.sha256("\n".join(
        [standard_id, revision] + sorted(item.instance_id or item.provision
                                         for item in constraints)).encode()).hexdigest()

    return NormativeConstraintSet(f"ncs-{digest[:16]}", standard_id, revision,
                                  objective, tuple(constraints), complete,
                                  "; ".join(reasons), len(coverage.retrieved))


def brief(packet: NormativeConstraintSet) -> str:
    """The coding core's request: the objective and the constraints, compactly."""
    lines = ["Objective:", packet.objective.strip(), "",
             f"Normative constraints ({packet.standard_id} {packet.revision}, constraint "
             f"set {packet.set_id}). They come from the bound standard; satisfy each one "
             f"exactly as stated -- its force, its condition and its counts:"]

    for item in packet.constraints:
        head = f"{item.constraint_id} [{item.provision}, {item.modality}"
        head += f", {item.condition}]" if item.condition else "]"
        lines.append(f"{head} {item.requirement}")
        lines.append(f"   {item.implication}")

        if item.values:
            lines.append(f"   Values as stated: {'; '.join(item.values)}.")

    lines += ["", "Evidence references: " + "; ".join(
        f"{item.constraint_id} = {item.provision}" for item in packet.constraints)]

    return "\n".join(lines)


# ------------------------------------------------------------------ checking

VERIFIER_SYSTEM = (
    "You check source code against normative constraints from a standard. You "
    "see the constraints and the final source, nothing else. For each "
    "constraint give one status:\n"
    "SATISFIED - the final source does what the constraint requires, as stated.\n"
    "VIOLATED - the final source contradicts it. In particular: a MAY made "
    "mandatory; a conditional requirement applied without its condition; a "
    "count implemented differently from how it is stated (exactly four is not "
    "at least four); a value, bit or identifier other than the stated one.\n"
    "NOT_DEMONSTRATED - the source shown neither shows it nor contradicts it.\n"
    "NOT_APPLICABLE - the change does not touch what the constraint governs.\n"
    "AMBIGUOUS - the constraint itself cannot be read one way.\n"
    "SATISFIED and VIOLATED must cite evidence: the file, the line number and "
    "that line's exact text as shown. Answer with JSON only: "
    "{\"constraints\": [{\"id\": \"C1\", \"status\": \"...\", \"reason\": "
    "\"one sentence\", \"evidence\": [{\"file\": \"...\", \"line\": 1, "
    "\"text\": \"...\"}]}]}")

_SOURCE_CHARS = 60_000


def numbered(text: str) -> str:
    return "\n".join(f"{index:5}| {line}" for index, line in
                     enumerate(text.splitlines(), 1))


def verifier_messages(packet: NormativeConstraintSet, files: dict[str, str]) -> list[dict]:
    """The check's request: the packet and the final text of the given files."""
    shown, budget = [], _SOURCE_CHARS

    for path, text in files.items():
        body = numbered(text)

        if len(body) > budget:
            body = body[:budget] + "\n... [truncated]"

        shown.append(f"=== {path}\n{body}")
        budget -= len(body)

        if budget <= 0:
            break

    constraints = json.dumps([dict({key: value for key, value in item.to_dict().items()
                                    if key in ("id", "provision", "modality", "requirement",
                                               "condition", "cardinality", "values",
                                               "identifiers", "implication")},
                                   checks=questions(item))
                              for item in packet.constraints], ensure_ascii=False, indent=1)

    return [{"role": "system", "content": VERIFIER_SYSTEM},
            {"role": "user", "content": f"Constraints:\n{constraints}\n\nFinal source:\n"
                                        + ("\n\n".join(shown) or "(no files)")}]


def questions(item: Constraint) -> list[str]:
    """What the check must answer for this constraint, from its own fields.

    A model reading "shall be 2 when X is set" alone judged code that sets 2
    always as satisfying it, and a validator accepting `>= 4` beside a list
    of four as keeping "exactly four". The questions name each part of what
    the provision states, so none of it is read past.
    """
    asked = []

    if item.modality == "MAY":
        asked.append("Is this left optional everywhere -- is there no code path "
                     "that requires it, rejects its absence or always forces it? "
                     "If something makes it mandatory: VIOLATED.")

    if item.condition:
        asked.append(f"Does the code apply this only {item.condition}? If it also "
                     f"applies it when that condition does not hold, it has dropped "
                     f"the condition: VIOLATED.")

    for phrase in item.cardinality:
        asked.append(f"Does every place that builds, checks or accepts this keep "
                     f"'{phrase}' exactly? A different bound anywhere (more, fewer, "
                     f"'at least', 'at most', '>=', '<='): VIOLATED.")

    for phrase in item.values:
        asked.append(f"Is the value or position exactly as stated ('{phrase}')? "
                     f"Any other value: VIOLATED.")

    return asked


CONFIRM_SYSTEM = (
    "A first check claimed that source code violates a normative constraint "
    "from a standard. Decide whether that violation is real. The constraint "
    "is about the identifiers, values, counts and condition it names: a bit, "
    "field or case it does not name cannot violate it. What it names, it "
    "names exactly -- and these ARE violations: a MAY made mandatory (the "
    "code requires it, rejects its absence or always forces it); a "
    "conditional requirement applied when its condition does not hold; a "
    "count different from the one stated anywhere it is built, checked or "
    "accepted (exactly four is not at least four); a value, bit or identifier "
    "other than the stated one. Answer the constraint's checks, tracing the "
    "code paths that matter. CONFIRMED if the code contradicts the "
    "constraint as stated; otherwise REFUTED. Answer with JSON only: "
    "{\"verdict\": \"CONFIRMED\" or \"REFUTED\", \"reason\": \"one or two "
    "sentences\"}.")


def confirmation_messages(constraint: Constraint, claim: str,
                          files: dict[str, str]) -> list[dict]:
    """A second reading of one possible violation: advisory only. The same
    model asked twice is not independent, and on real protocol code it
    confirmed false findings as readily as true ones."""
    shown, budget = [], _SOURCE_CHARS

    for path, text in files.items():
        body = numbered(text)[:budget]
        shown.append(f"=== {path}\n{body}")
        budget -= len(body)

        if budget <= 0:
            break

    stated = dict({key: value for key, value in constraint.to_dict().items()
                   if key in ("provision", "modality", "requirement", "condition",
                              "cardinality", "values", "identifiers")},
                  checks=questions(constraint))

    return [{"role": "system", "content": CONFIRM_SYSTEM},
            {"role": "user", "content": f"Constraint:\n{json.dumps(stated, ensure_ascii=False, indent=1)}"
                                        f"\n\nClaimed violation:\n{claim}\n\nSource:\n"
                                        + "\n\n".join(shown)}]


#: What the check may say about a constraint. A candidate is a lead, not a
#: status: it is kept for the audit trail and decides nothing.
POSSIBLE_SATISFACTION = "POSSIBLE_SATISFACTION"
POSSIBLE_VIOLATION = "POSSIBLE_VIOLATION"
POSSIBLE_NOT_APPLICABLE = "POSSIBLE_NOT_APPLICABLE"
POSSIBLE_AMBIGUITY = "POSSIBLE_AMBIGUITY"
NO_FINDING = "NO_FINDING"
_CANDIDATE = {SATISFIED: POSSIBLE_SATISFACTION, VIOLATED: POSSIBLE_VIOLATION,
              NOT_APPLICABLE: POSSIBLE_NOT_APPLICABLE, AMBIGUOUS: POSSIBLE_AMBIGUITY}


@dataclass(frozen=True)
class CandidateFinding:
    """What the check said about one constraint: source facts it quoted that
    the final tree holds, and its reading of them, kept apart."""
    constraint_id: str
    candidate: str = NO_FINDING
    interpretation: str = ""
    facts: tuple = ()               # normative_predicates.SourceFact, grounded only
    ungrounded: int = 0             # quotes the final source does not hold
    advisory: str = ""              # a second reading's word, for the record only

    def to_dict(self) -> dict:
        return {"candidate": self.candidate, "interpretation": self.interpretation,
                "facts": [item.to_dict() for item in self.facts],
                "ungrounded": self.ungrounded, "advisory": self.advisory}


@dataclass(frozen=True)
class ConstraintStatus:
    """A constraint's authoritative status, and what it rests on."""
    constraint_id: str
    status: str
    reason: str = ""
    evidence: tuple[tuple[str, int, str], ...] = ()
    authority: str = "none"         # predicate:<TYPE> | check:<id>, joined by + | packet | none
    candidate: CandidateFinding | None = None
    predicate: object = None        # the predicate's normative_evidence.Evidence


def _json_of(text: str):
    text = (text or "").strip()
    start, end = text.find("{"), text.rfind("}")

    try:
        return json.loads(text[start:end + 1]) if start != -1 and end > start else {}
    except ValueError:
        return {}


def _quoted(files: dict[str, str], path: str, line: int, quote: str) -> bool:
    """Whether the final text of `path` holds `quote` at or near `line`.

    Whitespace is not the claim: a quote may span lines and be re-indented.
    Its words, in order, must be there, within a few lines of where it says.
    """
    lines = (files.get(path) or "").splitlines()
    wanted = " ".join(str(quote).split())

    if not wanted or not lines:
        return False

    if isinstance(line, int):
        span = str(quote).count("\n") + 1
        lines = lines[max(0, line - 6):line + span + 5]

    return wanted in " ".join(" ".join(lines).split())


def read_candidates(packet: NormativeConstraintSet, answer: str,
                    files: dict[str, str]) -> dict[str, CandidateFinding]:
    """The check's answer as candidate findings, its quotes held to the final
    source: a quote the final tree does not hold is not a fact."""
    from normative.normative_predicates import fact

    given = {str(item.get("id")): item for item in (_json_of(answer).get("constraints") or [])
             if isinstance(item, dict)}
    found = {}

    for constraint in packet.constraints:
        item = given.get(constraint.constraint_id)

        if item is None:
            found[constraint.constraint_id] = CandidateFinding(constraint.constraint_id)
            continue

        facts, ungrounded = [], 0

        for quote in item.get("evidence") or []:
            if not isinstance(quote, dict):
                continue

            try:
                line = int(quote.get("line"))
            except (TypeError, ValueError):
                line = None

            path = str(quote.get("file") or "")

            if _quoted(files, path, line, quote.get("text")):
                facts.append(fact(path, line, str(quote.get("text"))))
            else:
                ungrounded += 1

        status = str(item.get("status") or "").upper()
        found[constraint.constraint_id] = CandidateFinding(
            constraint.constraint_id, _CANDIDATE.get(status, NO_FINDING),
            str(item.get("reason") or "")[:300], tuple(facts), ungrounded)

    return found


def adjudicate(packet: NormativeConstraintSet, candidates: dict[str, CandidateFinding],
               files: dict[str, str], *, evidence=None, epoch: str = "") -> tuple[ConstraintStatus, ...]:
    """Each constraint's authoritative status.

    SATISFIED and VIOLATED come only from decisive evidence of an
    authoritative provider (normative_evidence) on the final source epoch:
    a deterministic predicate, or a project check bound to the provision.
    `evidence` is what the providers returned; when it is None the source
    predicates alone are consulted. Decisive evidence that disagrees is an
    EVIDENCE_CONFLICT, never whichever ran last. The check's findings, and
    any second reading of them, are candidates and decide nothing; an
    unresolved provision is AMBIGUOUS.
    """
    from normative.normative_evidence import SourcePredicateProvider

    if evidence is None:
        evidence = SourcePredicateProvider().evidence(packet, files, epoch)

    by_constraint = {}

    for item in evidence:
        by_constraint.setdefault(item.constraint_id, []).append(item)

    statuses = []

    for constraint in packet.constraints:
        candidate = candidates.get(constraint.constraint_id) or CandidateFinding(
            constraint.constraint_id)

        if not constraint.resolved:
            statuses.append(ConstraintStatus(constraint.constraint_id, AMBIGUOUS,
                                             constraint.unresolved_reason, (), "packet",
                                             candidate))
            continue

        found = by_constraint.get(constraint.constraint_id, [])
        fresh = [item for item in found if item.decisive and item.source_epoch == epoch]
        stale = [item for item in found if item.decisive and item.source_epoch != epoch]
        said = {item.status for item in fresh}
        facts = tuple(fact for item in fresh for fact in item.facts)
        authority = "+".join(sorted({item.authority for item in fresh}))
        reason = "; ".join(item.reason for item in fresh)
        predicate = next((item for item in fresh if item.provider == "predicate"), None)

        if said == {SATISFIED} or said == {VIOLATED}:
            statuses.append(ConstraintStatus(constraint.constraint_id, said.pop(), reason,
                                             facts, authority, candidate, predicate))
            continue

        if len(said) > 1:
            statuses.append(ConstraintStatus(
                constraint.constraint_id, EVIDENCE_CONFLICT,
                "authoritative evidence disagrees: " + reason, facts, authority, candidate,
                predicate))
            continue

        notes = [item.reason for item in found if not item.decisive]
        notes += [f"{item.reason} -- on an earlier source state, stale" for item in stale]

        if candidate.candidate != NO_FINDING:
            what = candidate.candidate.removeprefix("POSSIBLE_").replace("_", " ").lower()
            notes.append(f"not independently established (the model check found a possible "
                         f"{what}" + (f": {candidate.interpretation[:160]}"
                                      if candidate.interpretation else "") + ")")

        statuses.append(ConstraintStatus(
            constraint.constraint_id, NOT_DEMONSTRATED,
            "; ".join(notes) or "nothing in the final source establishes it either way",
            tuple((item.path, item.line, item.excerpt) for item in candidate.facts),
            "none", candidate))

    return tuple(statuses)


def _decisive(status) -> bool:
    return any(part.split(":", 1)[0] in ("predicate", "check")
               for part in status.authority.split("+"))


def repairable(packet: NormativeConstraintSet, statuses) -> list[ConstraintStatus]:
    """The violations a repair may act on: an applicable requirement, violated
    on decisive evidence. A model's finding never enters the coding core as a
    fact, and neither does a requirement whose applicability is unresolved."""
    return [item for item in statuses if item.status == VIOLATED and _decisive(item)
            and getattr(packet.get(item.constraint_id), "modality", "") == "SHALL"
            and getattr(packet.get(item.constraint_id), "applicability", "") == "APPLICABLE"]


def normative_status(packet: NormativeConstraintSet, statuses) -> str:
    """SATISFIED only when the coverage is complete and every applicable
    requirement is established to hold.

    A requirement whose applicability is unresolved can be neither met nor
    failed: it holds the verdict at NOT_DEMONSTRATED. MAY and SHOULD
    constraints are not obligations and neither hold the verdict back nor
    make it.
    """
    by_id = {item.constraint_id: item for item in statuses}
    required = [(item, getattr(by_id.get(item.constraint_id), "status", NOT_DEMONSTRATED))
                for item in packet.constraints
                if item.modality == "SHALL" and item.applicability != "NOT_APPLICABLE"]

    if any(item.applicability == "APPLICABLE" and state == VIOLATED for item, state in required):
        return VIOLATED

    if any(state == EVIDENCE_CONFLICT for _, state in required):
        return EVIDENCE_CONFLICT

    if any(item.applicability == "APPLICABLE" and state == AMBIGUOUS for item, state in required):
        return AMBIGUOUS

    if packet.coverage_complete and required and all(
            item.applicability == "APPLICABLE" and state == SATISFIED for item, state in required):
        return SATISFIED

    return NOT_DEMONSTRATED


def coverage_report(packet: NormativeConstraintSet, statuses, excluded: int = 0) -> dict:
    """The counts behind the verdict."""
    by_id = {item.constraint_id: item.status for item in statuses}
    required = [item for item in packet.constraints if item.modality == "SHALL"]
    applicable = [item for item in required if item.applicability == "APPLICABLE"]

    def count(state, items=applicable):
        return sum(1 for item in items if by_id.get(item.constraint_id, NOT_DEMONSTRATED) == state)

    return {"coverage": "COMPLETE" if packet.coverage_complete else "INCOMPLETE",
            "coverage_reason": packet.coverage_reason,
            "cited": sum(1 for item in packet.constraints if item.origin == "CITED"),
            "closure_added": sum(1 for item in packet.constraints if item.origin == "CLOSURE"),
            "not_applicable_excluded": excluded,
            "required": len(required), "required_applicable": len(applicable),
            "satisfied": count(SATISFIED), "violated": count(VIOLATED),
            "not_demonstrated": count(NOT_DEMONSTRATED), "ambiguous": count(AMBIGUOUS),
            "conflicts": count(EVIDENCE_CONFLICT, required),
            "unresolved_applicability": sum(1 for item in required
                                            if item.applicability == "UNRESOLVED")}


def composite(implementation: str, normative: str) -> str:
    """The user-facing status of a MIXED turn: two dimensions, not one boolean."""
    if normative == VIOLATED:
        return "NOT COMPLIANT"

    if normative == EVIDENCE_CONFLICT:
        return "EVIDENCE CONFLICT"

    if normative == AMBIGUOUS:
        return "NORMATIVE AMBIGUITY"

    if normative == NOT_DEMONSTRATED:
        return "COMPLIANCE NOT DEMONSTRATED"

    if implementation == "VERIFIED":
        return "VERIFIED + COMPLIANT"

    if implementation == "NO_CHANGE":
        return "COMPLIANT, NO CHANGE MADE"

    return "COMPLIANT BUT IMPLEMENTATION UNVERIFIED"


#: A sentence that says the change complies with something.
_COMPLIANCE = re.compile(
    r"\b(?:complian\w*|compli(?:es|ed)|comply|conform\w*|meets?\s+(?:the\s+|all\s+)?"
    r"(?:standard|spec\w*|requirements?|provisions?|rules?)|satisf(?:ies|ied|y)\s+(?:the\s+|all\s+)?"
    r"(?:standard|spec\w*|requirements?|provisions?|rules?|constraints?)|"
    r"in\s+accordance\s+with)\b|✓|✔", re.I)
_HEDGED = re.compile(r"\b(?:not|n't|unverified|should|may|might|could|would|if|unless|once)\b", re.I)
_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9`*(\[])|\n")


def qualify(answer: str, normative: str) -> str:
    """Mark every unhedged compliance claim the normative verdict does not support."""
    if normative == SATISFIED:
        return answer

    pieces, last = [], 0

    for match in _SPLIT.finditer(answer or ""):
        pieces.append(_mark(answer[last:match.start()]))
        pieces.append(match.group(0))
        last = match.end()

    pieces.append(_mark((answer or "")[last:]))

    return "".join(pieces)


def _mark(sentence: str) -> str:
    stripped = sentence.strip()

    if stripped and _COMPLIANCE.search(stripped) and not _HEDGED.search(stripped) \
            and not stripped.startswith(("#", "|", "```")):
        body = stripped.replace("✓", "").replace("✔", "").replace("  ", " ")
        return sentence.replace(stripped, f"{body} *(compliance not established)*")

    return sentence


def source_fingerprint(root: str) -> str:
    """The delivered source as it stands: tracked changes and untracked files.

    What a build or a check writes into its own output areas is not source
    (completion.generated_path), or every check that imports a module would
    make its own result stale.
    """
    import subprocess

    from evidence.completion import generated_path

    digest = hashlib.sha256()

    for argv in (["git", "-C", root, "diff", "HEAD"],
                 ["git", "-C", root, "ls-files", "-o", "--exclude-standard"]):
        try:
            done = subprocess.run(argv, capture_output=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            return ""

        if argv[-2:] != ["-o", "--exclude-standard"]:
            digest.update(done.stdout)
        else:
            names = [name for name in done.stdout.decode(errors="replace").splitlines()
                     if not generated_path(name)]
            digest.update("\n".join(names).encode())

            for name in names:
                try:
                    with open(os.path.join(root, name), "rb") as handle:
                        digest.update(handle.read(1 << 20))
                except OSError:
                    pass

    return digest.hexdigest()
