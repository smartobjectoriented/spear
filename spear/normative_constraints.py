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
  read_verdicts     that check's answer, held to the source: a status that
                    quotes no line the final tree contains is not a status.
  normative_status  the packet's verdict from its constraints' statuses.
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

from normative_force import PERMISSION_FORCE, RECOMMENDATION_FORCE, REQUIREMENT_FORCE
from requirement_set import _family_of, _trigger_of, citations_in

SATISFIED = "SATISFIED"
VIOLATED = "VIOLATED"
NOT_DEMONSTRATED = "NOT_DEMONSTRATED"
NOT_APPLICABLE = "NOT_APPLICABLE"
AMBIGUOUS = "AMBIGUOUS"
STATUSES = (SATISFIED, VIOLATED, NOT_DEMONSTRATED, NOT_APPLICABLE, AMBIGUOUS)

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
                "implication": self.implication}


@dataclass(frozen=True)
class NormativeConstraintSet:
    set_id: str
    standard_id: str
    revision: str
    objective: str
    constraints: tuple[Constraint, ...] = ()

    def get(self, constraint_id: str) -> Constraint | None:
        return next((item for item in self.constraints
                     if item.constraint_id == constraint_id), None)

    def to_dict(self) -> dict:
        return {"set_id": self.set_id, "standard_id": self.standard_id,
                "revision": self.revision, "objective": self.objective,
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


def build(records, answer: str, *, standard_id: str, revision: str,
          objective: str, limit: int = _CARRY_LIMIT) -> NormativeConstraintSet:
    """The packet: the provisions the normative pass cited, or -- when its
    answer cited none -- the binding ones it retrieved, capped."""
    named = citations_in(answer)
    chosen = []

    for key, record in records.items():
        label = str(key).lower()
        bare = label.split()[-1] if label else ""

        if named and not (label in named or bare in named):
            continue

        chosen.append(record)

    if not named:
        chosen = [record for record in records.values()
                  if getattr(record, "effective_force", 0) >= REQUIREMENT_FORCE][:limit]

    constraints = []

    for record in sorted(chosen, key=_provision_order)[:limit]:
        item = constraint_from(record, f"C{len(constraints) + 1}")

        if item is not None:
            constraints.append(item)

    digest = hashlib.sha256("\n".join(
        [standard_id, revision] + sorted(item.instance_id or item.provision
                                         for item in constraints)).encode()).hexdigest()

    return NormativeConstraintSet(f"ncs-{digest[:16]}", standard_id, revision,
                                  objective, tuple(constraints))


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


@dataclass(frozen=True)
class ConstraintStatus:
    constraint_id: str
    status: str
    reason: str = ""
    evidence: tuple[tuple[str, int, str], ...] = ()


def _json_of(text: str):
    text = (text or "").strip()
    start, end = text.find("{"), text.rfind("}")

    try:
        return json.loads(text[start:end + 1]) if start != -1 and end > start else {}
    except ValueError:
        return {}


def _quoted(files: dict[str, str], path: str, line: int, quote: str) -> bool:
    """Whether the final text of `path` holds `quote` at or near `line`."""
    lines = (files.get(path) or "").splitlines()
    wanted = " ".join(str(quote).split())

    if not wanted or not lines:
        return False

    window = lines[max(0, int(line) - 3):int(line) + 2] if isinstance(line, int) else lines
    return any(wanted in " ".join(item.split()) for item in window)


def read_verdicts(packet: NormativeConstraintSet, answer: str,
                  files: dict[str, str]) -> tuple[ConstraintStatus, ...]:
    """The check's statuses, held to the final source and to the packet.

    An unresolved constraint is AMBIGUOUS whatever the check said. A SATISFIED
    or VIOLATED status whose evidence the final files do not contain is not
    demonstrated. A constraint the check left out is not demonstrated.
    """
    given = {str(item.get("id")): item for item in (_json_of(answer).get("constraints") or [])
             if isinstance(item, dict)}
    statuses = []

    for constraint in packet.constraints:
        item = given.get(constraint.constraint_id) or {}
        status = str(item.get("status") or NOT_DEMONSTRATED).upper()
        reason = str(item.get("reason") or "")[:300]
        evidence = []

        for found in item.get("evidence") or []:
            if not isinstance(found, dict):
                continue

            try:
                line = int(found.get("line"))
            except (TypeError, ValueError):
                line = None

            path = str(found.get("file") or "")

            if _quoted(files, path, line, found.get("text")):
                evidence.append((path, line, " ".join(str(found.get("text")).split())[:160]))

        if status not in STATUSES:
            status = NOT_DEMONSTRATED

        if not constraint.resolved:
            status, reason = AMBIGUOUS, constraint.unresolved_reason
        elif status in (SATISFIED, VIOLATED) and not evidence:
            reason = (f"{status.lower()} by the check, but its evidence is not in the "
                      f"final source")
            status = NOT_DEMONSTRATED

        statuses.append(ConstraintStatus(constraint.constraint_id, status, reason,
                                         tuple(evidence)))

    return tuple(statuses)


def normative_status(packet: NormativeConstraintSet, statuses) -> str:
    """SATISFIED only when every applicable requirement is shown to hold.

    A MAY or a SHOULD that the source neither shows nor contradicts does not
    hold the verdict back -- what it forbids is being made mandatory, which is
    a violation and would say so.
    """
    by_id = {item.constraint_id: item for item in statuses}
    states = [by_id[item.constraint_id].status for item in packet.constraints
              if item.constraint_id in by_id]

    if not states:
        return NOT_DEMONSTRATED

    if VIOLATED in states:
        return VIOLATED

    if AMBIGUOUS in states:
        return AMBIGUOUS

    blocking = [by_id[item.constraint_id].status for item in packet.constraints
                if item.modality == "SHALL" and by_id[item.constraint_id].status != NOT_APPLICABLE]

    if any(state != SATISFIED for state in blocking):
        return NOT_DEMONSTRATED

    if not blocking and all(state == NOT_APPLICABLE for state in states):
        return NOT_DEMONSTRATED

    return SATISFIED


def composite(implementation: str, normative: str) -> str:
    """The user-facing status of a MIXED turn: two dimensions, not one boolean."""
    if normative == VIOLATED:
        return "NOT COMPLIANT"

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
    r"in\s+accordance\s+with)\b", re.I)
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
        return sentence.replace(stripped, f"{stripped} *(compliance not established)*")

    return sentence


def source_fingerprint(root: str) -> str:
    """The delivered source as it stands: tracked changes and untracked files."""
    import subprocess

    digest = hashlib.sha256()

    for argv in (["git", "-C", root, "diff", "HEAD"],
                 ["git", "-C", root, "ls-files", "-o", "--exclude-standard"]):
        try:
            done = subprocess.run(argv, capture_output=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            return ""

        digest.update(done.stdout)

        if argv[-2:] == ["-o", "--exclude-standard"]:
            for name in done.stdout.decode(errors="replace").splitlines():
                try:
                    with open(os.path.join(root, name), "rb") as handle:
                        digest.update(handle.read(1 << 20))
                except OSError:
                    pass

    return digest.hexdigest()
