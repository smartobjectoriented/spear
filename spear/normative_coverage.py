"""Which provisions a bound change answers to: cited, closed over structure,
scoped. Three sets, and every move between them recorded.

  retrieved     what the normative pass looked at.
  cited         what its answer cites (normative_constraints.cited). The
                model's choice, and the start of coverage, not the end of it:
                a pass once cited five of six numbered rules of one list and
                the sixth was never implemented or checked.
  closed        the cited provisions, plus the provisions the document itself
                groups them with: the other declarations of the same numbered
                list (same section, same kind, same immediate heading), the
                other normative rows of the same table, and provisions a cited
                one names outright. Only relations the store represents --
                never "everything under this section number" -- and bounded:
                a group past the limit leaves coverage INCOMPLETE rather than
                absorbing a chapter.
  applicability each closed provision is APPLICABLE, NOT_APPLICABLE or
                UNRESOLVED, and only a deterministic basis settles it: the
                request names it, or the project declares it (projects.json,
                or a check bound to it). That the model cited it is advice,
                kept, and settles nothing; a requirement whose applicability
                is unresolved cannot be met by default.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import provision_identity as pi

CITED, CLOSURE = "CITED", "CLOSURE"
SAME_NORMATIVE_LIST = "SAME_NORMATIVE_LIST"
SAME_TABLE = "SAME_TABLE"
EXPLICIT_REFERENCE = "EXPLICIT_REFERENCE"

APPLICABLE, NOT_APPLICABLE, UNRESOLVED = "APPLICABLE", "NOT_APPLICABLE", "UNRESOLVED"

#: A structural group larger than this is not closed: coverage is INCOMPLETE.
GROUP_LIMIT = 16
#: Nor is a closure that would add more than this in all.
CLOSURE_LIMIT = 24

_NORMATIVE = frozenset({1, 2, 3})


@dataclass(frozen=True)
class CoverageEntry:
    """One provision in the coverage set, and how it got there."""
    record: object
    origin: str                     # CITED | CLOSURE
    coverage_reason: str = ""
    source_relation: str = ""
    originating: str = ""           # the instance it was reached from
    parent: str = ""                # what the relation runs through

    @property
    def instance(self) -> str:
        return instance_of(self.record)


@dataclass(frozen=True)
class Coverage:
    retrieved: tuple[str, ...]
    entries: tuple[CoverageEntry, ...]
    complete: bool
    incomplete_reason: str = ""
    ungrouped: tuple[str, ...] = ()  # cited provisions the store groups with nothing

    @property
    def cited(self):
        return tuple(item for item in self.entries if item.origin == CITED)

    @property
    def added(self):
        return tuple(item for item in self.entries if item.origin == CLOSURE)


def instance_of(record) -> str:
    try:
        return str(record.instance_id)
    except Exception:                               # noqa: BLE001
        return str(getattr(record, "key", ""))


def normative(record) -> bool:
    try:
        return record.effective_force in _NORMATIVE
    except Exception:                               # noqa: BLE001
        return False


class Universe:
    """Every provision record of the bound revision, indexed by structure."""

    def __init__(self, records, units):
        self.units = {str(unit.get("source_id")): unit for unit in units}
        self.lists, self.tables, self.by_key, self.sections = {}, {}, {}, {}

        for record in records:
            key = record.key
            self.by_key.setdefault(str(key), []).append(record)

            if normative(record) and record.declaration_status == pi.DECLARATION:
                self.sections.setdefault(str(getattr(key, "section", "")), []).append(record)

            if record.declaration_status != pi.DECLARATION:
                continue

            if isinstance(key, pi.ProvisionKey) and key.ordinal is not None:
                self.lists.setdefault(self._list_of(record), []).append(record)
            elif isinstance(key, pi.TableRowKey):
                self.tables.setdefault((key.section, key.table), []).append(record)

    def _heading(self, record):
        unit = self.units.get(record.source_id) or {}

        return tuple(unit.get("heading_path") or ())

    def _list_of(self, record):
        return (record.key.section, record.key.kind, self._heading(record))

    def group(self, record):
        """(relation, parent, members) of the structural group `record` is in."""
        key = record.key

        if isinstance(key, pi.ProvisionKey) and key.ordinal is not None:
            heading = self._heading(record)
            parent = f"{key.kind} list of §{key.section}" + (f" ({heading[-1]})" if heading else "")

            return SAME_NORMATIVE_LIST, parent, self.lists.get(self._list_of(record), [])

        if isinstance(key, pi.TableRowKey):
            return SAME_TABLE, f"Table {key.table} of §{key.section}", \
                self.tables.get((key.section, key.table), [])

        return None, "", []

    def in_section(self, section: str):
        """The normative declarations the store files under one section."""
        return self.sections.get(section, [])

    def resolve(self, identity: str):
        """The records a stated identity names: an instance id exactly, or a
        printed key that names exactly one instance. Anything else, none."""
        identity = str(identity).strip()

        if "@" in identity:
            key = identity.rsplit("@", 1)[0]
            return [record for record in self.by_key.get(key, [])
                    if instance_of(record) == identity]

        found = self.by_key.get(identity, [])

        return found if len(found) == 1 else []

    def order(self, record):
        """Document order: the ordinal the document prints, then position."""
        unit = self.units.get(record.source_id) or {}

        return (getattr(record.key, "ordinal", None) or 0, unit.get("unit_position", 0),
                record.source_id)


_SECTION_CITATION = re.compile(r"§\s*([A-Z]?\d+(?:\.\d+)*)")
_QUOTE = re.compile(r"[\"\u201c]([^\"\u201c\u201d]{30,})[\"\u201d]")


def quoted(answer: str, universe: Universe) -> list:
    """Provisions an answer cites by section and verbatim quotation.

    A provision without a printed label -- an unlabelled body of a
    specification -- can only be cited as "§G6.1.2.1: '...'". The quote
    settles which body is meant when its words, in order, are that body's
    own: a paraphrase identifies nothing. Each fragment of a quote shortened
    with an ellipsis counts on its own.
    """
    found, citations = {}, list(_SECTION_CITATION.finditer(answer or ""))

    for index, match in enumerate(citations):
        end = citations[index + 1].start() if index + 1 < len(citations) else len(answer)
        span, section = answer[match.end():end], match.group(1)
        bodies = [(record, " ".join((record.text or "").split()))
                  for record in universe.in_section(section)]

        for quote in _QUOTE.finditer(span):
            for fragment in re.split(r"\.\.\.|\u2026", quote.group(1)):
                fragment = " ".join(fragment.split()).strip(" .")

                if len(fragment) < 30:
                    continue

                for record, text in bodies:
                    if fragment in text or (len(text) >= 40 and text.rstrip(".") in fragment):
                        found.setdefault(instance_of(record), record)

    return list(found.values())


def close(cited, universe: Universe, *, retrieved=()) -> Coverage:
    """The coverage set: the cited provisions and their structural groups."""
    chosen = {}
    ungrouped, complete, reasons = [], True, []

    for record in cited:
        chosen.setdefault(instance_of(record), CoverageEntry(record, CITED,
                                                             "cited by the normative pass"))

    if not chosen:
        complete, reasons = False, ["the normative pass cited no provision"]

    for entry in list(chosen.values()):
        record = entry.record
        relation, parent, members = universe.group(record)
        members = [item for item in members if normative(item)]

        if relation is None:
            ungrouped.append(entry.instance)
        elif len(members) > GROUP_LIMIT:
            complete = False
            reasons.append(f"{parent} has {len(members)} normative members, over the "
                           f"limit of {GROUP_LIMIT}")
        else:
            for member in sorted(members, key=universe.order):
                chosen.setdefault(instance_of(member), CoverageEntry(
                    member, CLOSURE,
                    f"{member.key} is in the same {parent} as {record.key}",
                    relation, entry.instance, parent))

        # A provision the cited one names outright: one step, never onward.
        for match in pi._LABEL.finditer(getattr(record, "text", "") or ""):
            label = f"{match.group('kind')} {match.group('section')}-{match.group('ordinal')}"

            for target in universe.by_key.get(label, []):
                if normative(target) and target.declaration_status == pi.DECLARATION:
                    chosen.setdefault(instance_of(target), CoverageEntry(
                        target, CLOSURE, f"{record.key} names {target.key}",
                        EXPLICIT_REFERENCE, entry.instance, str(record.key)))

    added = [item for item in chosen.values() if item.origin == CLOSURE]

    if len(added) > CLOSURE_LIMIT:
        complete = False
        reasons.append(f"structural closure would add {len(added)} provisions, over the "
                       f"limit of {CLOSURE_LIMIT}")
        keep = {item.instance for item in added[:CLOSURE_LIMIT]}
        chosen = {key: item for key, item in chosen.items()
                  if item.origin == CITED or key in keep}

    return Coverage(tuple(retrieved), tuple(chosen.values()), complete, "; ".join(reasons),
                    tuple(ungrouped))


# ------------------------------------------------------------- applicability

@dataclass(frozen=True)
class Applicability:
    status: str                     # APPLICABLE | NOT_APPLICABLE | UNRESOLVED
    basis: str
    advisory: str = ""              # the model's view, recorded and never used


def named_by(objective: str, record) -> str:
    """How the user's request names this provision, if it does: by its printed
    label, or by naming its section."""
    key = record.key

    if isinstance(key, pi.ProvisionKey) and key.ordinal is not None and re.search(
            r"\b" + re.escape(str(key)) + r"(?![\d\-]|\.\d)", objective or "", re.I):
        return f"the request names {key}"

    section = getattr(key, "section", "")

    if section and re.search(r"(?:\b(?:rule|rules|permission|permissions|recommendation|"
                             r"recommendations|requirement|requirements|section|clause)\s+|§\s*)"
                             + re.escape(section) + r"(?![\d\-]|\.\d)", objective or "", re.I):
        return f"the request names §{section}"

    return ""


def decide(entry: CoverageEntry, objective: str, declared, bound) -> Applicability:
    """An entry's applicability, from deterministic facts only.

    `declared` maps an instance id to the project's statements about it,
    each (status, reason, authority); `bound` maps an instance id to the ids
    of the project checks bound to it.
    """
    record = entry.record
    advisory = ("POSSIBLY_APPLICABLE: cited by the normative pass"
                if entry.origin == CITED else "")
    statements = {status for status, _, _ in declared.get(entry.instance, ())}

    if len(statements) > 1:
        return Applicability(UNRESOLVED, "the project declares it both applicable and "
                                         "not applicable", advisory)

    if statements == {NOT_APPLICABLE}:
        reason, authority = next((reason, authority) for status, reason, authority
                                 in declared[entry.instance])
        return Applicability(NOT_APPLICABLE, f"{authority}: {reason}", advisory)

    if statements == {APPLICABLE}:
        reason, authority = next((reason, authority) for status, reason, authority
                                 in declared[entry.instance])
        return Applicability(APPLICABLE, f"{authority}: {reason}", advisory)

    if bound.get(entry.instance):
        return Applicability(APPLICABLE, "the project binds a conformance check to it ("
                             + ", ".join(sorted(bound[entry.instance])) + ")", advisory)

    named = named_by(objective, record)

    if named:
        return Applicability(APPLICABLE, named, advisory)

    return Applicability(UNRESOLVED, "nothing deterministic establishes that it applies to "
                                     "this change", advisory)
