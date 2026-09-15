"""A provision, not a section and not a source unit.

Normative identity in a standards document is finer than either container the
store has. One section holds many provisions; one extracted unit holds several
of them; and the document numbers each KIND independently, so the same section
and ordinal name different provisions with different modality:

    Rule 8.3.1.5-1          shall ...
    Permission 8.3.1.5-1    may ...
    Observation 8.3.1.5-1   (informative)

Measured on the bound standard: 308 of 806 bare labels carry more than one
kind, and one carries four. Everything that used a bare label as identity was
therefore wrong in 38% of cases -- including the clause ledger, whose
`covers()` proved only that a SECTION had been retrieved, which is how a
retrieved `shall` Rule came to license a requirement claim about its
neighbouring `should` Recommendation.

So identity is (section, kind, ordinal) and nothing less. A reference that
does not determine all three is ambiguous, and ambiguity is reported rather
than resolved by picking one.

The kinds are the ones the store actually contains -- Rule, Recommendation,
Permission, Observation, Definition as inline labels, plus the extractor's own
content_type for text that carries no label. Nothing here is specific to any
document.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

#: Inline provision labels, as the document writes them.
RULE = "Rule"
RECOMMENDATION = "Recommendation"
PERMISSION = "Permission"
OBSERVATION = "Observation"
DEFINITION = "Definition"

#: The extractor's taxonomy, for text that carries no inline label.
REQUIREMENT = "Requirement"
INFORMATIVE = "Informative"

#: Normative or scope-bearing text with no ordinal at all -- a section's
#: lead-in that says which packet its regulations are about. Real evidence,
#: reviewable on its own, and never given a made-up Rule number.
SCOPE_PREAMBLE = "ScopePreamble"

#: A row of an extracted table, keyed by the number that opens it. The CAM
#: layout of the bound standard arrives as one unit per row -- "20 ReqV
#: Request Validation ..." -- with no inline label and no content_type, so
#: without this they are not provisions at all and cannot be approved
#: individually. The ordinal is the row's own key, not an invented clause
#: number.
TABLE_ROW = "TableRow"

LABELLED_KINDS = (RULE, RECOMMENDATION, PERMISSION, OBSERVATION, DEFINITION)
KINDS = LABELLED_KINDS + (REQUIREMENT, INFORMATIVE, SCOPE_PREAMBLE,
         TABLE_ROW)

#: content_type -> the kind an unlabelled unit contributes.
_FROM_CONTENT_TYPE = {
    "REQUIREMENT": REQUIREMENT, "RECOMMENDATION": RECOMMENDATION,
    "INFORMATIVE": INFORMATIVE, "DEFINITION": DEFINITION,
}

_SECTION = r"\d+(?:\.\d+)*"

#: A row whose first token is its key, and whose second is a short name.
_TABLE_ROW = re.compile(r"^(?P<key>\d{1,3})\s+(?P<name>[A-Za-z][\w./-]{0,24})\b")
_LABEL = re.compile(
    rf"\b(?P<kind>{'|'.join(LABELLED_KINDS)})\s+(?P<section>{_SECTION})-(?P<ordinal>\d+)\b")

#: A citation as an answer writes it. The kind is optional, which is exactly
#: the problem: without it the reference may name several provisions.
_REFERENCE = re.compile(
    rf"\b(?:(?P<kind>{'|'.join(LABELLED_KINDS)})\s+)?"
    rf"(?P<section>{_SECTION})(?:-(?P<ordinal>\d+))?\b")


@dataclass(frozen=True, order=True)
class ProvisionKey:
    """(section, kind, ordinal). Two of the three is not an identity."""

    section: str
    kind: str
    ordinal: int | None = None

    def __str__(self):
        if self.ordinal is None:
            return f"{self.kind} §{self.section}"
        return f"{self.kind} {self.section}-{self.ordinal}"

    @property
    def bare_label(self):
        """What a careless citation would write. Not an identity."""
        if self.ordinal is None:
            return self.section
        return f"{self.section}-{self.ordinal}"


#: Modality read from a provision's OWN words. The parent unit's stored
#: modality is a property of the container: the unit carrying Permission
#: 8.3.1.5-1 is stored SHALL because a Rule sits beside the Permission in it,
#: and inheriting that would report a permission as a requirement.
_SPOKEN = ((("shall", "must"), "SHALL"), (("should",), "SHOULD"),
           (("may", "can"), "MAY"))


def _modality_of(text):
    lowered = (text or "").lower()

    for words, name in _SPOKEN:
        if any(re.search(rf"\b{word}\b", lowered) for word in words):
            return name

    return "NONE"


@dataclass(frozen=True)
class ProvisionRecord:
    """One provision, and the unit it was carried in."""

    key: ProvisionKey
    source_id: str = ""
    page: int | None = None
    text: str = ""
    modality: str = "NONE"          # the PROVISION's own modality
    unit_modality: str = "NONE"     # what the container is stored as
    content_type: str = ""
    needs_review: bool = False

    #: The container's text, kept so an approval recorded against the unit
    #: can be revalidated without re-deriving the whole store.
    unit_text: str = ""

    @property
    def text_sha256(self):
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    @property
    def unit_text_sha256(self):
        return hashlib.sha256((self.unit_text or "").encode("utf-8")).hexdigest()


def _clean(text):
    return " ".join((text or "").split())


def records_from_unit(unit):
    """Every provision a returned unit carries.

    A unit is the storage and retrieval container; it is not the provision.
    One unit of the bound standard carries a Permission, an Observation and a
    Rule that all share section and ordinal, and approving one of them must
    never approve the others.
    """
    if not isinstance(unit, dict):
        return []

    text = unit.get("text") or unit.get("snippet") or ""
    section = str(unit.get("section") or "")
    common = {"source_id": str(unit.get("source_id") or ""),
              "page": unit.get("page"),
              "unit_modality": str(unit.get("modality") or "NONE"),
              "content_type": str(unit.get("content_type") or ""),
              "needs_review": bool(unit.get("needs_review")),
              "unit_text": text}

    #: "20 ReqV Request Validation ..." -- a table row, keyed by its number.
    row = _TABLE_ROW.match(_clean(text))

    if row and section:
        return [ProvisionRecord(
            key=ProvisionKey(section, TABLE_ROW, int(row.group("key"))),
            text=_clean(text), modality=_modality_of(text), **common)]

    found, spans = [], [(m.start(), m) for m in _LABEL.finditer(text)]

    claimed = set()

    for index, (start, match) in enumerate(spans):
        end = spans[index + 1][0] if index + 1 < len(spans) else len(text)
        body = _clean(text[start:end])
        key = ProvisionKey(match.group("section"), match.group("kind"),
                           int(match.group("ordinal")))

        # A unit may name the same provision twice -- a rule and a later
        # cross-reference to it. The first occurrence is the provision; the
        # second is a mention, and emitting both put two records with one
        # identity into the ledger.
        if key in claimed:
            continue

        claimed.add(key)
        found.append(ProvisionRecord(key=key, text=body,
                                     modality=_modality_of(body), **common))

    # Text before the first label, or a unit with no label at all. It is
    # evidence -- a section lead-in states what its regulations are ABOUT --
    # and it gets a kind of its own rather than a invented ordinal.
    head = _clean(text[:spans[0][0]] if spans else text)

    if head and section:
        kind = (SCOPE_PREAMBLE if spans
                else _FROM_CONTENT_TYPE.get(common["content_type"].upper()))

        if kind:
            found.append(ProvisionRecord(
                key=ProvisionKey(section, kind, None), text=head,
                modality=_modality_of(head), **common))

    return found


class AmbiguousReference(Exception):
    """A citation that names more than one provision."""

    def __init__(self, reference, candidates):
        super().__init__(f"{reference} names {len(candidates)} provisions: "
                         + ", ".join(str(key) for key in sorted(candidates)))
        self.reference = reference
        self.candidates = list(candidates)


@dataclass
class ProvisionLedger:
    """Which PROVISIONS a turn retrieved -- not which sections."""

    records: dict = field(default_factory=dict)

    #: Keys seen in more than one source unit. Identity that is not unique is
    #: not identity; a caller that needs certainty must treat these as
    #: ambiguous rather than trusting the first arrival.
    duplicated: set = field(default_factory=set)

    def observe(self, payload):
        """Collect provisions from one tool result. The payload is untouched."""
        self._walk(payload)
        return self

    def _walk(self, node):
        if isinstance(node, dict):
            if node.get("text") or node.get("snippet"):
                for record in records_from_unit(node):
                    seen = self.records.get(record.key)

                    if seen is not None and seen.source_id != record.source_id:
                        self.duplicated.add(record.key)

                    self.records.setdefault(record.key, record)

            for value in node.values():
                if isinstance(value, (dict, list)):
                    self._walk(value)
        elif isinstance(node, list):
            for item in node:
                self._walk(item)

    # -- questions it can answer -------------------------------------

    def has(self, key):
        """Was THIS provision retrieved?"""
        return key in self.records

    def get(self, key):
        return self.records.get(key)

    def kinds_for(self, section, ordinal):
        """Every kind sharing one section and ordinal."""
        return sorted({key.kind for key in self.records
                       if key.section == section and key.ordinal == ordinal})

    def covers_section(self, section):
        """Weaker, informational only.

        That something from a section was read never establishes that the
        provision an answer relies on was among it. This must not satisfy a
        provision-level grounding requirement; it exists to say what was in
        front of the model.
        """
        return any(key.section == section
                   or key.section.startswith(section + ".")
                   for key in self.records)

    def resolve(self, reference):
        """The provision a citation names, or AmbiguousReference.

        A bare `8.3.1.5-2` is not resolved by picking one: on the bound
        standard that label names a Rule, a Recommendation and an Observation,
        and choosing silently is how a `should` became a `shall`.
        """
        match = _REFERENCE.fullmatch(_clean(reference))

        if not match:
            return None

        section, kind = match.group("section"), match.group("kind")
        ordinal = match.group("ordinal")
        ordinal = int(ordinal) if ordinal is not None else None

        if kind:
            key = ProvisionKey(section, kind, ordinal)
            return key if key in self.records else None

        candidates = [key for key in self.records
                      if key.section == section and key.ordinal == ordinal]

        if not candidates:
            return None

        if len(candidates) > 1:
            raise AmbiguousReference(_clean(reference), candidates)

        return candidates[0]

    def references_in(self, text):
        """Every citation an answer makes, resolved or reported ambiguous."""
        resolved, ambiguous = [], []

        for match in _REFERENCE.finditer(text or ""):
            if match.group("ordinal") is None and not match.group("kind"):
                continue                       # a bare section is not a citation

            try:
                key = self.resolve(match.group(0))
            except AmbiguousReference as problem:
                ambiguous.append(problem)
                continue

            if key is not None:
                resolved.append(key)

        return resolved, ambiguous
