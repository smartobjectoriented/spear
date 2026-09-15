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
#: A provision DECLARATION: the label, then a colon, then the provision. A
#: label without the colon is a cross-reference -- "as required by Rule
#: 5.1.4.1-1" -- and a later mention of a provision is not a second one.
#: Measured on the bound standard: 1332 labels are declarations and 41 are
#: references, every one of the 41 preceded by a referring word. Deriving
#: both put one rule on four pages with two different modalities.
#: The label itself. Whether an occurrence DECLARES the provision or merely
#: REFERS to it is decided by `classify_label`, never by the colon alone: the
#: colon is the usual typography and the extractor does lose it. Three
#: provisions of one bound standard have no colon anywhere, and a rule that
#: treated the colon as decisive discarded all three in silence.
_LABEL = re.compile(
    rf"\b(?P<kind>{'|'.join(LABELLED_KINDS)})\s+(?P<section>{_SECTION})-(?P<ordinal>\d+)")

DECLARATION = "DECLARATION"
CANDIDATE = "CANDIDATE"          # review required; may not ground a claim
REFERENCE = "REFERENCE"

#: Generic citation language. These are the words any technical document uses
#: to point AT a provision rather than to state one -- no phrase here names a
#: subject, a field or a packet.
_REFERRING = re.compile(
    r"(?:\b(?:see|per|pursuant to|according to|described in|given in|"
    r"expressed in|governed by|specified in|defined in|refer to|as in|"
    r"violat\w*|follow\w*|apply\w*|abide by|coupled with|together with|"
    r"in|of|by|with|from|to|and|than)\s*$)", re.I)

#: The left context of a sentence-initial occurrence: end of the previous
#: sentence, a bullet, or the start of the unit.
_SENTENCE_END = re.compile(r"(?:^|[.!?:•]\s*|\u2022\s*)$")

#: A label followed by a reporting verb is pointing at the provision, not
#: stating it: "Rule 6.1.2-7 implies that ...", "Rule 5.1.4.1-1 says that ...".
#: Generic citation language -- these verbs report what some OTHER text does,
#: and a provision never opens by reporting itself.
_REPORTING = re.compile(
    r"^\s+(?:says?|state[sd]?|implies|implied|requires?|required|establishes?|"
    r"permits?|permitted|allows?|allowed|prevents?|limits?|restricts?|"
    r"indicates?|specifies|specified|differs?|considers?|applies|apply|"
    r"holds?|creates?|makes?|saves?|and|or|is|are|was|were|shall|must|may)\b",
    re.I)

#: Body text after a label reads like a statement: it starts with a word,
#: not with a connective that would continue the referring sentence.
_BODY_START = re.compile(r"^\s+[\"\u201c(]?[A-Z]")


def classify_label(text, match):
    """DECLARATION, CANDIDATE or REFERENCE for one label occurrence.

    Several signals, because no single one is safe. A colon is decisive when
    present. Otherwise position decides: a label opening a sentence, followed
    by something that reads like a statement, is a declaration whose colon
    the extractor dropped -- and a label sitting mid-sentence after citation
    language is a reference. Anything else is a CANDIDATE, kept for review
    rather than thrown away.
    """
    after = text[match.end():match.end() + 2]

    if after.startswith(":"):
        return DECLARATION

    left = text[:match.start()]
    right = text[match.end():]
    sentence_initial = bool(_SENTENCE_END.search(left))
    referring = bool(_REFERRING.search(left))
    body_like = bool(_BODY_START.match(right))

    if referring and not sentence_initial:
        return REFERENCE

    if _REPORTING.match(right):
        return REFERENCE            # it reports a provision; it is not one

    if sentence_initial and body_like:
        return CANDIDATE            # a declaration, most likely, but unproven

    if not sentence_initial:
        return REFERENCE

    return CANDIDATE

#: A citation as an answer writes it. The kind is optional, which is exactly
#: the problem: without it the reference may name several provisions.
_REFERENCE = re.compile(
    rf"\b(?:(?P<kind>{'|'.join(LABELLED_KINDS)})\s+)?"
    rf"(?P<section>{_SECTION})(?:-(?P<ordinal>\d+))?\b")


# ── structural evidence, which has no provision identity ─────────────
# A numbered provision is identified by what the document calls it. A table
# row and a section's lead-in are not numbered, and giving them a
# ProvisionKey with a null ordinal collapsed every one of them in a section
# onto a single identity: 33 scope blocks and 37 table rows of one store.
#
# They are not provisions, so they do not get ProvisionKeys. They get their
# own keys, and the discriminator each needs is a property of the structure
# it sits in -- which table, which block -- never a bare source_id standing
# in for a normative name.


@dataclass(frozen=True, order=True)
class TableRowKey:
    """One row of one table. `table` is the table's own identity."""

    section: str
    table: str
    row: int

    kind: str = field(default="TableRow", init=False, compare=True)

    def __str__(self):
        return f"TableRow {self.table}:{self.row}"


@dataclass(frozen=True, order=True)
class ScopePreambleKey:
    """One unnumbered scope-bearing block.

    The discriminator localises the block within its section. It is the
    parent unit's id because that is what makes two blocks distinct; it is
    not a normative name and nothing resolves a citation to it.
    """

    section: str
    discriminator: str

    kind: str = field(default="ScopePreamble", init=False, compare=True)

    def __str__(self):
        return f"ScopePreamble §{self.section}@{self.discriminator[:12]}"


#: A caption's own label -- "Table 8.3.1-1: ..." -- which is what the document
#: calls the table and is stable across re-extraction. Nothing here knows any
#: particular table.
_TABLE_LABEL = re.compile(r"^\s*table\s+(?P<label>[0-9]+(?:[.\-][0-9]+)*)",
                          re.I)


def table_identity(caption):
    """A stable identity for a table, from the best evidence available.

    The caption's printed label first: it is what the document calls the
    table and it survives re-extraction. Failing that, the caption unit's id
    with a hash of its text, which is stable while the text is unchanged.
    Never array position -- that moves when something unrelated is extracted
    differently.
    """
    if not isinstance(caption, dict):
        return ""

    text = " ".join(str(caption.get("text") or "").split())
    match = _TABLE_LABEL.match(text)

    if match:
        return match.group("label")

    source = str(caption.get("source_id") or "")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]

    return f"{source}:{digest}" if source or text else ""


@dataclass(frozen=True, order=True)
class ProvisionKey:
    """The PRINTED CITATION KEY: what the document calls a provision.

    (section, kind, ordinal), and it stays a faithful record of the printed
    label. It is NOT a unique provision identity, because the document itself
    reuses labels: Rule 5.1.6-2 is declared twice in section 5.1.6, on pages
    75 and 76, with the same heading path and entirely different normative
    content. Twenty-two such labels exist in one bound standard.

    So a citation key resolves to ZERO, ONE or SEVERAL provision instances,
    and only the one-instance case is unambiguous. `ProvisionInstanceId`
    below is what identifies a declaration.
    """

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


@dataclass(frozen=True, order=True)
class ProvisionInstanceId:
    """One declaration, distinct from any other sharing its printed label.

    Smallest thing that is unique, deterministic and stable while the
    declaration is unchanged: the document, the printed key, and a digest of
    the declaring span -- its parent unit and its own text. Neither the page
    nor the source_id is the citation identity; both are provenance, and the
    digest is what makes two declarations of one label distinguishable.

    The digest changes when the extracted text changes, which is the point:
    an approval recorded against this instance is then reviewable rather
    than silently carried onto different words.
    """

    standard_id: str
    revision: str
    key: ProvisionKey
    digest: str

    def __str__(self):
        return f"{self.key}@{self.digest}"

    def render(self, page=None):
        """The human citation: the document's real printed label, plus only
        as much locator as is needed to tell two of them apart."""
        printed = str(self.key)

        return f"{printed} (p. {page})" if page is not None else printed


def instance_id(key, *, standard_id, revision, source_id, text):
    return ProvisionInstanceId(
        standard_id=str(standard_id or ""), revision=str(revision or ""),
        key=key,
        digest=hashlib.sha256(
            f"{standard_id}|{revision}|{key}|{source_id}|{text}".encode("utf-8")
        ).hexdigest()[:12])


@dataclass(frozen=True)
class ProvisionRecord:
    """One provision, and the unit it was carried in."""

    key: ProvisionKey
    source_id: str = ""
    page: int | None = None
    text: str = ""
    modality: str = "NONE"          # the PROVISION's own modality
    unit_modality: str = "NONE"     # what the container is stored as
    #: DECLARATION, or CANDIDATE when the occurrence could not be proven one.
    #: A CANDIDATE is preserved and surfaced; it may not ground a claim.
    declaration_status: str = DECLARATION
    content_type: str = ""
    needs_review: bool = False

    #: The container's text, kept so an approval recorded against the unit
    #: can be revalidated without re-deriving the whole store.
    unit_text: str = ""

    #: Every extraction span this one logical provision occupies. Ordinarily
    #: one; a declaration split across a page boundary and reconstructed has
    #: several, and is ONE instance rather than two competing ones.
    spans: tuple = ()

    standard_id: str = ""
    revision: str = ""

    @property
    def citation_key(self):
        """What the document prints. May not be unique."""
        return self.key

    @property
    def instance_id(self):
        """What identifies THIS declaration."""
        return instance_id(self.key, standard_id=self.standard_id,
                           revision=self.revision, source_id=self.source_id,
                           text=self.text)

    def rendered_citation(self, *, disambiguate=False):
        """The printed label, with a locator only when one is needed."""
        return (self.instance_id.render(self.page) if disambiguate
                else str(self.key))

    @property
    def text_sha256(self):
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    @property
    def unit_text_sha256(self):
        return hashlib.sha256((self.unit_text or "").encode("utf-8")).hexdigest()


def _clean(text):
    return " ".join((text or "").split())


def _collect_units(node, found):
    """Every unit-shaped dict in a payload, at any depth."""
    if isinstance(node, dict):
        if node.get("text") or node.get("snippet"):
            found.append(node)

        for value in node.values():
            if isinstance(value, (dict, list)):
                _collect_units(value, found)
    elif isinstance(node, list):
        for item in node:
            _collect_units(item, found)


def records_from_unit(unit, *, table=None):
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
              "unit_text": text,
              "standard_id": str(unit.get("standard_id") or ""),
              "revision": str(unit.get("revision") or ""),
              "spans": ({"source_id": str(unit.get("source_id") or ""),
                         "page": unit.get("page")},)}

    #: "20 ReqV Request Validation ..." -- a row, keyed by its table and its
    #: own number. Without the table, two tables in one section whose rows
    #: are numbered alike collide; `table` is supplied by whatever knows the
    #: caption, and a row derived with no table context falls back to its own
    #: unit so it is at least unique.
    row = _TABLE_ROW.match(_clean(text))

    if row and section:
        owner = table or f"unit:{common['source_id']}"

        return [ProvisionRecord(
            key=TableRowKey(section, owner, int(row.group("key"))),
            text=_clean(text), modality=_modality_of(text), **common)]

    found, spans = [], [(m.start(), m) for m in _LABEL.finditer(text)]

    claimed = set()
    statuses = {index: classify_label(text, match)
                for index, (_, match) in enumerate(spans)}

    for index, (start, match) in enumerate(spans):
        if statuses[index] == REFERENCE:
            continue                # a mention of a provision is not one

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
                                     modality=_modality_of(body),
                                     declaration_status=statuses[index],
                                     **common))

    # Text before the first label, or a unit with no label at all. It is
    # evidence -- a section lead-in states what its regulations are ABOUT --
    # and it gets a kind of its own rather than a invented ordinal.
    head = _clean(text[:spans[0][0]] if spans else text)

    if head and section:
        kind = (SCOPE_PREAMBLE if spans
                else _FROM_CONTENT_TYPE.get(common["content_type"].upper()))

        if kind:
            key = (ScopePreambleKey(section, common["source_id"])
                   if kind == SCOPE_PREAMBLE else ProvisionKey(section, kind, None))
            found.append(ProvisionRecord(key=key, text=head,
                                         modality=_modality_of(head), **common))

    return found


def records_for_units(units, *, tables=None):
    """Every provision in a set of units, with table context applied.

    Derivation of one unit cannot know which table a row belongs to; that is
    a property of the run it sits in. `tables` is whatever knows -- normally
    evidence_graph.tables_in -- and without it rows fall back to per-unit
    identity.
    """
    owner = {}

    for table in (tables or ()):
        identity = table_identity(table.caption)

        for row in table.rows:
            owner[str(row.get("source_id") or "")] = identity

    found = []

    for unit in units:
        if not isinstance(unit, dict):
            continue

        found.extend(records_from_unit(
            unit, table=owner.get(str(unit.get("source_id") or ""))))

    return found


#: A fragment that stops mid-clause: the declaration continues elsewhere.
_INCOMPLETE = re.compile(r"[a-z0-9,;(\-]\s*$")


def reconstruct_page_splits(records):
    """Join a declaration split across an extraction boundary, read-only.

    One logical provision occupying two extraction units is ONE instance
    with two provenance spans, not two competing declarations of the same
    printed label. The conditions are deliberately strict -- same citation
    key, adjacent pages, the first fragment syntactically incomplete, the
    second continuing without declaring anything itself -- and anything that
    fails them stays duplicated and review-required.

    The store is never written to. This is a derived view.
    """
    grouped = {}

    for record in records:
        grouped.setdefault(record.key, []).append(record)

    found, joined = [], set()

    for key, group in grouped.items():
        if len(group) != 2 or not isinstance(key, ProvisionKey):
            found.extend(group)
            continue

        first, second = sorted(group, key=lambda r: (r.page or 0))
        pages = [first.page or 0, second.page or 0]
        declares_again = second.text.startswith(f"{key.kind} {key.section}-")

        if (abs(pages[1] - pages[0]) == 1 and _INCOMPLETE.search(first.text)
                and not declares_again):
            merged = first.text + " " + second.text
            found.append(ProvisionRecord(
                key=key, source_id=first.source_id, page=first.page,
                text=merged, modality=_modality_of(merged),
                unit_modality=first.unit_modality,
                content_type=first.content_type,
                needs_review=first.needs_review or second.needs_review,
                unit_text=first.unit_text,
                spans=tuple(first.spans) + tuple(second.spans),
                standard_id=first.standard_id, revision=first.revision,
                declaration_status=first.declaration_status))
            joined.add(key)
        else:
            found.extend(group)

    return found, sorted(joined, key=str)


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

    #: instance_id -> record. The ledger holds INSTANCES, because a printed
    #: label may name several declarations and keeping one per label throws
    #: the others away.
    instances: dict = field(default_factory=dict)

    #: printed citation key -> [instance_id], in arrival order.
    by_key: dict = field(default_factory=dict)

    #: (citation key, parent source_id) -> record. One declaration cannot be
    #: made twice in one unit, so this is what collapses a snippet and the
    #: full unit into a single instance.
    _by_slot: dict = field(default_factory=dict)

    #: Citation keys naming more than one declaration. Not a defect: the
    #: document reuses labels. It does mean the printed citation alone
    #: cannot say which declaration is meant.
    duplicated: set = field(default_factory=set)

    @property
    def records(self):
        """{citation key: record} for labels naming exactly ONE declaration.

        Deliberately lossy and deliberately narrow: a reused label has no
        single record, and a caller that wants one must ask for instances.
        """
        return {key: self.instances[ids[0]]
                for key, ids in self.by_key.items() if len(ids) == 1}

    def observe(self, payload):
        """Collect provisions from one tool result. The payload is untouched.

        Table context comes from the payload itself: a fetch of a caption now
        carries its rows, so the rows in THIS result are identified by the
        table in THIS result.
        """
        import evidence_graph

        units = []
        _collect_units(payload, units)
        self._add(records_for_units(units,
                                    tables=evidence_graph.tables_in(units)))

        return self

    def _add(self, records):
        """One declaration per (printed key, parent unit).

        The same declaration reaches a turn more than once and not always
        whole: `search` returns a truncated snippet and `fetch` the full
        unit. Keyed on text alone those are two digests and therefore a
        false ambiguity -- one that made a turn refuse a citation the
        document does not actually reuse. A label cannot be declared twice
        in one unit, so the unit settles it, and the longest text wins
        because a snippet is a prefix of the truth.
        """
        for record in records:
            slot = (record.key, record.source_id)
            seen = self._by_slot.get(slot)

            if seen is not None:
                if len(record.text) <= len(seen.text):
                    continue

                # A fuller reading of the same declaration replaces the
                # snippet, identity and all.
                self.instances.pop(seen.instance_id, None)
                arrived = self.by_key.get(record.key, [])
                if seen.instance_id in arrived:
                    arrived.remove(seen.instance_id)

            identity = record.instance_id

            if identity in self.instances:
                continue

            self._by_slot[slot] = record
            self.instances[identity] = record
            arrived = self.by_key.setdefault(record.key, [])
            arrived.append(identity)

            if len(arrived) > 1:
                self.duplicated.add(record.key)
            elif record.key in self.duplicated and len(arrived) == 1:
                self.duplicated.discard(record.key)

    # -- instance-level questions ------------------------------------

    def instances_for(self, key):
        """Every declaration carrying this printed label."""
        return [self.instances[item] for item in self.by_key.get(key, ())]

    def contains_instance(self, identity):
        return identity in self.instances

    def covers_instance(self, identity):
        """Was THIS declaration retrieved, and is it a proven declaration?"""
        record = self.instances.get(identity)

        return record is not None and record.declaration_status == DECLARATION

    def covers_citation(self, key):
        """Is at least one declaration carrying this printed label present?

        Weaker than it looks when the label is reused: it says something
        with that label was read, never which one. It must not stand in for
        `covers_instance` where several instances share the key.
        """
        return any(record.declaration_status == DECLARATION
                   for record in self.instances_for(key))

    # -- questions it can answer -------------------------------------

    def has(self, key):
        """Was this printed label retrieved AND unambiguous?

        False when the label names several declarations: something with that
        label was read, but not which one, and grounding a claim on it would
        be picking. `covers_citation` reports the weaker fact.
        """
        found = self.instances_for(key)

        return (len(found) == 1
                and found[0].declaration_status == DECLARATION)

    def candidates(self):
        """Occurrences that could not be proven declarations.

        Preserved rather than discarded, and surfaced so an extraction
        diagnostic can show them. They ground nothing.
        """
        return {record.instance_id: record
                for record in self.instances.values()
                if record.declaration_status != DECLARATION}

    def get(self, key):
        """The sole declaration for a label, or None when it is reused."""
        found = self.instances_for(key)

        return found[0] if len(found) == 1 else None

    def kinds_for(self, section, ordinal):
        """Every provision kind sharing one section and ordinal."""
        return sorted({key.kind for key in self.by_key
                       if isinstance(key, ProvisionKey)
                       and key.section == section and key.ordinal == ordinal})

    def covers_section(self, section):
        """Weaker, informational only.

        That something from a section was read never establishes that the
        provision an answer relies on was among it. This must not satisfy a
        provision-level grounding requirement; it exists to say what was in
        front of the model.
        """
        return any(key.section == section
                   or key.section.startswith(section + ".")
                   for key in self.by_key)

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
            declared = [record for record in self.instances_for(key)
                        if record.declaration_status == DECLARATION]

            # Naming the kind is not enough when the DOCUMENT reuses the
            # label: "Rule 5.1.6-2" names two declarations, and returning
            # nothing would read as "not retrieved" rather than "which one?".
            if len(declared) > 1:
                raise AmbiguousReference(
                    reference, [record.instance_id for record in declared])

            return key if len(declared) == 1 else None

        # Only numbered provisions are citable. A scope block and a table row
        # carry evidence and have no name a citation could use.
        # Only numbered provisions are citable, and only PROVEN declarations
        # may ground a claim. A candidate is preserved and reported; it is
        # not silently promoted into evidence.
        candidates = [key for key, ids in self.by_key.items()
                      if isinstance(key, ProvisionKey)
                      and key.section == section and key.ordinal == ordinal
                      and any(self.instances[i].declaration_status == DECLARATION
                              for i in ids)]

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
