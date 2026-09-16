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

import normative_force

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
    "SCOPE_PREAMBLE": SCOPE_PREAMBLE,
}

#: A section's lead-in, recognised by what it SAYS. The kind of block that
#: says what the regulations below are about is scope-bearing whether or not
#: the extractor had a name for it: one extraction puts it inside the first
#: labelled unit, where it is already treated as a preamble, and another
#: emits it on its own -- where, keyed only on content_type, it produced no
#: identity at all and the scope of section 8.3.1.5 could not be cited.
_SCOPE_LEAD = re.compile(
    r"^\s*(?:the following|this (?:section|subsection|clause)|"
    r"the regulations|regulations (?:below|in this))\b", re.I)

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
    r"holds?|creates?|makes?|saves?|has|have|had|and|or|is|are|was|were|"
    r"shall|must|may)\b",
    re.I)

#: A label a sentence runs THROUGH rather than opens with: a comma straight
#: after it, or a word closing up against it with no space. Both are how a
#: document mentions a provision mid-clause -- "Permission 6.1.2-1, coupled
#: with ...", "Permission 9.7.3.4-1permits ..." -- and neither declares one.
_RUNS_ON = re.compile(r"^(?:\s*,|[a-z])")

#: A letter suffix on the ordinal, then the declaring colon: "Rule 5.2-1a:".
_SUFFIXED = re.compile(r"^[a-z]{1,2}\s*:")

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

    # A sub-lettered ordinal: the label regex stops at the digits, so
    # "Rule 5.2-1a:" leaves "a:" behind. The colon is still the decisive
    # signal, and without this the suffix reads as a word running into the
    # label -- which is the shape of a reference.
    if _SUFFIXED.match(text[match.end():]):
        return DECLARATION

    left = text[:match.start()]
    right = text[match.end():]
    sentence_initial = bool(_SENTENCE_END.search(left))
    referring = bool(_REFERRING.search(left))
    body_like = bool(_BODY_START.match(right))

    if referring and not sentence_initial:
        return REFERENCE

    if _REPORTING.match(right) or _RUNS_ON.match(right):
        return REFERENCE            # it mentions a provision; it is not one

    if sentence_initial and body_like:
        return CANDIDATE            # a declaration, most likely, but unproven

    if not sentence_initial:
        return REFERENCE

    return CANDIDATE

#: A citation as an answer writes it. The kind is optional, which is exactly
#: the problem: without it the reference may name several provisions.
#: What a document calls something that is NOT a provision but is numbered
#: the same way. "Table 8.4.1.5-1" and "Rule 8.4.1.5-1" are different objects
#: that share a printed number, and reading the first as a bare citation of
#: the second reported an ambiguity the writer never created -- then withheld
#: an answer over it.
_NOT_A_PROVISION = re.compile(
    r"\b(?:table|figure|section|clause|annex|appendix|chapter|equation)\s*$",
    re.I)

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
class UnlabelledBodyKey:
    """Normative body text the document did not number.

    A unit typed REQUIREMENT or INFORMATIVE by the extractor carries evidence
    and no printed label, so it has no ordinal -- and keying it (section,
    kind, None) collapses every such block in a section onto one identity.
    The old store hid this because it derived almost none of them; a parser
    that segments properly derives hundreds, and 63 false "reused citations"
    appeared the moment one did.

    It is structural evidence, like a scope preamble: not citable, and
    discriminated by the unit it came from.
    """

    section: str
    body_kind: str
    discriminator: str

    kind: str = field(default="UnlabelledBody", init=False, compare=True)

    def __str__(self):
        return f"{self.body_kind} §{self.section}@{self.discriminator[:12]}"


@dataclass(frozen=True, order=True)
class TableCellKey:
    """One cell: its table, its row, and its column.

    A row is the grouping identity; a cell is what a claim is actually about.
    The old extraction had neither -- it flattened a row into one blob, and
    on this document it interleaved two columns while doing so, so an
    approval of "bit 19 = ReqX" ended up recorded against text that also
    carried the Function column's words. Columns are named from the header
    where the table has one, and by position where it does not.
    """

    section: str
    table: str
    row: int
    column: str

    kind: str = field(default="TableCell", init=False, compare=True)

    def __str__(self):
        return f"TableCell {self.table}:{self.row}[{self.column}]"


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

    #: The strongest modal verb occurring in this provision's text. A LEXICAL
    #: fact, not a normative one: an Observation may contain "must" while
    #: describing an obligation some Rule imposes. What this provision is
    #: entitled to establish is `effective_force`, never this.
    modality: str = "NONE"
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

    #: For a table row: ordered cell records, each with its own key, text and
    #: canonical spans. A row never concatenates its columns into one
    #: normative blob.
    cells: tuple = ()

    #: Where the table this row belongs to came from. NATIVE means the parser
    #: recorded the grid and the store kept it; RECONSTRUCTED means it was
    #: inferred from what sat next to what, which is a guess and is reported
    #: as one.
    structure_origin: str = ""

    #: What the DOCUMENT calls this provision: RULE, RECOMMENDATION,
    #: PERMISSION, OBSERVATION, DEFINITION, or UNLABELLED. Structure, never
    #: words, and never inferred from a modal verb.
    provision_role: str = ""

    standard_id: str = ""
    revision: str = ""

    @property
    def role(self):
        return self.provision_role or normative_force.role_of(
            getattr(self.key, "kind", ""))

    @property
    def force(self):
        """What this provision may be used as evidence for, and why.

        Its own words, capped by what its printed role is entitled to
        establish under the taxonomy the standard declares.
        """
        return normative_force.assess(
            self.text, role=self.role,
            taxonomy=normative_force.taxonomy_for(self.standard_id, self.revision))

    @property
    def lexical_modalities(self):
        """Every modal verb in the text, whoever it belongs to."""
        return tuple(dict.fromkeys(item.word.lower() for item in self.force.lexical))

    @property
    def normative_authority(self):
        """The ceiling: the strongest force this ROLE could ever establish."""
        return self.force.authority

    @property
    def effective_force(self):
        """The ceiling and the words together. This is what grounds a claim."""
        return self.force.effective

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


#: Parts of a unit that are shaped like units and are not. A table cell has
#: text and a column; walked as a unit it became a record of its own, and a
#: two-cell row arrived as three pieces of evidence.
_NOT_UNITS = ("table_structure",)


def _collect_units(node, found):
    """Every unit-shaped dict in a payload, at any depth."""
    if isinstance(node, dict):
        if node.get("text") or node.get("snippet"):
            found.append(node)

        for name, value in node.items():
            if name in _NOT_UNITS:
                continue

            if isinstance(value, (dict, list)):
                _collect_units(value, found)
    elif isinstance(node, list):
        for item in node:
            _collect_units(item, found)


#: Where a row's table identity came from.
NATIVE = "NATIVE"
RECONSTRUCTED = "RECONSTRUCTED"


def _column_name(structure, cell):
    """What to call a cell's column: its printed header, else its position."""
    if cell.get("column"):
        return str(cell["column"])

    columns = structure.get("columns") or ()
    index = cell.get("column_index")

    if isinstance(index, int) and index < len(columns) and columns[index]:
        return str(columns[index])

    return f"col{index}" if isinstance(index, int) else "col?"


def _native_row(unit, section, common):
    """Identity for a row whose grid the store kept, or None.

    The row's own printed key is what the document numbers it by. A table
    that numbers nothing gives its rows no citable identity here, exactly as
    before -- position in a grid is not a name.
    """
    structure = unit.get("table_structure")

    if not isinstance(structure, dict) or not structure.get("table_id"):
        return None

    key = str(structure.get("row_key") or "").strip()

    if not section or not key.isdigit():
        return None

    owner = str(structure["table_id"])
    row_number = int(key)
    cells = tuple(
        {"key": TableCellKey(section, owner, row_number,
                             _column_name(structure, cell)),
         "text": _clean(cell.get("text")),
         "span_ids": tuple(cell.get("span_ids") or ())}
        for cell in (structure.get("cells") or ())
        if isinstance(cell, dict))
    text = _clean(unit.get("text") or unit.get("snippet") or "")

    return [ProvisionRecord(
        key=TableRowKey(section, owner, row_number),
        text=text, modality=_modality_of(text), cells=cells,
        structure_origin=NATIVE,
        provision_role=normative_force.role_of(TABLE_ROW), **common)]


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

    #: The grid the parser recorded, kept by the store. Preferred over every
    #: other route to a row's identity: a parser that read the table knows
    #: which table, which row and which column, and reconstructing that from
    #: adjacency throws away what it was told.
    structure = unit.get("table_structure")
    native = _native_row(unit, section, common) if structure else None

    if native is not None:
        return native

    #: "20 ReqV Request Validation ..." -- a row, keyed by its table and its
    #: own number. The legacy route, for a store whose parser recorded no
    #: grid: without the table, two tables in one section whose rows are
    #: numbered alike collide; `table` is supplied by whatever knows the
    #: caption, and a row derived with no table context falls back to its own
    #: unit so it is at least unique.
    # A unit the store DESCRIBED is never guessed at. If its grid gives it no
    # printed key, it has no citable row identity -- which is what a table
    # that numbers nothing has always had -- and inventing one from the text
    # would put the guess back in by the side door.
    row = None if structure else _TABLE_ROW.match(_clean(text))

    if row and section:
        owner = table or f"unit:{common['source_id']}"
        row_number = int(row.group("key"))
        header = unit.get("table_header") or ()
        cells = []

        for item in (unit.get("cells") or ()):
            index = item.get("col")
            name = ""

            if isinstance(index, int):
                name = (header[index] if index < len(header) and header[index]
                        else f"col{index}")

            cells.append({"key": TableCellKey(section, owner, row_number,
                                              name or "col?"),
                          "text": _clean(item.get("text")),
                          "span_ids": tuple(item.get("span_ids") or ())})

        return [ProvisionRecord(
            key=TableRowKey(section, owner, row_number),
            text=_clean(text), modality=_modality_of(text),
            cells=tuple(cells), structure_origin=RECONSTRUCTED,
            provision_role=normative_force.role_of(TABLE_ROW), **common)]

    found, spans = [], [(m.start(), m) for m in _LABEL.finditer(text)]

    claimed = set()
    statuses = {index: classify_label(text, match)
                for index, (_, match) in enumerate(spans)}

    for index, (start, match) in enumerate(spans):
        if statuses[index] == REFERENCE:
            continue                # a mention of a provision is not one

        # The provision runs to the next DECLARATION, not to the next label.
        # A reference inside it -- "Observation 6.1.2-8: Permission 6.1.2-1,
        # coupled with the other Rules, allows ..." -- is the provision's own
        # prose, and cutting there left the Observation as its label and a
        # colon while the sentence it was making went nowhere.
        following = [spans[later][0] for later in range(index + 1, len(spans))
                     if statuses[later] != REFERENCE]
        end = following[0] if following else len(text)
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
        found.append(ProvisionRecord(
            key=key, text=body, modality=_modality_of(body),
            declaration_status=statuses[index],
            provision_role=normative_force.role_of(key.kind), **common))

    # Text before the first label, or a unit with no label at all. It is
    # evidence -- a section lead-in states what its regulations are ABOUT --
    # and it gets a kind of its own rather than a invented ordinal.
    head = _clean(text[:spans[0][0]] if spans else text)

    if head and section:
        kind = (SCOPE_PREAMBLE
                if spans or _SCOPE_LEAD.match(head)
                else _FROM_CONTENT_TYPE.get(common["content_type"].upper()))

        if kind:
            # Unlabelled body text is structural evidence: it has no printed
            # ordinal, so it never becomes a citable ProvisionKey.
            key = (ScopePreambleKey(section, common["source_id"])
                   if kind == SCOPE_PREAMBLE
                   else UnlabelledBodyKey(section, kind, common["source_id"]))
            # The role is what the DOCUMENT printed, and it printed nothing
            # here. `kind` came from the extractor reading the words, which
            # is the conflation this separation exists to remove.
            found.append(ProvisionRecord(
                key=key, text=head, modality=_modality_of(head),
                provision_role=normative_force.UNLABELLED, **common))

    return found


def reconstructed_tables(units):
    """Tables inferred from what sits next to what -- the legacy fallback.

    Only for the units the store could not describe. Reconstruction is a
    guess about a relationship the parser may already have recorded, and a
    guess must never be consulted about something already known: a store
    whose rows all carry their own grid does no reconstruction at all.
    """
    import evidence_graph

    legacy = [unit for unit in units
              if isinstance(unit, dict) and not unit.get("table_structure")]

    return evidence_graph.tables_in(legacy) if legacy else ()


def records_for_units(units, *, tables=None):
    """Every provision in a set of units, with table context applied.

    A unit that carries its own grid needs nothing from here. For the rest,
    which table a row belongs to is a property of the run it sits in, not of
    the unit: `tables` is whatever knows, and when nothing does the legacy
    reconstruction is run over the units that need it.
    """
    if tables is None:
        tables = reconstructed_tables(units)

    owner = {}

    for table in (tables or ()):
        identity = table_identity(table.caption)

        for row in table.rows:
            owner[str(row.get("source_id") or "")] = identity

    # A unit may name its own table. A parser that already knows the table
    # structure should say so rather than have it reconstructed from
    # adjacency: the reconstruction exists because the old extractor lost the
    # relationship, and guessing is not better than being told.
    for unit in units:
        if isinstance(unit, dict) and unit.get("table"):
            owner[str(unit.get("source_id") or "")] = str(unit["table"])

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
    def group_key(record):
        # Unlabelled body text is discriminated per unit, so two fragments of
        # ONE split provision never share a key. They are still one
        # provision, so they are grouped by what they have in common --
        # section and body kind -- and the strict conditions below decide.
        if isinstance(record.key, UnlabelledBodyKey):
            return ("unlabelled", record.key.section, record.key.body_kind)

        return record.key

    grouped = {}

    for record in records:
        grouped.setdefault(group_key(record), []).append(record)

    found, joined = [], set()

    for key, group in grouped.items():
        unlabelled = isinstance(key, tuple)

        if len(group) != 2 or not (unlabelled or isinstance(key, ProvisionKey)):
            found.extend(group)
            continue

        first, second = sorted(group, key=lambda r: (r.page or 0))
        pages = [first.page or 0, second.page or 0]
        declares_again = (False if unlabelled
                          else second.text.startswith(f"{key.kind} {key.section}-"))

        if (abs(pages[1] - pages[0]) == 1 and _INCOMPLETE.search(first.text)
                and not declares_again):
            merged = first.text + " " + second.text
            found.append(ProvisionRecord(
                key=first.key, source_id=first.source_id, page=first.page,
                text=merged, modality=_modality_of(merged),
                unit_modality=first.unit_modality,
                content_type=first.content_type,
                needs_review=first.needs_review or second.needs_review,
                unit_text=first.unit_text,
                spans=tuple(first.spans) + tuple(second.spans),
                standard_id=first.standard_id, revision=first.revision,
                declaration_status=first.declaration_status))
            joined.add(first.key)
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
        units = []
        _collect_units(payload, units)
        self._add(records_for_units(units))

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

            # "Table 8.4.1.5-1" names a table. The document numbers its
            # tables and its provisions alike, and treating the one as a
            # citation of the other invents an ambiguity between provisions
            # the writer never mentioned.
            if not match.group("kind") and _NOT_A_PROVISION.search(
                    (text or "")[:match.start()]):
                continue

            try:
                key = self.resolve(match.group(0))
            except AmbiguousReference as problem:
                ambiguous.append(problem)
                continue

            if key is not None:
                resolved.append(key)

        return resolved, ambiguous
