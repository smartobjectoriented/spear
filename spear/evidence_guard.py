"""What a final answer may assert about a structure the evidence leaves open.

H4 survives every model-side intervention: five adapters, serving scale to 32x,
an explicit system-prompt rule, and withholding the precomputed complement from
the tool payload. In the last of those the model simply did the subtraction
itself and said so. The distinction it keeps losing is not arithmetic but
normative:

    supported    bits 31..16 are unclaimed by the established fields
    unsupported  the label "Reserved" occupies bits 31..16

The first follows from a stated word width and a stated range. The second binds
an unresolved normative label to a concrete range, which no evidence
establishes. So the boundary is enforced where it can be enforced
deterministically -- on the finished answer, after every tool call, out of the
model's sight -- rather than asked for again.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import diagram_geometry
import evidence_fetch
import normative_dimensions
import structural_entailment
from structural_entailment import (
    DETERMINISTIC_ENTAILMENT, EXPLICIT, REJECTED, Member, unique_placement,
)

# A label bound to a range, in the three places an answer puts one.
_C_MEMBER = re.compile(r"\b(?:u?int\d*_t|unsigned|signed|char|short|long)\s+"
                       r"([A-Za-z_]\w*)\s*:\s*(\d+)\s*;")
_LABEL_THEN_RANGE = re.compile(
    r"([A-Za-z][\w ,/\-]{1,44}?)\s*(?:\||:|=|-|—)?\s*"
    r"(?:occupies|occupy|spans?|covers?|at|is|are|in)?\s*"
    r"\(?\s*bits?\s*\[?\s*(\d{1,2})\s*(?:\.\.|:|-|–)\s*(\d{1,2})\s*\]?\s*\)?",
    re.I)
_RANGE_THEN_LABEL = re.compile(
    r"\(?\s*bits?\s*\[?\s*(\d{1,2})\s*(?:\.\.|:|-|–)\s*(\d{1,2})\s*\]?\s*\)?"
    r"\s*(?:are|is|shall be|=|:|->|→)?\s*(?:the\s+)?"
    r"([A-Za-z][\w ,/\-]{1,44}?)\b", re.I)
_WIDTH = re.compile(r"([A-Za-z][\w ,/\-]{1,44}?)\s*(?:is|has|:|=)?\s*"
                    r"(\d{1,3})\s*bits?\b", re.I)
# A table row states the range bare, with no "bits" to key on.
_TABLE_ROW = re.compile(r"^\s*\|\s*([A-Za-z][^|\n]{0,44}?)\s*\|"
                        r"[^|\n]*?\[?\s*(\d{1,2})\s*(?:\.\.|:|-|–)\s*"
                        r"(\d{1,2})\s*\]?[^|\n]*(?:\||$)", re.M)

# Words that describe a hole rather than name a normative field. A range
# attached to one of these is an arithmetic observation, which is allowed.
_ARITHMETIC = ("unclaimed", "unassigned", "unaccounted", "not claimed",
               "remaining", "remainder", "free", "undefined", "unspecified",
               "unresolved", "not established", "no stated", "gap", "unused by",
               "outside", "gap of", "left over", "leftover")

# Structure-member names a model reaches for when it invents a region.
_INVENTED = ("reserved", "rsvd", "padding", "pad", "spare", "vendor",
             "reserved2", "unused", "filler")

_NOISE = ("word", "bit", "bits", "field", "fields", "structure", "struct",
          "the", "a", "an", "of", "in", "at", "and", "or", "is", "are", "that",
          "this", "it", "they", "there", "value", "values", "range", "ranges",
          "total", "width", "layout", "offset", "note", "notes")

# An octet position, in the two places an answer puts one: a serialized-layout
# table, and a sentence. Bits and octets are separate coordinate systems, so
# they are read separately and never compared with each other.

_VERB = (r"occupies|occupy|spans?|covers?|fills?|filling|begins? at|"
         r"starts? at|sits? at|is at|located at|at|is|are|in|of")
_UNIT = r"(?:octet|byte|position)s?"
_RANGE_SEP = r"(?:\.\.|:|-|–|through|thru|to)"

# A label bound to octets, in the forms an answer writes them. Every one is
# anchored on the unit word or on "offset", so a bare pair of numbers never
# becomes a placement.

_OCTET_SENTENCE = re.compile(
    rf"([A-Za-z][\w ,/()\-]{{1,44}}?)\s*(?:\||:|=|-|—)?\s*"
    rf"(?:{_VERB})?\s*(?:the\s+)?(?:remaining\s+|free\s+|spare\s+)?"
    rf"(?:range\s+(?:from\s+)?)?"
    rf"\(?\s*(?:{_UNIT}\s*(?:offset\s*)?|offset\s+)\[?\s*(\d{{1,4}})"
    rf"\s*(?:{_RANGE_SEP}\s*(\d{{1,4}}))?\s*\]?\)?", re.I)

# "... from offset 4 to 7", where the label sits further back than the label
# group above will reach across an intervening clause.

_OCTET_FROM_TO = re.compile(
    rf"\bfrom\s+(?:{_UNIT}\s+)?(?:offset\s+)?(\d{{1,4}})\s*"
    rf"(?:{_RANGE_SEP})\s*(\d{{1,4}})", re.I)

# "Reserved begins at offset 4 and has length 4" / "Reserved: offset 4, length 4"
_OCTET_OFFSET_LENGTH = re.compile(
    rf"\boffset\s*[:=]?\s*(\d{{1,4}})\b[^.\n]{{0,30}}?"
    rf"\b(?:length|size|extent)\s*(?:of\s*)?[:=]?\s*(\d{{1,4}})\b", re.I)


# Denial of a fact the evidence states. Only an explicit denial counts; an
# answer that simply does not mention a field has not contradicted anything.

_DENIAL = ("no defined", "not defined", "does not define", "is not defined",
           "are not defined", "no word number", "no bit range", "no offset",
           "does not establish", "not established", "does not specify",
           "not specified", "does not state", "not stated", "no structure",
           "no approved structure", "cannot be determined", "is undefined",
           "are undefined", "no defined word", "nothing is defined",
           "does not assign", "not assigned by the standard")

# ... and it has to be the position that is denied. "The standard does not
# specify the content or format of the Primary Value beyond its size and
# position" denies neither, and firing on it would punish an accurate answer.

_POSITIONAL = (r"word(?:\s+(?:number|index))?s?|bits?|octets?|offsets?|"
               r"positions?|ranges?|layout|structures?|placement")
_DENIES_POSITION = re.compile(
    r"(?:" + "|".join(re.escape(tell) for tell in _DENIAL) + r")"
    r"[^.\n]{0,24}?\b(?:" + _POSITIONAL + r")\b", re.I)

# The denial alone. Which dimension it is about -- and whether it is about one
# at all -- is decided by the shared resolver, which reads the subject on
# either side of the phrase. _DENIES_POSITION only ever looked forwards, so
# "the word number for Coarse Time is not specified" never reached it.

_DENIAL_ONLY = re.compile(
    "|".join(re.escape(tell) for tell in _DENIAL), re.I)

CONTRADICTS_ESTABLISHED_EVIDENCE = "CONTRADICTS_ESTABLISHED_EVIDENCE"
CELL_LOCAL_RANGE_USED_AS_GLOBAL = "CELL_LOCAL_RANGE_USED_AS_GLOBAL"

# "occupies the upper half" places a field without printing a number, so the
# range patterns above never see it.

_HALF_CLAIM = re.compile(
    r"([A-Za-z][\w ,/\-]{1,44}?)\s*(?:\||:|=|-|—)?\s*"
    r"(?:occupies|occupy|is|are|sits? in|lies? in|takes?|uses?)\s*"
    r"(?:the\s+)?(?:physical\s+)?(upper|lower|high|low|most significant|"
    r"least significant)\s*(?:physical\s+)?(?:half|16 bits|16)", re.I)


# Answers arrive as markdown. "**Vertical Beamwidth**: occupies bits 31..16"
# is the same claim as the unadorned sentence, but the emphasis characters sit
# between the name and the range and every label pattern below stops on them --
# so the claim above was read with the label "occupies". Emphasis is replaced
# by spaces rather than removed, so every span stays where it was and the
# arithmetic-context window keeps looking at the right words.

# Underscore is deliberately absent: it is markdown emphasis, but it is also
# what holds "uint32_t reserved" together, and H4 is caught by exactly that
# declaration.

_EMPHASIS = re.compile(r"[*`~]")


def _readable(text):
    """The answer with its markdown flattened, character positions intact."""
    return _EMPHASIS.sub(" ", text or "")


def _key(label):
    """Compare labels the way a reader would: case, spacing and punctuation off."""
    return re.sub(r"[^a-z0-9]+", "", (label or "").lower())


def _tail(label):
    """The last few words of a captured span, where the actual name sits."""
    words = [w for w in re.split(r"[\s,/]+", (label or "").strip()) if w]

    while words and words[0].lower() in _NOISE:
        words.pop(0)

    return " ".join(words[-6:])


@dataclass
class EvidenceLedger:
    """Everything the session's tool results actually established."""

    established: dict = field(default_factory=dict)
    unresolved: dict = field(default_factory=dict)
    word_widths: dict = field(default_factory=dict)
    structures: list = field(default_factory=list)
    sources: list = field(default_factory=list)
    incomplete: bool = False

    # Facts a fetched requirement states outright. Kept apart from the
    # get_structure facts above so that a session with no fetch evidence
    # behaves exactly as it did before FT3F.

    octet_fields: dict = field(default_factory=dict)
    word_fields: dict = field(default_factory=dict)

    # Ranges a diagram prints inside one cell. Deliberately not merged into
    # `established`, which means a position in the enclosing word: a cell's
    # range is local to the cell until something says how the cells compose.

    cell_fields: dict = field(default_factory=dict)
    members: dict = field(default_factory=dict)
    record: dict = field(default_factory=dict)
    fetch_facts: list = field(default_factory=list)

    def observe(self, payload):
        """Read one standard.get_structure result. The payload is not mutated."""
        if not isinstance(payload, dict):
            return self

        definition = payload.get("definition_id")

        if definition and definition not in self.structures:
            self.structures.append(definition)

        for source in payload.get("citation_source_ids") or ():
            if source not in self.sources:
                self.sources.append(source)

        if payload.get("structural_completeness") == "STRUCTURALLY_INCOMPLETE":
            self.incomplete = True

        fields = (payload.get("fields")
                  or payload.get("established_fields") or ())

        for item in fields:
            label = item.get("display_label") or item.get("normative_label")

            if not label:
                continue

            if item.get("msb") is None or item.get("position_source") == "UNKNOWN":
                self.unresolved[_key(label)] = label
            else:
                self.established[_key(label)] = {
                    "label": label, "msb": item["msb"], "lsb": item["lsb"],
                    "width": item.get("width"),
                    "word_index": item.get("word_index"),
                    "source": item.get("normative_source_id")}

        for item in payload.get("unresolved") or ():
            if item.get("label"):
                self.unresolved[_key(item["label"])] = item["label"]

        for word in payload.get("words") or ():
            index = word.get("word_index")

            if index is not None and word.get("word_width"):
                self.word_widths[index] = word["word_width"]

            for label in word.get("unpositioned_labels") or ():
                self.unresolved[_key(label)] = label

            for item in word.get("unresolved_labels") or ():
                if item.get("label"):
                    self.unresolved[_key(item["label"])] = item["label"]

        return self

    def observe_fetch(self, payload):
        """Read one standard.fetch result. The payload is not mutated.

        Only what the requirement says is stored. A length without an offset
        stays a length: deriving the offset from the record's spare octets is
        arithmetic about a hole, not an assignment, and the whole point of the
        ledger is that it cannot tell itself where anything goes.
        """
        for fact in evidence_fetch.unit_facts(payload):
            self._record_fact(fact)

        return self

    def _record_fact(self, fact):
        key = fact.key

        if not key:
            return

        self.fetch_facts.append(fact)

        if fact.source_id and fact.source_id not in self.sources:
            self.sources.append(fact.source_id)

        if fact.fact_class == evidence_fetch.MEMBERSHIP:
            self.members.setdefault(key, fact.label)
            return

        if fact.fact_class in (evidence_fetch.RECORD_LENGTH,
                               evidence_fetch.RECORD_WORD_COUNT):
            self.record.setdefault("label", fact.label)

            for name in ("octets", "words", "word_width"):
                value = getattr(fact, name)

                if value is not None:
                    self.record[name] = value

            return

        if fact.fact_class == evidence_fetch.FIELD_WORD_INDEX:
            self.word_fields[key] = {
                "label": fact.label, "word_index": fact.word_index,
                "provenance": fact.provenance}
            return

        if fact.fact_class == evidence_fetch.FIELD_RANGE:
            self.established.setdefault(key, {
                "label": fact.label, "msb": fact.msb, "lsb": fact.lsb,
                "width": fact.msb - fact.lsb + 1, "word_index": None,
                "source": fact.source_id})
            return

        if fact.fact_class in (evidence_fetch.FIELD_OFFSET,
                               evidence_fetch.FIELD_LENGTH):
            slot = self.octet_fields.setdefault(
                key, {"label": fact.label, "offset": None, "length": None,
                      "provenance": fact.provenance})
            self.members.setdefault(key, fact.label)

            if fact.offset is not None:
                slot["offset"] = fact.offset
                slot["provenance"] = fact.provenance

            if fact.length is not None:
                slot["length"] = fact.length

    def observe_cells(self, cells):
        """Read the diagram cells behind one fetched unit. Nothing is derived."""
        for cell in cells or ():
            if not cell.key:
                continue

            self.cell_fields.setdefault(cell.key, {
                "label": cell.label, "msb": cell.msb, "lsb": cell.lsb,
                "column": cell.column, "provenance": cell.provenance})

            if cell.source_id and cell.source_id not in self.sources:
                self.sources.append(cell.source_id)

        return self

    def positioned_octets(self):
        """Octets the evidence actually places, as a set."""
        covered = set()

        for item in self.octet_fields.values():
            if item["offset"] is None or not item["length"]:
                continue

            covered |= set(range(item["offset"],
                                 item["offset"] + item["length"]))

        return covered

    def knows_anything(self):
        return bool(self.established or self.unresolved or self.octet_fields
                    or self.word_fields or self.record or self.cell_fields)


def _arithmetic_context(text, start, end):
    """Is this range described as a hole rather than assigned to a name?"""
    window = text[max(0, start - 120):min(len(text), end + 120)].lower()

    return any(tell in window for tell in _ARITHMETIC)


def _claims(text):
    """Every concrete (label, range-or-width) assertion the answer makes."""
    found = []

    for match in _C_MEMBER.finditer(text):
        found.append({"label": match.group(1), "kind": "STRUCTURE_MEMBER",
                      "detail": f"{match.group(1)} : {match.group(2)}",
                      "span": match.span()})

    for pattern, order in ((_LABEL_THEN_RANGE, "label_first"),
                           (_RANGE_THEN_LABEL, "range_first")):
        for match in pattern.finditer(text):
            if order == "label_first":
                label, high, low = match.group(1), match.group(2), match.group(3)
            else:
                high, low, label = match.group(1), match.group(2), match.group(3)

            found.append({"label": _tail(label), "kind": "FIELD_RANGE",
                          "detail": f"bits {high}..{low}",
                          "msb": int(high), "lsb": int(low),
                          "span": match.span()})

    for match in _TABLE_ROW.finditer(text):
        found.append({"label": _tail(match.group(1)), "kind": "FIELD_RANGE",
                      "detail": f"bits {match.group(2)}..{match.group(3)}",
                      "msb": int(match.group(2)), "lsb": int(match.group(3)),
                      "span": match.span()})

    for match in _WIDTH.finditer(text):
        found.append({"label": _tail(match.group(1)), "kind": "FIELD_WIDTH",
                      "detail": f"{match.group(2)} bits",
                      "width": int(match.group(2)), "span": match.span()})

    return found


def _table_octet_claims(text):
    """Rows of a serialized-layout table, read by what its header says.

    The column is identified from the header rather than by position, so a
    table that puts length before offset is read correctly and a table that
    names neither is not read at all.
    """
    found = []
    rows = []

    for line in text.splitlines(keepends=False):
        if line.strip().startswith("|"):
            rows.append(line)
            continue

        found.extend(_one_table(rows, text))
        rows = []

    found.extend(_one_table(rows, text))

    return found


def _cells(line):
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


# What a serialized-layout table calls its columns. The label column is found
# the same way as the others: a table that names none of them is not read.

_LABEL_HEADS = ("component", "field", "member", "name", "item", "subfield",
                "element")
_OFFSET_HEADS = ("offset", "position", "start", "octet", "byte")
_LENGTH_HEADS = ("length", "size", "extent", "width", "octets", "bytes")


def _column(header, wanted):
    for index, cell in enumerate(header):
        if any(word in cell for word in wanted):
            return index

    return None


def _one_table(rows, whole):
    """One markdown table's placements, read by what its header says.

    The columns are identified from their headings rather than by position, so
    "| Offset | Length | Component |" is read as readily as
    "| Component | Offset | Length |". Without a heading for both the label and
    the offset there is nothing to read, and guessing which column is which is
    exactly what would put a wrong label on a right number.
    """
    if len(rows) < 2:
        return []

    header = [cell.lower() for cell in _cells(rows[0])]
    offset_at = _column(header, _OFFSET_HEADS)
    label_at = _column(header, _LABEL_HEADS)
    length_at = _column(header, _LENGTH_HEADS)

    if offset_at is None or label_at is None or label_at == offset_at:
        return []

    found = []

    for line in rows[1:]:
        cells = _cells(line)

        if max(offset_at, label_at) >= len(cells):
            continue

        if set(cells[label_at]) <= set("-: ") or not cells[label_at]:
            continue

        digits = re.match(r"^(\d{1,4})\b", cells[offset_at])

        if digits is None:
            continue

        length = None

        if length_at is not None and length_at < len(cells):
            size = re.match(r"^(\d{1,4})\b", cells[length_at])
            length = int(size.group(1)) if size else None

        at = whole.find(line)
        found.append({"label": _tail(cells[label_at]), "kind": "OCTET_OFFSET",
                      "detail": f"octet offset {digits.group(1)}",
                      "offset": int(digits.group(1)), "length": length,
                      "span": (at, at + len(line)) if at >= 0 else (0, 0)})

    return found


def _label_before(text, at, span=110):
    """The name the sentence gave, when the number sits away from it."""
    head = text[max(0, at - span):at]
    head = head.split("\n")[-1]

    return _tail(re.sub(r"[^A-Za-z ]+", " ", head))


def _octet_claims(text):
    """Every concrete octet placement the answer makes."""
    found = _table_octet_claims(text)
    taken = [claim["span"] for claim in found]

    def add(label, low, length, span):
        if any(span[0] < end and start < span[1] for start, end in taken):
            return

        taken.append(span)
        found.append({"label": label, "kind": "OCTET_OFFSET",
                      "detail": f"octet offset {low}", "offset": low,
                      "length": length, "span": span})

    for match in _OCTET_OFFSET_LENGTH.finditer(text):
        add(_label_before(text, match.start()), int(match.group(1)),
            int(match.group(2)), match.span())

    for match in _OCTET_SENTENCE.finditer(text):
        high = match.group(3)
        low = int(match.group(2))
        length = (int(high) - low + 1) if high is not None else None
        add(_tail(match.group(1)), low, length, match.span())

    for match in _OCTET_FROM_TO.finditer(text):
        low, high = int(match.group(1)), int(match.group(2))
        add(_label_before(text, match.start()), low, high - low + 1,
            match.span())

    return found


def _sentences(text):
    return [part for part in re.split(r"(?<=[.!?])\s+|\n+", text or "")
            if part.strip()]


def _placed(ledger):
    """Every fact a denial could contradict, with the dimension it is about."""
    found = {}

    for key, item in ledger.word_fields.items():
        found[key] = (item["label"], normative_dimensions.WORD,
                      f"word {item['word_index']}")

    for key, item in ledger.octet_fields.items():
        if item["offset"] is not None:
            found[key] = (item["label"], normative_dimensions.OFFSET,
                          f"octet offset {item['offset']}")

    for key, item in ledger.established.items():
        if item.get("msb") is not None:
            found.setdefault(key, (item["label"], normative_dimensions.BITS,
                                   f"bits {item['msb']}..{item['lsb']}"))

    return found


def _denials(text, ledger):
    """Sentences that deny a fact the evidence states outright.

    A denial is about one dimension, and it can only contradict a fact in that
    same dimension. "The standard does not specify the bit ranges within those
    words" denies the bits; the word numbers beside it are untouched, and
    reading it as a retraction of them -- which this did before FT4D -- throws
    away a correct answer. The dimension is resolved by the same rule the
    evaluation oracle uses, imported rather than restated.

    An omission is still not a denial: the sentence has to name the field. And
    a denial whose dimension cannot be resolved contradicts nothing, because
    matching it to a fact on the strength of the label alone is exactly the
    over-reach being fixed.
    """
    violations = []
    placed = _placed(ledger)

    if not placed:
        return violations

    for sentence in _sentences(text):
        lowered = sentence.lower()

        for match in _DENIAL_ONLY.finditer(lowered):
            denied = normative_dimensions.dimension_of(lowered, match.end())

            if denied is None:
                continue

            for label, dimension, where in placed.values():
                if dimension != denied or label.lower() not in lowered:
                    continue

                at = text.find(sentence)
                violations.append({
                    "label": label, "kind": CONTRADICTS_ESTABLISHED_EVIDENCE,
                    "detail": sentence.strip()[:160],
                    "span": (at, at + len(sentence)) if at >= 0 else (0, 0),
                    "reason": f"the evidence establishes {label} at {where}"})

    return violations


def validate(answer, ledger):
    """Unsupported concrete structural claims in a finished answer."""
    text = _readable(answer)
    violations = []

    if not ledger.knows_anything():
        return violations

    for claim in _claims(text):
        key = _key(claim["label"])

        if not key:
            continue

        if claim["kind"] != "STRUCTURE_MEMBER" and _arithmetic_context(
                text, *claim["span"]):
            # "bits 31..16 remain unclaimed" states a gap, not an assignment.
            continue

        known = ledger.established.get(key)

        if known:
            if claim["kind"] == "FIELD_RANGE" and (
                    claim["msb"] != known["msb"] or claim["lsb"] != known["lsb"]):
                violations.append({**claim, "reason": "range contradicts the "
                                                      "established position"})

            continue

        if key in ledger.unresolved:
            violations.append({**claim, "reason": "the label is present in the "
                                                  "evidence with no established "
                                                  "position or width"})
            continue

        if any(word in key for word in _INVENTED):
            violations.append({**claim, "reason": "a structural region the "
                                                  "evidence does not name"})

    violations.extend(_octet_violations(text, ledger))
    violations.extend(_cell_violations(text, ledger))
    violations.extend(_denials(text, ledger))

    return violations


_UPPER = ("upper", "high", "most significant")


def _names_the_established_half(label, half, ledger):
    """Does the structure place this field in exactly the half being claimed?

    A field the evidence positions at bits 31..16 of a 32-bit word is in the
    upper half, and saying so is a restatement rather than an inference. Any
    other half is not, and falls through to the refusal below.
    """
    placed = ledger.established.get(
        _resolve(_key(label), ledger.established) or "")

    if not placed or placed.get("msb") is None:
        return False

    width = ledger.word_widths.get(placed.get("word_index")) or 32
    upper = placed["msb"] == width - 1 and placed["lsb"] == width // 2
    lower = placed["msb"] == width // 2 - 1 and placed["lsb"] == 0

    return upper if half.lower() in _UPPER else lower


def _cell_violations(text, ledger):
    """Cell-local diagram ranges used as positions in the enclosing word.

    A packet diagram prints each cell's range inside that cell. Two cells that
    both print (15..0) are two sixteen-bit quantities, not one 32-bit word
    split between them, and reading the second as the upper half is a claim the
    figure does not make. Restating a cell's own range is fine; moving it is
    not.
    """
    if not ledger.cell_fields:
        return []

    violations = []
    seen = set()

    def refuse(claim, known, detail):
        key = (known["label"], detail)

        if key in seen:
            return

        seen.add(key)
        violations.append({
            **claim, "kind": CELL_LOCAL_RANGE_USED_AS_GLOBAL,
            "detail": detail,
            "reason": f"the diagram prints {known['label']} with the range "
                      f"{known['msb']}..{known['lsb']} inside its own cell; no "
                      f"evidence places that cell in the enclosing word"})

    for claim in _claims(text):
        if claim["kind"] != "FIELD_RANGE":
            continue

        key = _resolve(_key(claim["label"]), ledger.cell_fields)
        known = ledger.cell_fields.get(key) if key else None

        if known is None:
            continue

        # A field the approved structure positions is positioned. Its diagram
        # cell still prints a value-local range, and a session that read both
        # would otherwise have the structure's own answer refused as if it were
        # the F2 error. Only the position the structure actually establishes is
        # allowed through: any other range is still a claim about a cell, and
        # is still refused here.

        placed = ledger.established.get(
            _resolve(_key(claim["label"]), ledger.established) or "")

        if placed and claim["msb"] == placed["msb"] and (
                claim["lsb"] == placed["lsb"]):
            continue

        if claim["msb"] == known["msb"] and claim["lsb"] == known["lsb"]:
            # The answer is repeating what the cell prints, which is supported.
            continue

        refuse(claim, known, claim["detail"])

    for match in _HALF_CLAIM.finditer(text):
        label = _tail(match.group(1))
        key = _resolve(_key(label), ledger.cell_fields)
        known = ledger.cell_fields.get(key) if key else None

        if known is None:
            continue

        # The same precedence as above, in the wording that carries no numbers.
        # "Stage 2 Gain occupies the upper half" restates a position the
        # structure establishes; only the half it actually places the field in
        # is allowed through.

        if _names_the_established_half(label, match.group(2), ledger):
            continue

        refuse({"label": label, "span": match.span()}, known,
               f"the {match.group(2).lower()} half of the word")

    return violations


def _resolve(key, known):
    """The ledger entry an answer's wording refers to, when exactly one does.

    An answer writes "Reserved (transport)" for what the requirement calls
    "reserved". One containing the other is the same field; two candidates
    means the wording is ambiguous, and an ambiguous label is not matched.
    """
    if key in known:
        return key

    hits = [name for name in known if name and (name in key or key in name)]

    return hits[0] if len(hits) == 1 else None


def _members(ledger):
    return [Member(item["label"], item["offset"], item["length"])
            for item in ledger.octet_fields.values()]


def _octet_paths(text, ledger):
    """Every octet placement in the answer, with the path that decided it.

    Three outcomes, in order: the evidence states the position; it does not,
    but the established constraints leave exactly one interval the claim could
    occupy; or neither, in which case the claim is unsupported.
    """
    if not (ledger.octet_fields or ledger.record):
        return []

    total = ledger.record.get("octets")
    decided = []

    for claim in _octet_claims(text):
        key = _resolve(_key(claim["label"]), ledger.octet_fields) or _key(
            claim["label"])
        known = ledger.octet_fields.get(key)
        named = key in ledger.members or known is not None or any(
            word in key for word in _INVENTED)

        if not key:
            continue

        # An unlabelled range is an observation about the container. Once a
        # normative label is bound to it, the wording around it stops
        # mattering: "Reserved fills the remaining octets 4..7" places
        # Reserved, whatever "remaining" suggests.

        outside = total is not None and (
            claim["offset"] >= total
            or claim["offset"] + (claim["length"] or 1) > total)

        if not named:
            # Octets the record does not have are refused whatever they are
            # called, and before the arithmetic reading is even considered:
            # there is no observation to make about them.

            if outside:
                decided.append({**claim, "path": REJECTED, "key": key,
                                "reason": f"the claimed octets lie outside the "
                                          f"established {total}-octet record"})
                continue

            if _arithmetic_context(text, *claim["span"]):
                continue

            decided.append({**claim, "path": REJECTED, "key": key,
                            "reason": "a structural region the evidence does "
                                      "not name"})
            continue

        if known is not None and known["offset"] is not None:
            if claim["offset"] == known["offset"]:
                decided.append({**claim, "path": EXPLICIT, "key": key,
                                "reason": ""})
            else:
                decided.append({**claim, "path": REJECTED, "key": key,
                                "reason": "offset contradicts the established "
                                          "position"})
            continue

        proof = unique_placement(
            Member(known["label"] if known else claim["label"], None,
                   known["length"] if known else None),
            _members(ledger), total)

        if proof.entailed and claim["offset"] == proof.low and (
                claim["length"] is None or claim["length"] == proof.width):
            decided.append({**claim, "path": DETERMINISTIC_ENTAILMENT,
                            "key": key, "reason": "",
                            "proof": {"interval": [proof.low, proof.high],
                                      "extent": proof.extent,
                                      "occupied": [list(pair) for pair
                                                   in proof.occupied],
                                      "width": proof.width}})
            continue

        if outside:
            reason = (f"the claimed octets lie outside the established "
                      f"{total}-octet record")
        elif proof.entailed:
            reason = (f"the only placement the evidence forces is octets "
                      f"{proof.low}..{proof.high}")
        else:
            reason = ("the label is present in the evidence with no "
                      f"established octet offset, and none is forced "
                      f"({proof.reason})")

        decided.append({**claim, "path": REJECTED, "key": key,
                        "reason": reason})

    return decided


def _octet_violations(text, ledger):
    """Octet placements the fetched evidence does not support."""
    return [{key: value for key, value in claim.items() if key != "path"}
            for claim in _octet_paths(text, ledger)
            if claim["path"] == REJECTED]


def entailed_octets(ledger):
    """Members with no stated offset whose position the constraints force."""
    total = ledger.record.get("octets")
    found = {}

    for key, item in ledger.octet_fields.items():
        if item["offset"] is not None:
            continue

        proof = unique_placement(Member(item["label"], None, item["length"]),
                                 _members(ledger), total)

        if proof.entailed:
            found[key] = (proof.low, proof.high)

    return found


def _fetch_lines(ledger):
    """The fetched facts, each written as narrowly as the evidence states it."""
    lines = []
    record = ledger.record

    if record.get("octets") is not None:
        lines.append(f"- {record.get('label', 'the record')} — a "
                     f"{record['octets']}-octet record")

    if record.get("words") is not None:
        width = record.get("word_width")
        lines.append(f"- {record.get('label', 'the record')} — {record['words']}"
                     f" words" + (f" of {width} bits each" if width else ""))

    entailed = entailed_octets(ledger)

    for key, item in sorted(ledger.octet_fields.items(),
                            key=lambda pair: (pair[1]["offset"] is None,
                                              pair[1]["offset"] or 0,
                                              pair[1]["label"])):
        length = f", length {item['length']} octets" if item["length"] else ""

        if item["offset"] is not None:
            lines.append(f"- {item['label']} — octet offset "
                         f"{item['offset']}{length}")
        elif key in entailed:
            low, high = entailed[key]
            lines.append(f"- {item['label']} — octets {low}..{high}{length}; "
                         f"the evidence gives no offset for it, and this is "
                         f"the only interval its stated length can occupy")
        elif item["length"]:
            lines.append(f"- {item['label']} — length {item['length']} octets, "
                         f"given without an octet offset")

    for item in sorted(ledger.word_fields.values(),
                       key=lambda f: (f["word_index"], f["label"])):
        lines.append(f"- {item['label']} — word {item['word_index']}")

    for item in sorted(ledger.cell_fields.values(),
                       key=lambda f: (f["column"], f["label"])):
        lines.append(f"- {item['label']} — the diagram prints bits "
                     f"{item['msb']}..{item['lsb']} within this field's own "
                     f"cell (page {item['provenance']['page']})")

    return lines


def _octet_observations(ledger):
    """Octets no positioned component covers. Arithmetic, not an assignment."""
    total = ledger.record.get("octets")

    if total is None:
        return []

    covered = set(ledger.positioned_octets())

    for low, high in entailed_octets(ledger).values():
        covered |= set(range(low, high + 1))

    free = sorted(set(range(total)) - covered)

    if not free or len(free) == total:
        return []

    return [f"- octets {free[0]}..{free[-1]} fall outside the components the "
            f"evidence positions."]


def safe_rendering(ledger, violations=()):
    """A grounded partial answer: what is established, what is not, and why."""
    lines = []
    fetched = _fetch_lines(ledger)

    if ledger.established or fetched:
        lines.append("Established by the normative evidence:")
        lines.extend(fetched)

        for item in sorted(ledger.established.values(),
                           key=lambda f: (f["word_index"] or 0, -(f["msb"] or 0))):
            where = (f" of word {item['word_index']}"
                     if item["word_index"] is not None else "")
            lines.append(f"- {item['label']} — bits {item['msb']}..{item['lsb']}"
                         f"{where}")

    if ledger.unresolved:
        lines.append("")
        lines.append("Unresolved:")

        for label in sorted(ledger.unresolved.values()):
            lines.append(f"- {label} — normative position and width are not "
                         f"established.")

    claimed = {}

    for item in ledger.established.values():
        index = item["word_index"]
        claimed.setdefault(index, []).append((item["msb"], item["lsb"]))

    observations = []

    for index, ranges in sorted(claimed.items()):
        width = ledger.word_widths.get(index)

        if not width:
            continue

        covered = set()

        for high, low in ranges:
            covered |= set(range(low, high + 1))

        free = sorted(set(range(width)) - covered, reverse=True)

        if free:
            observations.append(f"- word {index}: bits {free[0]}..{free[-1]} "
                                f"are currently unclaimed by the established "
                                f"fields.")

    observations.extend(_octet_observations(ledger))

    if observations:
        lines.append("")
        lines.append("Arithmetic observation (not a normative assignment):")
        lines.extend(observations)

    if ledger.unresolved:
        lines.append("")
        lines.append("Constraint: the available evidence does not establish "
                     "that any unresolved label occupies those bits, so no "
                     "complete structure definition can be produced from it.")

    if len(ledger.cell_fields) > 1:
        lines.append("")
        lines.append("Constraint: each range above is stated inside its own "
                     "diagram cell, and the cells are printed in separate "
                     "columns. The available evidence does not establish how "
                     "those cells are placed within an enclosing word, so it "
                     "does not establish that either field occupies the upper "
                     "or the lower half of one.")

    if ledger.sources:
        lines.append("")
        lines.append("Sources: " + ", ".join(ledger.sources))

    return "\n".join(lines)


def explain(answer, ledger):
    """Which path allowed or refused every structural claim in the answer."""
    return [{"label": claim["label"], "detail": claim["detail"],
             "path": claim["path"], "reason": claim["reason"],
             "proof": claim.get("proof")}
            for claim in _octet_paths(_readable(answer), ledger)]


def guard(answer, ledger):
    """Return the answer, or a grounded replacement, plus what was found."""
    violations = validate(answer, ledger)

    if not violations:
        return answer, violations, False

    return safe_rendering(ledger, violations), violations, True
