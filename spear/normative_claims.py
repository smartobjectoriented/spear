"""What a normative answer CLAIMS, checked against what the evidence SAYS.

The guards that came before this one all ask the same question: was supporting
evidence retrieved? None of them asks whether it agrees. That gap let three
distinct failures through on one battery:

  * an answer opened "Yes, an Acknowledge packet may carry more than one
    subtype indicator", quoted the rule that says it shall carry only one,
    and concluded correctly two sentences later;
  * an answer named a field that occurs nowhere in the document;
  * an answer asked for a maximum the document never states, counted the
    request flags, and reported their number as the limit.

Each was grounded by the older guards, because in each case the right clause
had been retrieved and was cited nearby. Retrieval is not agreement.

So this module holds the evidence as more than a set of section numbers. The
normative tools already return `modality`, `content_type` and `text` for every
unit; nothing kept them. `NormativeEvidence` does, and the four checks below
read them:

    identifiers   a technical identifier the answer introduces must occur in
                  the evidence
    modality      a conclusion may not be stronger than the evidence it rests
                  on -- informative < may < should < shall
    coherence     the principal conclusion may not contradict the evidence,
                  and an answer may not contradict itself
    cardinality   a maximum, minimum or exact count is a normative claim and
                  needs a clause that states one; counting flags is not a
                  clause

Nothing here names a standard, a section, a field or a packet. The VITA cases
that motivated it are in the tests, not in the code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import provision_identity

UNGROUNDED_IDENTIFIER = "UNGROUNDED_IDENTIFIER"
STRENGTHENED_MODALITY = "STRENGTHENED_MODALITY"
INCOHERENT_CONCLUSION = "INCOHERENT_CONCLUSION"
UNSUPPORTED_CARDINALITY = "UNSUPPORTED_CARDINALITY"
AMBIGUOUS_CITATION = "AMBIGUOUS_CITATION"

#: informative < permission < recommendation < requirement. A conclusion may
#: sit at or below the level of its evidence, never above it.
INFORMATIVE, PERMISSION, RECOMMENDATION, REQUIREMENT = 0, 1, 2, 3

_LEVEL_NAME = {INFORMATIVE: "informative", PERMISSION: "permission",
               RECOMMENDATION: "recommendation", REQUIREMENT: "requirement"}

#: Stored modality, as the normative tools report it.
_STORED = {"SHALL": REQUIREMENT, "MUST": REQUIREMENT, "SHOULD": RECOMMENDATION,
           "MAY": PERMISSION, "NONE": INFORMATIVE}

#: What a claim's own words assert. Ordered: the strongest match wins, so
#: "shall not be required" reads as a requirement claim, not a permission.
_CLAIM = (
    (REQUIREMENT, re.compile(
        r"\b(?:shall|must|mandatory|required|requires|requirement|obliged?|"
        r"has to|have to|is expected to|are expected to)\b", re.I)),
    (RECOMMENDATION, re.compile(
        r"\b(?:should|recommended|recommendation|advisable|ought to)\b", re.I)),
    (PERMISSION, re.compile(
        r"\b(?:may|can|permitted|allowed|optional|is able to)\b", re.I)),
)

#: A question that asks whether something is obligatory. A bare "Yes" to one
#: of these asserts the obligation, which is how a recommendation gets
#: promoted without a single modal verb appearing in the answer.
_ASKS = (
    (REQUIREMENT, re.compile(
        r"\b(?:required|requirement|must|shall|mandatory|obliged?)\b", re.I)),
    (RECOMMENDATION, re.compile(r"\b(?:should|recommended)\b", re.I)),
    (PERMISSION, re.compile(r"\b(?:may|can|permitted|allowed)\b", re.I)),
)

_AFFIRM = re.compile(r"^\s*(?:\*\*)?\s*yes\b", re.I)
_DENY = re.compile(r"^\s*(?:\*\*)?\s*no[,.\s]", re.I)

#: Evidence that restricts rather than permits. Deliberately about quantity
#: and prohibition, which is what a permissive conclusion contradicts.
_RESTRICTIVE = re.compile(
    r"\b(?:shall not|must not|may not|cannot|shall have only|only one|"
    r"exactly one|no more than|at most|not permitted|prohibited|"
    r"only when|consist of only)\b", re.I)

#: A request for a bound. Each is a normative claim in its own right.
_ASKS_BOUND = re.compile(
    r"\b(?:maximum|minimum|at most|at least|how many|upper limit|lower limit|"
    r"exactly how|no more than|no fewer than|exhaustive)\b", re.I)

#: Evidence that actually states a bound. "three flags exist" does not.
_STATES_BOUND = re.compile(
    r"\b(?:at most|at least|no more than|no fewer than|maximum of|minimum of|"
    r"shall not exceed|shall be limited to|up to (?:a maximum of )?\d+|"
    r"only one|exactly one|no greater than)\b", re.I)

#: A clause that bounds something, and the value it bounds it to.
_BOUND_VALUE = re.compile(
    r"\b(?:at most|no more than|no fewer than|at least|maximum of|minimum of|"
    r"limited to|shall not exceed|no greater than|up to(?: a maximum of)?)\s+"
    r"(?P<value>\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b"
    r"|\b(?:only|exactly)\s+(?P<value2>one|two|\d+)\b", re.I)

_NUMBER = re.compile(
    r"\b(?:is|are|of)\s+(?:\*\*)?(\d+|one|two|three|four|five|six|seven|eight|"
    r"nine|ten)(?:\*\*)?\b", re.I)

#: An identifier shaped like a field, flag or symbol rather than like English.
#: Conservative on purpose: an ordinary capitalised word at the start of a
#: sentence must never look technical, so a bare capitalised word is not
#: enough. What qualifies is an internal capital after a lower-case letter
#: (AckV, ReqX, SchX), an all-caps run of three or more (NACK, CAM), or a
#: hyphen/underscore joining a short part to a letter or digit (Req-V, A1).
#: Deliberately NOT bare all-caps. An acronym is defined once, in a clause
#: this turn probably did not retrieve, so flagging CAM or NACK withholds
#: correct answers -- which is a worse failure than missing a fabricated
#: acronym. What stays is the shape the fabrication actually took: a
#: field name.
_IDENTIFIER = re.compile(
    r"\b(?:"
    r"[A-Za-z][a-z]+[A-Z][A-Za-z0-9]*"          # AckV, ReqX, SchX, CmdReport
    r"|[A-Z][a-z]*(?:[-_][A-Z][a-z0-9]*)+"       # Req-V, Ctrl-P
    r")\b")

#: Shapes the identifier pattern catches that are not field names. Kept short
#: and generic: acronyms every technical document uses about ITSELF, and the
#: words a normative answer says about its own sources.
_NOT_IDENTIFIERS = frozenset({
    "PDF", "HTML", "JSON", "XML", "ASCII", "UTF", "URL", "URI", "API", "CPU",
    "GPU", "RAM", "USB", "TODO", "NOTE", "WARNING", "ERROR", "INFO", "YES",
    "NO", "AND", "OR", "NOT", "THE", "ALL", "ANY", "ONE", "TWO", "MAY",
    "SHALL", "SHOULD", "MUST", "RULE", "SECTION", "TABLE", "FIGURE", "ANSI",
    "IEEE", "ISO", "IEC", "RFC", "NIST", "VITA",
})


def _normalise(token):
    """One spelling for comparison: `Req-V`, `ReqV` and `req_v` are one name."""
    return re.sub(r"[-_\s]", "", str(token)).lower()


@dataclass
class EvidenceUnit:
    section: str = ""
    source_id: str = ""
    modality: str = "NONE"
    content_type: str = ""
    text: str = ""

    @property
    def level(self):
        """What this unit is entitled to establish.

        Stored modality first. Where the store disagrees with the words --
        and it does: units typed INFORMATIVE carry `shall`, and a unit typed
        REQUIREMENT holds a Permission -- the words win, because the words
        are the document and the type is an extraction artefact.
        """
        stored = _STORED.get((self.modality or "NONE").upper(), INFORMATIVE)
        spoken = INFORMATIVE

        for level, pattern in _CLAIM:
            if pattern.search(self.text or ""):
                spoken = level
                break

        return max(stored, spoken)


@dataclass
class NormativeEvidence:
    """Every unit the normative tools returned this turn, with its words."""

    units: list = field(default_factory=list)

    #: The same retrievals at PROVISION granularity. A unit is the container:
    #: one of them carries a Permission, an Observation and a Rule sharing an
    #: ordinal, and a claim about one of those is not grounded by the others.
    provisions: provision_identity.ProvisionLedger = field(
        default_factory=provision_identity.ProvisionLedger)

    def observe(self, payload):
        """Collect units from one tool result. The payload is not mutated."""
        self._walk(payload)
        self.provisions.observe(payload)
        return self

    def _walk(self, node):
        if isinstance(node, dict):
            text = node.get("text") or node.get("snippet")

            if isinstance(text, str) and text.strip():
                self.units.append(EvidenceUnit(
                    section=str(node.get("section") or ""),
                    source_id=str(node.get("source_id") or ""),
                    modality=str(node.get("modality") or "NONE"),
                    content_type=str(node.get("content_type") or ""),
                    text=text))

            for value in node.values():
                if isinstance(value, (dict, list)):
                    self._walk(value)
        elif isinstance(node, list):
            for item in node:
                self._walk(item)

    # -- views -----------------------------------------------------------

    @property
    def text(self):
        return "\n".join(unit.text for unit in self.units)

    def knows_anything(self):
        return bool(self.units)

    def level(self):
        """The strongest thing the retrieved evidence establishes."""
        return max((unit.level for unit in self.units), default=INFORMATIVE)

    def identifiers(self):
        """Every technical identifier the evidence itself uses, normalised.

        A document spells one field two ways -- `Req-V` in a heading and
        `ReqV` in a table row -- and an answer may pick either. Comparing the
        literal spellings flagged three real fields as fabricated and withheld
        a correct answer.
        """
        return {_normalise(token) for token in _IDENTIFIER.findall(self.text)
                if token.upper() not in _NOT_IDENTIFIERS}

    def restrictive_units(self):
        return [unit for unit in self.units if _RESTRICTIVE.search(unit.text)]

    def states_a_bound(self):
        return any(_STATES_BOUND.search(unit.text) for unit in self.units)

    def cited_provisions(self, answer):
        """The provisions an answer cites, and the citations that name more
        than one. Substring matching against unit text cannot do this: the
        Rule and the Recommendation share a label and often a unit."""
        return self.provisions.references_in(answer or "")

    def cited_units(self, answer):
        """The provision RECORDS the answer points at, as evidence.

        `get` returns a record only when the printed label names exactly one
        declaration. Where the document reuses a label the citation is
        ambiguous and is reported as such, so nothing is grounded by
        guessing which instance was meant.
        """
        resolved, _ = self.cited_provisions(answer)
        found = [self.provisions.get(key) for key in resolved]

        return [record for record in found if record is not None]

    def cited_instances(self, answer):
        """The exact ProvisionInstanceIds an answer rests on.

        Carried through rather than re-derived from the rendered citation
        later: a printed label may name several declarations, and parsing the
        prose back into an identity is the guess this layer exists to avoid.
        """
        return [record.instance_id for record in self.cited_units(answer)]

    def level_for(self, citations):
        """Deprecated shape kept for the unit tests that drive it directly."""
        # A kinded citation must match kind AND number; a bare number may
        # match anything carrying it.
        def names(unit, item):
            if " " in item:
                return re.search(re.escape(item).replace(r"\ ", r"\s+"),
                                 unit.text, re.I) is not None
            return item in unit.text

        wanted = [unit for unit in self.units
                  if any(names(unit, item) for item in citations)]

        if not wanted:
            return self.level()

        return max(unit.level for unit in wanted)

    def bound_values(self):
        """The values the retrieved clauses actually bound things to.

        "only one bit out of three" states a bound of ONE. The three is the
        size of the set being chosen from, and reading it as the bound is the
        same mistake as counting flags.
        """
        found = set()

        for unit in self.units:
            for match in _BOUND_VALUE.finditer(unit.text):
                found.add(((match.group("value")
                            or match.group("value2")) or "").lower())

        return {item for item in found if item}

    def states_bound_of(self, value):
        """A clause that states a bound AND names this value.

        "only one" in a neighbouring rule is a bound of one; it does not
        support a claimed maximum of three.
        """
        spelled = {"1": "one", "2": "two", "3": "three", "4": "four",
                   "5": "five", "6": "six", "7": "seven", "8": "eight",
                   "9": "nine", "10": "ten"}
        forms = {str(value).lower()}
        forms.add(spelled.get(str(value), ""))
        forms |= {key for key, word in spelled.items()
                  if word == str(value).lower()}
        forms.discard("")

        return bool(forms & self.bound_values())


# ── the principal conclusion ─────────────────────────────────────────
# The sentence a reader acts on. An answer that is right in its third
# paragraph and wrong in its first has told the reader the wrong thing.

def principal(answer):
    """The opening claim, as prose."""
    for line in (answer or "").splitlines():
        stripped = line.strip().lstrip("#*->  ").strip()

        # Guard prefaces are the harness speaking, not the answer.
        if not stripped or stripped.startswith(("No source file was read",
                                                "Sources read:", "Clauses read")):
            continue

        return re.split(r"(?<=[.!?])\s+", stripped)[0]

    return ""


def claim_level(sentence, question=""):
    """What the sentence asserts -- with the question's own modality when the
    sentence merely says yes. "Is it required?" / "Yes." asserts a
    requirement without containing one modal word."""
    for level, pattern in _CLAIM:
        if pattern.search(sentence or ""):
            return level

    if _AFFIRM.match(sentence or ""):
        for level, pattern in _ASKS:
            if pattern.search(question or ""):
                return level

    return None


# ── 1. identifiers ───────────────────────────────────────────────────

def identifier_findings(answer, evidence, *, permitted=()):
    """Technical identifiers the answer introduces and the evidence lacks."""
    if not evidence.knows_anything():
        return []

    known = evidence.identifiers() | {_normalise(item) for item in permitted}
    found = []

    for token in dict.fromkeys(_IDENTIFIER.findall(answer or "")):
        if token.upper() in _NOT_IDENTIFIERS or _normalise(token) in known:
            continue

        found.append({"kind": UNGROUNDED_IDENTIFIER, "identifier": token})

    return found


# ── 2. modality ──────────────────────────────────────────────────────

#: Clause identifiers as the rest of the harness already understands them.
#: A bare dotted number is not enough: an answer naming the standard writes
#: its version, and "49.2" then selected whatever unit happened to contain it.
#: A rule identifier as an answer writes it: "8.3.1.5-2", "4.3-2", "8.4.1.1".
#: Two constraints, both needed. A trailing -N or at least two dots, so the
#: standard's own version ("49.2") is not read as a citation -- it selected
#: whatever unit happened to contain it, and a section's `shall` then
#: licensed a requirement claim about its neighbouring recommendation.
_RULE_ID = re.compile(
    r"\b(?P<kind>Rule|Recommendation|Observation|Permission)?\s*"
    r"(?P<id>\d+(?:\.\d+)+-\d+|\d+(?:\.\d+){2,})\b", re.I)


def _citations(answer):
    """Citations as written, keeping the provision kind when it is given.

    The number alone is ambiguous: this document numbers its Rules and its
    Recommendations independently inside a section, so `Recommendation
    8.3.1.5-2` and `Rule 8.3.1.5-2` are different provisions with different
    modality. Matching on the bare number pulled the Rule's unit in beside
    the Recommendation's and promoted a `should` to a `shall`.
    """
    found = []

    for match in _RULE_ID.finditer(answer or ""):
        kind, identifier = match.group("kind"), match.group("id")
        found.append(f"{kind.capitalize()} {identifier}" if kind else identifier)

    return sorted(set(found))


def modality_findings(answer, evidence, *, question=""):
    """A conclusion stronger than the evidence it rests on.

    The comparison is against the units the ANSWER CITES, not against the
    strongest thing retrieved. A turn that reads one `shall` anywhere in a
    section would otherwise licence a requirement claim about a neighbouring
    recommendation -- which is exactly how one got promoted.
    """
    if not evidence.knows_anything():
        return []

    sentence = principal(answer)
    claimed = claim_level(sentence, question)

    if claimed is None:
        return []

    cited = evidence.cited_units(answer)

    if cited:
        # The provision's own words, not its container's. The unit carrying a
        # Permission is stored SHALL when a Rule sits beside it.
        supported = max(_STORED.get(record.modality.upper(), INFORMATIVE)
                        for record in cited)
    else:
        supported = evidence.level_for(_citations(answer))

    if claimed <= supported:
        return []

    return [{"kind": STRENGTHENED_MODALITY, "sentence": sentence,
             "claimed": _LEVEL_NAME[claimed],
             "supported": _LEVEL_NAME[supported]}]


# ── 3. coherence ─────────────────────────────────────────────────────

def coherence_findings(answer, evidence, *, question=""):
    """The principal conclusion against the evidence, and against itself.

    Two ways the same defect shows: the opening affirms what a retrieved
    requirement restricts, or the answer reverses itself further down. The
    second needs no evidence at all -- an answer that says yes and then says
    shall-not has already failed, whatever the document says.
    """
    sentence = principal(answer)

    if not sentence:
        return []

    affirmed = bool(_AFFIRM.match(sentence))
    denied = bool(_DENY.match(sentence))

    if not (affirmed or denied):
        return []

    body = (answer or "")[len(sentence):]
    reversal = [part for part in re.split(r"(?<=[.!?])\s+", body)
                if _RESTRICTIVE.search(part)
                and re.search(r"\b(?:shall|must|only)\b", part, re.I)]

    if affirmed and reversal:
        return [{"kind": INCOHERENT_CONCLUSION, "sentence": sentence,
                 "contradicted_by": reversal[0][:200], "source": "self"}]

    # The evidence branch, and it has to be narrow. A turn that reads a whole
    # chapter reads restrictive rules beside permissive ones, and "an
    # affirmative opening while some retrieved rule restricts something"
    # withheld two correct answers: one about what a CONTROL packet may
    # request, refused because of a rule about the ACKNOWLEDGE packet.
    #
    # So it fires only when every provision the answer CITES is restrictive.
    # An answer that rests on a permission is entitled to say yes.

    if affirmed:
        cited = evidence.cited_units(answer)

        def restricts(record):
            return (_RESTRICTIVE.search(record.text)
                    and _STORED.get(record.modality.upper(),
                                    INFORMATIVE) >= REQUIREMENT)

        if cited and all(restricts(record) for record in cited):
            return [{"kind": INCOHERENT_CONCLUSION, "sentence": sentence,
                     "contradicted_by": cited[0].text[:200],
                     "source": cited[0].source_id or cited[0].section}]

    return []


# ── 4. cardinality ───────────────────────────────────────────────────

def cardinality_findings(answer, evidence, *, question=""):
    """A bound asserted where no retrieved clause states one.

    Structural enumeration is not normative cardinality: that a field has
    three flags says nothing about how many responses are allowed. The bound
    has to come from a clause.
    """
    if not _ASKS_BOUND.search(question or ""):
        return []

    sentence = principal(answer)
    number = _NUMBER.search(sentence or "")

    if not number:
        return []

    if evidence.states_bound_of(number.group(1)):
        return []

    return [{"kind": UNSUPPORTED_CARDINALITY, "sentence": sentence,
             "asserted": number.group(1)}]


# ── the gate ─────────────────────────────────────────────────────────

def ambiguity_findings(answer, evidence):
    """A citation that names more than one provision.

    The document numbers each kind independently, so a bare `8.3.1.5-2` may
    name a Rule AND a Recommendation with different modality. Picking one
    silently is how a `should` became a `shall`; the reference is reported
    instead, and the answer is withheld.
    """
    if not evidence.knows_anything():
        return []

    _, ambiguous = evidence.cited_provisions(answer)

    return [{"kind": AMBIGUOUS_CITATION, "reference": problem.reference,
             "candidates": [str(key) for key in problem.candidates]}
            for problem in ambiguous]


def findings(answer, evidence, *, question="", permitted=()):
    return (ambiguity_findings(answer, evidence)
            + identifier_findings(answer, evidence, permitted=permitted)
            + modality_findings(answer, evidence, question=question)
            + coherence_findings(answer, evidence, question=question)
            + cardinality_findings(answer, evidence, question=question))


def _note(problems, evidence):
    """What replaces an answer the evidence does not support."""
    lines = ["The retrieved normative evidence does not support this answer "
             "as written, so it is withheld rather than shown."]

    for problem in problems:
        kind = problem["kind"]

        if kind == UNGROUNDED_IDENTIFIER:
            lines.append(f"  - {problem['identifier']}: named in the answer, "
                         "present in no retrieved unit.")
        elif kind == STRENGTHENED_MODALITY:
            lines.append(f"  - the conclusion states a {problem['claimed']}; "
                         f"the evidence establishes a {problem['supported']}.")
        elif kind == INCOHERENT_CONCLUSION:
            lines.append("  - the opening conclusion contradicts the evidence "
                         "the answer itself relies on.")
        elif kind == AMBIGUOUS_CITATION:
            lines.append(f"  - {problem['reference']} names more than one "
                         "provision (" + ", ".join(problem["candidates"])
                         + "); which one is meant decides the answer.")
        elif kind == UNSUPPORTED_CARDINALITY:
            lines.append(f"  - a bound of {problem['asserted']} is asserted; "
                         "no retrieved clause states one. Counting flags, "
                         "types or examples does not establish a limit.")

    if evidence.units:
        sections = sorted({unit.section for unit in evidence.units if unit.section})
        if sections:
            lines.append("Clauses read this turn: "
                         + ", ".join("§" + item for item in sections[:12]) + ".")

    return "\n".join(lines)


def guard(answer, evidence, *, question="", permitted=()):
    """The answer, or a note saying why it is withheld."""
    problems = findings(answer, evidence, question=question,
                        permitted=permitted)

    if not problems:
        return answer, [], False

    return _note(problems, evidence), problems, True
