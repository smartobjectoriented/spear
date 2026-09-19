"""What one grounded answer established, in a form the next turn can be held to.

A two-prompt workflow is one task in two halves. The first asks what an
authoritative source requires; the second says "now make the code comply with
THIS". The word *this* points at the first answer, and until now the only
thing that travelled between the two turns was a list of section numbers and
four thousand characters of the answer's prose.

Prose is a poor contract. Measured across four runs of the same two turns, the
second turn implemented between one and eight of the behaviours the first turn
had established, and the variation was not random: in the two weakest runs the
first answer had been WITHHELD by the identifier guard, so the prose carried
forward was an assessment record and nothing else. The second turn started
from nothing, rediscovered an arbitrary subset of the document, and built that.

So what travels is not the answer. It is the set of PROVISIONS the first turn
actually retrieved, with their printed citation, their normative force and
their own words — which exists whether or not the answer survived the guards,
because it is built from the evidence ledger rather than from the reply. Each
one then has to reach a disposition before the second turn may write, and
every disposition is a statement the turn made on the record:

    SATISFIED_ALREADY      the code already does this, and here is where
    CHANGE_PLANNED         a change is intended
    CHANGE_IMPLEMENTED     the change was made
    EXPLICITLY_OUT_OF_SCOPE  deliberately not supported, said plainly
    UNDETERMINED           could not be decided — the honest fifth answer

The one thing a requirement may not do is disappear. A turn that quietly stops
mentioning a rule it was carrying is the failure this module exists to stop,
and UNDETERMINED is there so that "I could not tell" never has to be spelled
as silence.

Three bounds keep this from becoming an attempt to model the whole document:

* only provisions of REQUIREMENT force are carried — a Recommendation is not
  a thing a turn can fail to close;
* only provisions this turn RETRIEVED, never the whole standard;
* and when the answer survived and cited provisions by name, only those,
  because the user's "this" points at the answer, not at everything the
  retrieval happened to return.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from enum import StrEnum

import normative_claims
from normative_force import REQUIREMENT_FORCE


class Disposition(StrEnum):
    SATISFIED_ALREADY = "satisfied_already"
    CHANGE_PLANNED = "change_planned"
    CHANGE_IMPLEMENTED = "change_implemented"
    EXPLICITLY_OUT_OF_SCOPE = "explicitly_out_of_scope"
    UNDETERMINED = "undetermined"


#: The dispositions that leave work unfinished. DONE is not honest while a
#: requirement is still in one of them: the first is a promise and the second
#: is an open question.
UNFINISHED = frozenset({Disposition.CHANGE_PLANNED, Disposition.UNDETERMINED})

#: What a turn may say through the plan tool. `CHANGE_IMPLEMENTED` is absent
#: on purpose: whether a change was made is a fact about the tree, observed by
#: the harness, not a claim the model gets to file.
STATED = frozenset({Disposition.SATISFIED_ALREADY, Disposition.CHANGE_PLANNED,
                    Disposition.EXPLICITLY_OUT_OF_SCOPE,
                    Disposition.UNDETERMINED})

#: How many provisions may be carried when the answer named none of them. A
#: cap, not a judgement: one question's retrieval returns a handful of units,
#: and a set this size is a task contract while ten times it is a reading list.
CARRY_LIMIT = 12

#: A printed citation, as the documents write them and as an answer repeats
#: them: "Rule 8.4.1.1-3", "Observation 8.4.2-1", "§8.5". Matched here only to
#: ask whether an answer NAMED a provision the turn retrieved.
_CITED = re.compile(
    r"(?:\b(?:rule|observation|recommendation|permission|definition|"
    r"requirement|table|figure|section|clause)\s*|§\s*)"
    r"(\d+(?:\.\d+)*(?:-\d+)?)", re.I)

#: A bare dotted ordinal: "4.2.1-1", "8.4". Used ONLY where the whole string
#: is meant as a citation -- a plan item's evidence field -- and never when
#: scanning prose, where a bare number is as likely to be a version, a count
#: or a byte offset as a provision.
_HANDLE = re.compile(r"\b[a-z]{2,4}-[0-9a-f]{8,}\b")

_BARE = re.compile(r"\b(\d+(?:\.\d+)+(?:-\d+)?)\b")

#: A coordinated list of technical identifiers inside one provision — "read,
#: write and execute", "Alpha, Bravo or Charlie". Two or more, because a list
#: of one is a mention. This is how a provision says that several cases are
#: one family; which cases they are is the provision's business, and naming
#: any of them here would make this file know one document.
#: Two shapes, because documents write lists both ways: "A, B, or C" and the
#: bare "A, B, C". The second needs three members before it is a list rather
#: than a comma doing ordinary work in a sentence.
_WORD = r"[A-Za-z][\w\-]{1,31}"
_FAMILY = re.compile(
    # "A, B or C", "A, B, or C", "A, B, C and D"
    rf"\b{_WORD}\b(?:\s*,\s*\b{_WORD}\b)+\s*,?\s+(?:or|and)\s+\b{_WORD}\b"
    # ...and the bare "A, B, C", which needs three before it is a list
    # rather than a comma doing ordinary work in a sentence.
    rf"|\b{_WORD}\b\s*,\s*\b{_WORD}\b\s*,\s*\b{_WORD}\b")

#: Words a coordination may list that are not technical identifiers.
_NOT_A_MEMBER = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "with", "by",
    "is", "are", "be", "shall", "should", "may", "must", "will", "not",
    "one", "each", "any", "all", "more", "than", "when", "that", "this",
    "packet", "packets", "bit", "bits", "field", "fields", "set", "sent",
})


def citations_in(text):
    """The provisions a piece of text names, normalised for comparison."""
    return {found.lower() for found in _CITED.findall(text or "")}


def _family_of(text):
    """The members of a coordinated list this provision states, if any.

    Inferred from the provision's OWN words and nothing else. A rule that
    reads "one for each of A, B or C" is stating that A, B and C are three
    cases of one obligation, and a turn that dispositions the rule without
    saying anything about C has closed it too early.
    """
    found = _FAMILY.search(text or "")

    if found is None:
        return ()

    members = []

    for token in re.findall(r"\b[A-Za-z][\w\-]{1,31}\b", found.group(0)):
        if token.lower() in _NOT_A_MEMBER or token in members:
            continue

        members.append(token)

    return tuple(members) if len(members) > 1 else ()


@dataclass(frozen=True)
class Requirement:
    """One provision the earlier turn established, and what became of it."""

    key: str
    section: str = ""
    source_id: str = ""
    force: str = "requirement"
    statement: str = ""
    members: tuple = ()
    disposition: str = str(Disposition.UNDETERMINED)
    note: str = ""

    #: Whether the turn SAID this, as against never having mentioned it.
    #: Without the distinction "I could not determine it" and "I forgot about
    #: it" are the same record, and they are not the same thing: the first is
    #: an answer the gate accepts, the second is the failure it exists for.
    stated: bool = False

    @property
    def open(self):
        return self.disposition in UNFINISHED

    def to_dict(self):
        return {"key": self.key, "section": self.section,
                "source_id": self.source_id, "force": self.force,
                "statement": self.statement, "members": list(self.members),
                "disposition": self.disposition, "note": self.note,
                "stated": self.stated}

    @classmethod
    def from_dict(cls, raw):
        raw = raw if isinstance(raw, dict) else {}

        return cls(
            key=str(raw.get("key") or ""),
            section=str(raw.get("section") or ""),
            source_id=str(raw.get("source_id") or ""),
            force=str(raw.get("force") or "requirement"),
            statement=str(raw.get("statement") or ""),
            members=tuple(raw.get("members") or ()),
            disposition=str(raw.get("disposition")
                            or Disposition.UNDETERMINED),
            note=str(raw.get("note") or ""),
            stated=bool(raw.get("stated")))


@dataclass
class RequirementSet:
    """The task contract one turn hands the next, with its provenance."""

    items: list = field(default_factory=list)
    #: Which turn published it, for a reader of the record.
    origin: str = ""

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    @property
    def keys(self):
        return tuple(item.key for item in self.items)

    def get(self, key):
        wanted = (key or "").strip().lower()

        for item in self.items:
            if item.key.lower() == wanted:
                return item

        return None

    def open_items(self):
        return tuple(item for item in self.items if item.open)

    def unstated(self):
        """The requirements this turn has said nothing at all about."""
        return tuple(item for item in self.items if not item.stated)

    # -- matching ---------------------------------------------------------

    def match(self, citation):
        """The carried requirement a plan item's citation refers to.

        Written against what a model actually writes in that field, which is
        rarely the bare printed key. Measured on one run, every one of
        twenty-two plan items cited its provision in the form

            "<document> <revision> §<section>, p.<page>, source <handle>"

        -- the document, the SECTION, the page and the retrieval handle, and
        not once the printed key of the rule itself. Every one of them failed
        to match, so nothing was ever dispositioned, the coverage gate never
        opened, and a turn that had planned the work correctly was refused ten
        edits and finished having written nothing. A gate nobody can close is
        not a gate, it is a wall.

        So: the retrieval handle, the printed key, the ordinal, the section
        the provision sits in, and a section that contains it.
        """
        wanted = citations_in(citation) | {
            found.lower() for found in _BARE.findall(citation or "")}
        handles = {found.lower() for found in _HANDLE.findall(citation or "")}

        # The handle is the strongest signal there is: it names one retrieved
        # unit and nothing else.
        for item in self.items:
            if item.source_id and item.source_id.lower() in handles:
                return item

        if not wanted:
            return None

        for item in self.items:
            if item.key.lower() in wanted:
                return item

        for item in self.items:
            bare = item.key.lower().split()[-1] if item.key else ""

            if bare and bare in wanted:
                return item

            if item.section and item.section.lower() in wanted:
                return item

        # The section a provision sits in, which for "Rule 8.4.1.1-3" is
        # 8.4.1.1 whether or not the record carried one. Citing the section is
        # how a turn refers to the rule printed under it.
        for item in self.items:
            bare = item.key.lower().split()[-1] if item.key else ""
            owning = {part for part in
                      (bare.rsplit("-", 1)[0] if "-" in bare else bare,
                       item.section.lower() if item.section else "") if part}

            if owning & wanted:
                return item

        # ...and a broader section that contains it: "§8.4" names the chapter
        # Rule 8.4.1.1-3 lives in.
        for item in self.items:
            bare = item.key.lower().split()[-1] if item.key else ""
            under = item.section.lower() or (
                bare.rsplit("-", 1)[0] if "-" in bare else bare)

            for candidate in wanted:
                if under and under.startswith(candidate + "."):
                    return item

        return None

    def dispose(self, key, disposition, note=""):
        """Record what became of one requirement. Returns the item, or None."""
        found = self.get(key) or self.match(key)

        if found is None:
            return None

        self.items[self.items.index(found)] = replace(
            found, disposition=str(disposition), note=note or found.note,
            stated=True)

        return self.items[self.items.index(self.get(found.key))]

    # -- serialisation ----------------------------------------------------

    def to_dict(self):
        return {"schema_version": 1, "origin": self.origin,
                "items": [item.to_dict() for item in self.items]}

    @classmethod
    def from_dict(cls, raw):
        if not isinstance(raw, dict) or raw.get("schema_version") != 1:
            return cls()

        return cls([Requirement.from_dict(item)
                    for item in raw.get("items") or ()],
                   str(raw.get("origin") or ""))

    def matrix(self):
        """The requirement table a review owes its reader."""
        rows = []

        for item in self.items:
            rows.append({
                "requirement": item.key,
                "force": item.force,
                "evidence": item.source_id or item.section,
                "statement": item.statement,
                "disposition": item.disposition,
                "note": item.note,
                "stated": item.stated,
            })

        return rows


def publish(policy, answer="", *, origin="", limit=CARRY_LIMIT):
    """The requirement set a bound turn established, from its OWN evidence.

    Built from the provision ledger rather than from the reply, so a turn
    whose answer was withheld still hands its successor a contract. When the
    answer survived and named provisions, those are the set — the user's
    "this" points at the answer. When it named none, the requirement-force
    provisions the turn retrieved are the set, capped.
    """
    ledger = getattr(getattr(policy, "claim_evidence", None), "provisions", None)

    if ledger is None:
        return RequirementSet()

    records = getattr(ledger, "records", None) or {}
    binding = []

    for key, record in records.items():
        try:
            force = record.effective_force
        except Exception:
            continue

        if force < REQUIREMENT_FORCE:
            continue

        text = " ".join((getattr(record, "text", "") or "").split())
        binding.append(Requirement(
            key=str(key),
            section=str(getattr(record, "section", "") or ""),
            source_id=str(getattr(record, "source_id", "") or ""),
            force="requirement",
            statement=text[:400],
            members=_family_of(text)))

    if not binding:
        return RequirementSet()

    named = citations_in(answer)

    if named:
        chosen = [item for item in binding
                  if item.key.lower() in named
                  or (item.key.lower().split()[-1] if item.key else "") in named]

        if chosen:
            return RequirementSet(_ordered(chosen), origin)

    return RequirementSet(_ordered(binding)[:limit], origin)


def _ordered(items):
    """A stable order two runs agree on: by section, then by printed key."""
    def sort_key(item):
        section = item.section or ""
        parts = tuple(int(part) for part in re.findall(r"\d+", section))

        return (parts, item.key)

    return sorted(items, key=sort_key)


#: How a follow-up turn points back at the answer before it. Deliberately
#: narrow: a demonstrative with nothing of its own to name, or an explicit
#: word for the thing that was just established. "Add a --verbose flag" names
#: its own subject and carries nothing forward.
_REFERS_BACK = re.compile(
    r"\b(?:with|to|against|for|per|following|using)\s+(?:this|that|these|those|it)\b"
    r"|\b(?:this|that|these|those)\s+(?:requirement|requirements|rule|rules|"
    r"clause|clauses|provision|provisions|spec|specification|standard|answer|"
    r"finding|findings|analysis)\b"
    r"|\b(?:implement|apply|do|make|bring)\s+(?:it|this|that|these|those)\b"
    r"|\bcomply\s+with\s+(?:this|that|these|those|it)\b"
    r"|\b(?:as|like)\s+(?:described|stated|explained|said)\s+above\b"
    r"|\b(?:the\s+)?above\b",
    re.IGNORECASE)


def refers_back(question) -> bool:
    """Does this turn point at what the previous one established?

    The carry is a contract, and a contract inherited by a turn that never
    asked for it is a turn told to do somebody else's work. A new,
    self-contained task names its own subject and gets no inheritance.
    """
    return bool(_REFERS_BACK.search(question or ""))
