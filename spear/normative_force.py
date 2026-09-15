"""What a provision is allowed to establish, and what its words actually say.

Three things were one thing, and conflating them let an answer stand that the
document does not support.

* The ROLE is what the document PRINTS: Rule, Recommendation, Permission,
  Observation, Definition, or nothing at all. It is structure, never words.
* The LEXICAL MODALS are the modal verbs occurring anywhere in the text.
  There may be several, they may disagree, and some of them belong to a
  DIFFERENT provision that this one is talking about.
* The EFFECTIVE FORCE is what this provision may be used as evidence for.

Measured failure: an Observation whose text reads "the Controllee **must**
generate multiple Acknowledge packets" was read as MUST-level evidence, so a
turn concluded that the Observation itself stated the binding requirement. It
does not; a Rule elsewhere does, and the Observation is describing it. The
modal was real, the reading of it was not.

The rule is a ceiling, not a floor, and it cuts both ways:

* a strong role does not promote weak wording -- a Rule that says "should"
  supports a recommendation, not a requirement;
* strong wording does not promote a weak role -- an Observation that says
  "must" supports nothing binding on its own.

What a role MEANS is a property of the document, not of this file. A standard
that numbers its provisions differently, or that gives "Observation" binding
weight, says so in its own taxonomy; the default here is the conventional
reading and is a default, not a law.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ── force ────────────────────────────────────────────────────────────
# Ordered, so "at most this strong" is a comparison.

INFORMATIVE_FORCE, PERMISSION_FORCE, RECOMMENDATION_FORCE, REQUIREMENT_FORCE = 0, 1, 2, 3

FORCE_NAME = {INFORMATIVE_FORCE: "informative", PERMISSION_FORCE: "permission",
              RECOMMENDATION_FORCE: "recommendation",
              REQUIREMENT_FORCE: "requirement"}

# ── roles ────────────────────────────────────────────────────────────
# The names a document prints. Not an exhaustive list of what documents do:
# a taxonomy may add its own.

RULE = "RULE"
REQUIREMENT = "REQUIREMENT"
RECOMMENDATION = "RECOMMENDATION"
PERMISSION = "PERMISSION"
OBSERVATION = "OBSERVATION"
DEFINITION = "DEFINITION"
INFORMATIVE = "INFORMATIVE"
UNLABELLED = "UNLABELLED"

#: How the printed labels of a document map onto roles. Spelling only -- the
#: MEANING of each role is the taxonomy's business.
_PRINTED = {
    "rule": RULE, "requirement": REQUIREMENT, "recommendation": RECOMMENDATION,
    "permission": PERMISSION, "observation": OBSERVATION,
    "definition": DEFINITION, "informative": INFORMATIVE,
    "scopepreamble": UNLABELLED, "unlabelledbody": UNLABELLED,
    "tablerow": UNLABELLED, "tablecell": UNLABELLED,
}


def role_of(kind):
    """The role a printed label names, or UNLABELLED when it names none."""
    return _PRINTED.get(re.sub(r"[\s_-]", "", str(kind or "")).lower(),
                        UNLABELLED)


# ── the taxonomy a standard declares ─────────────────────────────────

@dataclass(frozen=True)
class RoleTaxonomy:
    """What the roles of one document are entitled to establish.

    A ceiling per role. Anything not named falls to `default_ceiling`, which
    is deliberately the most permissive reading: a role this file has never
    heard of is a role whose semantics nobody has declared, and silencing it
    would quietly discard normative text.
    """

    name: str = "conventional"
    ceilings: dict = field(default_factory=dict)
    default_ceiling: int = REQUIREMENT_FORCE

    def ceiling(self, role):
        """The strongest force a provision of this role may establish."""
        return self.ceilings.get(role, self.default_ceiling)

    def may_establish(self, role, force):
        return force <= self.ceiling(role)


#: The conventional reading, used when a standard declares no taxonomy of its
#: own. Rules and Requirements bind; a Recommendation recommends; a Permission
#: permits; an Observation, a Definition and informative prose describe, and
#: describing an obligation is not imposing one. Unlabelled body text is left
#: at the top: a document's normative prose is frequently unnumbered, and a
#: ceiling there would discard it.
CONVENTIONAL = RoleTaxonomy(
    name="conventional",
    ceilings={
        RULE: REQUIREMENT_FORCE,
        REQUIREMENT: REQUIREMENT_FORCE,
        RECOMMENDATION: RECOMMENDATION_FORCE,
        PERMISSION: PERMISSION_FORCE,
        OBSERVATION: INFORMATIVE_FORCE,
        DEFINITION: INFORMATIVE_FORCE,
        INFORMATIVE: INFORMATIVE_FORCE,
        UNLABELLED: REQUIREMENT_FORCE,
    },
    default_ceiling=REQUIREMENT_FORCE)

#: standard_id -> taxonomy. A standard whose roles mean something else
#: registers it; nothing in this module knows any particular document.
_TAXONOMIES = {}


def register_taxonomy(standard_id, taxonomy):
    _TAXONOMIES[str(standard_id)] = taxonomy

    return taxonomy


def taxonomy_for(standard_id=None, revision=None):
    """The declared taxonomy for a standard, else the conventional reading."""
    return (_TAXONOMIES.get(f"{standard_id}@{revision}")
            or _TAXONOMIES.get(str(standard_id))
            or CONVENTIONAL)


# ── lexical modals ───────────────────────────────────────────────────

#: A modal verb and the force it would carry if this provision established
#: it. Ordered strongest first so one occurrence is classified once.
_MODALS = (
    (REQUIREMENT_FORCE, re.compile(r"\b(?:shall|must|requires?|required|"
                                   r"requirement)\b", re.I)),
    (RECOMMENDATION_FORCE, re.compile(r"\b(?:should|ought to|recommended)\b", re.I)),
    (PERMISSION_FORCE, re.compile(r"\b(?:may|permitted|allowed|optional)\b", re.I)),
)

#: Language that attributes a following clause to some OTHER text. A modal
#: after one of these is being reported, not imposed.
#:
#: The verb must actually take a clause -- "states that", "specifies:" -- and
#: that requirement is the whole point. Matching the bare verbs read the NOUN
#: "state" as the verb "states", so "State and Event Indicators ... shall be
#: positioned" became a reported requirement, and 45 ordinary Rules of one
#: document stopped grounding anything. A word that is a verb here and a noun
#: three lines down is not a signal; a verb with its clause is.
#:
#: Citational phrases are deliberately absent -- "in accordance with", "as
#: specified in" -- because they attach to a reference, and a provision that
#: cites a section while stating its own obligation is still stating it.
_ATTRIBUTION = re.compile(
    r"\b(?:states?|stated|says?|said|specif(?:y|ies|ied)|provides?|"
    r"declares?|establishes?|indicates?|indicated|mandates?|implies|implied|"
    r"clarif(?:y|ies)|explains?|notes?)\s*(?:\b(?:that|how|which|whether)\b|:)",
    re.I)

#: A provision label, and the shape of one being TALKED ABOUT: the citation
#: standing as the subject of what follows.
#:
#: Merely occurring in the sentence is far too weak. "Rule 11-3: The
#: representation of the file size for Rule 11-1 shall be by two consecutive
#: 32-bit words" cites a rule and then imposes its own obligation; so does
#: "When a Controllee ID is used per Permission 9.8.3-1, it shall use the
#: same format". Both were demoted to informative by presence alone.
#:
#: So the citation has to sit where a subject sits: immediately before the
#: modal, at most a couple of words away, with no comma between them, and not
#: introduced by a preposition -- "for Rule 11-1 shall" is a modifier, "Rule
#: 5.2-1 requires" is a subject.
_CITED_SOURCE = (r"(?:Rule|Recommendation|Permission|Observation|Definition|"
                 r"Requirement)\s+\d+(?:\.\d+)*-\d+")
_CITED_PROVISION = re.compile(_CITED_SOURCE, re.I)
_CITATION_SUBJECT = re.compile(
    rf"(?:(?P<lead>[A-Za-z]+)\s+)?{_CITED_SOURCE}"
    rf"(?:\s+[A-Za-z]+){{0,2}}\s*$", re.I)

#: Words that make the citation after them a modifier rather than a subject.
_MODIFIER_LEAD = frozenset({
    "for", "in", "per", "of", "under", "by", "with", "to", "from", "via",
    "see", "and", "or", "than", "beyond", "within", "against", "unless"})

#: Quoted material. What a document quotes is the quoted text's force, not
#: the quoting paragraph's.
_QUOTED = re.compile(r"[\"“‘']([^\"”’']{3,})[\"”’']")

_SENTENCE = re.compile(r"(?<=[.!?;:])\s+")

#: Clause boundaries inside one sentence. An attributing verb governs only
#: up to the next of these.
_CLAUSE = re.compile(r"[,;]")

ESTABLISHED = "ESTABLISHED_MODALITY"
REPORTED = "REPORTED_MODALITY"


@dataclass(frozen=True)
class Modal:
    """One modal occurrence, and whether this provision is imposing it."""

    force: int
    word: str
    sentence: str
    status: str
    why: str = ""

    @property
    def established(self):
        return self.status == ESTABLISHED


def _quoted_spans(sentence):
    return [(match.start(1), match.end(1)) for match in _QUOTED.finditer(sentence)]


def _reported_because(sentence, position):
    """Why this modal is being reported rather than imposed, or ''.

    Deterministic signals only, in the order they are worth trusting: inside
    a quotation; in a sentence that names another provision; after language
    that attributes the statement to some other text.
    """
    for start, end in _quoted_spans(sentence):
        if start <= position < end:
            return "inside a quotation"

    before = sentence[:position]
    subject = _CITATION_SUBJECT.search(before)

    if subject and (subject.group("lead") or "").lower() not in _MODIFIER_LEAD:
        return "another provision is the subject of this statement"

    # The attribution has to GOVERN the modal, so only the clause the modal
    # sits in counts. "If the LSH Code indicates that leap seconds are not
    # applicable, the field shall be ..." attributes the CONDITION to the
    # code and then states the rule's own obligation; searching the whole
    # sentence read the obligation as reported and demoted six rules.
    if _ATTRIBUTION.search(_CLAUSE.split(before)[-1]):
        return "attributed to another text"

    return ""


def modals(text):
    """Every modal occurrence in a text, with why it is or is not imposed.

    Sentence by sentence, because attribution is a property of the sentence a
    modal sits in: a provision may state a requirement in one sentence and
    describe someone else's in the next.
    """
    found = []

    for sentence in _SENTENCE.split(text or ""):
        for force, pattern in _MODALS:
            for match in pattern.finditer(sentence):
                why = _reported_because(sentence, match.start())
                found.append(Modal(
                    force=force, word=match.group(0), sentence=sentence.strip(),
                    status=REPORTED if why else ESTABLISHED, why=why))

    return found


# ── the assessment ───────────────────────────────────────────────────

@dataclass(frozen=True)
class ForceAssessment:
    """What one provision may be used as evidence for, and why."""

    role: str
    authority: int
    lexical: tuple = ()
    established: tuple = ()
    reported: tuple = ()

    @property
    def spoken(self):
        """The strongest force this provision's own words impose."""
        return max((item.force for item in self.established),
                   default=INFORMATIVE_FORCE)

    @property
    def effective(self):
        """What it may support: its words, capped by its authority.

        Never the other way round. A Rule that says "should" supports a
        recommendation, and an Observation that says "must" supports nothing
        binding -- the ceiling does not lift weak wording, and strong wording
        does not lift the ceiling.
        """
        return min(self.authority, self.spoken)

    @property
    def explanation(self):
        words = ", ".join(sorted({item.word.lower() for item in self.lexical})) or "none"
        reported = ", ".join(sorted({f"{item.word.lower()} ({item.why})"
                                     for item in self.reported}))
        found = (f"role {self.role} may establish at most "
                 f"{FORCE_NAME[self.authority]}; words: {words}; "
                 f"supports {FORCE_NAME[self.effective]}")

        return f"{found}; reported rather than imposed: {reported}" if reported else found


def assess(text, *, role=None, kind=None, taxonomy=None):
    """What a provision with this printed role and this text may support."""
    role = role or role_of(kind)
    taxonomy = taxonomy or CONVENTIONAL
    ceiling = taxonomy.ceiling(role)
    found = modals(text)

    # A modal stronger than this role may impose is not this provision's
    # statement, whatever the sentence looks like: an Observation that says
    # "must" is describing an obligation imposed somewhere else.
    graded = tuple(
        item if (item.status == REPORTED or item.force <= ceiling)
        else Modal(item.force, item.word, item.sentence, REPORTED,
                   f"role {role} cannot establish {FORCE_NAME[item.force]}")
        for item in found)

    return ForceAssessment(
        role=role, authority=ceiling, lexical=graded,
        established=tuple(item for item in graded if item.established),
        reported=tuple(item for item in graded if not item.established))
