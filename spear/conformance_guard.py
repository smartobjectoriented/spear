"""What a bound answer may say about compliance, checked against what it read.

Bound to ANSI-VITA-49.2, asked to validate the acknowledge path, a session ran
twelve greps, called no normative tool of its own, and concluded:

    The implementation is correct and compliant with ANSI/VITA 49.2-2017 (R2024).

It is not. Rule 8.4.1.1-3 is not implemented, Req-V is never honoured, the
execution acknowledge is unconditional. None of the guards before this one had
anything to say: the sentence binds no label to a range, and it cites nothing a
provenance check could find fabricated. It is plain prose, which is the one
form a false clearance takes.

No deterministic layer can decide whether code satisfies a clause; that takes
reading the code. What can be decided is whether the turn is ENTITLED to the
sentence at all, and the distinction is the one evidence_guard already lives on:

    scoped     Rule 8.4.1.1-2 is satisfied: the encoder sets exactly one bit
    unscoped   the implementation correctly follows the VITA 49.2 specification

The first names a clause and a fact. The second quantifies over the whole
standard, which no finite set of retrievals establishes. So an affirmative
verdict survives only when it names a clause, that clause was retrieved this
turn, and its subject is not the standard or the implementation as a whole.

Removal would be worse than the fault: an answer with its conclusion cut reads
as approval by silence. The unscoped sentence is replaced instead, with a
statement built from the turn's own ledger of what was read -- the same idiom
as evidence_progress.bounded_absence, and for the same reason: it is a record,
not a guess.

Affirmative verdicts only. A wrong "not compliant" costs a re-read; a wrong
"compliant" costs a deployment, and the asymmetry decides the asymmetry.

This is the weakest of the three guards, and knows it. A scoped, cited and
false verdict passes untouched. It is a floor on entitlement, not a proof of
anything.
"""

from __future__ import annotations

import re

import provision_identity
from dataclasses import dataclass, field

UNSCOPED_VERDICT = "UNSCOPED_VERDICT"
UNREAD_CLAUSE = "UNREAD_CLAUSE"

# A clause identifier as the standards write them: "Rule 8.4.1.1-2",
# "Observation 8.2.1-3", "§8.4", "Section 8.3.1.5", "Table 8.4.1-1". The
# section part alone is what the ledger keys on, so a verdict citing Rule
# 8.4.1.1-2 is backed by a retrieval of §8.4.1.1.
_CLAUSE = re.compile(
    r"(?:\b(?:rule|observation|recommendation|permission|definition|table|"
    r"figure|section|sect\.?|clause)\s*|§\s*)"
    r"(\d+(?:\.\d+)+)(?:-\d+)?", re.I)

# An affirmative verdict. Negations are looked for separately and exempt the
# sentence: "is not compliant" is a different claim with a different cost.
_ACT = (r"(?:follows?|implements?|handles?|echo(?:es|ing)?|sets?|encodes?|"
        r"manages?|matches)")

_VERDICT = re.compile(
    r"\b(?:compliant|compliance|conformant|conforms?|conformance|"
    r"(?:correctly|properly)\s+" + _ACT
    # "implements it CORRECTLY": the adverb after the verb is as common as
    # before it, and only the first form was caught.
    # The gap is [\w.\-]+, not \w+: "implements the VITA49.2 ACK management
    # correctly" has a dot in the middle of it, and \w+ stops there.
    + r"|" + _ACT + r"\s+(?:[\w.\-]+\s+){0,5}?(?:correctly|properly)"
    # "is satisfied" and "is met" are how a rule is reported as done, and
    # only the third-person forms were listed: "Rule 8.4.1.1-2 is satisfied"
    # was not a verdict at all, so a sentence citing a clause nobody read
    # went unchecked.
    + r"|adheres?\s+to|in\s+accordance\s+with|satisf(?:ies|ied)|"
    r"(?:is|are|was|were)\s+met|meets\s+the|"
    r"follows\s+the\s+(?:standard|spec(?:ification)?)|"
    r"no\s+(?:adaptations?|changes?|modifications?|fixes?)\s+"
    r"(?:are|is|were)?\s*needed|"
    r"fully\s+(?:compliant|conformant|implements?))\b", re.I)

# What turns a verdict into its opposite. A bare "no" is NOT here: "No
# adaptations are needed -- the current implementation is compliant" is a
# clearance, and reading its first word as a negation exempted the very
# sentence this guard exists for.
_NEGATED = re.compile(
    r"\b(?:not|never|isn't|aren't|doesn't|does\s+not|is\s+not|are\s+not|"
    r"non-?compliant|non-?conformant|fails?\s+to|violates?|breaks?|"
    r"partially|incomplete)\b", re.I)

# Subjects that quantify over everything. A sentence with one of these and a
# verdict is unscoped even if a clause appears somewhere else in it: "the
# implementation is compliant, see Rule 8.4.1-2" still asserts the whole.
# Any adjective before the noun, not a list of them: "the whole", "the
# entire", "the overall" were enumerated and "the current implementation"
# walked straight through. A list of adjectives is the same mistake as a list
# of follow-up words.
# Subjects that quantify over everything. A sentence with one of these and a
# verdict is unscoped even if a clause appears somewhere else in it: "the
# implementation is compliant, see Rule 8.4.1-2" still asserts the whole.
# Any adjective before the noun, not a list of them: "the whole", "the
# entire", "the overall" were enumerated and "the current implementation"
# walked straight through. A list of adjectives is the same mistake as a list
# of follow-up words.
_GLOBAL_SUBJECT = re.compile(
    r"\b(?:(?:the|this|our|its)\s+(?:\w+\s+){0,2}?"
    r"(?:implementation|code(?:base)?|converter|controllee|design|system)|"
    r"the\s+(?:standard|spec(?:ification)?)(?:'s)?\s+"
    r"(?:as\s+a\s+whole|requirements?|in\s+(?:full|general))|"
    r"(?:all|every)\s+(?:of\s+)?(?:the\s+)?(?:rules?|requirements?|clauses?))\b",
    re.I)

# Naming the bound standard as the thing complied with -- "compliant with
# ANSI-VITA-49.2", "follows the VITA 49.2 specification" -- without a clause
# narrowing it.
#
# Built from the binding, never from a list. The first version spelled VITA
# 49.2 into the pattern, which made a guard that exists to catch an unscoped
# claim work for exactly one standard and silently pass the same sentence
# about any other. standard_scope already derives a subject's names from its
# own identifier; this is the same rule, and the reason is the same: a
# subject's name is not a guess about vocabulary.
_CLAIMS = r"(?:with|to|of|follows?|per|against)"


def bare_standard(binding=None):
    """Does a sentence name the bound standard itself, whole?

    With no binding there is nothing to name, and the pattern matches
    nothing rather than falling back to a hardcoded one.
    """
    spelled = standard_names(binding)

    if not spelled:
        return re.compile(r"(?!x)x")

    return re.compile(
        rf"\b{_CLAIMS}\s+(?:the\s+)?(?:{spelled})"
        rf"(?:\s+(?:standard|spec(?:ification)?))?\b", re.I)


def standard_names(binding=None):
    """The names this standard goes by, as one alternation, longest first.

    Longest first so "ansi-vita-49.2" wins over "vita", and each name is
    spelled loosely so the writer's punctuation does not decide the match.

    Public because two guards need the same spelling: this one asks whether
    a sentence claims conformance TO the standard, `normative_precedence`
    asks whether a sentence speaks FOR it, and a name spelled two ways is a
    subject recognised in one guard and missed in the other.
    """
    terms = _identity_terms(binding)

    if not terms:
        return ""

    return "|".join(_loose(term) for term in
                    sorted(terms, key=len, reverse=True))


def _loose(term):
    """One identity term, matching however the writer punctuated it."""
    return r"[\s\-/.]*".join(re.escape(part) for part in
                              re.split(r"[\s\-/.]+", term) if part)


class _Named:
    """Adapts a binding dict to what identity_terms reads."""

    def __init__(self, raw):
        self.standard_id = str(raw.get("standard_id") or "")
        self.revision = str(raw.get("revision") or "")


def _identity_terms(binding):
    """The names this standard goes by, from the binding and nowhere else.

    A binding travels as a dict through the runtime and as a StandardBinding
    inside the operator plane; identity_terms reads attributes, so a dict
    silently yielded nothing and the pattern built from it matched nothing at
    all. Both shapes are accepted here rather than at every call site.
    """
    if not binding:
        return ()

    import standard_scope

    subject = _Named(binding) if isinstance(binding, dict) else binding

    return tuple(term for term in standard_scope.identity_terms(subject)
                 if len(term) > 3)


# A clause identifier sitting immediately before the standard's name, which
# is what makes the name attributive rather than the thing being claimed.
_ATTRIBUTIVE = re.compile(
    r"(?:\b(?:rule|observation|recommendation|permission|definition|table|"
    r"figure|section|sect\.?|clause)\s*|§\s*)\d+(?:\.\d+)+(?:-\d+)?"
    r"[\s,]*(?:of|in|from|per)?\s*$", re.I)


def names_standard_as_object(sentence, pattern):
    """Is the bound standard what this sentence claims conformance TO?

    "conforms to ANSI-VITA-49.2, see Rule 8.4.1.1-2" claims the whole
    document and cites one rule as support; the citation does not narrow it.
    "Rule 8.4.1.1-2 of ANSI-VITA-49.2 is satisfied" claims the rule, and
    names the standard only to say which rule it is.

    The difference is positional and that is the whole of it: in the second,
    a clause identifier ends immediately before the name. Requiring the
    absence of any clause anywhere in the sentence -- the earlier rule --
    could not tell the two apart, so it let the first through.
    """
    for found in pattern.finditer(sentence or ""):
        if not _ATTRIBUTIVE.search(sentence[:found.start()]):
            return True

    return False


def _sentences(text):
    return [part for part in re.split(r"(?<=[.!?])\s+|\n+", text or "")
            if part.strip()]


def clauses_in(text):
    """Section numbers named in a text, as the standard numbers them."""
    return {match.group(1) for match in _CLAUSE.finditer(text or "")}


@dataclass
class ClauseLedger:
    """Exactly the clauses the tools put in front of the model this turn."""

    sections: set = field(default_factory=set)

    provisions: provision_identity.ProvisionLedger = field(

        default_factory=provision_identity.ProvisionLedger)

    def observe(self, payload, text=""):
        """One tool result. Sections come from the payload's own metadata
        first -- the retrieval names its section -- and from clause
        identifiers in the text second. Nothing is derived."""
        if isinstance(payload, dict):
            self._walk(payload)

        if text:
            self.sections |= clauses_in(text)

        if isinstance(payload, dict):
            self.provisions.observe(payload)

        return self

    def _walk(self, node):
        if isinstance(node, dict):
            section = node.get("section")

            if isinstance(section, str) and re.fullmatch(r"\d+(?:\.\d+)*",
                                                          section.strip()):
                self.sections.add(section.strip())

            for key in ("text", "snippet", "rendered", "heading_path"):
                value = node.get(key)

                if isinstance(value, str):
                    self.sections |= clauses_in(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, str):
                            self.sections |= clauses_in(item)

            for value in node.values():
                if isinstance(value, (dict, list)):
                    self._walk(value)
        elif isinstance(node, list):
            for item in node:
                self._walk(item)

    def covers_provision(self, key):
        """Was THIS provision retrieved?

        Provision-level grounding. `covers` below is section-level and must
        never stand in for this: a section holds many provisions, and on the
        bound standard 38% of bare labels name more than one of them.
        """
        return self.provisions.has(key)

    def kinds_sharing(self, section, ordinal):
        """Every provision kind retrieved under one section and ordinal."""
        return self.provisions.kinds_for(section, ordinal)

    def covers(self, section):
        """A retrieval of §8.4.1 covers a verdict on Rule 8.4.1-2, and a
        retrieval of §8.4.1.1 covers one on §8.4.1.1; a retrieval of the
        parent §8.4 alone does not cover a child clause, because the child's
        text was never in front of the model."""
        return any(known == section or known.startswith(section + ".")
                   for known in self.sections)

    def knows_anything(self):
        return bool(self.sections)


def findings(answer, ledger, binding=None):
    """Every affirmative verdict the turn is not entitled to."""
    found = []
    named_standard = bare_standard(binding)

    for sentence in _sentences(answer):
        if not _VERDICT.search(sentence) or _NEGATED.search(sentence):
            continue

        named = clauses_in(sentence)

        if _GLOBAL_SUBJECT.search(sentence) or names_standard_as_object(
                sentence, named_standard):
            found.append({"kind": UNSCOPED_VERDICT, "sentence": sentence,
                          "clauses": sorted(named)})
            continue

        if not named:
            found.append({"kind": UNSCOPED_VERDICT, "sentence": sentence,
                          "clauses": []})
            continue

        unread = sorted(section for section in named
                        if not ledger.covers(section))

        if unread:
            found.append({"kind": UNREAD_CLAUSE, "sentence": sentence,
                          "clauses": unread})

    return found


def _replacement(ledger, standard_id="", revision=""):
    name = " ".join(part for part in (standard_id, revision) if part) or (
        "the bound standard")
    read = sorted(ledger.sections, key=lambda s: [int(p) for p in s.split(".")])

    if read:
        listed = ", ".join(f"§{section}" for section in read[:12])
        if len(read) > 12:
            listed += f" and {len(read) - 12} more"

        return (f"No compliance verdict is issued. Clauses read this turn: "
                f"{listed}. Conformance with {name} as a whole was not "
                f"established by them.")

    return (f"No compliance verdict is issued: no clause of {name} was read "
            f"this turn.")


def sanitize(answer, ledger, *, standard_id="", revision="",
             binding=None):
    """The answer with each unentitled verdict replaced, once, by the record
    of what was read. A second unentitled sentence is cut rather than
    repeating the record."""
    text = answer or ""
    problems = findings(text, ledger, binding)

    if not problems:
        return text, []

    replaced = []
    note = _replacement(ledger, standard_id, revision)
    placed = False

    for problem in problems:
        sentence = problem["sentence"]

        if sentence not in text:
            continue

        text = text.replace(sentence, "" if placed else note, 1)
        placed = True
        replaced.append(sentence)

    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    return text, replaced


def guard(answer, ledger, *, standard_id="", revision="",
          binding=None):
    """Return the answer, or the same answer without its unentitled verdicts."""
    problems = findings(answer, ledger, binding)

    if not problems:
        return answer, [], [], False

    cleaned, replaced = sanitize(answer, ledger, standard_id=standard_id,
                                 revision=revision, binding=binding)

    return cleaned, problems, replaced, True
