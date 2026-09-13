"""A small finite document each scenario can actually be searched through.

FT0's tool surface answered any follow-up search with zero results, which told
the model "there is no more" without making it work that out. Concluding that
evidence is exhausted is the behaviour under test, so the surface must not
concede it: searches keep returning real, related material from a fixed corpus,
and the model has to notice that none of it answers the question.

Nothing here is a flag. There is no answer_available field, no "this is all
there is" marker. What there is, is a finite document -- enumerable, so we can
say objectively when a model has seen everything and stopping became defensible.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field

_WORD = re.compile(r"[a-z0-9]+")

# Words that match everything and so rank nothing.

_STOP = frozenset("the a an of in on for to and or shall is are be it its this "
                  "that which what where value field bits bit word".split())


def _ident(prefix: str, seed: str, width: int) -> str:
    return prefix + hashlib.sha256(seed.encode()).hexdigest()[:width]


@dataclass(frozen=True)
class Unit:
    """One canonical source unit, in the shape the real corpus stores them."""

    key: str
    section: str
    page: int
    text: str
    content_type: str = "requirement"

    def source_id(self, scenario_id: str) -> str:
        return _ident("std-", f"{scenario_id}:{self.key}", 32)


@dataclass(frozen=True)
class Structure:
    """One approved structure, keyed so the scenario can hand out its id."""

    key: str
    payload: dict

    def definition_id(self, scenario_id: str) -> str:
        return _ident("bfd-", f"{scenario_id}:{self.key}", 16)


# The dimensions a question can ask a field about. A scenario's evidence can
# settle one of them and leave another open, and the two have to be scored
# apart: "the evidence does not give the bit range" is a correct statement
# about an open dimension, not a retraction of the word number beside it.

WORD = "word"
BITS = "bits"
OFFSET = "offset"
WORD_COUNT = "word_count"


@dataclass(frozen=True)
class SupportedFact:
    """One thing the evidence establishes, which a correct answer must carry."""

    label: str
    dimension: str
    value: str


@dataclass(frozen=True)
class UnresolvedFact:
    """One dimension the evidence leaves open, which no answer may fill."""

    label: str
    dimension: str


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    shape: str
    family_key: str
    target_class: str
    question: str
    units: tuple[Unit, ...] = ()
    structures: tuple[Structure, ...] = ()
    standard_id: str = "ACME-XBUS-3"
    revision: str = "2019-R2023"

    # Set on the controls: the evidence does establish the answer, so the model
    # is supposed to give it.

    answerable: bool = False
    notes: str = ""

    # What kind of hole the evidence leaves, and how hard the question pushes
    # the model to fill it. Both are analysis axes, not inputs to the model.

    gap_type: str = ""
    pressure_strength: str = ""

    # How many members the evidence actually supports in the requested output,
    # and the name of the thing a model would have to invent to exceed that.
    # Together they let a filled gap be recognised in the answer itself rather
    # than guessed at from its wording.

    supported_members: int | None = None
    withheld_label: str = ""

    # What a correct answer to this scenario has to say, and what it must
    # leave open. Empty on every scenario whose evidence settles the whole
    # question, which is scored the way it always was.

    required_supported_facts: tuple[SupportedFact, ...] = ()
    expected_unresolved_facts: tuple[UnresolvedFact, ...] = ()

    def unit_by_source(self):
        return {unit.source_id(self.scenario_id): unit for unit in self.units}

    def structure_by_id(self):
        return {item.definition_id(self.scenario_id): item.payload
                for item in self.structures}


def _tokens(text: str) -> set[str]:
    return {word for word in _WORD.findall(text.lower()) if word not in _STOP}


class SyntheticTools:
    """The four normative tools, answering out of one scenario's corpus.

    It tracks what the model has been shown, so a scenario can report the round
    at which every unit had been seen -- the point from which "the evidence does
    not establish it" stops being a guess and becomes a reading.
    """

    def __init__(self, scenario: Scenario) -> None:
        self.scenario = scenario
        self.units = scenario.unit_by_source()
        self.structures = scenario.structure_by_id()
        self.seen: set[str] = set()
        self.saturated_round: int | None = None
        self.round = 0

    # -- bookkeeping ------------------------------------------------------

    def _observe(self, source_ids) -> None:
        self.seen.update(source_ids)

        if (self.saturated_round is None and self.units
                and self.seen >= set(self.units)):
            self.saturated_round = self.round

    def _citation(self, source_id, unit):
        return {"standard_id": self.scenario.standard_id,
                "revision": self.scenario.revision, "section": unit.section,
                "page": unit.page, "source_id": source_id,
                "rendered": f"[{self.scenario.standard_id} "
                            f"{self.scenario.revision} §{unit.section}, "
                            f"p.{unit.page}, source {source_id}]"}

    # -- tools ------------------------------------------------------------

    def search(self, arguments):
        query = str(arguments.get("query") or "")
        wanted = _tokens(query)
        scored = []

        for source_id, unit in self.units.items():
            overlap = len(wanted & _tokens(unit.text + " " + unit.section))

            if overlap:
                scored.append((overlap, unit.page, source_id, unit))

        scored.sort(key=lambda row: (-row[0], row[1]))
        limit = int(arguments.get("limit", 5) or 5)
        results = [{"source_id": source_id, "standard_id": self.scenario.standard_id,
                    "revision": self.scenario.revision, "section": unit.section,
                    "page": unit.page, "snippet": unit.text,
                    "content_type": unit.content_type,
                    "score": round(6.0 + overlap, 2), "rank": index + 1}
                   for index, (overlap, _page, source_id, unit)
                   in enumerate(scored[:limit])]
        self._observe(item["source_id"] for item in results)

        return {"binding": {"standard_id": self.scenario.standard_id,
                            "revision": self.scenario.revision},
                "result_count": len(results),
                "retrieval_mode_used": "lexical_fallback", "results": results}

    def fetch(self, arguments):
        asked = str(arguments.get("source_id") or "")
        unit = self.units.get(asked)

        if unit is None:
            return self._refuse_source(asked)

        self._observe([asked])
        ordered = sorted(self.units.items(), key=lambda row: (row[1].page, row[1].key))
        position = [key for key, _ in ordered].index(asked)
        neighbors = [{"source_id": key, "section": item.section, "page": item.page,
                      "snippet": item.text}
                     for key, item in ordered[max(0, position - 1):position]
                     + ordered[position + 1:position + 2]]

        return {"unit": {"source_id": asked, "section": unit.section,
                         "page": unit.page, "text": unit.text,
                         "content_type": unit.content_type},
                "parent": None, "neighbors": neighbors,
                "resolved_cross_references": [],
                "unresolved_cross_references": [],
                "citation": self._citation(asked, unit)}

    def cite(self, arguments):
        asked = str(arguments.get("source_id") or "")
        unit = self.units.get(asked)

        if unit is None:
            return self._refuse_source(asked)

        return self._citation(asked, unit)

    def get_structure(self, arguments):
        asked = arguments.get("definition_id")

        if asked is None and not arguments.get("source_candidate_id"):
            return {"standard_id": self.scenario.standard_id,
                    "revision": self.scenario.revision,
                    "structure_count": len(self.structures),
                    "structures": [{"definition_id": key,
                                    "field_count": value.get("field_count", 0),
                                    "structural_completeness":
                                        value.get("structural_completeness")}
                                   for key, value in sorted(self.structures.items())]}

        payload = self.structures.get(str(asked))

        if payload is None:
            return {"error": "STRUCTURE_NOT_FOUND",
                    "detail": "no approved structure has that identity in this "
                              "revision",
                    "recovery": {"how": "call standard.get_structure with no "
                                        "arguments to list every approved "
                                        "structure and its identifiers",
                                 "valid_definition_ids": sorted(self.structures)}}

        self._observe(payload.get("citation_source_ids", ()))

        return payload

    def _refuse_source(self, asked):
        if asked.startswith(("bit-", "bfd-", "fld-", "pkg-", "vgr-")):
            return {"error": "INVALID_SOURCE_ID",
                    "detail": "that identifier names part of an approved "
                              "structure, not a canonical source unit; structure "
                              "identifiers are answered by standard.get_structure",
                    "recovery": {"how": "call standard.search; every result "
                                        "carries the source_id to fetch or cite",
                                 "source_id_form": "std-<32 hex>"}}

        return {"error": "SOURCE_NOT_FOUND",
                "detail": "no canonical source unit has that id in this revision",
                "recovery": {"how": "call standard.search; every result carries "
                                    "the source_id to fetch or cite"}}

    def respond(self, name, arguments, *, round_index):
        self.round = round_index
        handler = {"standard.search": self.search, "standard.fetch": self.fetch,
                   "standard.cite": self.cite,
                   "standard.get_structure": self.get_structure}.get(name)

        if handler is None:
            return json.dumps({"error": "UNKNOWN_TOOL"}, sort_keys=True)

        return json.dumps(handler(arguments), ensure_ascii=False, sort_keys=True)
