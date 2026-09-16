"""Atomic retrieval, then a bounded walk to evidence it is incomplete without.

Two experiments established the shape of the problem. Atomic BM25 finds a
precise object well -- one table row among a hundred -- and misses the
provision that completes it. Ranking by structural container finds what lives
together and buries the precise object among its siblings. Neither ordering
serves both, because they are not the same question.

So the primary search is not touched. Its five hits keep their order and
their places; nothing here can displace one. What this adds is a small number
of COMPANIONS, reached from a primary hit along a relation the document
itself states -- a citation, an adjacent declaration, a shared field name --
each carrying the reason it was reached.

A companion is ordinary atomic evidence. An edge says "this may be relevant
too"; it never says "this inherits that provision's force". Guards go on
reading one provision at a time, and a Rule beside a Permission changes
nothing about the Permission.

Nothing here can see the benchmark. It reads no file, imports no evaluation
module, and is handed units and a query -- so it cannot be tuned toward
questions it is not allowed to know.
"""
from __future__ import annotations

import collections
import math
import re

import normative_claims as nc
import provision_identity as pi
from standard_retrieval import tokenize

class Bm25:
    """BM25 over a set of documents, with the atomic index's constants."""

    def __init__(self, documents):
        self.terms = [collections.Counter(tokenize(text)) for text in documents]
        self.frequency = collections.Counter()

        for item in self.terms:
            self.frequency.update(item.keys())

        self.lengths = [sum(item.values()) for item in self.terms]
        self.average = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0

    def score(self, query_terms, index):
        total, length = 0.0, self.lengths[index]

        for term in query_terms:
            count = self.terms[index].get(term, 0)

            if not count:
                continue

            n = self.frequency[term]
            idf = math.log(1 + (len(self.terms) - n + 0.5) / (n + 0.5))
            total += idf * count * (K1 + 1) / (
                count + K1 * (1 - B_PARAM + B_PARAM * length / (self.average or 1)))

        return total


def _searchable(unit):
    """What a unit contributes to its container's ranking text."""
    from standard_vector_index import structural_context

    return " ".join((*structural_context(unit), unit.get("text") or ""))


#: BM25 constants, the same the atomic index uses.
K1, B_PARAM = 1.5, 0.75

EXPLICIT_REFERENCE = "EXPLICIT_REFERENCE"
SHARED_IDENTIFIER = "SHARED_EXACT_TECHNICAL_IDENTIFIER"
DECLARATION_SIBLING = "IMMEDIATE_DECLARATION_SIBLING"
TABLE_IDENTIFIER = "TABLE_IDENTIFIER_RELATION"

#: Ordered by how much the document is actually claiming, and checked
#: against measured precision on one bound standard rather than assumed.
#:
#: A citation is the document saying outright that one provision bears on
#: another. An adjacent declaration under the same immediate parent is the
#: document's own ordering of a subject. A field name shared between a table
#: row and a provision is that row's own designation being discussed.
#:
#: A field name shared between two arbitrary provisions is the weakest, and
#: measurably so: on eight reviewed cases it offered 21 companions of which
#: 1 was useful, because a name like an acknowledge flag occurs in dozens of
#: provisions across a document. Ranked above the others it simply consumed
#: the budget. It is last for that reason -- a property of the relation, not
#: of any question.
PRIORITY = (EXPLICIT_REFERENCE, DECLARATION_SIBLING, TABLE_IDENTIFIER,
            SHARED_IDENTIFIER)

#: A printed reference to a table, alongside the provision labels that
#: `provision_identity` already recognises.
_TABLE_REFERENCE = re.compile(r"\bTable\s+(\d+(?:[.\-]\d+)*)", re.I)


def _identifiers(text):
    """Field-shaped names, by the extractor the guards already use."""
    return {token for token in nc._IDENTIFIER.findall(text or "")
            if token.upper() not in nc._NOT_IDENTIFIERS}


def _normalise(token):
    return re.sub(r"[-_\s]", "", token).lower()


class RelationGraph:
    """What the document says relates to what. Built from units alone."""

    def __init__(self, units):
        self.units = {unit["source_id"]: unit for unit in units}
        self.records = collections.defaultdict(list)
        self.by_identity = collections.defaultdict(set)
        self.by_identifier = collections.defaultdict(set)
        self.is_row = {}
        declarations = []

        for record in pi.records_for_units(units):
            self.records[record.source_id].append(record)
            key = record.key

            if isinstance(key, pi.ProvisionKey) and key.ordinal is not None:
                self.by_identity[str(key)].add(record.source_id)

                if record.declaration_status == pi.DECLARATION:
                    declarations.append((key.section, record.source_id, str(key)))
            elif isinstance(key, pi.TableRowKey):
                self.by_identity[f"TableRow {key.table}:{key.row}"].add(record.source_id)
                self.by_identity[f"Table {key.table}"].add(record.source_id)
                self.is_row[record.source_id] = True

        for source, unit in self.units.items():
            for token in _identifiers(unit.get("text") or ""):
                self.by_identifier[_normalise(token)].add(source)

        # Declarations under one immediate structural parent, in document
        # order. Only the neighbours on either side become edges: a section
        # is not a claim that everything in it is related.
        self.siblings = collections.defaultdict(list)

        for section, source, label in sorted(
                declarations, key=lambda item: (
                    item[0], self.units[item[1]].get("unit_position", 0), item[1])):
            self.siblings[section].append(source)

    def _cited(self, unit):
        text = unit.get("text") or ""
        found = set()

        for match in pi._LABEL.finditer(text):
            found |= self.by_identity.get(
                f"{match.group('kind')} {match.group('section')}"
                f"-{match.group('ordinal')}", set())

        for match in _TABLE_REFERENCE.finditer(text):
            found |= self.by_identity.get(f"Table {match.group(1)}", set())

        return found

    def neighbours(self, source_id):
        """(relation, target) for everything this unit is related to.

        Sorted before it is returned. The indexes behind it are sets, and set
        iteration of strings varies between processes -- so an unsorted walk
        made the RECORDED PROVENANCE of a companion differ from run to run
        while the evidence itself stayed identical. Evidence that is stable
        and a reason that is not is worse than either: it is an audit trail
        that cannot be reproduced.
        """
        unit = self.units.get(source_id)

        if unit is None:
            return []

        found = []

        for target in self._cited(unit):
            if target != source_id:
                found.append((EXPLICIT_REFERENCE, target))

        relation = TABLE_IDENTIFIER if self.is_row.get(source_id) else SHARED_IDENTIFIER

        for token in _identifiers(unit.get("text") or ""):
            for target in self.by_identifier.get(_normalise(token), ()):
                if target != source_id:
                    found.append((relation, target))

        row = self.siblings.get(unit.get("section") or "") or []

        if source_id in row:
            index = row.index(source_id)

            for step in (index - 1, index + 1):
                if 0 <= step < len(row):
                    found.append((DECLARATION_SIBLING, row[step]))

        return sorted(set(found), key=lambda item: (PRIORITY.index(item[0]),
                                                    item[1]))


class Completion:
    """Primary atomic evidence, plus companions reached through the graph."""

    def __init__(self, units):
        searchable = [unit for unit in units if unit.get("retrievable", True)]
        self.graph = RelationGraph(units)
        self.order = [unit["source_id"] for unit in searchable]
        self.position = {source: index for index, source in enumerate(self.order)}
        self.scorer = Bm25([_searchable(unit) for unit in searchable])

    def score(self, terms, source_id):
        index = self.position.get(source_id)

        return self.scorer.score(terms, index) if index is not None else 0.0

    def candidates(self, query, primary):
        """Every companion the graph offers, with why it was reached.

        One target reached by several relations keeps the strongest, so a
        provision that is both cited and adjacent is not counted twice.
        """
        terms = tokenize(query)
        best = {}

        for source in sorted(primary):
            for relation, target in self.graph.neighbours(source):
                if target in primary:
                    continue

                rank = PRIORITY.index(relation)
                seen = best.get(target)

                # A companion reachable from several primaries keeps the
                # strongest relation, and on a tie the lowest source id --
                # never whichever the walk happened to reach first.
                if seen is None or (rank, source) < (seen["priority"],
                                                     seen["reached_from"]):
                    best[target] = {"source_id": target, "relation": relation,
                                    "priority": rank, "reached_from": source,
                                    "score": self.score(terms, target)}

        return sorted(best.values(),
                      key=lambda item: (item["priority"], -item["score"],
                                        item["source_id"]))

    def complete(self, query, primary, budget=3):
        """The primary hits, in their own order, then up to `budget` companions."""
        chosen = self.candidates(query, list(primary))[:budget]

        return ([{"source_id": source, "role": "PRIMARY", "relation": None}
                 for source in primary]
                + [{"source_id": item["source_id"], "role": "COMPANION",
                    "relation": item["relation"],
                    "reached_from": item["reached_from"]} for item in chosen])
