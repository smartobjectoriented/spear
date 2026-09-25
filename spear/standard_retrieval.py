"""Deterministic lexical/vector retrieval over canonical standard units."""
from __future__ import annotations

import os
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone

from standard_schema import (
    StandardCitation, StandardIndexManifest, StandardSchemaError, sha256_json,
)
import standard_query_expansion
from standard_progress import report
from standard_store import StandardStore, StandardStoreError
from standard_vector_index import (
    LocalStandardEmbedder, structural_context,
)

INDEXER_VERSION = "standard-bm25-v1"
TOKENIZER_CONFIG = {"name": "casefold-alnum-compound", "version": 1}
_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[._/-][A-Za-z0-9]+)*")
# A part letter is allowed: lettered documents number D24.2.67, not 24.2.67.
_SECTION_QUERY = re.compile(r"^\s*(?:section|§)?\s*([A-Z]?\d+(?:\.\d+)+)\s*$", re.I)
_SOURCE_QUERY = re.compile(r"^std-[0-9a-f]{32}$")


@dataclass(frozen=True)
class StandardRetrievalPolicy:
    lexical_candidate_count: int = 20
    vector_candidate_count: int = 20
    final_top_k: int = 5
    rrf_constant: int = 60
    neighbor_context_size: int = 1
    cross_reference_expansion_limit: int = 2
    version: str = "standard-retrieval-policy-v1"

    def fingerprint(self) -> str:
        return sha256_json(self.__dict__)


def tokenize(text: str) -> tuple[str, ...]:
    return tuple(match.group(0).casefold() for match in _TOKEN.finditer(text))


def rebuild_lexical_index(store: StandardStore, standard_id: str, revision: str, *,
                          indexer_version: str = INDEXER_VERSION,
                          created_at: str | None = None,
                          progress=None) -> StandardIndexManifest:
    report(progress, "verifying corpus")
    source = store.verify_corpus(standard_id, revision)

    # Page furniture and contents entries stay in the canonical corpus for
    # provenance, but they are navigation, not normative text to retrieve.

    units = [unit for unit in store.load_units(standard_id, revision)
             if unit.retrievable]

    if not units:
        raise StandardStoreError("canonical corpus has no retrievable units")

    documents = {}
    document_frequency: Counter[str] = Counter()

    for position, unit in enumerate(units, 1):
        report(progress, "lexical index", position, len(units))
        # The same structural context the vector side indexes, so one
        # representation is searched both ways. Empty for a unit whose store
        # recorded no grid, which keeps an older corpus byte-identical.
        tokens = tokenize(" ".join((unit.section or "", *unit.heading_path,
                                    *structural_context(unit), unit.text)))
        frequencies = Counter(tokens)
        documents[unit.source_id] = {"length": len(tokens),
                                     "terms": dict(sorted(frequencies.items()))}
        document_frequency.update(frequencies.keys())

    index = {
        "schema_version": 1, "standard_id": standard_id, "revision": revision,
        "document_count": len(units),
        "average_document_length": sum(v["length"] for v in documents.values()) / len(units),
        "documents": dict(sorted(documents.items())),
        "document_frequency": dict(sorted(document_frequency.items())),
        "tokenizer_config": TOKENIZER_CONFIG, "bm25": {"k1": 1.5, "b": 0.75},
    }
    manifest = StandardIndexManifest(
        standard_id, revision, source.source_pdf_sha256, source.corpus_manifest_sha256,
        "bm25", 1, indexer_version, sha256_json(TOKENIZER_CONFIG),
        sha256_json({"indexer_version": indexer_version,
                     "source_corpus_sha256": source.corpus_manifest_sha256,
                     "index": index}),
        created_at or datetime.now(timezone.utc).isoformat(timespec="seconds"))
    store.save_index(manifest, index)

    return manifest


@dataclass(frozen=True)
class StandardSearchResult:
    source_id: str
    standard_id: str
    revision: str
    section: str | None
    page: int
    heading_path: tuple[str, ...]
    content_type: str
    modality: str
    snippet: str
    score: float
    rank: int
    parent_source_id: str | None
    lexical_rank: int | None = None
    vector_rank: int | None = None
    fused_rank: int | None = None
    match_reason: str = "lexical"
    exact_section_match: bool = False
    exact_term_match: bool = False
    referenced_from: str | None = None
    cross_reference_relation: str | None = None
    lexical_score: float | None = None
    vector_score: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "source_id": self.source_id, "standard_id": self.standard_id,
            "revision": self.revision, "section": self.section, "page": self.page,
            "heading_path": list(self.heading_path), "content_type": self.content_type,
            "modality": self.modality, "snippet": self.snippet, "score": self.score,
            "lexical_score": self.lexical_score, "vector_score": self.vector_score,
            "rank": self.rank, "lexical_rank": self.lexical_rank,
            "vector_rank": self.vector_rank, "fused_rank": self.fused_rank,
            "match_reason": self.match_reason,
            "exact_section_match": self.exact_section_match,
            "exact_term_match": self.exact_term_match,
            "parent_source_id": self.parent_source_id,
            "referenced_from": self.referenced_from,
            "cross_reference_relation": self.cross_reference_relation,
        }


#: What retrieval a run actually HAD, as opposed to what it asked for.
#: "lexical_fallback" says a fallback happened and not what was lost: asking
#: for hybrid on a store with no vector index, and asking for vector on the
#: same store, are one string. An evaluation that reports "hybrid retrieval"
#: while the store has never been embedded is describing something that did
#: not happen, and that is how a retrieval regression gets read as a parsing
#: one.
#: Evidence completion is a stage AFTER retrieval, not a retrieval mode. It
#: never reorders or displaces a primary hit; it appends a bounded number of
#: atoms reached along a relation the document itself states. Reported
#: separately for exactly that reason -- describing it as hybrid retrieval
#: would claim a ranking change that did not happen.
STRUCTURAL_COMPLETION = "STRUCTURAL_COMPLETION"
NO_COMPLETION = "NONE"

HYBRID_LEXICAL_VECTOR = "HYBRID_LEXICAL_VECTOR"
LEXICAL_ONLY = "LEXICAL_ONLY"
HYBRID_DEGRADED_NO_VECTOR = "HYBRID_DEGRADED_NO_VECTOR"
VECTOR_ONLY = "VECTOR_ONLY"
EXACT_SOURCE = "EXACT_SOURCE"


@dataclass(frozen=True)
class StandardSearchResponse:
    results: tuple[StandardSearchResult, ...]
    retrieval_mode_requested: str
    retrieval_mode_used: str
    lexical_candidate_count: int
    vector_candidate_count: int
    expanded_cross_reference_ids: tuple[str, ...]
    retrieval_fingerprint: str

    #: One of the names above. Kept beside `retrieval_mode_used` rather than
    #: replacing it: the older string is what existing readers match on.
    retrieval_capability: str = ""

    #: STRUCTURAL_COMPLETION when companions were appended, else NONE.
    evidence_completion: str = NO_COMPLETION
    completion_budget: int = 0


def _capability(requested, used):
    """What the run HAD: both rankings, one by choice, or one by absence."""
    if used == "lexical_fallback":
        return HYBRID_DEGRADED_NO_VECTOR

    if used == "lexical":
        return LEXICAL_ONLY

    if used == "vector":
        return VECTOR_ONLY

    return HYBRID_LEXICAL_VECTOR


#: How much of a unit a search hit shows. Unchanged; what changed is that a
#: cut now says so instead of trailing off.

_SNIPPET_CHARS = 497


class StandardRetrieval:
    _completion = None

    def __init__(self, store: StandardStore,
                 embedder: LocalStandardEmbedder | None = None,
                 policy: StandardRetrievalPolicy | None = None) -> None:
        self.store, self.embedder = store, embedder
        self.policy = policy or StandardRetrievalPolicy()
        self.last_response: StandardSearchResponse | None = None
        #: Which query terms brought in which corpus-native family terms,
        #: for the record a diagnosis needs and the answer never sees.
        self.last_expansion: dict = {}

    def _lexical(self, standard_id, revision, query, section):
        _, index = self.store.load_index(standard_id, revision)
        units = {u.source_id: u for u in self.store.load_units(standard_id, revision)}
        count, avg = int(index["document_count"]), float(index["average_document_length"] or 1)

        # A question asks with the stem; the document defines the family. The
        # terms added here are the bound standard's own, never a synonym from
        # somewhere else, and nothing is removed -- a query with no family in
        # this corpus scores exactly as it did before.
        terms, self.last_expansion = standard_query_expansion.expand(
            tokenize(query), index["document_frequency"],
            index["document_frequency"], count, query=query)
        k1, b = float(index["bm25"]["k1"]), float(index["bm25"]["b"])
        match = _SECTION_QUERY.match(query)
        exact_section = match.group(1).upper() if match else None
        scored = []

        for source_id, document in index["documents"].items():
            unit = units[source_id]

            if section is not None and unit.section != section:
                continue

            score = 0.0

            # Standard BM25: a term contributes in proportion to how often it
            # appears here and how rare it is across the corpus, damped by k1
            # and normalised for document length by b.

            for term in terms:
                frequency = int(document["terms"].get(term, 0))

                if frequency:
                    frequency_docs = int(index["document_frequency"].get(term, 0))
                    inverse = math.log(1 + (count - frequency_docs + .5) /
                                       (frequency_docs + .5))
                    norm = frequency + k1 * (1 - b + b * int(document["length"]) / avg)
                    score += inverse * frequency * (k1 + 1) / norm

            # A query that names a section is asking for that section, so it
            # outranks anything term matching could produce.

            if exact_section is not None and exact_section == unit.section:
                score += 1000

            if score > 0:
                scored.append((score, source_id))

        return sorted(scored, key=lambda x: (-x[0], x[1]))[:self.policy.lexical_candidate_count]

    def _vector(self, standard_id, revision, query, section):
        if self.embedder is None:
            raise StandardStoreError("no local standard embedder is configured")

        manifest, index = self.store.load_vector_index(standard_id, revision)

        # Vectors are only comparable within one model at one revision, so a
        # mismatch is refused rather than searched against.

        if (manifest.embedding_model_id, manifest.embedding_model_revision) != (
                self.embedder.model_id, self.embedder.model_revision):
            raise StandardStoreError("configured embedding model does not match vector index")

        query_vector = self.embedder.embed_query(query)

        if len(query_vector) != manifest.embedding_dimension:
            raise StandardStoreError("vector dimension mismatch")

        import numpy

        ids = index["ids"]
        scores = index["matrix"] @ numpy.asarray(query_vector, dtype=numpy.float32)
        rows = numpy.arange(len(ids))

        if section is not None:
            units = {u.source_id: u for u in self.store.load_units(standard_id, revision)}
            rows = numpy.array([row for row, source_id in enumerate(ids)
                                if units[source_id].section == section], dtype=int)

        count = self.policy.vector_candidate_count

        # Everything scoring at least the count-th best is kept before the
        # exact sort, so a tie at the cut is broken by source id exactly as a
        # full sort would break it.
        if len(rows) > count:
            kth = numpy.partition(scores[rows], len(rows) - count)[len(rows) - count]
            rows = rows[scores[rows] >= kth]

        scored = [(float(scores[row]), ids[row]) for row in rows]

        return sorted(scored, key=lambda x: (-x[0], x[1]))[:count]

    def search_response(self, standard_id: str, revision: str, query: str, *,
                        section: str | None = None, limit: int | None = None,
                        mode: str = "hybrid", expand_cross_references: bool = True
                        ) -> StandardSearchResponse:
        """Retrieve against the bound standard, and record how it was retrieved."""

        if not query.strip():
            raise ValueError("standard search query must not be empty")

        limit = self.policy.final_top_k if limit is None else limit

        if not 1 <= limit <= 20 or mode not in {"lexical", "vector", "hybrid"}:
            raise ValueError("invalid standard retrieval limit or mode")

        binding = self.store.binding(standard_id, revision)

        # A bare source id is a direct lookup, not a search: it is answered
        # exactly and skips scoring altogether.

        if _SOURCE_QUERY.fullmatch(query.strip()):
            unit = self.store.resolve_source(standard_id, revision, query.strip())
            values = () if section is not None and unit.section != section else (
                StandardSearchResult(
                    unit.source_id, unit.standard_id, unit.revision, unit.section,
                    unit.page, unit.heading_path, unit.content_type.value,
                    unit.modality.value, unit.text[:500], 2000.0, 1,
                    unit.parent_source_id, 1, None, None, "exact_source",
                    False, True),)
            response = StandardSearchResponse(
                values, mode, "exact_source", len(values), 0, (),
                binding.retrieval_fingerprint or binding.index_fingerprint,
                retrieval_capability=EXACT_SOURCE)
            self.last_response = response

            return response

        lexical = self._lexical(standard_id, revision, query, section) if mode != "vector" else []
        vector, used = [], mode
        #: Set when a vector index was asked for and could not be used, so
        #: the reason survives into the report rather than being inferred.
        vector_unavailable = None

        # Vector search is optional infrastructure; without it retrieval falls
        # back to lexical and SAYS so, rather than silently returning less.

        if mode != "lexical":
            try:
                vector = self._vector(standard_id, revision, query, section)
            except (FileNotFoundError, StandardStoreError, RuntimeError) as exc:
                used = "lexical_fallback"
                vector_unavailable = f"{type(exc).__name__}: {exc}"

                if mode == "vector":
                    lexical = self._lexical(standard_id, revision, query, section)

        lr = {sid: rank for rank, (_, sid) in enumerate(lexical, 1)}
        vr = {sid: rank for rank, (_, sid) in enumerate(vector, 1)}
        ls = {sid: score for score, sid in lexical}
        vs = {sid: score for score, sid in vector}

        # Hybrid mode fuses the two rankings by reciprocal rank: each list
        # contributes 1/(k + rank), so a result ranked well by either is kept
        # without having to make two incomparable scores comparable.

        if used in {"lexical", "lexical_fallback"}:
            ordered = list(lexical)
        elif used == "vector":
            ordered = list(vector)
        else:
            ordered = [(sum(1 / (self.policy.rrf_constant + rank) for rank in
                            (lr.get(sid), vr.get(sid)) if rank is not None), sid)
                       for sid in set(lr) | set(vr)]

        units = {u.source_id: u for u in self.store.load_units(standard_id, revision)}
        structural = _SECTION_QUERY.match(query)
        exact_section = structural.group(1).upper() if structural else None
        ordered.sort(key=lambda x: (-int(exact_section is not None and
                                          units[x[1]].section == exact_section),
                                    -x[0], x[1]))
        primary = ordered[:limit]
        expanded = []

        # A clause that refers to another is incomplete without it, so a
        # bounded number of resolved references are pulled in alongside.

        if expand_cross_references and primary:
            try:
                _, crossrefs = self.store.load_cross_reference_index(standard_id, revision)
                selected = {sid for _, sid in primary}

                for _, source_id in primary:
                    for relation in crossrefs["entries"].get(source_id, []):
                        if relation.get("status") == "resolved":
                            for target in relation.get("target_source_ids", []):
                                if target not in selected:
                                    expanded.append((0.0, target, source_id))
                                    selected.add(target)

                                    if len(expanded) >= self.policy.cross_reference_expansion_limit:
                                        break

                        if len(expanded) >= self.policy.cross_reference_expansion_limit:
                            break

                    if len(expanded) >= self.policy.cross_reference_expansion_limit:
                        break
            except (FileNotFoundError, StandardStoreError):
                pass

        # Evidence a primary hit is incomplete without, appended after it
        # and never in front of it. Off unless an operator asks for it, so a
        # store that has not been measured with it behaves exactly as before.
        budget = self._completion_budget(standard_id, revision)
        companions = self._companions(query, [sid for _, sid in primary],
                                      expanded, units, budget)

        results = []
        query_terms = set(tokenize(query))

        relations = {item["source_id"]: item for item in companions}

        for rank, (score, source_id, referenced_from) in enumerate(
                [(score, sid, None) for score, sid in primary] + expanded
                + [(0.0, item["source_id"], item["reached_from"])
                   for item in companions], 1):
            unit = units[source_id]
            exact = exact_section is not None and unit.section == exact_section
            exact_term = bool(query_terms) and query_terms.issubset(set(tokenize(unit.text)))
            companion = relations.get(source_id)
            reason = (companion["relation"] if companion
                      else "cross_reference" if referenced_from
                      else "exact_section" if exact
                      else "hybrid" if source_id in lr and source_id in vr
                      else "vector" if source_id in vr else "lexical")
            # A cut snippet that ends in an ellipsis reads as a finished
            # clause with a trailing pause. One did: the cap fell inside a
            # rule of the form "... shall be generated when ...", mid-
            # sentence and just before the condition that mattered, and the
            # turn concluded from the two complete rules before it, never
            # learning that the third contradicted them. Say what was cut,
            # and say what to call to read the rest.

            if len(unit.text) <= _SNIPPET_CHARS:
                snippet = unit.text
            else:
                snippet = (
                    unit.text[:_SNIPPET_CHARS].rstrip()
                    + f" […TRUNCATED at {_SNIPPET_CHARS} of {len(unit.text)} "
                      f"characters. This is NOT the complete clause and may "
                      f"stop mid-sentence; call standard.fetch on this "
                      f"source_id to read it in full.]")
            results.append(StandardSearchResult(
                unit.source_id, unit.standard_id, unit.revision, unit.section, unit.page,
                unit.heading_path, unit.content_type.value, unit.modality.value,
                snippet, round(score, 8), rank, unit.parent_source_id,
                lr.get(source_id), vr.get(source_id),
                rank if used == "hybrid" and referenced_from is None else None,
                reason, exact, exact_term, referenced_from,
                "direct_one_hop" if referenced_from else None,
                round(ls[source_id], 8) if source_id in ls else None,
                round(vs[source_id], 8) if source_id in vs else None))

        response = StandardSearchResponse(
            tuple(results), mode, used, len(lexical), len(vector),
            tuple(sid for _, sid, _ in expanded),
            binding.retrieval_fingerprint or binding.index_fingerprint,
            retrieval_capability=_capability(mode, used),
            evidence_completion=(STRUCTURAL_COMPLETION if companions
                                 else NO_COMPLETION if not budget
                                 else STRUCTURAL_COMPLETION),
            completion_budget=budget)
        self.last_response = response

        return response

    def _completion_budget(self, standard_id: str, revision: str) -> int:
        """How many companions an operator has asked for. Zero means off.

        SPEAR_STANDARD_EVIDENCE_COMPLETION wins when set, for one session;
        otherwise the document's own setting, since that is what was measured.
        """
        configured = os.environ.get("SPEAR_STANDARD_EVIDENCE_COMPLETION", "").strip()

        if configured:
            try:
                return max(0, int(configured))
            except ValueError:
                return 0

        try:
            declared = self.store.load_retrieval_settings(standard_id, revision)
        except StandardStoreError:
            return 0

        return int(declared.get("evidence_completion", 0))

    def _companions(self, query, primary, expanded, units, budget):
        """Atoms reached from the primary hits along a document relation.

        Built from the corpus alone. The benchmark is not readable from here
        and must never be: a retriever that could see its own gold could be
        tuned to questions it is meant to answer blind.
        """
        if not budget or not primary:
            return []

        import standard_evidence_completion

        if self._completion is None:
            self._completion = standard_evidence_completion.Completion(
                [unit.to_dict() for unit in units.values()])

        already = set(primary) | {sid for _, sid, _ in expanded}
        found = self._completion.candidates(query, list(already))

        return found[:budget]

    def search(self, standard_id: str, revision: str, query: str, **kwargs):
        return self.search_response(standard_id, revision, query, **kwargs).results

    def fetch(self, standard_id: str, revision: str, source_id: str, *,
              include_parent: bool = True, neighbor_limit: int = 1) -> dict[str, object]:
        unit = self.store.resolve_source(standard_id, revision, source_id)
        parent_unit = (self.store.resolve_source(standard_id, revision,
                                                 unit.parent_source_id)
                       if include_parent and unit.parent_source_id else None)
        parent = self._with_citation(parent_unit) if parent_unit else None
        ordered = sorted(self.store.load_units(standard_id, revision),
                         key=lambda x: (x.page, x.section or "", x.source_id))
        position = next(i for i, item in enumerate(ordered) if item.source_id == source_id)
        window = (ordered[max(0, position-neighbor_limit):position] +
                  ordered[position+1:position+1+neighbor_limit])
        neighbors = [self._with_citation(item) for item in window
                     if item.section == unit.section]
        resolved, unresolved = [], []

        try:
            _, crossrefs = self.store.load_cross_reference_index(standard_id, revision)

            for relation in crossrefs["entries"].get(source_id, []):
                (resolved if relation.get("status") == "resolved" else unresolved).append(
                    dict(relation))
        except (FileNotFoundError, StandardStoreError):
            unresolved = [{"literal": literal, "status": "unresolved"}
                          for literal in unit.cross_references]

        citation = StandardCitation(unit.standard_id, unit.revision, unit.section,
                                    unit.page, unit.source_id)

        return {"unit": self._with_citation(unit), "parent": parent,
                "neighbors": neighbors,
                "resolved_cross_references": resolved,
                "unresolved_cross_references": unresolved,
                "citation": citation.to_dict()}

    @staticmethod
    def _with_citation(unit) -> dict[str, object]:
        """One unit, carrying the citation it would be cited by.

        Every field a citation needs is already on the unit -- standard_id,
        revision, section, page, source_id -- so the only thing a separate
        `cite` call ever added for a unit already in hand was the RENDERED
        string, and each of those cost a model round trip. One turn spent
        eighteen of them, seventeen on units a fetch had just returned.

        The rendered string alone, not the whole citation dict: repeating
        five fields the unit already carries would grow every fetch payload
        by a fifth and push results that used to arrive whole over the
        result-store threshold -- buying a citation at the price of the text
        it cites. Built by StandardCitation.render(), the same formatter
        `cite` uses, so the two can never render one source differently.

        A unit whose identity will not make a citation (a page the extractor
        never established) is returned without one rather than failing the
        fetch: the text is still evidence, and the missing citation is
        exactly what `cite` is for.
        """

        payload = unit.to_dict()

        try:
            payload["citation_rendered"] = StandardCitation(
                unit.standard_id, unit.revision, unit.section,
                unit.page, unit.source_id).render()
        except (StandardSchemaError, TypeError, ValueError):
            pass

        return payload

    def cite(self, standard_id: str, revision: str, source_id: str) -> StandardCitation:
        unit = self.store.resolve_source(standard_id, revision, source_id)
        return StandardCitation(unit.standard_id, unit.revision, unit.section,
                                unit.page, unit.source_id)
