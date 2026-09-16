"""Read-only model tools scoped to one operator-pinned StandardBinding."""

from __future__ import annotations

import evidence_graph

import hashlib
import json
import re
from typing import Mapping

from standard_retrieval import StandardRetrieval
from standard_structure_access import StandardStructureAccess, StructureAccessError
from standard_schema import StandardBinding
from standard_store import StandardStore
from standard_vector_index import configured_local_embedder
from tool_registry import (
    ToolCategory, ToolMutability, ToolRegistry, ToolResultPolicy, ToolSpec,
)
from tool_router import ToolExecutionContext, ToolHandlerResult


STANDARD_TOOL_NAMES = frozenset({"standard.search", "standard.fetch",
                                 "standard.cite", "standard.get_structure"})

# A request that cannot be made this way, as opposed to one this revision has
# no answer for. The caller has to be able to tell those apart: the second is
# evidence about the standard, the first is only evidence about the caller.

INVALID_SOURCE_ID = "INVALID_SOURCE_ID"
SOURCE_NOT_FOUND = "SOURCE_NOT_FOUND"
INVALID_QUERY = "INVALID_QUERY"

# The shape the canonical corpus hands out.

_SOURCE_ID = re.compile(r"^std-[0-9a-f]{32}$")

# Identifiers that name parts of an approved structure. They are real
# identifiers, just not for this tool, so a caller holding one is told where it
# does belong rather than that it is nonsense.

_STRUCTURE_PREFIXES = ("bit-", "bfd-", "fld-", "pkg-", "vgr-")


# How much of a neighbour or a parent is worth showing beside the unit that
# was actually asked for. Enough to recognise it and decide whether to fetch
# it; never enough to crowd out the text the call was about.

_CONTEXT_PREVIEW_CHARS = 240

# The fetch output budget, named once: the spec below and the renderer above
# must agree, or the renderer fills a window the policy then cuts.

_FETCH_RESULT_POLICY = ToolResultPolicy(4000, True, 1800)


def _attach_table_rows(fetched):
    """Link a fetched table caption to the row units beside it.

    Conservative: a neighbour joins only if it is shaped like a row of a
    table. Adjacency alone would attach the paragraph after the table, and a
    table's neighbourhood is mostly prose.
    """
    unit = fetched.get("unit")

    if not isinstance(unit, dict) or not evidence_graph.is_caption(unit):
        return fetched

    neighbours = [item for item in (fetched.get("neighbors") or [])
                  if isinstance(item, dict)]
    tables = evidence_graph.tables_in([unit] + neighbours)
    rows = tables[0].rows if tables else []

    if not rows:
        return fetched

    found = dict(fetched)
    found["table_rows"] = rows

    return found


def _fetch_policy() -> ToolResultPolicy:
    return _FETCH_RESULT_POLICY


def _unit_line(unit: Mapping[str, object]) -> str:
    """One neighbour or parent: what it is, then as much of it as fits."""

    text = " ".join(str(unit.get("text") or "").split())
    citation = str(unit.get("citation_rendered")
                   or f"source {unit.get('source_id')}")

    if len(text) > _CONTEXT_PREVIEW_CHARS:
        text = text[:_CONTEXT_PREVIEW_CHARS].rstrip() + " […]"

    return f"- {citation}\n  {text}" if text else f"- {citation}"


def render_fetch_for_model(fetched: Mapping[str, object], *, budget: int,
                           text_offset: int = 0) -> str:
    """What the model is shown for one fetched unit, in priority order.

    The text that was asked for comes FIRST and comes whole; its citation
    follows; parent, neighbours and cross-references get what is left, and
    are dropped rather than allowed to push the text out. The structured
    result is unchanged and still stored in full.

    This exists because the opposite order was measured. The result was
    serialised with sorted keys, which put `citation`, `neighbors` and
    `parent` ahead of `unit`, and the size cap then kept the first 1800
    characters: six fetches out of nine in one turn showed the model no unit
    text at all. One of them was the clause that contradicted the answer the
    turn went on to give -- the model had asked for exactly the right source
    and been handed its neighbours' checksums.

    JSON is not the shape for this either: the same 1754 characters of clause
    cost about a quarter more once escaped and wrapped in field names, and
    that quarter is the part that gets cut.
    """

    unit = dict(fetched.get("unit") or {})
    citation = dict(fetched.get("citation") or {})
    full = str(unit.get("text") or "")
    offset = max(0, min(int(text_offset or 0), len(full)))
    section = unit.get("section") or citation.get("section") or "?"
    header = (f"{citation.get('standard_id', '?')} {citation.get('revision', '?')} "
              f"§{section}, p.{unit.get('page', citation.get('page', '?'))}, "
              f"source {unit.get('source_id', citation.get('source_id', '?'))}")

    if offset:
        header += f" — continued from character {offset} of {len(full)}"

    body = full[offset:]
    citation_line = f"CITATION: {citation.get('rendered', header)}"

    # The core is the header, the text, and the citation: it is never traded
    # away for context around it. Everything the marker and the citation will
    # need is measured BEFORE deciding how much text fits, so the marker can
    # never be the thing that gets cut -- a truncation nobody can see is the
    # failure this whole rendering exists to stop.

    marker = ("[INCOMPLETE: characters {start}-{end} of {total} of this unit's "
              "text. The rest has NOT been shown and must not be guessed at. "
              "To read it, call standard.fetch again with the same source_id "
              "and text_offset={end}]")
    overhead = (len(header) + len(citation_line)
                + len(marker.format(start=offset, end=len(full), total=len(full)))
                + 8)
    room = budget - overhead

    if 0 < room < len(body):
        shown = body[:room].rstrip()
        nxt = offset + len(shown)
        parts = [header, "", shown, "",
                 marker.format(start=offset, end=nxt, total=len(full)),
                 "", citation_line]
    else:
        parts = [header, "", body, "", citation_line]

    core = "\n".join(parts)
    extras: list[str] = []
    omitted = "- […] further entries omitted; the text above is complete"

    # Whatever room is left, and not one character more. Every addition is
    # measured against the rendering it would actually produce: three groups
    # each measuring themselves against the same starting figure is three
    # ways to overflow, and an overflow here hands the whole thing back to
    # the blind cap this rendering exists to avoid.

    def fits(*candidate: str) -> bool:
        return len("\n".join([core] + list(candidate))) <= budget

    for label, items in (("PARENT", [fetched.get("parent")] if fetched.get("parent") else []),
                         ("TABLE ROWS belonging to this table",
                          fetched.get("table_rows") or []),
                         ("NEIGHBOURS in the same section", fetched.get("neighbors") or []),
                         ("CROSS-REFERENCES", fetched.get("resolved_cross_references") or [])):
        group: list[str] = []

        for item in items:
            if not isinstance(item, Mapping):
                continue

            line = _unit_line(item)

            if fits(*extras, "", f"{label}:", *group, line):
                group.append(line)
                continue

            if group and fits(*extras, "", f"{label}:", *group, omitted):
                group.append(omitted)

            break

        if group and fits(*extras, "", f"{label}:", *group):
            extras += ["", f"{label}:", *group]

    return "\n".join([core] + extras) if extras else core

# How a caller comes by a source id at all. There is no listing of the corpus,
# so pointing at one would be inventing a path that does not exist.

_SOURCE_RECOVERY = {
    "how": "call standard.search; every result carries the source_id to fetch "
           "or cite",
    "source_id_form": "std-<32 hex>",
    "structure_identifiers_belong_to": "standard.get_structure",
}


class StandardToolRefusal(RuntimeError):
    """A refusal the caller is meant to read and act on, not an outage."""

    def __init__(self, reason: str, detail: str,
                 recovery: Mapping[str, object] | None = None) -> None:
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail or reason
        self.recovery = dict(recovery or {})

    def to_dict(self) -> dict[str, object]:
        found: dict[str, object] = {"error": self.reason, "detail": self.detail}

        if self.recovery:
            found["recovery"] = self.recovery

        return found


def _refused(exc: StandardToolRefusal) -> "ToolHandlerResult":
    """Hand a refusal back as data, so a caller can branch on it."""
    return ToolHandlerResult(
        json.dumps(exc.to_dict(), ensure_ascii=False, sort_keys=True),
        metadata={"standard_event": "standard_tool_refused",
                  "standard_refusal": exc.reason})


def _source_id(arguments: Mapping[str, object]) -> str:
    """The one identifier form fetch and cite accept, or a refusal saying so."""
    supplied = arguments.get("source_id")

    if supplied is None:
        raise StandardToolRefusal(
            INVALID_SOURCE_ID, "no source_id was given", _SOURCE_RECOVERY)

    text = str(supplied)

    if _SOURCE_ID.fullmatch(text):
        return text

    if text.startswith(_STRUCTURE_PREFIXES):
        detail = ("that identifier names part of an approved structure, not a "
                  "canonical source unit; structure identifiers are answered "
                  "by standard.get_structure")
    else:
        detail = ("that is not a canonical source id; a source id is 'std-' "
                  "followed by 32 hex digits, and only standard.search hands "
                  "them out")

    raise StandardToolRefusal(INVALID_SOURCE_ID, detail, _SOURCE_RECOVERY)


def standard_tool_specs() -> tuple[ToolSpec, ...]:
    string = {"type": "string"}
    binding_properties = {
        "standard_id": {"type": "string", "description": "optional; must equal active binding"},
        "revision": {"type": "string", "description": "optional; must equal active binding"},
    }

    return (
        ToolSpec(
            "standard.search",
            "Search the currently bound standard/revision using local hybrid retrieval. "
            "Returns bounded canonical snippets and stable source IDs.",
            {"type": "object", "properties": {
                "query": string, "section": string,
                "limit": {"type": "integer"}, **binding_properties,
            }, "required": ["query"]},
            ToolCategory.RETRIEVAL, ToolMutability.READ_ONLY,
            result_policy=ToolResultPolicy(8000, True, 4000),
            roles=frozenset({"main"}), handler_key="standard.search",
        ),

        ToolSpec(
            "standard.fetch",
            "Read the normative text. Fetches one canonical source unit from the "
            "currently bound standard/revision with its page, section, hierarchy "
            "and parent context. EVERY unit it returns -- the unit itself, its "
            "parent and each neighbour -- already carries `citation_rendered`, "
            "the finished citation string. Quote it directly; do NOT call "
            "standard.cite for a unit whose citation is already in your "
            "context.",
            {"type": "object", "properties": {
                "source_id": string, "include_parent": {"type": "boolean"},
                "text_offset": {"type": "integer", "description":
                                "resume the unit's text at this character; "
                                "only for a unit reported INCOMPLETE"},
                **binding_properties,
            }, "required": ["source_id"]},
            ToolCategory.RETRIEVAL, ToolMutability.READ_ONLY,
            result_policy=_FETCH_RESULT_POLICY,
            roles=frozenset({"main"}), handler_key="standard.fetch",
        ),

        ToolSpec(
            "standard.get_structure",
            "Return a human-approved bitfield structure from the bound standard: "
            "its words, and each field's label, role, msb/lsb/width, where that "
            "position came from, and the canonical sources behind it. "
            "CALL IT WITH NO ARGUMENTS FIRST to list every approved structure "
            "and its identifiers. definition_id must be an identifier that "
            "listing returned (it starts with 'bfd-'); source_candidate_id must "
            "be one it returned starting with 'bit-'. Passing a field name, a "
            "section number, a 'std-' source id or any guessed label is an "
            "error and will keep failing -- list first, then ask by identifier. "
            "Only reviewed structures are returned, never a candidate. A result "
            "may be STRUCTURALLY_INCOMPLETE, meaning the diagram names something "
            "it never positions: report that, and do not infer the missing "
            "field. Fields carry both declared_* (what the source states) and "
            "msb/lsb (physical position); value_groups are segments of ONE "
            "value, packing_groups are INDEPENDENT values sharing one container.",
            {"type": "object", "properties": {
                "definition_id": string, "source_candidate_id": string,
                "require_structurally_complete": {"type": "boolean"},
                **binding_properties,
            }},
            ToolCategory.RETRIEVAL, ToolMutability.READ_ONLY,
            result_policy=ToolResultPolicy(8000, True, 4000),
            roles=frozenset({"main"}), handler_key="standard.get_structure",
        ),

        ToolSpec(
            "standard.cite",
            "Citation metadata ONLY, for a source ID whose citation you do not "
            "already have -- one found in a cross-reference, say. It returns no "
            "text and is NOT a reading of the standard: nothing it returns can "
            "support a statement about what the standard requires. Every unit "
            "standard.fetch returns already carries its citation, so calling "
            "this for one of those buys nothing and costs a round trip.",
            {"type": "object", "properties": {
                "source_id": string, **binding_properties,
            }, "required": ["source_id"]},
            ToolCategory.RETRIEVAL, ToolMutability.READ_ONLY,
            roles=frozenset({"main"}), handler_key="standard.cite",
        ),
    )


def _binding(context: ToolExecutionContext, arguments: Mapping[str, object]) -> StandardBinding:
    raw = context.metadata.get("standard_binding")

    if not isinstance(raw, Mapping):
        raise PermissionError("no standard is bound to this task")

    binding = StandardBinding.from_dict(raw)

    for key in ("standard_id", "revision"):
        supplied = arguments.get(key)

        if supplied is not None and supplied != getattr(binding, key):
            raise PermissionError(
                f"requested {key} does not equal the active StandardBinding; "
                "the bound standard and revision are supplied by the task -- "
                "omit standard_id and revision instead of restating them")

    return binding


class StandardToolService:
    def __init__(self, store: StandardStore, *, embedder=None) -> None:
        self.store = store

        if embedder is None:
            try:
                embedder = configured_local_embedder()
            except (RuntimeError, ValueError):
                embedder = None

        self.retrieval = StandardRetrieval(store, embedder)
        self.structures = StandardStructureAccess(store)

    def _active_binding(
        self, context: ToolExecutionContext, arguments: Mapping[str, object],
    ) -> StandardBinding:
        binding = _binding(context, arguments)
        current = self.store.binding(binding.standard_id, binding.revision,
                                     bound_at=binding.bound_at)

        if (binding.pdf_sha256 != current.pdf_sha256
                or binding.corpus_manifest_sha256 != current.corpus_manifest_sha256):
            raise PermissionError("active StandardBinding canonical source changed")

        if ((binding.retrieval_fingerprint or binding.index_fingerprint)
                != (current.retrieval_fingerprint or current.index_fingerprint)):
            raise PermissionError(
                "active StandardBinding retrieval changed; fresh binding/retrieval required")

        return binding

    def search(self, context: ToolExecutionContext, arguments: Mapping[str, object]):
        binding = self._active_binding(context, arguments)
        query = str(arguments.get("query") or "")
        section = arguments.get("section")

        if not query.strip():
            raise StandardToolRefusal(
                INVALID_QUERY, "query must be non-empty text",
                {"how": "call standard.search again with the wording to look "
                        "for; a query that matches nothing comes back as a "
                        "result with result_count 0, not as this refusal"})

        try:
            limit = int(arguments.get("limit", 5))
        except (TypeError, ValueError):
            raise StandardToolRefusal(
                INVALID_QUERY, "limit must be a whole number",
                {"how": "omit limit to take the default, or pass an integer"}
            ) from None

        retrieval_fingerprint = binding.retrieval_fingerprint or binding.index_fingerprint
        key = ("standard.search", retrieval_fingerprint, query, section, limit)

        if key in context.cache:
            return context.cache[key]

        try:
            response = self.retrieval.search_response(
                binding.standard_id, binding.revision, query,
                section=(str(section) if section is not None else None),
                limit=limit,
            )
        except ValueError as exc:
            raise StandardToolRefusal(
                INVALID_QUERY, str(exc),
                {"how": "reword the query, or omit limit to take the default"}
            ) from None

        results = response.results
        payload = {
            "binding": {"standard_id": binding.standard_id,
                        "revision": binding.revision,
                        "corpus_manifest_sha256": binding.corpus_manifest_sha256,
                        "index_fingerprint": binding.index_fingerprint,
                        "retrieval_fingerprint": retrieval_fingerprint},
            "result_count": len(results),
            "retrieval_mode_used": response.retrieval_mode_used,
            "retrieval_capability": response.retrieval_capability,
            "evidence_completion": response.evidence_completion,
            "completion_budget": response.completion_budget,
            "results": [item.to_dict() for item in results],
        }
        result = ToolHandlerResult(
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            metadata={
                "standard_event": "standard_search_completed",
                "standard_binding": payload["binding"],
                "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
                "result_count": len(results),
                "standard_source_ids": [item.source_id for item in results],
                "retrieval_mode_used": response.retrieval_mode_used,
            "retrieval_capability": response.retrieval_capability,
            "evidence_completion": response.evidence_completion,
            "completion_budget": response.completion_budget,
                "retrieval_fingerprint": retrieval_fingerprint,
                "lexical_candidate_count": response.lexical_candidate_count,
                "vector_candidate_count": response.vector_candidate_count,
                "expanded_cross_reference_ids": list(
                    response.expanded_cross_reference_ids),
            },
        )
        context.cache[key] = result

        return result

    def fetch(self, context: ToolExecutionContext, arguments: Mapping[str, object]):
        binding = self._active_binding(context, arguments)
        source_id = _source_id(arguments)
        offset = arguments.get("text_offset", 0)

        try:
            offset = max(0, int(offset))
        except (TypeError, ValueError):
            raise StandardToolRefusal(
                INVALID_SOURCE_ID, "text_offset must be a non-negative integer",
                "call again without text_offset, or with the value the "
                "previous result named") from None

        try:
            fetched = self.retrieval.fetch(
                binding.standard_id, binding.revision, source_id,
                include_parent=bool(arguments.get("include_parent", True)),
            )
        except KeyError:
            # Well formed and absent: a real answer about this revision, and a
            # different fact from having been asked in the wrong currency.

            raise StandardToolRefusal(
                SOURCE_NOT_FOUND,
                "no canonical source unit has that id in this revision",
                _SOURCE_RECOVERY) from None

        # A table arrives from the extractor as a caption unit and a run of
        # unlabelled row units with nothing linking them, so a session can
        # hold every row of a table and still not know it has the table. The
        # relationship is derived here, read-only, from what the fetch
        # already returned -- the licensed corpus is not rewritten and a
        # caller that ignores the field sees what it saw before.

        fetched = _attach_table_rows(fetched)

        # The structured form is what is kept, traced and stored; the
        # rendering is what the model is shown. The tool's own output budget
        # bounds the second, so the clause it asked for is never displaced by
        # the metadata around it.

        return ToolHandlerResult(
            json.dumps(fetched, ensure_ascii=False, sort_keys=True),
            model_text=render_fetch_for_model(
                fetched,
                budget=_fetch_policy().model_context_chars,
                text_offset=offset),
            metadata={
                "standard_event": "standard_source_fetched",
                "standard_binding": {
                    "standard_id": binding.standard_id, "revision": binding.revision,
                    "corpus_manifest_sha256": binding.corpus_manifest_sha256,
                    "index_fingerprint": binding.index_fingerprint,
                    "retrieval_fingerprint": (
                        binding.retrieval_fingerprint or binding.index_fingerprint),
                },
                "standard_source_ids": [source_id],
                "source_id": source_id,
                "citation": fetched["citation"],
                "text_offset": offset,
            },
        )

    def cite(self, context: ToolExecutionContext, arguments: Mapping[str, object]):
        binding = self._active_binding(context, arguments)
        source_id = _source_id(arguments)

        try:
            citation = self.retrieval.cite(
                binding.standard_id, binding.revision, source_id)
        except KeyError:
            raise StandardToolRefusal(
                SOURCE_NOT_FOUND,
                "no canonical source unit has that id in this revision",
                _SOURCE_RECOVERY) from None

        return ToolHandlerResult(
            json.dumps(citation.to_dict(), ensure_ascii=False, sort_keys=True),
            metadata={"standard_source_ids": [citation.source_id],
                      "citation": citation.to_dict()},
        )

    def get_structure(self, context: ToolExecutionContext,
                      arguments: Mapping[str, object]):
        """Hand back a reviewed structure, or the named reason there is none."""
        binding = self._active_binding(context, arguments)
        definition_id = arguments.get("definition_id")
        candidate_id = arguments.get("source_candidate_id")

        try:
            if definition_id is None and candidate_id is None:
                payload = self.structures.index(binding.standard_id,
                                                binding.revision)
                event, sources = "standard_structure_listed", []
            else:
                payload = self.structures.get(
                    binding.standard_id, binding.revision,
                    definition_id=(str(definition_id) if definition_id else None),
                    source_candidate_id=(str(candidate_id) if candidate_id else None),
                    require_structurally_complete=bool(
                        arguments.get("require_structurally_complete", False)))
                event = "standard_structure_fetched"
                sources = list(payload["citation_source_ids"])
        except StructureAccessError as exc:
            # A refusal is an answer the model must be able to act on, so it
            # comes back as data with its reason, not as an opaque failure.

            return ToolHandlerResult(
                json.dumps(exc.to_dict(), ensure_ascii=False, sort_keys=True),
                metadata={"standard_event": "standard_structure_refused",
                          "structure_refusal": exc.reason,
                          "standard_refusal": exc.reason})

        return ToolHandlerResult(
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            metadata={
                "standard_event": event,
                "standard_binding": {
                    "standard_id": binding.standard_id,
                    "revision": binding.revision,
                    "corpus_manifest_sha256": binding.corpus_manifest_sha256},
                "standard_source_ids": sources,
                "definition_id": payload.get("definition_id"),
                "structural_completeness": payload.get("structural_completeness"),
            })

    @staticmethod
    def _answering(handler):
        """Refusals come back as data; everything else keeps its own behaviour."""

        def call(context: ToolExecutionContext, arguments: Mapping[str, object]):
            try:
                return handler(context, arguments)
            except StandardToolRefusal as exc:
                return _refused(exc)

        call.__name__ = getattr(handler, "__name__", "call")

        return call

    def register(self, registry: ToolRegistry) -> None:
        handlers = {
            "standard.search": self.search,
            "standard.fetch": self.fetch,
            "standard.cite": self.cite,
            "standard.get_structure": self.get_structure,
        }

        for spec in standard_tool_specs():
            registry.register(spec, self._answering(handlers[spec.name]))
