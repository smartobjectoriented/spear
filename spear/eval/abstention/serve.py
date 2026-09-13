"""A tool surface for the synthetic scenarios, so the model can exhaust them.

Sampling a single round mostly produced another tool call rather than an
answer -- which is the model behaving correctly, not failing. A model asked for
a bit position with one search result in hand should look further. The failure
we are after only appears once looking further stops paying, so the scenario has
to be able to say "nothing more".

Everything served here is invented. It answers in the real tools' shapes so the
model sees a familiar surface, and it never reaches the real store.
"""

from __future__ import annotations

import json

from synthetic import REV, SID


def _known(item):
    """Everything this scenario is willing to serve, keyed by identifier."""
    sources, structures = {}, {}

    for step in item["evidence"]:
        result = step["result"]

        for hit in result.get("results", ()):
            sources[hit["source_id"]] = hit

        if "definition_id" in result:
            structures[result["definition_id"]] = result

        if "unit" in result:
            sources.setdefault(result["unit"].get("source_id", ""), result["unit"])

    return sources, structures


def respond(item, name, arguments):
    """One tool answer, in the shape the real tool would have used."""
    sources, structures = _known(item)

    if name == "standard.search":
        # The scenario has already returned what it has. A further query finds
        # nothing, which is a finding rather than a fault.

        return json.dumps({"binding": {"standard_id": SID, "revision": REV},
                           "result_count": 0, "retrieval_mode_used":
                           "lexical_fallback", "results": []}, sort_keys=True)

    if name in ("standard.fetch", "standard.cite"):
        asked = str(arguments.get("source_id", ""))

        if asked in sources:
            hit = sources[asked]
            citation = {"standard_id": SID, "revision": REV,
                        "section": hit.get("section"), "page": hit.get("page"),
                        "source_id": asked,
                        "rendered": f"[{SID} {REV} §{hit.get('section')}, "
                                    f"p.{hit.get('page')}, source {asked}]"}

            if name == "standard.cite":
                return json.dumps(citation, sort_keys=True)

            return json.dumps({"unit": hit, "parent": None, "neighbors": [],
                               "resolved_cross_references": [],
                               "unresolved_cross_references": [],
                               "citation": citation}, sort_keys=True)

        if asked.startswith(("bit-", "bfd-", "fld-", "pkg-", "vgr-")):
            return json.dumps({
                "error": "INVALID_SOURCE_ID",
                "detail": "that identifier names part of an approved structure, "
                          "not a canonical source unit; structure identifiers "
                          "are answered by standard.get_structure",
                "recovery": {"how": "call standard.search; every result carries "
                                    "the source_id to fetch or cite",
                             "source_id_form": "std-<32 hex>"}}, sort_keys=True)

        return json.dumps({
            "error": "SOURCE_NOT_FOUND",
            "detail": "no canonical source unit has that id in this revision",
            "recovery": {"how": "call standard.search; every result carries the "
                                "source_id to fetch or cite"}}, sort_keys=True)

    if name == "standard.get_structure":
        asked = arguments.get("definition_id")

        if asked is None and not arguments.get("source_candidate_id"):
            return json.dumps({
                "standard_id": SID, "revision": REV,
                "structure_count": len(structures),
                "structures": [{"definition_id": key,
                                "field_count": value.get("field_count", 0)}
                               for key, value in sorted(structures.items())]},
                sort_keys=True)

        if str(asked) in structures:
            return json.dumps(structures[str(asked)], sort_keys=True)

        return json.dumps({
            "error": "STRUCTURE_NOT_FOUND",
            "detail": "no approved structure has that identity in this revision",
            "recovery": {"how": "call standard.get_structure with no arguments "
                                "to list every approved structure and its "
                                "identifiers",
                         "valid_definition_ids": sorted(structures)}},
            sort_keys=True)

    return json.dumps({"error": "UNKNOWN_TOOL"}, sort_keys=True)
