"""MODEL_USE_FT0: the synthetic half of the abstention preference set.

Nothing here comes from an evaluated document. The standard named below does
not exist; the identifiers are made up in the shapes the real tools hand out,
so a training example looks exactly like a serving context without carrying a
line of licensed prose. That is the whole point of the file: the behaviour we
want to move is "decide from the evidence in front of you", which has nothing
to do with which standard the evidence came from.

Four behaviours are represented, and one of them is answering. A set that only
ever rewards "insufficient evidence" teaches a model to say that when it is
untrue, which is a worse failure than the one being fixed.
"""

from __future__ import annotations

import hashlib

SID, REV = "ACME-XBUS-3", "2019-R2023"


def _id(prefix: str, seed: str, width: int) -> str:
    return prefix + hashlib.sha256(seed.encode()).hexdigest()[:width]


def src(seed: str) -> str:
    return _id("std-", "src:" + seed, 32)


def dfn(seed: str) -> str:
    return _id("bfd-", "def:" + seed, 16)


def fld(seed: str) -> str:
    return _id("fld-", "fld:" + seed, 16)


def field(seed, label, word, msb, lsb, **over):
    """One field the way standard.get_structure serves it.

    msb/lsb of None is how an unresolved position is served: the field is
    known to exist and to have a declared width, and nothing places it.
    """
    placed = msb is not None and lsb is not None
    found = {
        "field_id": fld(seed), "display_label": label, "normative_label": label,
        "word_index": word, "msb": msb, "lsb": lsb,
        "width": over.pop("width", (msb - lsb + 1) if placed else None),
        "declared_msb": over.pop("declared_msb", msb),
        "declared_lsb": over.pop("declared_lsb", lsb),
        "coordinate_domain": over.pop("coordinate_domain", "WORD_LOCAL"),
        "position_source": over.pop("position_source", "STATED_RANGE"),
        "semantic_role": "FIELD", "value_group_id": None, "packing_group_id": None,
        "segment_index": None, "segment_count": None,
        "value_width": over.pop("value_width", (msb - lsb + 1) if placed else None),
        "packing_slot": None, "projection_source": None, "projection_reference": None,
        "normative_source_id": over.pop("normative_source_id", None),
        "canonical_source_ids": over.pop("canonical_source_ids", []),
        "stated_range_text": over.pop("stated_range_text",
                                      f"({msb}..{lsb})" if placed else None),
        "warnings": [],
    }
    found.update(over)

    return found


def structure(seed, fields, *, complete=True, unresolved=0, words=1,
              value_groups=(), packing_groups=()):
    return {
        "standard_id": SID, "revision": REV, "definition_id": dfn(seed),
        "source_candidate_id": _id("bit-", "cand:" + seed, 16),
        "human_verdict": "PASS", "bit_order": "MSB_TO_LSB",
        "completeness": "COMPLETE" if complete else "PARTIAL",
        "structural_completeness": ("STRUCTURALLY_COMPLETE" if not unresolved
                                    else "STRUCTURALLY_INCOMPLETE"),
        "unresolved_word_count": unresolved, "word_count": words,
        "field_count": len(fields), "fields": list(fields),
        "value_groups": list(value_groups), "packing_groups": list(packing_groups),
        "warnings": [], "citation_source_ids": sorted(
            {s for f in fields for s in f["canonical_source_ids"]}),
    }


def value_group(seed, members, width=64):
    return {"value_group_id": _id("vgr-", "vgr:" + seed, 16),
            "semantic_kind": "COMPOSITE_VALUE", "value_width": width,
            "member_field_ids": [m["field_id"] for m in members],
            "members": members,
            "interpretation": "these members are segments of ONE semantic value "
                              "split across containers; do not describe them as "
                              "independent values"}


def packing_group(seed, members, width=32):
    return {"packing_group_id": _id("pkg-", "pkg:" + seed, 16),
            "semantic_kind": "INDEPENDENT_VALUES_SHARED_CONTAINER",
            "physical_width": width,
            "member_field_ids": [m["field_id"] for m in members],
            "members": members,
            "interpretation": "these members are INDEPENDENT semantic values that "
                              "share one physical container; physical_width is the "
                              "container, not a combined value; do not concatenate "
                              "them"}


def hit(seed, section, page, snippet):
    return {"source_id": src(seed), "standard_id": SID, "revision": REV,
            "section": section, "page": page, "snippet": snippet,
            "content_type": "requirement", "score": 11.5}


def search(results):
    return {"binding": {"standard_id": SID, "revision": REV},
            "result_count": len(results), "retrieval_mode_used": "lexical_fallback",
            "results": results}
