"""Per-standard role profiles, loaded from the store — never hardcoded here.

A standard whose roles mean something other than the conventional reading
(normative_force.CONVENTIONAL) ships a profile beside its ingested revision in
the store, as `profile.json`. It travels with the document into an image and is
never named in this public source; a standard with no profile uses the
conventional taxonomy. This module only knows how to load one.
"""
from __future__ import annotations

import json

from normative import normative_force as force

PROFILE_FILE = "profile.json"


def register_for(store, standard_id, revision):
    """Register the role taxonomy a standard ships in its store, if any.

    Best-effort and idempotent: a missing or unreadable profile leaves the
    conventional reading in place, so a broken file never blocks a binding.
    """
    try:
        path = store.revision_dir(standard_id, revision) / PROFILE_FILE
    except Exception:
        return None

    if not path.is_file():
        return None

    try:
        taxonomy = force.RoleTaxonomy.from_mapping(
            json.loads(path.read_text("utf-8")))
    except Exception:
        return None

    return force.register_taxonomy(standard_id, taxonomy)
