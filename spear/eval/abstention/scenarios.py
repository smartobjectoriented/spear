"""The candidate pool: which shapes get dressed in which vocabulary.

Two groups are deliberately complete across families -- partial_slot and
absence run over the same eleven families, invented ones included. That is the
experiment: identical evidence, identical question shape, only the names
change. Whatever difference shows up between them is the prior's doing.
"""

from __future__ import annotations

import json

import shapes
from families import BY_KEY

# The evidence-equivalent group. Two invented families anchor it; the rest are
# pairs an engineer meets often enough to expect something about.

SPECTRUM = ("invented_x", "invented_zeta", "gain_stages", "txrx", "latlon",
            "iq", "timestamp", "threshold", "pitchyaw", "magphase", "mantissa")

_WORD_ORDER = ("invented_x", "timestamp", "latlon", "srcdst", "mantissa")
_PACKING = ("invented_zeta", "gain_stages", "iq", "threshold")
_PRESSURE_ASKS = (("neutral", "pitchyaw"), ("fill_reserved", "cmdstat"),
                  ("make_complete", "magphase"), ("conventional", "horizvert"),
                  ("judgement", "txrx"), ("already_know", "invented_x"))
_ENUM = ("latlon", "iq", "threshold")
_WORD_ASSIGN = ("timestamp", "srcdst", "invented_zeta")

# Chosen to give the three families outside SPECTRUM a control of
# their own; the spectrum families already have one.

_COMPLETE = ("srcdst", "horizvert", "cmdstat")
_ABSENCE_SUPPORTED = ("timestamp", "threshold", "invented_x")


def build():
    """Every candidate scenario, in a stable order."""
    found = []

    for key in SPECTRUM:
        found.append(shapes.partial_slot(BY_KEY[key]))

    for key in _WORD_ORDER:
        found.append(shapes.word_order(BY_KEY[key]))

    for key in _PACKING:
        found.append(shapes.packing_slot_unknown(BY_KEY[key]))

    for ask, key in _PRESSURE_ASKS:
        found.append(shapes.incomplete_struct(BY_KEY[key], ask))

    for key in _ENUM:
        found.append(shapes.enum_unknown(BY_KEY[key]))

    for key in _WORD_ASSIGN:
        found.append(shapes.word_assignment_unknown(BY_KEY[key]))

    for key in SPECTRUM:
        found.append(shapes.absence(BY_KEY[key]))

    for key in SPECTRUM:
        found.append(shapes.partial_slot_supported(BY_KEY[key]))

    for key in _COMPLETE:
        found.append(shapes.complete_struct(BY_KEY[key]))

    for key in _ABSENCE_SUPPORTED:
        found.append(shapes.absence_supported(BY_KEY[key]))

    return tuple(found)


def serialize(scenario):
    """A stable description of one scenario, for tests and for the record."""
    return json.dumps({
        "scenario_id": scenario.scenario_id, "shape": scenario.shape,
        "family": scenario.family_key, "target_class": scenario.target_class,
        "answerable": scenario.answerable, "question": scenario.question,
        "units": [[unit.key, unit.section, unit.page, unit.text]
                  for unit in scenario.units],
        "structure_keys": [item.key for item in scenario.structures],
    }, sort_keys=True)


SCENARIOS = build()
