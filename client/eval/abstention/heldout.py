"""The frozen synthetic held-out evaluation.

The first version of this file was built from families the training pairs did
not use, and the base model handled every one of them -- an evaluation with no
baseline failure cannot show an adapter doing anything. This version is built
from the shape and vocabulary pairings each behaviour has already been shown to
come out under, on families training does not use, and every item kept here was
kept because the base model actually failed it on replay.

Four audited pairs sit alongside these as held-out items of their own; the
scenarios below are scored by replay rather than as pairs. Neither ever enters
training.
"""

from __future__ import annotations

import complement
import pressure
import shapes
from families import BY_KEY
from heldout_scenarios import absence_rich

# Chosen because the base model fails them, reproducibly, on a clean tool path.

COMPLEMENT_ONLY = (("byte_offset", "primsec"),)
PRESSURE_ONLY = ("realimag", "coarsefine", "mantissa", "pitchyaw")
ABSENCE_ONLY = ("iq", "timestamp", "gain_stages", "threshold")

# Controls, to measure whether an adapter starts refusing what it can read.

SUPPORTED_ONLY = (("two_slot", "cmdstat"), ("byte_offset", "primsec"),
                  ("two_word", "coarsefine"), ("partial_slot", "opcode"),
                  ("whole_word", "addrdata"), ("two_slot", "srcdst"),
                  ("byte_offset", "cmdstat"), ("two_word", "primsec"))


def discipline():
    found = [complement.byte_offset(BY_KEY[key]) for _, key in COMPLEMENT_ONLY]
    found += [pressure.enum_gap(BY_KEY[key], pressure.EXPLICIT_FILL)
              for key in PRESSURE_ONLY]
    found += [absence_rich(BY_KEY[key]) for key in ABSENCE_ONLY]

    return tuple(found)


def controls():
    makers = {"two_slot": complement.two_slot,
              "byte_offset": complement.byte_offset,
              "two_word": complement.two_word,
              "partial_slot": shapes.partial_slot_supported,
              "whole_word": pressure.whole_word}
    found = []

    for shape, key in SUPPORTED_ONLY:
        maker = makers[shape]

        if shape == "partial_slot":
            found.append(maker(BY_KEY[key]))
        elif shape == "whole_word":
            found.append(maker(BY_KEY[key], pressure.EXPLICIT_FILL,
                               supported=True))
        else:
            found.append(maker(BY_KEY[key], supported=True))

    return tuple(found)


HELDOUT = discipline() + controls()
