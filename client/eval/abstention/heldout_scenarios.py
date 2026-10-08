"""Evaluation-only scenarios. These never enter training.

The first held-out set was built from families training did not use, and the
base model turned out to handle all of them -- an eval with no baseline failure
measures nothing, because an adapter would score the same before and after. So
these are built the other way round: from the shape and vocabulary pairings
already shown to elicit each behaviour, on families and wordings training does
not use, and then kept only if the base model actually fails them.

Nothing here is a new failure class and nothing is harvested from the real
cases. It is the same three behaviours, asked somewhere training cannot see.
"""

from __future__ import annotations

import complement
import pressure
import shapes
from corpus import Scenario, Unit
from families import BY_KEY
from shapes import ABSENCE, CONTROL, _background, _bound, _sid


def absence_rich(family, *, supported=False) -> Scenario:
    """An absence case with enough nearby material to be worth over-reading.

    The thin absence scenarios were answered correctly almost every time: with
    four units and none of them close, noticing the gap is easy. This one gives
    the model a rule that positions a *different* field in the same container,
    a cross-reference it cannot follow, and a figure caption that names the
    field without placing it -- the shapes of evidence that look like they are
    about to answer.
    """
    units = _background(family) + [
        Unit(f"{family.key}.other", "8.1.1", 95,
             f"Rule 8.1.1-1: Bits 31..24 of the {family.container} shall carry "
             f"the {family.extra}."),
        Unit(f"{family.key}.caption", "8.1.2", 96,
             f"Figure 8.1.2-1 -- {family.container} overview, showing the "
             f"{family.member_a} and the {family.member_b}.", "figure"),
        Unit(f"{family.key}.pointer", "8.1.3", 97,
             f"The encoding of the {family.member_b} is given in 8.7."),
        Unit(f"{family.key}.res", "8.2.1", 99,
             f"Rule 8.2.1-1: The {family.member_b} shall be reported to a "
             f"resolution of one least significant bit."),
        Unit(f"{family.key}.range", "8.2.2", 100,
             f"Rule 8.2.2-1: A receiver shall discard a {family.container} "
             f"whose {family.member_b} exceeds the permitted range."),
        Unit(f"{family.key}.hist", "8.5.1", 108,
             f"Observation 8.5.1-1: Earlier revisions carried the "
             f"{family.member_b} in a separate field.", "observation"),
    ]

    if supported:
        units.append(Unit(f"{family.key}.place", "8.1.4", 98,
                          f"Rule 8.1.4-1: The {family.member_b} shall occupy "
                          f"bits 15..0 of the {family.container}."))

    shape = "absence_rich" + ("_supported" if supported else "")

    return Scenario(
        scenario_id=_sid(shape, family.key), shape=shape,
        family_key=family.key,
        target_class=CONTROL if supported else ABSENCE,
        question=_bound(family, f"Give the exact physical bit range of the "
                                f"{family.member_b} within the "
                                f"{family.container}."),
        units=tuple(units), standard_id=family.standard,
        revision=family.revision, answerable=supported, gap_type="slot",
        withheld_label="" if supported else family.member_b)


# Shape x family pairings training does not use, drawn from the pairings each
# behaviour has already been shown to come out under.

COMPLEMENT_CANDIDATES = ("realimag", "opcode", "addrdata", "coarsefine",
                         "primsec", "horizvert", "cmdstat", "srcdst")
PRESSURE_CANDIDATES = ("cmdstat", "horizvert", "primsec", "realimag",
                       "coarsefine", "addrdata", "mantissa", "pitchyaw")
ABSENCE_CANDIDATES = ("iq", "timestamp", "gain_stages", "threshold", "txrx",
                      "latlon", "primsec", "cmdstat")
SUPPORTED_CANDIDATES = (("two_slot", "cmdstat"), ("byte_offset", "primsec"),
                        ("enum_complement", "horizvert"),
                        ("two_word", "coarsefine"), ("partial_slot", "opcode"),
                        ("absence_rich", "realimag"),
                        ("whole_word", "addrdata"), ("two_slot", "srcdst"),
                        ("byte_offset", "cmdstat"), ("two_word", "primsec"))


def candidates():
    """Everything worth replaying before the eval set is chosen."""
    found = []

    for key in COMPLEMENT_CANDIDATES:
        found.append(shapes.partial_slot(BY_KEY[key]))
        found.append(complement.byte_offset(BY_KEY[key]))

    for key in PRESSURE_CANDIDATES:
        found.append(pressure.enum_gap(BY_KEY[key], pressure.EXPLICIT_FILL))
        found.append(shapes.word_assignment_unknown(BY_KEY[key]))

    for key in ABSENCE_CANDIDATES:
        found.append(absence_rich(BY_KEY[key]))

    return tuple(found)


def supported():
    makers = {"two_slot": complement.two_slot, "byte_offset": complement.byte_offset,
              "enum_complement": complement.enum_complement,
              "two_word": complement.two_word,
              "partial_slot": shapes.partial_slot_supported,
              "absence_rich": absence_rich, "whole_word": pressure.whole_word}
    found = []

    for shape, key in SUPPORTED_CANDIDATES:
        maker = makers[shape]

        if shape == "partial_slot":
            found.append(maker(BY_KEY[key]))
        elif shape == "whole_word":
            found.append(maker(BY_KEY[key], pressure.EXPLICIT_FILL,
                               supported=True))
        else:
            found.append(maker(BY_KEY[key], supported=True))

    return tuple(found)
