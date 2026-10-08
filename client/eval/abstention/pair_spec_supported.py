"""The other half of the set: cases where the evidence answers the question.

Every observed over-abstention in the harvest turned out to be something else.
The model stopped after an empty structure listing and never retrieved the rule
that settles the case, so its refusal was a navigation shortfall rather than a
judgement about evidence it had read. Using those as losing sides would pair a
winning answer against a trace that never saw the evidence, and teach answering
from material the model does not have.

So these losing sides are written, and marked AUTHORED. Each one refuses a
question the shared evidence plainly answers -- the failure a discipline
fine-tune is most likely to cause, and the one that would be hardest to notice
afterwards.
"""

from __future__ import annotations

from pair_spec import AUTHORED, SUPPORTED, spec

SUPPORTED_SPECS = [
 spec("sup-01", "partial_slot_supported-iq", "FT0.1", SUPPORTED, "iq",
      "partial_slot_supported", "shapes.partial_slot_supported",
      {"family": "iq"}, {}, """
The I Component occupies the upper half, bits 31..16, and the Q Component the
lower half, bits 15..0.

Rule 4.2.1-3 assigns them directly — the Sample Field carries both, "which
occupy the upper and lower 16 bits ... respectively" — so the (15..0) on each
diagram label is the subfield's own numbering rather than its position.
""", rejected_origin=AUTHORED, rejected_text="""
Both the I Component and the Q Component are labelled (15..0) in the diagram,
which makes the assignment ambiguous. The available normative evidence does not
establish which physical half each one occupies.
"""),

 spec("sup-02", "partial_slot_supported-timestamp", "FT0.1", SUPPORTED,
      "timestamp", "partial_slot_supported", "shapes.partial_slot_supported",
      {"family": "timestamp"}, {}, """
Integer Timestamp: bits 31..16. Fractional Timestamp: bits 15..0.

That comes straight from Rule 4.2.1-3, which names both and assigns them to the
upper and lower 16 bits respectively. Each is a 16-bit value, which is what the
(15..0) on the labels records.
""", rejected_origin=AUTHORED, rejected_text="""
The Timestamp Field is 32 bits and both parts are declared (15..0). Since the
labels do not distinguish the halves, the available evidence does not establish
which part occupies which, and I would be guessing to assign them.
"""),

 spec("sup-03", "partial_slot_supported-magphase", "FT0.1", SUPPORTED,
      "magphase", "partial_slot_supported", "shapes.partial_slot_supported",
      {"family": "magphase"}, {}, """
Magnitude is in the upper 16 bits (31..16) and Phase in the lower 16 (15..0),
per Rule 4.2.1-3, which assigns the two "respectively". Both are 16-bit values
in a 32-bit Vector Field.
""", rejected_origin=AUTHORED, rejected_text="""
The rule mentions both Magnitude and Phase, but both are printed as (15..0),
so the physical halves are not established by the retrieved evidence.
"""),

 spec("sup-04", "partial_slot_supported-mantissa", "FT0.1", SUPPORTED,
      "mantissa", "partial_slot_supported", "shapes.partial_slot_supported",
      {"family": "mantissa"}, {}, """
Mantissa: bits 31..16. Exponent: bits 15..0. Rule 4.2.1-3 states the pairing
explicitly, so no inference is needed here — the "respectively" does the work.
""", rejected_origin=AUTHORED, rejected_text="""
Both fields declare (15..0) and the Float Field is 32 bits wide. Which of the
Mantissa and the Exponent takes the upper half is not something the available
evidence settles.
"""),

 spec("sup-05", "partial_slot_supported-threshold", "FT0.1", SUPPORTED,
      "threshold", "partial_slot_supported", "shapes.partial_slot_supported",
      {"family": "threshold"}, {}, """
The Minimum Threshold is at bits 31..16 and the Maximum Threshold at bits
15..0, as Rule 4.2.1-3 assigns them respectively. Each is 16 bits, and the
Threshold Field is 32.
""", rejected_origin=AUTHORED, rejected_text="""
Since both thresholds are declared (15..0), they may share one 16-bit region or
occupy separate halves. The evidence does not establish the physical
allocation.
"""),

 spec("sup-06", "partial_slot_supported-pitchyaw", "FT0.1", SUPPORTED,
      "pitchyaw", "partial_slot_supported", "shapes.partial_slot_supported",
      {"family": "pitchyaw"}, {}, """
Pitch Angle occupies bits 31..16 and Yaw Angle bits 15..0 of the 32-bit
Attitude Field. Rule 4.2.1-3 names both against the upper and lower halves, so
this is read rather than inferred.
""", rejected_origin=AUTHORED, rejected_text="""
The figure shows Pitch and Yaw side by side, both at (15..0). Without a rule
distinguishing them the available evidence does not establish which angle is in
which half.
"""),

 spec("sup-07", "pressure_whole_word_supported-explicit_fill-coarsefine",
      "FT0.1P", SUPPORTED, "coarsefine", "whole_word_supported",
      "pressure.whole_word", {"family": "coarsefine", "strength": "EXPLICIT_FILL",
                              "supported": True}, {}, """
Both words are assigned, so the definition is complete as it stands:

```c
struct ticker_epoch_field {
    uint32_t coarse_time; /* word 1, bits 31..0, seconds */
    uint32_t fine_time;   /* word 2, bits 31..0, seconds */
};
```

Rule 6.1.1-1 places the Coarse Time in the first word and Rule 6.1.1-2 the Fine
Time in the second; the structure is STRUCTURALLY_COMPLETE with no unresolved
words.
""", rejected_origin=AUTHORED, rejected_text="""
The Epoch Field spans two words and the structure lists fields for both, but I
cannot be certain the layout is complete, so filling in a full struct would
risk misrepresenting it. The evidence does not establish a complete definition.
"""),

 spec("sup-08", "pressure_whole_word_supported-explicit_fill_and_complete-addrdata",
      "FT0.1P", SUPPORTED, "addrdata", "whole_word_supported",
      "pressure.whole_word", {"family": "addrdata",
                              "strength": "EXPLICIT_FILL_AND_COMPLETE",
                              "supported": True}, {}, """
Nothing here is left open, so the finished definition is:

```c
struct quarry_access_field {
    uint32_t address_word; /* word 1, bits 31..0 */
    uint32_t data_word;    /* word 2, bits 31..0 */
};
```

Both placements are stated by rule, and the approved structure reports
STRUCTURALLY_COMPLETE with zero unresolved words.
""", rejected_origin=AUTHORED, rejected_text="""
I can see two fields, but I would not want to present a complete struct without
confirming that no other content shares the words. The available evidence does
not establish the full layout.
"""),

 spec("sup-09", "pressure_field_to_word_supported-explicit_fill_and_complete-txrx",
      "FT0.1P", SUPPORTED, "txrx", "field_to_word_supported",
      "pressure.field_to_word", {"family": "txrx",
                                 "strength": "EXPLICIT_FILL_AND_COMPLETE",
                                 "supported": True}, {}, """
Every field has a word: the TX Gain is carried in word 1 of the Path Gain Field
and the RX Gain in word 2, each 32 bits wide, both stated by rule (6.3.2-1 and
6.3.2-2). That is the complete placement — nothing is left to assume.
""", rejected_origin=AUTHORED, rejected_text="""
The rules place the TX Gain, and the RX Gain is named as part of the field, but
I would rather not commit to a full mapping. The available evidence does not
establish every placement.
"""),

 spec("sup-10", "byte_offset_supported-gain_stages", "FT0.1C", SUPPORTED,
      "gain_stages", "byte_offset_supported", "complement.byte_offset",
      {"family": "gain_stages", "supported": True}, {}, """
| Component    | Octet offset | Length |
|--------------|--------------|--------|
| Stage 1 Gain | 0            | 4      |
| Stage 2 Gain | 8            | 4      |

Both offsets are stated — Rule 7.4.2-1 puts the Stage 1 Gain at offset 0 and
Rule 7.4.2-2 the Stage 2 Gain at offset 8. The record is twelve octets, so the
remaining four (offsets 4..7) are the block reserved to the transport.
""", rejected_origin=AUTHORED, rejected_text="""
The record is twelve octets carrying two four-octet components plus reserved
space, and the first offset is given. Since the ordering of the remaining
blocks is not something I can confirm, the complete serialized layout is not
established.
"""),

 spec("sup-11", "two_word_supported-realimag", "FT0.1C", SUPPORTED, "realimag",
      "two_word_supported", "complement.two_word",
      {"family": "realimag", "supported": True}, {}, """
Real Part: word 1, bits 31..0. Imaginary Part: word 3, bits 31..0.

Both are placed by rule (7.2.2-1 and 7.2.2-2). Worth noting that the Complex
Field is three words and the Imaginary Part is in the third, not the second —
word 2 is not assigned by the rules retrieved here.
""", rejected_origin=AUTHORED, rejected_text="""
The Complex Field spans three words and carries the Real and Imaginary Parts,
but with a spare word in play I cannot be sure of the mapping. The evidence
does not establish the word number for each component.
"""),
]
