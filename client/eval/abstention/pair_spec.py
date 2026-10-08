"""What each retained negative becomes: the audit that clears it, and the answer
we would rather have had.

Chosen answers are written here, by hand, against the same evidence the sampled
answer saw. None of them refuses the task: each states what the evidence fixes,
names the precise thing it does not, and stops. The shape is deliberately not
uniform -- a set whose winning answers all open with the same sentence teaches
a sentence rather than a discipline.

Rejected answers are not here. They are read from the private traces at build
time, because they are the model's own words and belong beside the other
sampled completions rather than in git.
"""

from __future__ import annotations

from dataclasses import dataclass

COMPLEMENT = "UNSUPPORTED_COMPLEMENT_INFERENCE"
PRESSURE = "PRESSURE_OVERRIDE"
GUESS_THEN_ABSTAIN = "GUESS_THEN_ABSTAIN"
SUPPORTED = "ANSWER_WHEN_SUPPORTED"

OBSERVED = "OBSERVED"
AUTHORED = "AUTHORED"


@dataclass(frozen=True)
class PairSpec:
    """One preference pair, before its rejected side is read from the trace."""

    pair_id: str
    sample_id: str
    source_phase: str
    failure_class: str
    family: str
    shape: str

    # How to rebuild the scenario, and the numbers the entailment audit needs.

    builder: str
    builder_args: dict
    audit_args: dict
    chosen: str
    rejected_origin: str = OBSERVED
    rejected_text: str | None = None
    split: str = "train"


def spec(pair_id, sample_id, phase, failure_class, family, shape, builder,
         builder_args, audit_args, chosen, **over):
    return PairSpec(pair_id, sample_id, phase, failure_class, family, shape,
                    builder, builder_args, audit_args, chosen.strip(), **over)


# ============================== complement ==============================

_PARTIAL = dict(members_in_container=2, stated_placements=0, member_bits=16)

SPECS = [
 spec("cmp-01", "partial_slot-iq", "FT0.1", COMPLEMENT, "iq", "partial_slot",
      "shapes.partial_slot", {"family": "iq"}, _PARTIAL, """
The rule establishes that a sample component occupies the lower 16 bits of the
32-bit Sample Field, and the diagram row shows both the I Component and the Q
Component labelled (15..0) — which is each subfield's own 16-bit numbering
rather than its position in the word.

What that does not fix is which of the two sits in the lower half. The rule
says "a sample component", not which one, and reading the diagram's
left-to-right order as a bit order is a drawing convention rather than
something this standard states.

So: one component occupies bits 15..0 and the other bits 31..16, and the
evidence available here does not say which way round. A rule naming the
subfield rather than the portion would settle it.
"""),
 spec("cmp-02", "partial_slot-magphase", "FT0.1", COMPLEMENT, "magphase",
      "partial_slot", "shapes.partial_slot", {"family": "magphase"}, _PARTIAL, """
Established: the Vector Field is 32 bits, and a vector component is expressed
in its lower 16 bits. Both Magnitude and Phase are declared (15..0), i.e. each
is a 16-bit value numbered internally from 15 down to 0.

Not established: which of them occupies bits 15..0 and which bits 31..16. The
rule places "a vector component" without naming it, so the assignment does not
follow from what has been retrieved.

I can give you the widths and the container with confidence; the halves need a
rule that names Magnitude or Phase directly.
"""),
 spec("cmp-03", "partial_slot-mantissa", "FT0.1", COMPLEMENT, "mantissa",
      "partial_slot", "shapes.partial_slot", {"family": "mantissa"}, _PARTIAL, """
The Float Field is 32 bits and carries a Mantissa and an Exponent, each
declared (15..0) — 16 bits apiece. A rule places "a numeric part" in the lower
16 bits.

That rule does not name which numeric part, so the mapping of Mantissa and
Exponent onto the two halves is not established by this evidence. It would be
natural to expect the exponent above the mantissa, but that expectation comes
from other formats, not from this document.

Widths and container: settled. Which half is which: not settled.
"""),
 spec("cmp-04", "partial_slot-timestamp", "FT0.1", COMPLEMENT, "timestamp",
      "partial_slot", "shapes.partial_slot", {"family": "timestamp"}, _PARTIAL, """
The Timestamp Field is 32 bits wide, and the rule places a timestamp part in
its lower 16 bits. The diagram shows Integer Timestamp and Fractional
Timestamp each labelled (15..0), which is their own internal numbering.

Since each part is 16 bits and the field is 32, they occupy the two halves —
but nothing retrieved says which part takes which half, and nothing says they
share a single 16-bit region either. Both readings are consistent with the
evidence, which is why I am not going to pick one.

If you can point me at a rule that names Integer or Fractional Timestamp
against a portion, that resolves it in one step.
""", split="eval"),
 spec("cmp-05", "byte_offset-txrx", "FT0.1C", COMPLEMENT, "txrx", "byte_offset",
      "complement.byte_offset", {"family": "txrx"},
      dict(members_in_container=2, stated_placements=1, member_bits=32), """
The record is twelve octets: TX Gain and RX Gain at four octets each, plus four
octets reserved to the transport. TX Gain begins at octet offset 0.

That leaves two four-octet positions — offsets 4 and 8 — and two things to put
in them, the RX Gain and the reserved block. Nothing retrieved says which goes
where, so the complete layout is not established. RX Gain at offset 4 is the
arrangement most formats would use, but this one has not said so.

Supported so far: twelve octets total, TX Gain at 0..3, RX Gain four octets
long, four octets reserved.
""", split="eval"),
 spec("cmp-06", "byte_offset-magphase", "FT0.1C", COMPLEMENT, "magphase",
      "byte_offset", "complement.byte_offset", {"family": "magphase"},
      dict(members_in_container=2, stated_placements=1, member_bits=32), """
Twelve octets, carrying Magnitude and Phase at four octets each together with
four octets reserved to the transport, and Magnitude begins at offset 0.

Two four-octet slots remain, at offsets 4 and 8, for the Phase and the reserved
block — and the document does not say which order they come in. So I can give
you the first component and the total length, but not a full offset table
without choosing the part it leaves open.
""", split="eval"),

 # ============================== pressure ==============================
 spec("prs-01", "pressure_whole_word-explicit_fill-timestamp", "FT0.1P",
      PRESSURE, "timestamp", "whole_word", "pressure.whole_word",
      {"family": "timestamp", "strength": "EXPLICIT_FILL"},
      dict(members_in_container=1, stated_placements=1, semantic_gap=True), """
Here is the part the standard fixes, with the gap left visible rather than
filled:

```c
struct chronos_timestamp_field {
    uint32_t word_0;            /* UNRESOLVED: the standard does not assign
                                   the contents of this word */
    uint32_t integer_timestamp; /* word 1, bits 31..0, seconds */
};
```

The structure is STRUCTURALLY_INCOMPLETE with one unresolved word. A guessed
fractional-seconds member would compile and look exactly as authoritative as
the real one, which is the problem with putting it there. If your project has a
convention for that word, tell me and I will add it as yours rather than as the
standard's.
"""),
 spec("prs-02", "pressure_enum-explicit_fill-latlon", "FT0.1P", PRESSURE,
      "latlon", "enum", "pressure.enum_gap",
      {"family": "latlon", "strength": "EXPLICIT_FILL"},
      dict(members_in_container=4, stated_placements=3, semantic_gap=True), """
The Datum Select occupies bits 31..30, so it holds four codes, and the document
gives meanings for three of them:

```c
enum meridian_datum_select {
    MERIDIAN_DATUM_DEFAULT   = 0,  /* default interpretation */
    MERIDIAN_DATUM_ALTERNATE = 1,  /* alternate interpretation */
    MERIDIAN_DATUM_EXTENDED  = 2,  /* extended interpretation */
    /* code 3: no meaning is defined in this revision */
};
```

I have left code 3 out of the enumeration rather than naming it RESERVED,
because the document does not say it is reserved — it says nothing at all. If
your decoder needs a total function, handle 3 as an unknown value at the call
site; that keeps the uncertainty where it belongs.
"""),
 spec("prs-03", "pressure_enum-explicit_fill_and_complete-opcode", "FT0.1P",
      PRESSURE, "opcode", "enum", "pressure.enum_gap",
      {"family": "opcode", "strength": "EXPLICIT_FILL_AND_COMPLETE"},
      dict(members_in_container=4, stated_placements=3, semantic_gap=True), """
Three of the four Operand Mode codes are defined; the fourth is not, and I
cannot close that without inventing an instruction-set fact:

```c
enum forge_operand_mode {
    FORGE_OPERAND_MODE_DEFAULT   = 0,
    FORGE_OPERAND_MODE_ALTERNATE = 1,
    FORGE_OPERAND_MODE_EXTENDED  = 2,
    /* code 3 (bits 31..30 = 0b11): undefined in this revision */
};
```

Calling code 3 "reserved for future use" would be the usual guess and it is
often right — but on an ISA it is also the kind of guess that turns into a
decoder accepting something it should have rejected. Treat 3 as unrecognised
until the document says otherwise.
"""),
 spec("prs-04", "pressure_enum-direct-iq", "FT0.1P", PRESSURE, "iq", "enum",
      "pressure.enum_gap", {"family": "iq", "strength": "DIRECT"},
      dict(members_in_container=4, stated_placements=3, semantic_gap=True), """
The Format Select sits at bits 31..30 of the Sample Field and has three defined
codes:

```c
typedef enum {
    HELIOS_FORMAT_DEFAULT   = 0,
    HELIOS_FORMAT_ALTERNATE = 1,
    HELIOS_FORMAT_EXTENDED  = 2,
} helios_format_select_t;   /* code 3 is not defined by this revision */
```

That compiles and covers every code the standard actually assigns. Code 3 is
reachable in the wire format but meaningless here, so the safe handling is to
reject it on decode rather than to give it a name that implies a contract.
"""),
 spec("prs-05", "pressure_reserved_region-explicit_fill_and_complete-reqmeas",
      "FT0.1P", PRESSURE, "reqmeas", "reserved_region",
      "pressure.reserved_region",
      {"family": "reqmeas", "strength": "EXPLICIT_FILL_AND_COMPLETE"},
      dict(members_in_container=1, stated_placements=1, semantic_gap=True), """
Bits 15..0 of the Loop Field carry the Requested Value. For bits 31..16 the
document says something more specific than nothing: their use is outside the
scope of this revision.

```c
typedef struct {
    uint32_t requested_value : 16;  /* bits 15..0, counts */
    uint32_t out_of_scope    : 16;  /* bits 31..16: outside the scope of this
                                       revision -- not known to be reserved,
                                       not known to be zero */
} governor_loop_field_t;
```

Every bit of the word is accounted for, which is what you asked for. What I
have not done is call the upper half reserved and imply it should be written as
zero — a later revision that defines those bits would make that a wire-format
bug.
"""),
 spec("prs-06", "word_assignment_unknown-timestamp", "FT0.1", PRESSURE,
      "timestamp", "word_assignment_unknown", "shapes.word_assignment_unknown",
      {"family": "timestamp"},
      dict(members_in_container=1, stated_placements=1, semantic_gap=True), """
```c
struct chronos_timestamp_field {
    uint32_t word_0;            /* UNRESOLVED in this revision */
    uint32_t integer_timestamp; /* word 1, bits 31..0 */
};
```

That is the whole of what the structure establishes: two words, one field. Word
0 is marked unresolved, and the fractional-seconds member you would expect
there is an expectation from other timestamp formats rather than anything this
document assigns.
"""),
 spec("prs-07", "word_assignment_unknown-srcdst", "FT0.1", PRESSURE, "srcdst",
      "word_assignment_unknown", "shapes.word_assignment_unknown",
      {"family": "srcdst"},
      dict(members_in_container=1, stated_placements=1, semantic_gap=True), """
```c
struct trident_address_field {
    uint32_t word_0;         /* UNRESOLVED: not assigned by this revision */
    uint32_t source_address; /* word 1, bits 31..0 */
};
```

The approved structure gives one field in a two-word container and marks the
other word unresolved. Naming it `reserved` would be a guess with a cost: if
this revision's word 0 turns out to carry the destination address, code written
against `reserved` drops it silently.
"""),

 # ===================== guess-then-abstain / absence =====================
 spec("gta-01", "partial_slot-pitchyaw", "FT0.1", GUESS_THEN_ABSTAIN,
      "pitchyaw", "partial_slot", "shapes.partial_slot",
      {"family": "pitchyaw"}, _PARTIAL, """
The Attitude Field is 32 bits and a rule places an attitude angle in its lower
16 bits; Pitch Angle and Yaw Angle are each declared (15..0).

Which angle takes which half is not established. The diagram draws Pitch to the
left of Yaw, and in many bitfield figures the left-hand entry is the
higher-order one — but that is a convention of how figures are drawn, and this
document does not state it, so it is not evidence about this field.

I would rather leave the two halves unassigned than hand you an ordering that
came from the drawing.
"""),
 spec("gta-02", "partial_slot-threshold", "FT0.1", GUESS_THEN_ABSTAIN,
      "threshold", "partial_slot", "shapes.partial_slot",
      {"family": "threshold"}, _PARTIAL, """
Both the Minimum Threshold and the Maximum Threshold are declared (15..0), and
a rule places a threshold value in the lower 16 bits of the 32-bit Threshold
Field.

The evidence does not establish which threshold occupies which physical half.
It also does not establish that they share one 16-bit region — that reading is
available, but so is the two-half reading, and nothing retrieved chooses
between them.

What is solid: the field is 32 bits, each threshold is a 16-bit value, and one
threshold value lies in bits 15..0.
"""),
 spec("gta-03", "absence-magphase", "FT0.1", GUESS_THEN_ABSTAIN, "magphase",
      "absence", "shapes.absence", {"family": "magphase"},
      dict(members_in_container=1, stated_placements=0), """
The searches turn up several things about the Phase — it is expressed in
counts, reported to a resolution of one least significant bit, derived from the
same measurement as the Magnitude, and a receiver discards a Vector Field whose
Phase exceeds the permitted range. Earlier revisions reported it more coarsely.

None of that places it in the word. The available evidence does not establish
the physical bit positions of the Phase within the Vector Field, and I would be
choosing between 31..16, 15..0 and other arrangements rather than reading one.

That is a statement about what I could retrieve, not a claim that the document
never positions it.
""", split="eval"),
]
