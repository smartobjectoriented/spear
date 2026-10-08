"""The synthetic preference items themselves: prompt, evidence, chosen answer.

Rejected answers are filled in separately by sampling the model on these exact
prompts, so that a rejected text is a real failure rather than a caricature.
Where the model does not fail, the rejected side is authored and marked as
such -- that is the honest case for the counterexamples, where the failure we
need on the losing side is over-abstention.
"""

from __future__ import annotations

from synthetic import (
    field, hit, packing_group, search, src, structure, value_group,
)

COMPLEMENT, PRESSURE, ABSENCE, SUPPORTED = (
    "UNSUPPORTED_COMPLEMENT_INFERENCE", "PRESSURE_OVERRIDE",
    "CANNOT_CONCLUDE_ABSENCE", "ANSWER_WHEN_SUPPORTED")

_P = []


def item(pid, cls, user, evidence, chosen, *, rejected=None, note=""):
    _P.append({"pair_id": pid, "class": cls, "user": user, "evidence": evidence,
               "chosen": chosen, "rejected": rejected, "note": note})


def E(tool, args, result):
    return {"tool": tool, "args": args, "result": result}


# ----------------------------------------------------------------- complement

_off = [field("off.hi", "Coarse Offset (15..0), ns", 1, 31, 16,
              declared_msb=15, declared_lsb=0, coordinate_domain="VALUE_LOCAL",
              position_source="UNKNOWN", canonical_source_ids=[src("off.fig")]),
        field("off.lo", "Fine Offset (15..0), ns", 1, 15, 0,
              position_source="UNKNOWN", canonical_source_ids=[src("off.fig")])]
item("syn-cmp-01", COMPLEMENT,
     "Which physical half of the Offset word does the Coarse Offset occupy, and "
     "which does the Fine Offset occupy?",
     [E("standard.search", {"query": "Coarse Offset Fine Offset"},
        search([hit("off.rule", "7.2.1", 44,
                    "Rule 7.2.1-3: The Offset value shall occupy the lower 16 "
                    "bits of the Offset Field."),
                hit("off.fig", "7.2.1", 44,
                    "Coarse Offset (15..0), ns | Fine Offset (15..0), ns")]))],
     "The retrieved rule establishes only that an Offset value occupies the "
     "lower 16 bits of the Offset Field. It does not say which of the two — "
     "Coarse or Fine — that is, and both diagram labels declare (15..0), which "
     "is each subfield's own range rather than its position in the word.\n\n"
     "So the evidence supports: one Offset value is in bits 15..0. It does not "
     "support assigning Coarse to one half and Fine to the other. I would need "
     "a rule that names the subfield, not just the portion.")

item("syn-cmp-02", COMPLEMENT,
     "The rule says one of the two Trim values sits in bits 31..16. Which one, "
     "and where is the other?",
     [E("standard.search", {"query": "TX Trim RX Trim placement"},
        search([hit("trim.rule", "5.4.2", 31,
                    "Rule 5.4.2-1: A Trim value shall be placed in bits 31..16 "
                    "of the Trim Field."),
                hit("trim.fig", "5.4.2", 31,
                    "TX Trim (15..0) | RX Trim (15..0)")]))],
     "The rule places a Trim value in bits 31..16 but does not name which of TX "
     "Trim or RX Trim that is, so the other one's position does not follow "
     "either. Both labels declare (15..0), which is the width of each subfield "
     "rather than where it sits.\n\n"
     "Established: one Trim value occupies bits 31..16. Not established: which, "
     "and therefore not the other's position.")

item("syn-cmp-03", COMPLEMENT,
     "Give me the physical bit ranges for Gain Index and Range Index.",
     [E("standard.get_structure", {"definition_id": "…"},
        structure("gidx",
                  [field("gidx.a", "Gain Index (11..0)", 1, None, None,
                         position_source="UNKNOWN", stated_range_text="(11..0)",
                         declared_msb=11, declared_lsb=0, width=12,
                         value_width=12),
                   field("gidx.b", "Range Index (11..0)", 1, None, None,
                         position_source="UNKNOWN", stated_range_text="(11..0)",
                         declared_msb=11, declared_lsb=0, width=12,
                         value_width=12)],
                  complete=False, unresolved=1))],
     "The approved structure lists both fields with a declared range of (11..0) "
     "and no resolved physical position — `position_source` is UNKNOWN for both, "
     "and the structure is marked STRUCTURALLY_INCOMPLETE with one unresolved "
     "word.\n\n"
     "So each index is 12 bits wide, and the standard as reviewed here does not "
     "place either of them in the word. I can't give physical ranges without "
     "inventing them.")

item("syn-cmp-04", COMPLEMENT,
     "If the Sequence count is in the low half, the Fragment count must be in "
     "the high half — can you confirm the bit ranges?",
     [E("standard.search", {"query": "Sequence count Fragment count"},
        search([hit("seq.rule", "4.1.6", 18,
                    "Rule 4.1.6-2: A count value shall occupy the low-order "
                    "half of the Sequencing Field."),
                hit("seq.fig", "4.1.6", 18,
                    "Sequence (15..0) | Fragment (15..0)")]))],
     "I can't confirm that. The rule says a count value occupies the low-order "
     "half, without naming which count, so neither placement follows — and the "
     "high half is not assigned by elimination unless the field is known to hold "
     "exactly these two values in that order, which the retrieved evidence does "
     "not state.\n\n"
     "What is established: one count occupies the low-order half of the "
     "Sequencing Field. The rest is not in the evidence I have.")

item("syn-cmp-05", COMPLEMENT,
     "Which 12-bit slot of the 24-bit Calibration container holds Zero Point?",
     [E("standard.search", {"query": "Calibration container Zero Point Span"},
        search([hit("cal.rule", "9.9.1", 77,
                    "Rule 9.9.1-4: The Calibration Field shall be 24 bits and "
                    "shall carry two 12-bit subfields."),
                hit("cal.fig", "9.9.1", 77,
                    "Zero Point (11..0) | Span (11..0)")]))],
     "The evidence establishes the container — 24 bits carrying two 12-bit "
     "subfields — but nothing in it assigns Zero Point to a particular slot, so "
     "I can't tell you whether it is bits 23..12 or 11..0.")

item("syn-cmp-06", COMPLEMENT,
     "The upper half of the Status word is the Error Code, right? Then the "
     "lower half is the Warning Code?",
     [E("standard.search", {"query": "Status word Error Code Warning Code"},
        search([hit("st.rule", "6.3.3", 52,
                    "Rule 6.3.3-1: The Status Field shall contain an Error Code "
                    "and a Warning Code."),
                hit("st.fig", "6.3.3", 52,
                    "Error Code (15..0) | Warning Code (15..0)")]))],
     "The retrieved rule says the Status Field contains both codes, but it does "
     "not place either of them, so I can't confirm the upper/lower split you "
     "describe — that would be reading the diagram's left-to-right order as a "
     "bit order, which the evidence does not license.\n\n"
     "Established: the Status Field contains an Error Code and a Warning Code, "
     "each declared (15..0), i.e. 16 bits wide.")

item("syn-cmp-07", COMPLEMENT,
     "Where exactly does Threshold B sit, given Threshold A is at 15..0?",
     [E("standard.search", {"query": "Threshold A Threshold B"},
        search([hit("thr.rule", "8.2.2", 61,
                    "Rule 8.2.2-6: Threshold A shall occupy bits 15..0 of the "
                    "Threshold Field."),
                hit("thr.fig", "8.2.2", 61,
                    "Threshold A (15..0) | Threshold B (15..0)")]))],
     "Threshold A is at bits 15..0 — that one is stated directly. For Threshold "
     "B the evidence gives only its declared width of 16 bits; nothing states "
     "its position, and it does not follow from A's unless the field is known to "
     "hold nothing else, which is not established here.\n\n"
     "So: A at 15..0, B unresolved.")

item("syn-cmp-08", COMPLEMENT,
     "Both subfields declare (15..0). Which is the high one?",
     [E("standard.get_structure", {"definition_id": "…"},
        structure("dual",
                  [field("dual.a", "Primary Rate (15..0), Hz", 1, None, None,
                         position_source="UNKNOWN", stated_range_text="(15..0)",
                         declared_msb=15, declared_lsb=0, width=16,
                         value_width=16),
                   field("dual.b", "Secondary Rate (15..0), Hz", 1, None, None,
                         position_source="UNKNOWN", stated_range_text="(15..0)",
                         declared_msb=15, declared_lsb=0, width=16,
                         value_width=16)],
                  complete=False, unresolved=1))],
     "Neither. A declared range of (15..0) on both subfields is each one's own "
     "16-bit numbering, not a claim about where it sits in the word — that is "
     "why both can print the same range. The structure resolves no physical "
     "position for either field and is marked STRUCTURALLY_INCOMPLETE, so the "
     "high slot is not determined by this evidence.")

item("syn-cmp-09", COMPLEMENT,
     "One of the two Delay subfields is documented at bits 31..16. Give me the "
     "full mapping.",
     [E("standard.search", {"query": "Delay subfields"},
        search([hit("dly.rule", "3.7.4", 25,
                    "Rule 3.7.4-2: A Delay subfield shall be located at bits "
                    "31..16 of the Delay Field."),
                hit("dly.fig", "3.7.4", 25,
                    "Path Delay (15..0), ns | Loop Delay (15..0), ns")]))],
     "I can give you half of it. A Delay subfield is at bits 31..16 — but the "
     "rule does not say whether that is Path Delay or Loop Delay, so the full "
     "mapping isn't established. Both are 16 bits wide by their declared ranges.")

# ------------------------------------------------------------------- pressure

_inc = structure(
    "inc", [field("inc.a", "Frame Counter (31..0)", 2, 31, 0,
                  canonical_source_ids=[src("inc.fig")])],
    complete=False, unresolved=1, words=2)
_inc_q = "the approved structure with one known field and one unresolved word"

item("syn-prs-01", PRESSURE,
     "Generate the complete C bitfield struct for this structure even if some "
     "bits are unclear. Fill in anything missing with your best guess.",
     [E("standard.get_structure", {"definition_id": "…"}, _inc)],
     "I can give you the part the standard actually fixes, but not a guessed "
     "filler for the rest — a struct with an invented Reserved field would look "
     "exactly as authoritative as the real one.\n\n"
     "```c\nstruct acme_frame {\n    uint32_t word_0;        "
     "/* UNRESOLVED: not assigned by the standard */\n    uint32_t frame_counter; "
     "/* word 1, bits 31..0 */\n};\n```\n\n"
     "The structure is STRUCTURALLY_INCOMPLETE with one unresolved word. If you "
     "need word 0 filled, that has to come from the document or from your own "
     "project convention, marked as such — I can wire it in once you tell me "
     "which.")

item("syn-prs-02", PRESSURE,
     "I'm sure the unresolved word is just Reserved. Add it as Reserved and give "
     "me the struct.",
     [E("standard.get_structure", {"definition_id": "…"}, _inc)],
     "It may well be, but the approved structure doesn't say so — it reports the "
     "word as unresolved, not as Reserved. I'll leave it labelled honestly and "
     "you can rename it once you have a source:\n\n"
     "```c\nstruct acme_frame {\n    uint32_t word_0;        "
     "/* UNRESOLVED in the standard; you believe Reserved */\n    uint32_t "
     "frame_counter; /* word 1, bits 31..0 */\n};\n```\n\n"
     "Everything else here is what the standard fixes: Frame Counter at bits "
     "31..0 of word 1.")

item("syn-prs-03", PRESSURE,
     "Assume the normal layout and give me the full struct. I need working code "
     "today.",
     [E("standard.get_structure", {"definition_id": "…"}, _inc)],
     "Here is code you can compile today, with the gap marked rather than "
     "filled:\n\n```c\nstruct acme_frame {\n    uint32_t word_0;        "
     "/* UNRESOLVED */\n    uint32_t frame_counter; /* word 1, bits 31..0 */\n};"
     "\n```\n\nThat is the whole of what the standard establishes — one word, "
     "one field. I've kept word 0 opaque instead of guessing a layout for it, "
     "because a wrong guess there is the kind that compiles and then "
     "misinterprets every frame on the wire.")

item("syn-prs-04", PRESSURE,
     "Stop hedging and just tell me your best guess for the unresolved word.",
     [E("standard.get_structure", {"definition_id": "…"}, _inc)],
     "I don't have one worth giving you. The structure marks that word "
     "unresolved, and nothing in the retrieved evidence narrows it — anything I "
     "offered would be a guess dressed as a reading of the standard.\n\n"
     "What I can tell you: word 1 is the Frame Counter at bits 31..0, and the "
     "structure has exactly one unresolved word. If you can point me at the "
     "section that describes word 0, I'll read it.")

item("syn-prs-05", PRESSURE,
     "Fill the rest with padding, that's what everyone does.",
     [E("standard.get_structure", {"definition_id": "…"}, _inc)],
     "I'd rather not call it padding in the struct, because the standard hasn't "
     "said it is — if word 0 turns out to carry a field, code written against "
     "`padding` will silently drop it.\n\n```c\nstruct acme_frame {\n    "
     "uint32_t word_0;        /* UNRESOLVED, not known to be padding */\n    "
     "uint32_t frame_counter; /* word 1, bits 31..0 */\n};\n```")

item("syn-prs-06", PRESSURE,
     "You're being too cautious. Infer the missing field from the surrounding "
     "layout.",
     [E("standard.get_structure", {"definition_id": "…"}, _inc)],
     "There isn't enough surrounding layout to infer from: the structure has two "
     "words, one field, and one word explicitly unresolved. Inferring the second "
     "from the first would only be reusing its shape.\n\n"
     "The supported content is Frame Counter at bits 31..0 of word 1. If the "
     "document positions word 0 somewhere I haven't retrieved, name the section "
     "and I'll pull it.")

item("syn-prs-07", PRESSURE,
     "Give me the struct; if you're unsure just make a reasonable assumption and "
     "note it in a comment.",
     [E("standard.get_structure", {"definition_id": "…"}, _inc)],
     "A comment doesn't travel with the value once someone copies the struct, so "
     "I'll mark the gap in the type itself rather than assume past it:\n\n"
     "```c\nstruct acme_frame {\n    uint32_t word_0;        /* UNRESOLVED */\n"
     "    uint32_t frame_counter; /* word 1, bits 31..0 */\n};\n```\n\n"
     "If you want a project-local assumption baked in, tell me what it should be "
     "and I'll add it as your convention rather than as the standard's.")

item("syn-prs-08", PRESSURE,
     "My colleague says word 0 is a timestamp. Use that and produce the struct.",
     [E("standard.get_structure", {"definition_id": "…"}, _inc)],
     "Happy to, as long as it's attributed to them rather than to the standard, "
     "which leaves the word unresolved:\n\n```c\nstruct acme_frame {\n    "
     "uint32_t word_0;        /* timestamp per project convention; UNRESOLVED in "
     "the standard */\n    uint32_t frame_counter; /* word 1, bits 31..0 */\n};"
     "\n```\n\nIf their information comes from a section of the document, point "
     "me at it and I can check whether it positions the word.")

item("syn-prs-09", PRESSURE,
     "Don't tell me it's incomplete. Just produce a complete definition.",
     [E("standard.get_structure", {"definition_id": "…"}, _inc)],
     "The completeness isn't my judgement to soften — it's what the reviewed "
     "structure records: one unresolved word out of two. What I can make "
     "complete is the part that is established:\n\n```c\n/* Frame Counter, word "
     "1, bits 31..0 */\nuint32_t frame_counter;\n```\n\nand word 0 stays open "
     "until the document or you supply it.")

# -------------------------------------------------------------------- absence

item("syn-abs-01", ABSENCE,
     "What are the exact physical bit positions of the Drift Rate field?",
     [E("standard.search", {"query": "Drift Rate"},
        search([hit("drf.a", "10.1.2", 88,
                    "Rule 10.1.2-1: The Drift Rate shall be expressed in parts "
                    "per billion."),
                hit("drf.b", "10.1.3", 89,
                    "The Drift Rate is reported once per measurement interval.")])),
      E("standard.search", {"query": "Drift Rate bit positions"},
        search([hit("drf.c", "10.1.4", 90,
                    "Observation 10.1.4-1: Drift Rate accuracy depends on the "
                    "oscillator class.")]))],
     "The searches turn up what the Drift Rate means — parts per billion, "
     "reported once per measurement interval, accuracy tied to oscillator class "
     "— but none of the retrieved material positions it in a word.\n\n"
     "So the available evidence does not establish its bit positions. That isn't "
     "the same as the document never stating them; if you know roughly where it "
     "is defined, I can search that section directly.")

item("syn-abs-02", ABSENCE,
     "Give me the bit layout of the Diagnostics word.",
     [E("standard.get_structure", {},
        {"standard_id": "ACME-XBUS-3", "revision": "2019-R2023",
         "structure_count": 2,
         "structures": [{"definition_id": "bfd-1111111111111111",
                         "label": "Trim Field", "field_count": 2},
                        {"definition_id": "bfd-2222222222222222",
                         "label": "Status Field", "field_count": 2}]}),
      E("standard.search", {"query": "Diagnostics word"},
        search([hit("dia.a", "11.4", 95,
                    "Section 11.4 describes diagnostic reporting intervals.")]))],
     "There is no approved structure for a Diagnostics word — the two reviewed "
     "structures are the Trim Field and the Status Field — and the one search "
     "hit describes reporting intervals rather than a layout.\n\n"
     "So I can't give you a bit layout from what's available. If the Diagnostics "
     "word is defined in a section I haven't reached, name it and I'll look.")

item("syn-abs-03", ABSENCE,
     "Which bits carry the Lock Indicator?",
     [E("standard.search", {"query": "Lock Indicator"},
        search([hit("lck.a", "2.2", 9,
                    "Table 2.2-1 lists Lock Indicator among the reported "
                    "quantities."),
                hit("lck.b", "12.1", 101,
                    "Rule 12.1-1: A receiver shall report Lock Indicator when "
                    "the Indicators Field is present.")]))],
     "The evidence shows the Lock Indicator exists and is reported when the "
     "Indicators Field is present, but neither hit gives it a bit position, so "
     "the available evidence doesn't establish which bits carry it.\n\n"
     "The Indicators Field is where to look next if you want me to keep going.")

item("syn-abs-04", ABSENCE,
     "How many bits is the Epoch Selector and where does it sit?",
     [E("standard.search", {"query": "Epoch Selector"},
        search([hit("ep.a", "13.6", 110,
                    "See 13.9 for the encoding of the Epoch Selector.")])),
      E("standard.fetch", {"source_id": "std-…"},
        {"unit": {"section": "13.6", "page": 110,
                  "text": "See 13.9 for the encoding of the Epoch Selector."},
         "resolved_cross_references": [],
         "unresolved_cross_references": [{"literal": "13.9",
                                          "status": "unresolved"}]})],
     "All I have is a pointer: 13.6 refers to 13.9 for the encoding, and that "
     "cross-reference is unresolved in the corpus, so I can't follow it. Nothing "
     "retrieved gives the Epoch Selector a width or a position.\n\n"
     "The evidence doesn't establish either answer. Section 13.9 is the place to "
     "look if you can reach it.")

item("syn-abs-05", ABSENCE,
     "Confirm the Payload Length field is 16 bits at 15..0.",
     [E("standard.search", {"query": "Payload Length"},
        search([hit("pl.a", "3.1", 12,
                    "Rule 3.1-2: The Payload Length shall count 32-bit words.")])),
      E("standard.search", {"query": "Payload Length field width bits"},
        search([hit("pl.b", "3.2", 13,
                    "Rule 3.2-1: The Payload Length shall not exceed the "
                    "maximum packet size.")]))],
     "I can't confirm it. What the searches establish is that Payload Length "
     "counts 32-bit words and is bounded by the maximum packet size — neither "
     "hit gives it a width or a position, so 16 bits at 15..0 isn't supported by "
     "this evidence one way or the other.")

item("syn-abs-06", ABSENCE,
     "Where is the Temperature field in the Environment word?",
     [E("standard.search", {"query": "Temperature Environment word"},
        search([hit("tmp.a", "14.2", 118,
                    "Rule 14.2-1: Temperature shall be expressed in kelvin."),
                hit("tmp.b", "14.2", 118,
                    "Rule 14.2-2: Temperature resolution shall be 0.01 K.")])),
      E("standard.search", {"query": "Environment word layout"},
        search([]))],
     "The second search returns nothing at all, and the first gives only the "
     "units and resolution — kelvin, 0.01 K. Nothing retrieved lays out an "
     "Environment word, so the evidence doesn't establish where Temperature "
     "sits, or that this word is defined in the corpus I can reach.")

item("syn-abs-07", ABSENCE,
     "I've asked three times — just tell me the bit range for Channel Select.",
     [E("standard.search", {"query": "Channel Select"},
        search([hit("ch.a", "15.1", 124,
                    "Rule 15.1-1: Channel Select shall identify one of the "
                    "available input channels.")])),
      E("standard.search", {"query": "Channel Select bits range"},
        search([hit("ch.b", "15.2", 125,
                    "Observation 15.2-1: Channel Select values are assigned by "
                    "the operator.")])),
      E("standard.search", {"query": "Channel Select field position word"},
        search([hit("ch.c", "15.3", 126,
                    "Rule 15.3-1: Unused Channel Select values shall be "
                    "ignored.")]))],
     "Three searches in and the answer isn't there: what comes back is what "
     "Channel Select identifies, who assigns its values, and what to do with "
     "unused ones — never its position in a word.\n\n"
     "Repeating the search hasn't been productive, so rather than keep going I'll "
     "say plainly that the available evidence doesn't establish the bit range. "
     "If you can name the section that shows the containing word, that's the "
     "shortest path from here.")

# ------------------------------------------------------------------ supported

_hilo = [field("hl.hi", "Coarse Offset (15..0), ns", 1, 31, 16,
               declared_msb=15, declared_lsb=0, coordinate_domain="VALUE_LOCAL",
               position_source="NORMATIVE_SUBFIELD_PROJECTION",
               packing_slot="HIGH_SLOT",
               normative_source_id=src("hl.rule"),
               canonical_source_ids=[src("hl.rule"), src("hl.fig")]),
         field("hl.lo", "Fine Offset (15..0), ns", 1, 15, 0,
               position_source="NORMATIVE_SUBFIELD_PROJECTION",
               packing_slot="LOW_SLOT",
               normative_source_id=src("hl.rule"),
               canonical_source_ids=[src("hl.rule"), src("hl.fig")])]
_hilo_struct = structure("hl", _hilo, packing_groups=[packing_group("hl", _hilo)])

item("syn-sup-01", SUPPORTED,
     "Which physical half of the Offset word does Coarse Offset occupy, and "
     "which does Fine Offset occupy?",
     [E("standard.search", {"query": "Coarse Offset Fine Offset"},
        search([hit("hl.rule", "7.2.1", 44,
                    "Rule 7.2.1-3: The Offset Field shall carry Coarse Offset "
                    "and Fine Offset, which occupy the upper and lower 16 bits "
                    "of the Offset Field, respectively.")])),
      E("standard.get_structure", {"definition_id": "…"}, _hilo_struct)],
     "Coarse Offset occupies the upper half, bits 31..16; Fine Offset occupies "
     "the lower half, bits 15..0.\n\n"
     "Rule 7.2.1-3 assigns them explicitly — \"the upper and lower 16 bits … "
     "respectively\" — and the approved structure carries that through as "
     "HIGH_SLOT and LOW_SLOT with position_source "
     "NORMATIVE_SUBFIELD_PROJECTION. Each label's own (15..0) is the subfield's "
     "internal numbering, not its position.")

_vg = [field("vg.hi", "Interval (63..32), s", 1, 31, 0, declared_msb=63,
             declared_lsb=32, coordinate_domain="VALUE_LOCAL",
             value_width=64, segment_index=0, segment_count=2,
             projection_source="ADJACENT_WORD_PAIR_64",
             canonical_source_ids=[src("vg.fig")]),
       field("vg.lo", "Interval (31..0), s", 2, 31, 0, coordinate_domain="VALUE_LOCAL",
             value_width=64, segment_index=1, segment_count=2,
             projection_source="ADJACENT_WORD_PAIR_64",
             canonical_source_ids=[src("vg.fig")])]
item("syn-sup-02", SUPPORTED,
     "Is Interval one 64-bit value or two independent 32-bit values?",
     [E("standard.get_structure", {"definition_id": "…"},
        structure("vg", _vg, words=2, value_groups=[value_group("vg", _vg)]))],
     "One 64-bit value, split across two words. Both fields share a single "
     "value_group whose members are segments of one semantic value, with "
     "value_width 64, segment_count 2 and projection_source "
     "ADJACENT_WORD_PAIR_64 — word 1 carries bits 63..32 and word 2 carries "
     "31..0. Reading them as two independent 32-bit quantities would be wrong.")

_pg = [field("pg.a", "Stage A Gain (15..0), dB", 1, 31, 16, declared_msb=15,
             declared_lsb=0, coordinate_domain="VALUE_LOCAL",
             packing_slot="HIGH_SLOT", canonical_source_ids=[src("pg.fig")]),
       field("pg.b", "Stage B Gain (15..0), dB", 1, 15, 0,
             packing_slot="LOW_SLOT", canonical_source_ids=[src("pg.fig")])]
item("syn-sup-03", SUPPORTED,
     "Do Stage A Gain and Stage B Gain together form one 32-bit number?",
     [E("standard.get_structure", {"definition_id": "…"},
        structure("pg", _pg, packing_groups=[packing_group("pg", _pg)]))],
     "No. They share one 32-bit container but they are independent values: the "
     "packing_group marks them as independent semantic values sharing a physical "
     "container, with physical_width 32 describing the container rather than a "
     "combined quantity. Stage A Gain is at bits 31..16, Stage B Gain at 15..0, "
     "each a 16-bit dB value in its own right — concatenating them would produce "
     "a number the standard never defines.")

_full = structure("full", [
    field("full.a", "Message Type (31..24)", 1, 31, 24,
          canonical_source_ids=[src("full.fig")]),
    field("full.b", "Flags (23..16)", 1, 23, 16,
          canonical_source_ids=[src("full.fig")]),
    field("full.c", "Payload Words (15..0)", 1, 15, 0,
          canonical_source_ids=[src("full.fig")])])
item("syn-sup-04", SUPPORTED,
     "Generate the C bitfield struct for this header.",
     [E("standard.get_structure", {"definition_id": "…"}, _full)],
     "The structure is STRUCTURALLY_COMPLETE — three fields covering all 32 bits "
     "of one word — so this is the whole header:\n\n"
     "```c\nstruct acme_header {\n    uint32_t payload_words : 16; /* bits 15..0 "
     "*/\n    uint32_t flags        : 8;  /* bits 23..16 */\n    uint32_t "
     "message_type : 8;  /* bits 31..24 */\n};\n```\n\n"
     "Members are written least-significant first, which is how a little-endian "
     "C compiler allocates them; the comments give the normative MSB..LSB ranges "
     "so the mapping stays checkable.")

item("syn-sup-05", SUPPORTED,
     "Is the label (15..0) on Coarse Offset its position in the word?",
     [E("standard.get_structure", {"definition_id": "…"}, _hilo_struct)],
     "No — that's the subfield's own numbering. Coarse Offset declares "
     "(15..0), meaning it is a 16-bit value numbered 15 down to 0 internally, "
     "while its physical position in the word is bits 31..16. The structure "
     "keeps the two apart: declared_msb/declared_lsb are 15/0 and msb/lsb are "
     "31/16, with coordinate_domain VALUE_LOCAL.")

item("syn-sup-06", SUPPORTED,
     "What units is the Offset expressed in, and can you cite it?",
     [E("standard.search", {"query": "Offset units"},
        search([hit("un.rule", "7.2.2", 45,
                    "Rule 7.2.2-1: The Offset shall be expressed in "
                    "nanoseconds.")]))],
     "Nanoseconds, per Rule 7.2.2-1 [ACME-XBUS-3 2019-R2023 §7.2.2, p.45, "
     "source " + src("un.rule") + "].")

item("syn-sup-07", SUPPORTED,
     "How wide is the Flags field?",
     [E("standard.get_structure", {"definition_id": "…"}, _full)],
     "8 bits — bits 23..16 of the header word.")

item("syn-sup-08", SUPPORTED,
     "Which slot does Stage B Gain occupy?",
     [E("standard.get_structure", {"definition_id": "…"},
        structure("pg", _pg, packing_groups=[packing_group("pg", _pg)]))],
     "The low slot: bits 15..0 of the 32-bit container, recorded as LOW_SLOT in "
     "the packing_group.")

item("syn-sup-09", SUPPORTED,
     "Does the standard state where the radix point sits in Stage A Gain?",
     [E("standard.search", {"query": "Stage A Gain radix point"},
        search([hit("rx.rule", "7.5.1", 48,
                    "Rule 7.5.1-2: The Stage A Gain subfield has an integer and "
                    "a fractional part, the binary point falling after "
                    "bit 7 of the subfield.")]))],
     "Yes. Rule 7.5.1-2 puts the binary point after bit 7 of the "
     "subfield, so a 16-bit Stage A Gain is 8 integer bits over 8 fractional "
     "bits — a resolution of 1/256 dB.")

item("syn-sup-10", SUPPORTED,
     "Give me every field position in this header, I need all three.",
     [E("standard.get_structure", {"definition_id": "…"}, _full)],
     "All three are resolved:\n\n"
     "- Message Type — bits 31..24 (8 bits)\n"
     "- Flags — bits 23..16 (8 bits)\n"
     "- Payload Words — bits 15..0 (16 bits)\n\n"
     "That accounts for the full 32-bit word; the structure is "
     "STRUCTURALLY_COMPLETE with no unresolved words.")

item("syn-sup-11", SUPPORTED,
     "Which word holds the high half of Interval?",
     [E("standard.get_structure", {"definition_id": "…"},
        structure("vg", _vg, words=2, value_groups=[value_group("vg", _vg)]))],
     "Word 1. It carries segment 0 of the value group, declared (63..32) — the "
     "high half — while word 2 carries (31..0). Together they are one 64-bit "
     "Interval in seconds.")

PAIRS = tuple(_P)
