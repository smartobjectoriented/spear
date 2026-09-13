"""Synthetic page geometry covering the table shapes STD2A must handle.

Positions are explicit so ``pdftotext`` reconstructs real columns, wrapped
cells, spanning labels and page-crossing tables. No licensed text appears here.
"""

from __future__ import annotations

from tests.standard_extraction_fixture import positioned_pdf_bytes


def _row(y: float, cells: tuple[tuple[float, str], ...]):
    return tuple((x, y, text) for x, text in cells)


GEOMETRY_PAGES: tuple[tuple[tuple[float, float, str], ...], ...] = (
    # 1 -- caption, header, three clean columns
    ((72, 730, "Table 1: Simple Values"),
     *_row(700, ((72, "Code"), (220, "Meaning"), (400, "Size"))),
     *_row(680, ((72, "0001"), (220, "Alpha"), (400, "4"))),
     *_row(660, ((72, "0010"), (220, "Beta"), (400, "8"))),
     *_row(640, ((72, "0011"), (220, "Gamma"), (400, "16")))),
    # 2 -- uneven widths, a wrapped cell, a blank cell, a spanning remark
    ((72, 730, "Table 2: Channel Flags"),
     *_row(700, ((72, "Bit"), (140, "Name"), (330, "Notes"))),
     *_row(675, ((72, "7"), (140, "Enable"), (330, "Turns the channel on"))),
     (330, 661, "when the flag is set"),
     *_row(635, ((72, "6"), (140, "Reserved"))),
     *_row(615, ((72, "5"), (140, "Mode"), (330, "Selects the operating mode"))),
     (72, 585, "Values above are reserved for future use and apply to every profile.")),
    # 3 -- bit positions, a label spanning several of them, then a data row
    ((72, 730, "Table 3: Word Layout"),
     *_row(700, ((72, "31"), (110, "30"), (148, "29"), (186, "28"),
                 (224, "27"), (262, "26"), (300, "25"), (338, "24"))),
     *_row(675, ((72, "Stream Class"), (224, "Reserved"))),
     *_row(655, ((72, "1"), (110, "0"), (148, "1"), (186, "1"),
                 (224, "0"), (262, "0"), (300, "0"), (338, "0")))),
    # 4 -- aligned prose and a numbered list; neither is a table
    ((72, 730, "The encoder writes each frame in order and the decoder reads them"),
     (72, 714, "back in the same order, which keeps the stream aligned."),
     (72, 686, "1 The first condition applies to every implementation."),
     (72, 668, "2 The second condition applies only to extended profiles."),
     (72, 650, "3 The third condition is informative.")),
    # 5 -- a figure caption and its prose, then a captioned table
    ((72, 730, "Figure 1: Processing Chain"),
     (72, 700, "Figure 1 shows the processing chain used by the reference design."),
     (72, 660, "Table 4: Supported Rates"),
     *_row(630, ((72, "Rate"), (220, "Unit"))),
     *_row(610, ((72, "48000"), (220, "Hz"))),
     *_row(590, ((72, "96000"), (220, "Hz")))),
    # 6 -- two independent tables on one page, neither captioned
    ((*_row(730, ((72, "Left"), (220, "Right"))),
      *_row(710, ((72, "one"), (220, "two"))),
      *_row(690, ((72, "three"), (220, "four"))),
      *_row(560, ((72, "Alpha"), (220, "Beta"), (360, "Gamma"))),
      *_row(540, ((72, "a"), (220, "b"), (360, "c"))),
      *_row(520, ((72, "d"), (220, "e"), (360, "f"))))),
    # 7 -- a table running to the bottom of the page
    ((72, 730, "Table 5: Long Table"),
     *_row(700, ((72, "Index"), (220, "Label"), (360, "Name"))),
     *_row(680, ((72, "1"), (220, "first"), (360, "ten"))),
     *_row(650, ((72, "2"), (220, "second"), (360, "twenty"))),
     *_row(620, ((72, "3"), (220, "third"), (360, "thirty"))),
     *_row(590, ((72, "4"), (220, "fourth"), (360, "forty"))),
     *_row(560, ((72, "5"), (220, "fifth"), (360, "fifty"))),
     *_row(530, ((72, "6"), (220, "sixth"), (360, "sixty"))),
     *_row(500, ((72, "7"), (220, "seventh"), (360, "seventy"))),
     *_row(470, ((72, "8"), (220, "eighth"), (360, "eighty"))),
     *_row(440, ((72, "9"), (220, "ninth"), (360, "ninety"))),
     *_row(410, ((72, "10"), (220, "tenth"), (360, "hundred"))),
     *_row(380, ((72, "11"), (220, "eleventh"), (360, "alpha"))),
     *_row(350, ((72, "12"), (220, "twelfth"), (360, "beta"))),
     *_row(320, ((72, "13"), (220, "thirteenth"), (360, "gamma"))),
     *_row(290, ((72, "14"), (220, "fourteenth"), (360, "delta"))),
     *_row(260, ((72, "15"), (220, "fifteenth"), (360, "epsilon"))),
     *_row(230, ((72, "16"), (220, "sixteenth"), (360, "zeta"))),
     *_row(200, ((72, "17"), (220, "seventeenth"), (360, "eta"))),
     *_row(170, ((72, "18"), (220, "eighteenth"), (360, "theta"))),
     *_row(140, ((72, "19"), (220, "nineteenth"), (360, "iota"))),
     *_row(110, ((72, "20"), (220, "twentieth"), (360, "kappa")))),
    # 8 -- the same table continuing, header repeated
    ((*_row(740, ((72, "Index"), (220, "Label"), (360, "Name"))),
      *_row(715, ((72, "21"), (220, "twenty-first"), (360, "ten"))),
      *_row(695, ((72, "22"), (220, "twenty-second"), (360, "twenty"))),
      *_row(675, ((72, "23"), (220, "twenty-third"), (360, "thirty"))))),
    # 9 -- a table whose columns do not line up with the previous page
    ((*_row(740, ((100, "Completely"), (330, "Different"))),
      *_row(720, ((100, "x"), (330, "y"))),
      *_row(700, ((100, "z"), (330, "w"))))),
    # 10 -- heading, caption, sub-header row, and a sparse row
    ((72, 730, "5     Encoding Limits"),
     (72, 700, "Table 6: Limits"),
     *_row(675, ((72, "Group"), (220, "Min"), (330, "Max"))),
     *_row(655, ((72, "Timing"),)),
     *_row(635, ((72, "Start"), (220, "0"), (330, "15"))),
     *_row(615, ((72, "Stop"), (330, "31")))),
    # 11 -- a numeric table that is not a bit layout
    ((72, 730, "Table 7: Measurements"),
     *_row(700, ((72, "Sample"), (180, "Low"), (280, "High"), (380, "Mean"))),
     *_row(680, ((72, "A"), (180, "12"), (280, "48"), (380, "30"))),
     *_row(660, ((72, "B"), (180, "15"), (280, "51"), (380, "33"))),
     *_row(640, ((72, "C"), (180, "11"), (280, "44"), (380, "27")))),
)


def geometry_pdf_bytes() -> bytes:
    return positioned_pdf_bytes(GEOMETRY_PAGES)


# A single reviewable bit layout: an eight-bit ruler and two labels that each
# span four of its positions.
SEMANTIC_BITFIELD_PAGES: tuple[tuple[tuple[float, float, str], ...], ...] = (
    ((72, 730, "Table 1: Word Layout"),
     *_row(700, ((72, "31"), (106, "30"), (140, "29"), (174, "28"),
                 (208, "27"), (242, "26"), (276, "25"), (310, "24"))),
     *_row(675, ((72, "FIELD_A_ALPHA_ONEX"), (208, "FIELD_B_BRAVO_TWO"))),
     *_row(650, ((72, "1"), (106, "0"), (140, "1"), (174, "1"),
                 (208, "0"), (242, "0"), (276, "0"), (310, "0")))),
)


def semantic_bitfield_pdf_bytes() -> bytes:
    return positioned_pdf_bytes(SEMANTIC_BITFIELD_PAGES)


# A whole ruler with labels that state their own bit ranges. Each label is
# centred inside the field it names, so its text box is narrower than the range
# it states -- the shape STD2B-R found in a real standard, and the reason STD2C
# believes the text over the measurement. The second row repeats a local range
# in a different word, which is a different place in the packet.
_RULER16 = tuple((72 + (15 - bit) * 30, 700, str(bit)) for bit in range(15, -1, -1))

STATED_RANGE_PAGES: tuple[tuple[tuple[float, float, str], ...], ...] = (
    ((72, 730, "Table 1: Stated Word Layout"),
     *_RULER16,
     (85, 675, "SR_HEADER (15..12)"),
     (250, 675, "SR_COUNT (11..4)"),
     (435, 675, "SR_INDEX (3..0)"),
     (85, 650, "SR_ALPHA (15..12)"),
     (280, 650, "SR_OMEGA (11..0)")),
)


def stated_range_pdf_bytes() -> bytes:
    return positioned_pdf_bytes(STATED_RANGE_PAGES)
