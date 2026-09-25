"""Clause numbers that carry a part letter: A2.2.5, D24.2.67.

The Arm A-profile manual numbers every section that way. The extractor read
digits only, so its 17 145 pages came out with fifty "sections" -- "11"
alone held 91 982 units, lines of lists and tables taken for headings.
"""

import unittest

from standard_ingest import (
    _LETTERED, _NUMBERED, _clause_parts, _numbering_scheme, _page_lines,
    canonical_units,
)
from standard_retrieval import _SECTION_QUERY

SHA = "0" * 64


def lettered_document(pages=24):
    """A part title and a one-space running header on every page, a heading
    set with several spaces, body text -- the Arm manual's page shape."""
    # Words, not numbers: furniture is found by repetition with digits
    # blanked out, so text that differs only by a number is furniture.
    words = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet "
             "kilo lima mike november oscar papa quebec romeo sierra tango "
             "uniform victor whiskey xray yankee zulu").split()
    result = []

    for page in range(1, pages + 1):
        section = f"A{1 + page // 12}.{1 + page % 12 // 6}"
        word = words[page % len(words)]
        result.append("\n".join([
            "A-profile Architecture",
            f"{section} Running title of the section",
            "",
            f"{section}.{page % 6 + 1}      Heading about {word}",
            "",
            f"Body text about {word}, which says what the heading covers.",
            f"B2 Reserved          a {word} row that must not become a clause",
            "",
            "ARM DDI 0487      Copyright notice      A-" + str(page),
        ]))

    return tuple(result)


class LetteredNumberingTests(unittest.TestCase):

    def test_the_scheme_is_decided_by_the_document(self):
        self.assertIs(_numbering_scheme(_page_lines(lettered_document())), _LETTERED)
        digits = tuple(f"{page}.1      Heading {page}\n\nBody." for page in range(1, 30))
        self.assertIs(_numbering_scheme(_page_lines(digits)), _NUMBERED)

    def test_lettered_headings_become_sections(self):
        units = canonical_units(lettered_document(), standard_id="X",
                                revision="R", pdf_sha256=SHA)
        sections = {unit.section for unit in units if unit.section}

        self.assertIn("A1.1.2", sections)
        self.assertIn("A2.2.1", sections)
        self.assertTrue(all(section[0] == "A" for section in sections))

    def test_running_headers_and_part_titles_are_furniture_not_headings(self):
        units = canonical_units(lettered_document(), standard_id="X",
                                revision="R", pdf_sha256=SHA)
        running = [unit for unit in units
                   if unit.text.startswith(("A1.1 Running", "A-profile Architecture"))]

        self.assertTrue(running)
        self.assertTrue(all(unit.content_type.value == "PAGE_FURNITURE"
                            for unit in running))
        # A page's running header names its parent section; had it been read
        # as a heading, the body would sit under A1.1 rather than A1.1.N.
        body = [unit for unit in units if unit.text.startswith("Body text about delta,")]
        self.assertEqual(body[0].section, "A1.1.4")

    def test_a_lone_letter_and_number_is_not_a_section(self):
        units = canonical_units(lettered_document(), standard_id="X",
                                revision="R", pdf_sha256=SHA)
        self.assertNotIn("B2", {unit.section for unit in units})

    def test_lettered_clauses_order_and_nest(self):
        self.assertLess(_clause_parts("A2.2.4"), _clause_parts("A2.2.5"))
        self.assertLess(_clause_parts("D24.2"), _clause_parts("E1.1"))
        self.assertEqual(_clause_parts("D24.2.67")[1:], (2, 67))
        self.assertIsNone(_clause_parts("a1.1"))
        self.assertEqual(_clause_parts("7.1.5"), (7, 1, 5))

    def test_a_section_query_may_carry_the_letter(self):
        self.assertEqual(_SECTION_QUERY.match("D24.2.67").group(1), "D24.2.67")
        self.assertEqual(_SECTION_QUERY.match("section 7.1.5").group(1), "7.1.5")


if __name__ == "__main__":
    unittest.main()
