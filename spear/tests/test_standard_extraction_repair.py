"""STD1E: canonical extraction repair for real technical standards.

Every fixture here is synthetic. No licensed normative text appears in this file.
"""

import re
import tempfile
import unittest
from pathlib import Path

from dataclasses import replace

from standard_ingest import (
    EXTRACTOR_VERSION, canonical_units, extract_pdf_pages, verify_against_contents,
)
from standard_schema import (
    StandardContentType, StandardLayoutKind, StandardModality,
)
from standard_store import StandardStore
from tests.standard_extraction_fixture import (
    EXTRACTION_PAGES, classifier_pdf_bytes, contents_target_pdf_bytes,
    extraction_pdf_bytes, positioned_pdf_bytes, wide_column_pdf_bytes,
)


class ExtractionRepairTests(unittest.TestCase):
    """Canonical segmentation and section attribution over a real PDF layout."""

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        pdf = Path(cls.temp.name) / "extraction.pdf"
        pdf.write_bytes(extraction_pdf_bytes())
        cls.pages, cls.warnings = extract_pdf_pages(pdf)
        cls.units = canonical_units(
            cls.pages, standard_id="SYNTH-STD", revision="R1",
            pdf_sha256="d" * 64)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def unit_for(self, phrase):
        """The single body unit carrying this phrase; front matter is excluded."""
        matches = [unit for unit in self.units
                   if phrase in unit.text and not unit.is_front_matter]
        self.assertEqual(len(matches), 1, f"expected exactly one body unit for {phrase!r}")
        return matches[0]

    def front_matter_unit_for(self, phrase):
        matches = [unit for unit in self.units
                   if phrase in unit.text and unit.is_front_matter]
        self.assertEqual(len(matches), 1, f"expected one front-matter unit for {phrase!r}")
        return matches[0]

    def anchors_for(self, section):
        """Units the cross-reference resolver would treat as this clause's anchor."""
        pattern = re.compile(rf"^{re.escape(section)}(?:\s|$)")
        return [unit for unit in self.units
                if unit.section == section and pattern.match(unit.text.strip())]

    # -- headings ---------------------------------------------------------
    def test_top_level_heading_is_recognised(self):
        heading = self.unit_for("1 Introduction")
        self.assertEqual(heading.section, "1")
        self.assertEqual(heading.layout_kind, StandardLayoutKind.HEADING)

    def test_dotted_subheading_is_recognised(self):
        self.assertEqual(self.unit_for("1.1 Scope").section, "1.1")
        self.assertEqual(self.unit_for("2.3.1 Field Rules").section, "2.3.1")

    def test_binary_table_row_is_not_a_heading(self):
        row = self.unit_for("00000100")
        self.assertNotEqual(row.section, "00000100")
        self.assertEqual(row.section, "2")
        self.assertNotEqual(row.layout_kind, StandardLayoutKind.HEADING)

    def test_leading_zero_value_is_not_a_heading(self):
        row = self.unit_for("Nominal")
        self.assertNotEqual(row.section, "0001")
        self.assertEqual(row.section, "2")

    def test_bare_large_integer_is_not_a_heading(self):
        for phrase in ("65536", "255 "):
            self.assertNotIn(self.unit_for(phrase).section, {"65536", "255"})

    def test_decimal_measurement_is_not_a_heading(self):
        measurement = self.unit_for("102.4 MHz")
        self.assertNotEqual(measurement.section, "102.4")
        self.assertEqual(measurement.section, "1.1")

    def test_numeric_body_sentence_is_not_a_heading(self):
        for phrase in ("2015 devices", "187 octets"):
            self.assertEqual(self.unit_for(phrase).section, "1.1")

    def test_numbered_list_items_are_not_headings(self):
        for phrase in ("apples are counted", "oranges are counted",
                       "pears are counted"):
            item = self.unit_for(phrase)
            self.assertEqual(item.section, "3")
            self.assertNotEqual(item.layout_kind, StandardLayoutKind.HEADING)

    # -- table of contents ------------------------------------------------
    def test_dot_leader_line_is_recognised_as_a_toc_entry(self):
        entry = self.front_matter_unit_for("1 Introduction .")
        self.assertTrue(entry.is_toc_entry)
        self.assertTrue(entry.is_front_matter)
        self.assertEqual(entry.content_type, StandardContentType.FRONT_MATTER)

    def test_toc_entry_does_not_set_the_current_section(self):
        for entry in (unit for unit in self.units if unit.is_toc_entry):
            self.assertIsNone(entry.section)

    def test_toc_entry_does_not_duplicate_a_body_section_anchor(self):
        for section in ("1", "1.1", "2", "2.3.1"):
            self.assertEqual(len(self.anchors_for(section)), 1, section)

    def test_genuine_body_heading_remains_a_section_anchor(self):
        anchor = self.anchors_for("2.3.1")[0]
        self.assertEqual(anchor.page, 5)
        self.assertFalse(anchor.is_front_matter)

    # -- page furniture ---------------------------------------------------
    def test_repeated_page_furniture_is_detected_on_every_page(self):
        furniture = [unit for unit in self.units
                     if unit.content_type == StandardContentType.PAGE_FURNITURE]
        self.assertEqual(len(furniture), len(EXTRACTION_PAGES))
        self.assertEqual({unit.page for unit in furniture},
                         set(range(1, len(EXTRACTION_PAGES) + 1)))

    def test_page_furniture_is_excluded_from_retrieval_but_kept_in_the_corpus(self):
        furniture = [unit for unit in self.units
                     if unit.content_type == StandardContentType.PAGE_FURNITURE]
        self.assertTrue(furniture)
        for unit in furniture:
            self.assertFalse(unit.retrievable)
            self.assertIsNone(unit.section)
            self.assertEqual(unit.layout_kind, StandardLayoutKind.FURNITURE)

    def test_ordinary_units_stay_retrievable(self):
        self.assertTrue(self.unit_for("1.1 Scope").retrievable)
        self.assertTrue(self.unit_for("shall reject a reserved encoding").retrievable)

    # -- columnar / table evidence ----------------------------------------
    def test_columnar_rows_are_flagged_for_structured_review(self):
        for phrase in ("00000100", "1111", "65536", "Continued"):
            row = self.unit_for(phrase)
            self.assertEqual(row.layout_kind, StandardLayoutKind.COLUMNAR, phrase)
            self.assertTrue(row.needs_structured_review, phrase)

    def test_ordinary_indented_paragraph_is_not_table_flagged(self):
        paragraph = self.unit_for("ordinary indented paragraph")
        self.assertEqual(paragraph.layout_kind, StandardLayoutKind.PROSE)
        self.assertFalse(paragraph.needs_structured_review)

    def test_cross_page_table_continuation_is_recorded(self):
        self.assertTrue(self.unit_for("Continued").possible_table_continuation)
        self.assertFalse(self.unit_for("00000100").possible_table_continuation)

    # -- annex ------------------------------------------------------------
    def test_annex_heading_does_not_corrupt_numeric_hierarchy(self):
        annex = self.unit_for("Annex A Informative Material")
        self.assertNotEqual(annex.layout_kind, StandardLayoutKind.COLUMNAR)
        self.assertEqual(self.unit_for("4 Encoding Notes").section, "4")

    # -- provenance -------------------------------------------------------
    def test_extractor_version_was_bumped(self):
        self.assertEqual(EXTRACTOR_VERSION, "poppler-structure-v2")
        self.assertTrue(all(unit.extractor_version == EXTRACTOR_VERSION
                            for unit in self.units))

    def test_identical_input_re_extracts_deterministically(self):
        again = canonical_units(self.pages, standard_id="SYNTH-STD", revision="R1",
                                pdf_sha256="d" * 64)
        self.assertEqual([unit.to_dict() for unit in again],
                         [unit.to_dict() for unit in self.units])

    def test_hierarchy_runs_forward_over_the_body(self):
        ordered = [unit.section for unit in self.units
                   if unit.section and unit.layout_kind == StandardLayoutKind.HEADING]
        self.assertEqual(ordered, ["1", "1.1", "2", "2.3.1", "3", "4"])


class ExtractionEdgeCaseTests(unittest.TestCase):
    """Cases the classifier must handle without a document-specific rule."""

    def units_for(self, pages):
        with tempfile.TemporaryDirectory() as name:
            pdf = Path(name) / "case.pdf"
            pdf.write_bytes(positioned_pdf_bytes(pages))
            extracted, _ = extract_pdf_pages(pdf)
            return canonical_units(extracted, standard_id="S", revision="R",
                                   pdf_sha256="e" * 64)

    def test_a_document_without_front_matter_keeps_every_page(self):
        units = self.units_for((
            ((72, 720, "1 Scope"), (72, 690, "This clause is normative.")),
        ))
        self.assertTrue(any(unit.section == "1" for unit in units))
        self.assertFalse(any(unit.is_front_matter for unit in units))

    def test_a_short_document_does_not_invent_page_furniture(self):
        units = self.units_for((
            ((72, 720, "1 Scope"), (72, 40, "unique footer one")),
            ((72, 720, "2 Terms"), (72, 40, "different footer two")),
        ))
        self.assertFalse(any(
            unit.content_type == StandardContentType.PAGE_FURNITURE for unit in units))

    def test_hierarchy_resync_after_a_gap_is_allowed_with_layout_evidence(self):
        units = self.units_for((
            ((72, 720, "7 Transport"), (72, 690, "Body text for transport."),
             (72, 640, "9 Timing"), (72, 610, "Body text for timing.")),
        ))
        self.assertEqual([unit.section for unit in units
                          if unit.layout_kind == StandardLayoutKind.HEADING],
                         ["7", "9"])


if __name__ == "__main__":
    unittest.main()


class ClauseUniquenessTests(unittest.TestCase):
    """A clause number opens one clause; a second claim is a column, not a heading."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        pdf = self.root / "repeat.pdf"
        pdf.write_bytes(positioned_pdf_bytes((
            ((72, 720, "7 Transport"),
             (72, 680, "7.1 First Anchor"),
             (72, 640, "Body text that belongs to the first clause."),
             (72, 580, "7.1 Second Anchor"),
             (72, 540, "Body text that follows the repeated number."),
             (72, 480, "8 Reference"),
             (72, 440, "See Section 7.1 for the normative rule.")),)))
        self.pdf = pdf
        pages, _ = extract_pdf_pages(pdf)
        self.units = canonical_units(pages, standard_id="AMB", revision="R1",
                                     pdf_sha256="f" * 64)

    def tearDown(self):
        self.temp.cleanup()

    def test_a_repeated_clause_number_opens_only_the_first_clause(self):
        headings = [unit for unit in self.units
                    if unit.layout_kind == StandardLayoutKind.HEADING]
        self.assertEqual([unit.section for unit in headings], ["7", "7.1", "8"])
        demoted = next(unit for unit in self.units if "Second Anchor" in unit.text)
        self.assertNotEqual(demoted.layout_kind, StandardLayoutKind.HEADING)
        self.assertIn("clause number was already opened earlier", demoted.warnings)

    def test_the_surviving_anchor_makes_the_reference_resolvable(self):
        from standard_crossrefs import rebuild_cross_reference_index
        from standard_ingest import build_manifest, ingest_pdf
        from standard_store import StandardStore

        store = StandardStore(self.root / "store")
        ingest_pdf(store, self.pdf, standard_id="AMB", revision="R1",
                   source_origin="TEST_FIXTURE")
        index = rebuild_cross_reference_index(store, "AMB", "R1", created_at="x")
        self.assertEqual(index.ambiguous_count, 0)
        self.assertEqual(index.resolved_count, 1)


class ColumnSetNumberTests(unittest.TestCase):
    """A number set under a column header is a cell, whatever it would continue."""

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        pdf = Path(cls.temp.name) / "wide.pdf"
        pdf.write_bytes(wide_column_pdf_bytes())
        pages, _ = extract_pdf_pages(pdf)
        cls.units = canonical_units(pages, standard_id="WIDE", revision="R1",
                                    pdf_sha256="c" * 64)
        cls.headings = [unit for unit in cls.units
                        if unit.layout_kind == StandardLayoutKind.HEADING]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_a_wide_set_row_is_not_a_heading_even_when_it_would_continue(self):
        # "3" on page 2 follows clause 2 and would continue the tree perfectly;
        # only its distance from its title tells us it is a cell.
        borrowed = {"3", "4", "5"}
        opened_on_the_table_page = {unit.section for unit in self.headings
                                    if unit.page == 2}
        self.assertEqual(opened_on_the_table_page & borrowed, set())
        self.assertEqual(opened_on_the_table_page, {"2"})
        for value in borrowed:
            row = next(unit for unit in self.units if unit.page == 2
                       and unit.text.strip().startswith(value))
            self.assertNotEqual(row.layout_kind, StandardLayoutKind.HEADING, value)

    def test_the_clause_the_row_borrowed_still_opens_at_its_real_heading(self):
        opened = [unit for unit in self.headings if unit.section == "3"]
        self.assertEqual(len(opened), 1)
        self.assertEqual(opened[0].page, 3)

    def test_the_heading_gap_limit_is_measured_from_the_document(self):
        from standard_ingest import _heading_gap_limit, _page_lines
        narrow = _heading_gap_limit(_page_lines(("1 A\n2 B\n",)))
        wide = _heading_gap_limit(_page_lines(("x" * 400 + "\n" + "y" * 400 + "\n",)))
        self.assertGreater(wide, narrow)
        self.assertGreaterEqual(narrow, 12)

    def test_an_annex_renumbering_does_not_displace_a_body_clause(self):
        opened = [unit for unit in self.headings if unit.section == "1.1"]
        self.assertEqual(len(opened), 1)
        self.assertEqual(opened[0].page, 1)
        borrowed = next(unit for unit in self.units
                        if unit.page == 4 and unit.text.strip().startswith("1.1"))
        self.assertNotEqual(borrowed.layout_kind, StandardLayoutKind.HEADING)
        self.assertIn("clause number was already opened earlier", borrowed.warnings)

    def test_an_annex_stops_filing_its_content_under_the_last_body_clause(self):
        annex = next(unit for unit in self.units if "Annex A" in unit.text)
        self.assertIn("annex or appendix heading resets clause numbering",
                      annex.warnings)
        for unit in self.units:
            if unit.page == 4:
                self.assertIsNone(unit.section, unit.text[:40])


class ContentsPageTargetTests(unittest.TestCase):
    """The contents table names each clause's page; that outranks any heuristic."""

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        pdf = Path(cls.temp.name) / "contents.pdf"
        pdf.write_bytes(contents_target_pdf_bytes())
        pages, _ = extract_pdf_pages(pdf)
        cls.units = canonical_units(pages, standard_id="TOC", revision="R1",
                                    pdf_sha256="a" * 64)
        cls.headings = {unit.section: unit for unit in cls.units
                        if unit.layout_kind == StandardLayoutKind.HEADING}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_a_contested_clause_opens_on_the_page_the_contents_names(self):
        # Page 3's "7" continues 6 perfectly and comes first; the real heading on
        # page 5 does not continue anything. Only the contents table settles it.
        self.assertEqual(self.headings["7"].page, 5)
        borrowed = next(unit for unit in self.units if unit.page == 3
                        and unit.text.strip().startswith("7"))
        self.assertNotEqual(borrowed.layout_kind, StandardLayoutKind.HEADING)
        self.assertIn("clause number was already opened earlier", borrowed.warnings)

    def test_uncontested_clauses_are_unaffected(self):
        self.assertEqual({key: unit.page for key, unit in self.headings.items()},
                         {"1": 2, "2": 3, "3": 3, "6": 3, "7": 5})

    def test_the_printed_page_offset_is_measured_not_assumed(self):
        report = verify_against_contents(self.units)
        self.assertEqual(report["printed_page_offset"], 1)
        self.assertEqual(report["clauses_in_both"], 5)
        self.assertEqual(report["page_agreement"], 5)
        self.assertEqual(report["page_disagreement"], 0)
        self.assertEqual(report["disagreements"], [])
        self.assertEqual(report["agreement_rate"], 1.0)
        self.assertEqual(report["advertised_not_detected"], [])

    def test_the_check_reports_a_heading_on_the_wrong_page(self):
        moved = tuple(
            replace(unit, page=unit.page + 3,
                    source_content_sha256=unit.source_content_sha256)
            if unit.section == "6" and unit.layout_kind == StandardLayoutKind.HEADING
            else unit
            for unit in self.units)
        report = verify_against_contents(moved)
        self.assertEqual(report["page_disagreement"], 1)
        self.assertEqual(report["disagreements"][0]["clause"], "6")
        self.assertEqual(report["disagreements"][0]["delta"], 3)

    def test_a_document_without_a_contents_table_still_extracts(self):
        with tempfile.TemporaryDirectory() as name:
            pdf = Path(name) / "plain.pdf"
            pdf.write_bytes(positioned_pdf_bytes((
                ((72, 720, "1     Scope"), (72, 690, "Body text for the clause.")),)))
            pages, _ = extract_pdf_pages(pdf)
            units = canonical_units(pages, standard_id="P", revision="R1",
                                    pdf_sha256="b" * 64)
        report = verify_against_contents(units)
        self.assertEqual(report["clauses_in_both"], 0)
        self.assertIsNone(report["printed_page_offset"])
        self.assertIsNone(report["agreement_rate"])
        self.assertTrue(any(unit.section == "1" for unit in units))


class ContentTypeTests(unittest.TestCase):
    """A caption is not a sentence about a table, and "a means of" is not a definition."""

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        pdf = Path(cls.temp.name) / "classify.pdf"
        pdf.write_bytes(classifier_pdf_bytes())
        pages, _ = extract_pdf_pages(pdf)
        cls.units = canonical_units(pages, standard_id="CLS", revision="R1",
                                    pdf_sha256="e" * 64)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def unit_for(self, phrase):
        matches = [unit for unit in self.units if phrase in unit.text]
        self.assertEqual(len(matches), 1, f"expected one unit for {phrase!r}")
        return matches[0]

    def test_a_caption_is_typed_but_a_sentence_about_it_is_not(self):
        self.assertEqual(self.unit_for("Table 4-1: Reserved").content_type,
                         StandardContentType.TABLE)
        self.assertEqual(self.unit_for("Figure 4-2: Packet").content_type,
                         StandardContentType.FIGURE)
        for phrase in ("lists the reserved encodings", "shows how the packet"):
            reference = self.unit_for(phrase)
            self.assertNotIn(reference.content_type,
                             {StandardContentType.TABLE, StandardContentType.FIGURE})
            self.assertFalse(reference.needs_structured_review, phrase)

    def test_means_is_a_definition_only_as_a_verb(self):
        self.assertEqual(self.unit_for("value unavailable for ordinary").content_type,
                         StandardContentType.DEFINITION)
        self.assertNotEqual(self.unit_for("standard means of conveying").content_type,
                            StandardContentType.DEFINITION)

    def test_a_labelled_paragraph_takes_the_type_the_document_gives_it(self):
        self.assertEqual(self.unit_for("A Frame is a bounded").content_type,
                         StandardContentType.DEFINITION)
        self.assertEqual(self.unit_for("Rule 4-2").content_type,
                         StandardContentType.REQUIREMENT)
        self.assertEqual(self.unit_for("Permission 4-4").content_type,
                         StandardContentType.INFORMATIVE)
        self.assertEqual(self.unit_for("Recommendation 4-5").content_type,
                         StandardContentType.RECOMMENDATION)
        # Labelled an observation, so it stays one even though it says "shall".
        observation = self.unit_for("Observation 4-3")
        self.assertEqual(observation.content_type, StandardContentType.INFORMATIVE)
        self.assertEqual(observation.modality, StandardModality.SHALL)

    def test_a_quoted_keyword_is_mentioned_not_imposed(self):
        quoted = self.unit_for("are reserved for stating rules")
        self.assertEqual(quoted.modality, StandardModality.NONE)
        self.assertNotEqual(quoted.content_type, StandardContentType.REQUIREMENT)

    def test_an_unlabelled_normative_sentence_is_still_a_requirement(self):
        rule = self.unit_for("Rule 4-2")
        self.assertEqual(rule.modality, StandardModality.SHALL)
