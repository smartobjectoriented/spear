import hashlib
import unittest

from standard_schema import (
    StandardCitation, StandardContentType, StandardDocumentUnit,
    StandardModality, make_source_id, source_content_sha256,
)


class StandardSchemaTests(unittest.TestCase):
    def test_source_id_is_deterministic_content_sensitive_and_index_independent(self):
        arguments = dict(standard_id="TEST", revision="1", page=3,
                         section="3.1", unit_position=7,
                         text="Implementations shall reject 0xF.")
        first = make_source_id(**arguments)
        self.assertEqual(first, make_source_id(**arguments))
        self.assertNotEqual(first, make_source_id(**{**arguments, "text": "changed"}))
        self.assertTrue(first.startswith("std-"))

    def test_unit_round_trip_preserves_page_heading_modality_and_original_text(self):
        text = "Implementations SHALL reject reserved value 0xF."
        unit = StandardDocumentUnit(
            make_source_id(standard_id="TEST", revision="1", page=2,
                           section="3.1", unit_position=1, text=text),
            "TEST", "1", "3.1", 2, ("3 General", "3.1 Reserved"),
            StandardContentType.REQUIREMENT, StandardModality.SHALL, text,
            hashlib.sha256(b"pdf").hexdigest(), source_content_sha256(text), "test-v1",
        )
        self.assertEqual(unit, StandardDocumentUnit.from_dict(unit.to_dict()))

    def test_citation_rendering_is_deterministic(self):
        citation = StandardCitation("TEST", "1", "3.1", 2, "std-" + "a" * 32)
        self.assertEqual(citation.render(),
                         "[TEST 1 §3.1, p.2, source std-" + "a" * 32 + "]")
