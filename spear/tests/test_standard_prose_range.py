"""STD2F: a rule may position a field the diagram names but never places.

Every fixture here is invented. No licensed normative text appears in this file.
The bar for linking is deliberately high, so most of these tests are refusals.
"""

import unittest
from dataclasses import replace

from standard_prose_range import (
    COMPETING_NORMATIVE_RANGES, FIELD_HAS_LOCAL_RANGE, MULTIPLE_NAMES_IN_STATEMENT,
    MULTIPLE_RANGES_IN_STATEMENT, NAME_AMBIGUOUS, NAME_MATCHES_NO_FIELD,
    NAME_TOO_SHORT, NOT_NORMATIVE, NO_QUOTED_FIELD_NAME,
    PROSE_RANGE_WORD_UNRESOLVED, RANGE_EXCEEDS_WORD_WIDTH, SECTION_MISMATCH,
    LinkStatus, MatchMethod, ProseFieldTarget, normalize_field_name,
    normative_range_links, normative_statements, ordinal_word,
)
from standard_semantic import (
    PositionSource, SemanticRole, SpanRole, StandardBitfieldApproval,
    StructuralCompleteness, normative_links_for, promote_bitfield,
)
from standard_prose_range import StandardReviewedLink
from standard_word_association import field_candidates, unpositioned_labels
from tests.standard_word_fixture import (
    FINGERPRINT, LAYOUT, REV, SID, SOURCE, bitfield, ruler_row, table, word_row,
)

OPERATOR = "test-operator"


class _Unit:
    """The shape build_semantics sees: a canonical unit with a modality."""

    class _Kind:
        def __init__(self, value): self.value = value

    def __init__(self, text, *, source_id="std-rule0001", section="1.2",
                 modality="REQUIREMENT", page=1):
        self.text = text
        self.source_id = source_id
        self.section = section
        self.content_type = self._Kind(modality)
        self.page = page


def target(label, word=0, cell="cel-target", sources=(SOURCE,)):
    return ProseFieldTarget(cell_id=cell, label=label, word_index=word,
                            source_ids=sources, provenance="DIRECT_TEXT_MATCH")


def link_for(statement, targets=None, *, section="1.2", modality="REQUIREMENT",
             sections=frozenset({"1.2"}), words=(0, 1), source_id="std-rule0001"):
    targets = targets if targets is not None else [target("Slot Depth", word=1)]
    return normative_range_links(
        targets, [_Unit(statement, section=section, modality=modality,
                        source_id=source_id)],
        candidate_id="bit-0123456789abcdef", sections=sections,
        diagram_words=words)


RULE = ('Rule 1.2-5: The “Slot Depth” subfield of the second header word, '
        'bits 31-28 shall state the depth of each slot.')


class LinkingTests(unittest.TestCase):
    def test_a_named_subfield_in_a_rule_positions_a_visible_field(self):
        linked, refused = link_for(RULE)
        self.assertEqual(len(linked), 1)
        item = linked[0]
        self.assertEqual((item.high_bit, item.low_bit, item.width), (31, 28, 4))
        self.assertEqual(item.status, LinkStatus.LINKED)
        self.assertEqual(item.match_method, MatchMethod.EXACT_NORMALIZED_NAME)
        self.assertEqual(item.target_label, "Slot Depth")
        self.assertEqual(item.requirement_source_id, "std-rule0001")
        self.assertTrue(item.link_fingerprint.startswith("lnk-"))
        self.assertEqual(item.warnings, ())

    def test_the_link_names_the_word_the_diagram_gave_the_field(self):
        linked, _ = link_for(RULE, [target("Slot Depth", word=1)])
        self.assertEqual(len(linked), 1)
        self.assertEqual(linked[0].word_index, 1)
        self.assertEqual(linked[0].ordinal_word_index, 1)

    def test_a_rule_naming_a_different_word_than_the_diagram_is_refused(self):
        linked, refused = link_for(RULE, [target("Slot Depth", word=0)])
        self.assertEqual(linked, ())
        self.assertIn(PROSE_RANGE_WORD_UNRESOLVED, refused[0].warnings)

    def test_a_rule_that_names_no_word_is_refused(self):
        statement = ('Rule 1.2-5: The “Slot Depth” subfield, bits 31-28 '
                     'shall state the depth of each slot.')
        linked, refused = link_for(statement)
        self.assertEqual(linked, ())
        self.assertIn(PROSE_RANGE_WORD_UNRESOLVED, refused[0].warnings)

    def test_the_ordinal_is_read_in_the_diagrams_own_numbering(self):
        self.assertEqual(ordinal_word("the second header word", [1, 2, 3]), 2)
        self.assertEqual(ordinal_word("the second header word", [0, 1, 2]), 1)
        self.assertEqual(ordinal_word("the first word", [0, 1]), 0)
        self.assertIsNone(ordinal_word("no ordinal here", [0, 1]))
        self.assertIsNone(ordinal_word("the eighth word", [0, 1]))


class RefusalTests(unittest.TestCase):
    """Everything the corpus scan showed a prose range is usually about."""

    def refuse(self, statement, targets=None, **kwargs):
        linked, refused = link_for(statement, targets, **kwargs)
        self.assertEqual(linked, (), f"{statement[:60]!r} was linked")
        return refused

    def test_a_radix_point_sentence_never_positions_a_field(self):
        refused = self.refuse(
            'Rule 1.2-9: This value has an integer and a fractional part, the '
            'binary point falling after bit 22 of the second word.')
        self.assertIn(NO_QUOTED_FIELD_NAME, refused[0].warnings)

    def test_a_value_scale_sentence_is_refused(self):
        refused = self.refuse(
            'Rule 1.2-8: The scale of the second word shall use bits 3-0 to '
            'encode the value range of the measurement.')
        self.assertIn(NO_QUOTED_FIELD_NAME, refused[0].warnings)

    def test_a_generic_sentence_mentioning_bits_is_refused(self):
        refused = self.refuse(
            'Rule 1.2-7: Implementations shall ignore bits 15-8 of the second '
            'word when the feature is disabled.')
        self.assertIn(NO_QUOTED_FIELD_NAME, refused[0].warnings)

    def test_a_rule_naming_a_field_the_diagram_does_not_show_is_refused(self):
        statement = ('Rule 1.2-5: The “Something Else” subfield of the '
                     'second header word, bits 31-28 shall state the depth.')
        refused = self.refuse(statement)
        self.assertIn(NAME_MATCHES_NO_FIELD, refused[0].warnings)

    def test_a_name_matching_two_visible_fields_is_refused(self):
        refused = self.refuse(RULE, [target("Slot Depth", word=1, cell="cel-a"),
                                     target("slot  depth", word=1, cell="cel-b")])
        self.assertIn(NAME_AMBIGUOUS, refused[0].warnings)

    def test_a_one_letter_name_is_too_short_to_match_on(self):
        statement = ('Rule 1.2-5: The “C” subfield of the second header '
                     'word, bits 31-28 shall be set.')
        refused = self.refuse(statement, [target("C", word=1)])
        self.assertIn(NAME_TOO_SHORT, refused[0].warnings)

    def test_a_statement_with_two_ranges_is_refused(self):
        statement = ('Rule 1.2-5: The “Slot Depth” subfield of the second '
                     'header word, bits 31-28 and bits 27-24 shall be set.')
        refused = self.refuse(statement)
        self.assertIn(MULTIPLE_RANGES_IN_STATEMENT, refused[0].warnings)

    def test_a_statement_naming_two_subfields_is_refused(self):
        statement = ('Rule 1.2-5: The “Slot Depth” and “Other” '
                     'subfields of the second header word, bits 31-28 apply.')
        refused = self.refuse(statement)
        self.assertIn(MULTIPLE_NAMES_IN_STATEMENT, refused[0].warnings)

    def test_a_statement_from_another_section_is_refused(self):
        refused = self.refuse(RULE, section="9.9")
        self.assertIn(SECTION_MISMATCH, refused[0].warnings)

    def test_page_proximity_alone_proves_nothing(self):
        # Same page, different section, no section match: still refused.
        refused = self.refuse(RULE, section="4.4")
        self.assertIn(SECTION_MISMATCH, refused[0].warnings)

    def test_a_statement_that_is_not_a_requirement_is_refused(self):
        refused = self.refuse(RULE, modality="RECOMMENDATION")
        self.assertIn(NOT_NORMATIVE, refused[0].warnings)
        refused = self.refuse(RULE, modality="INFORMATIVE")
        self.assertIn(NOT_NORMATIVE, refused[0].warnings)

    def test_a_range_wider_than_the_word_is_refused(self):
        statement = ('Rule 1.2-5: The “Slot Depth” subfield of the second '
                     'header word, bits 47-40 shall state the depth.')
        linked, refused = link_for(statement)
        self.assertEqual(linked, ())

    def test_two_rules_giving_the_same_field_different_bits_are_refused(self):
        other = ('Rule 1.2-6: The “Slot Depth” subfield of the second '
                 'header word, bits 27-24 shall state the depth.')
        linked, refused = normative_range_links(
            [target("Slot Depth", word=1)],
            [_Unit(RULE, source_id="std-rule0001"),
             _Unit(other, source_id="std-rule0002")],
            candidate_id="bit-0123456789abcdef", sections=frozenset({"1.2"}),
            diagram_words=(0, 1))
        self.assertEqual(linked, ())
        self.assertTrue(any(COMPETING_NORMATIVE_RANGES in item.warnings
                            for item in refused))

    def test_two_rules_agreeing_are_not_a_conflict(self):
        same = RULE.replace("Rule 1.2-5", "Rule 1.2-6")
        linked, _ = normative_range_links(
            [target("Slot Depth", word=1)],
            [_Unit(RULE, source_id="std-rule0001"),
             _Unit(same, source_id="std-rule0002")],
            candidate_id="bit-0123456789abcdef", sections=frozenset({"1.2"}),
            diagram_words=(0, 1))
        self.assertEqual(len(linked), 1)
        self.assertEqual((linked[0].high_bit, linked[0].low_bit), (31, 28))


class NormalizationTests(unittest.TestCase):
    def test_names_match_on_case_spacing_and_the_field_noun_only(self):
        self.assertEqual(normalize_field_name("Slot Depth"), "slot depth")
        self.assertEqual(normalize_field_name("  SLOT   depth "), "slot depth")
        self.assertEqual(normalize_field_name("Slot Depth subfield"), "slot depth")
        self.assertEqual(normalize_field_name("Slot-Depth field"), "slot depth")

    def test_different_names_do_not_match(self):
        self.assertNotEqual(normalize_field_name("number of entries"),
                            normalize_field_name("# entries"))
        self.assertNotEqual(normalize_field_name("Slot Depth"),
                            normalize_field_name("Entry Count"))

    def test_statements_are_split_so_a_range_cannot_cross_a_rule(self):
        text = ("Rule 1.2-4: The “a” subfield, bits 19-0. "
                "Rule 1.2-5: The “b” subfield, bits 31-28.")
        parts = normative_statements(text)
        self.assertEqual(len(parts), 2)
        self.assertIn("19-0", parts[0][1])
        self.assertNotIn("31-28", parts[0][1])


class PromotionTests(unittest.TestCase):
    """The linked range becomes a position, but never outranks the diagram."""

    def build(self, labels, statement=RULE, roles=None, extra_units=(),
              accept=True, link_role="FIELD", review=True):
        """Roles cover the diagram's own candidates; links are reviewed apart."""
        source = table((ruler_row(), word_row(1, ((0, labels[0]),
                                                  (1, labels[1])))))
        mark = bitfield(source)
        units = [_Unit(statement)] + list(extra_units)
        links = normative_links_for(mark, source, units)
        originals = field_candidates(mark.to_dict(), source.to_dict())
        reviewed = tuple(StandardReviewedLink.of(item, accepted=accept,
                                                 role=link_role)
                         for item in links.values()) if review else ()
        approval = StandardBitfieldApproval(
            candidate_id=mark.bitfield_id, verdict="PASS", reviewer=OPERATOR,
            reviewed_at="2026-01-01T00:00:00+00:00",
            structure_fingerprint=FINGERPRINT,
            span_roles=tuple(roles if roles is not None
                             else ["FIELD"] * len(originals)),
            reviewed_links=reviewed)
        return originals, links, promote_bitfield(
            mark, source, approval, standard_id=SID, revision=REV,
            corpus_fingerprint=FINGERPRINT, layout_fingerprint=LAYOUT,
            structure_fingerprint=FINGERPRINT, prose_links=links)

    def test_a_linked_field_is_promoted_with_its_own_position_source(self):
        entries, links, (definition, blocked) = self.build(
            (("FIELD_A (15-0)",), ("Slot Depth", "FIELD_B (27-0)")))
        self.assertIsNone(blocked)
        self.assertEqual(len(links), 1)
        field = next(f for f in definition.fields if f.display_label == "Slot Depth")
        self.assertEqual((field.msb, field.lsb, field.width), (31, 28, 4))
        self.assertEqual(field.position_source, PositionSource.NORMATIVE_PROSE_RANGE)
        self.assertEqual(field.normative_source_id, "std-rule0001")
        self.assertTrue(field.normative_link_fingerprint)

    def test_a_field_the_diagram_positions_is_never_linked(self):
        entries, links, (definition, blocked) = self.build(
            (("FIELD_A (15-0)",), ("Slot Depth (23-20)", "FIELD_B (19-0)")))
        self.assertEqual(links, {})
        field = next(f for f in definition.fields
                     if f.display_label == "Slot Depth")
        self.assertEqual((field.msb, field.lsb), (23, 20))
        self.assertEqual(field.position_source, PositionSource.STATED_RANGE)

    def test_a_diagram_and_a_rule_that_disagree_block_the_candidate(self):
        # The label states its own bits; the rule states different ones.
        source = table((ruler_row(),
                        word_row(1, ((1, ("Slot Depth (23-20)",)),))))
        mark = bitfield(source)
        cell = next(c for r in source.rows for c in r.cells
                    if c.text.startswith("Slot Depth"))
        from standard_prose_range import StandardNormativeRangeLink, MatchMethod
        from standard_prose_range import LinkStatus as LS
        forced = {cell.cell_id: StandardNormativeRangeLink(
            candidate_id=mark.bitfield_id, target_cell_id=cell.cell_id,
            target_label=cell.text, normalized_name="slot depth",
            quoted_name="Slot Depth", requirement_source_id="std-rule0001",
            statement_index=0, range_text="bits 31-28", high_bit=31, low_bit=28,
            width=4, word_index=1, ordinal_word_index=1,
            match_method=MatchMethod.EXACT_NORMALIZED_NAME, status=LS.LINKED,
            link_fingerprint="lnk-0000000000000000")}
        # The reviewer accepted this link; the conflict is with the diagram,
        # not with the review, so the review gate must let it through to it.
        approval = StandardBitfieldApproval(
            candidate_id=mark.bitfield_id, verdict="PASS", reviewer=OPERATOR,
            reviewed_at="2026-01-01T00:00:00+00:00",
            structure_fingerprint=FINGERPRINT, span_roles=("FIELD",),
            reviewed_links=(StandardReviewedLink.of(
                forced[cell.cell_id], accepted=True),))
        definition, blocked = promote_bitfield(
            mark, source, approval, standard_id=SID, revision=REV,
            corpus_fingerprint=FINGERPRINT, layout_fingerprint=LAYOUT,
            structure_fingerprint=FINGERPRINT, prose_links=forced)
        self.assertIsNone(definition)
        self.assertEqual(blocked["reason"], "PROSE_RANGE_CONFLICTS_LOCAL")

    def test_a_reserved_label_stays_reserved_when_prose_positions_it(self):
        statement = ('Rule 1.2-5: The “Reserved” subfield of the second '
                     'header word, bits 31-28 shall be zero.')
        entries, links, (definition, blocked) = self.build(
            (("FIELD_A (15-0)",), ("Reserved", "FIELD_B (27-0)")),
            statement=statement, link_role="RESERVED")
        self.assertIsNone(blocked)
        field = next(f for f in definition.fields if f.display_label == "Reserved")
        self.assertEqual(field.semantic_role, SemanticRole.RESERVED)
        self.assertEqual(field.position_source, PositionSource.NORMATIVE_PROSE_RANGE)

    def test_the_linked_field_keeps_the_canonical_source_of_its_own_cell(self):
        entries, links, (definition, _) = self.build(
            (("FIELD_A (15-0)",), ("Slot Depth", "FIELD_B (27-0)")))
        field = next(f for f in definition.fields if f.display_label == "Slot Depth")
        self.assertEqual(field.supporting_source_ids, (SOURCE,))
        self.assertNotIn("std-rule0001", field.supporting_source_ids)
        self.assertEqual(field.normative_source_id, "std-rule0001")


class StructuralCompletenessTests(unittest.TestCase):
    def word_with_orphan(self):
        return table((ruler_row(),
                      word_row(1, ((1, ("Slot Depth", "FIELD_B (27-0)")),))))

    def promote(self, source, links=None, roles=None, accept=True):
        mark = bitfield(source)
        links = links if links is not None else {}
        originals = field_candidates(mark.to_dict(), source.to_dict())
        approval = StandardBitfieldApproval(
            candidate_id=mark.bitfield_id, verdict="PASS", reviewer=OPERATOR,
            reviewed_at="2026-01-01T00:00:00+00:00",
            structure_fingerprint=FINGERPRINT,
            span_roles=tuple(roles if roles is not None
                             else ["FIELD"] * len(originals)),
            reviewed_links=tuple(
                StandardReviewedLink.of(item, accepted=accept)
                for item in links.values()))
        return promote_bitfield(mark, source, approval, standard_id=SID,
                                revision=REV, corpus_fingerprint=FINGERPRINT,
                                layout_fingerprint=LAYOUT,
                                structure_fingerprint=FINGERPRINT,
                                prose_links=links)

    def test_a_named_but_unpositioned_label_makes_a_word_structurally_incomplete(self):
        definition, blocked = self.promote(self.word_with_orphan())
        self.assertIsNone(blocked)
        self.assertEqual(definition.completeness.value, "COMPLETE")
        self.assertEqual(definition.structural_completeness,
                         StructuralCompleteness.STRUCTURALLY_INCOMPLETE)
        self.assertTrue(any("never positions" in item
                            for item in definition.warnings))

    def test_linking_that_label_makes_the_word_structurally_complete(self):
        source = self.word_with_orphan()
        mark = bitfield(source)
        links = normative_links_for(mark, source, [_Unit(RULE)])
        self.assertEqual(len(links), 1)
        definition, blocked = self.promote(source, links)
        self.assertIsNone(blocked)
        self.assertEqual(definition.structural_completeness,
                         StructuralCompleteness.STRUCTURALLY_COMPLETE)

    def test_a_gap_with_no_named_label_is_a_legitimate_gap(self):
        source = table((ruler_row(),
                        word_row(1, ((1, ("FIELD_B (27-0)",)),))))
        definition, blocked = self.promote(source)
        self.assertIsNone(blocked)
        self.assertEqual(definition.structural_completeness,
                         StructuralCompleteness.STRUCTURALLY_COMPLETE)

    def test_no_reserved_field_is_ever_synthesized_to_fill_a_gap(self):
        definition, _ = self.promote(self.word_with_orphan())
        self.assertEqual([f.display_label for f in definition.fields], ["FIELD_B"])
        self.assertNotIn(SemanticRole.RESERVED,
                         [f.semantic_role for f in definition.fields])

    def test_the_completeness_flags_survive_serialization(self):
        definition, _ = self.promote(self.word_with_orphan())
        value = definition.to_dict()
        self.assertEqual(value["completeness"], "COMPLETE")
        self.assertEqual(value["structural_completeness"],
                         "STRUCTURALLY_INCOMPLETE")


class DeterminismTests(unittest.TestCase):
    def test_a_link_fingerprint_is_reproducible(self):
        first, _ = link_for(RULE)
        second, _ = link_for(RULE)
        self.assertEqual(first[0].link_fingerprint, second[0].link_fingerprint)
        self.assertEqual(first[0].to_dict(), second[0].to_dict())

    def test_a_different_range_gives_a_different_link_fingerprint(self):
        first, _ = link_for(RULE)
        other = RULE.replace("bits 31-28", "bits 27-24")
        second, _ = link_for(other)
        self.assertNotEqual(first[0].link_fingerprint, second[0].link_fingerprint)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
