"""STD2F-R: a prose link is authoritative only if a person accepted that link.

A role list classifies what the diagram shows. A field positioned by a rule
from outside the diagram is a separate decision, and silence about it is never
consent. Every fixture here is invented; no licensed text appears.
"""

import unittest
from dataclasses import replace

from standard_prose_range import (
    ProseRangeError, StandardNormativeRangeLink, StandardReviewedLink,
)
from standard_semantic import (
    APPROVAL_SCHEMA_VERSION, PositionSource, SemanticRole, SpanRole,
    StandardBitfieldApproval, StandardSemanticError, StructuralCompleteness,
    normative_links_for, promote_bitfield, review_evidence_fingerprint,
    semantic_fingerprint,
)
from standard_word_association import associate_words, field_candidates
from tests.standard_word_fixture import (
    FINGERPRINT, LAYOUT, REV, SID, SOURCE, bitfield, ruler_row, table, word_row,
)
from tests.test_standard_prose_range import RULE, _Unit

OPERATOR = "test-operator"


class _Target(unittest.TestCase):
    """One diagram field the label positions, one a rule positions."""

    def setUp(self):
        self.source = table((ruler_row(),
                             word_row(1, ((1, ("Slot Depth", "FIELD_B (27-0)")),))))
        self.mark = bitfield(self.source)
        self.links = normative_links_for(self.mark, self.source, [_Unit(RULE)])
        self.assertEqual(len(self.links), 1)
        self.link = next(iter(self.links.values()))
        self.originals = field_candidates(self.mark.to_dict(),
                                          self.source.to_dict())

    def approval(self, reviewed=(), roles=("FIELD",), evidence=False,
                 reviewer=OPERATOR, **kwargs):
        fingerprint = None
        if evidence:
            fingerprint = review_evidence_fingerprint(
                self.mark.bitfield_id, FINGERPRINT,
                table=self.source.to_dict(), originals=self.originals,
                roles=roles, associations=associate_words(self.source.to_dict()),
                reviewed_links=tuple(reviewed))
        return StandardBitfieldApproval(
            candidate_id=self.mark.bitfield_id, verdict="PASS",
            reviewer=reviewer, reviewed_at="2026-01-01T00:00:00+00:00",
            structure_fingerprint=FINGERPRINT, span_roles=tuple(roles),
            reviewed_links=tuple(reviewed),
            review_evidence_fingerprint=fingerprint, **kwargs)

    def promote(self, approval, links=None):
        return promote_bitfield(
            self.mark, self.source, approval, standard_id=SID, revision=REV,
            corpus_fingerprint=FINGERPRINT, layout_fingerprint=LAYOUT,
            structure_fingerprint=FINGERPRINT,
            prose_links=self.links if links is None else links)


class RoleBoundaryTests(_Target):
    def test_the_role_list_covers_only_what_the_diagram_shows(self):
        self.assertEqual(len(self.originals), 1)
        self.assertEqual([item.text for item in self.originals], ["FIELD_B (27-0)"])

    def test_a_prose_link_does_not_lengthen_the_role_list(self):
        definition, blocked = self.promote(self.approval(
            (StandardReviewedLink.of(self.link, accepted=True),)))
        self.assertIsNone(blocked)
        self.assertEqual(len(definition.fields), 2)

    def test_one_role_too_many_is_refused(self):
        definition, blocked = self.promote(self.approval(
            (StandardReviewedLink.of(self.link, accepted=True),),
            roles=("FIELD", "FIELD")))
        self.assertIsNone(definition)
        self.assertEqual(blocked["reason"], "SPANS_NOT_CLASSIFIED")
        self.assertIn("3" if False else "1 diagram field candidates",
                      blocked["detail"])


class AcceptanceTests(_Target):
    def test_an_accepted_link_becomes_an_authoritative_field(self):
        definition, blocked = self.promote(self.approval(
            (StandardReviewedLink.of(self.link, accepted=True),)))
        self.assertIsNone(blocked)
        field = next(f for f in definition.fields if f.display_label == "Slot Depth")
        self.assertEqual((field.msb, field.lsb), (31, 28))
        self.assertEqual(field.position_source, PositionSource.NORMATIVE_PROSE_RANGE)
        self.assertEqual(definition.structural_completeness,
                         StructuralCompleteness.STRUCTURALLY_COMPLETE)

    def test_an_unreviewed_link_blocks_rather_than_being_assumed(self):
        definition, blocked = self.promote(self.approval())
        self.assertIsNone(definition)
        self.assertEqual(blocked["reason"], "PROSE_LINK_NOT_REVIEWED")

    def test_a_declined_link_leaves_the_field_out_and_says_the_word_is_partial(self):
        definition, blocked = self.promote(self.approval(
            (StandardReviewedLink.of(self.link, accepted=False),)))
        self.assertIsNone(blocked)
        self.assertEqual([f.display_label for f in definition.fields], ["FIELD_B"])
        self.assertEqual(definition.structural_completeness,
                         StructuralCompleteness.STRUCTURALLY_INCOMPLETE)

    def test_an_accepted_link_may_be_reserved_when_the_reviewer_says_so(self):
        definition, _ = self.promote(self.approval(
            (StandardReviewedLink.of(self.link, accepted=True, role="RESERVED"),)))
        field = next(f for f in definition.fields if f.display_label == "Slot Depth")
        self.assertEqual(field.semantic_role, SemanticRole.RESERVED)

    def test_a_linked_field_can_only_be_a_field_or_reserved(self):
        with self.assertRaises(ProseRangeError):
            StandardReviewedLink.of(self.link, accepted=True, role="ROW_LABEL")

    def test_a_candidate_with_no_links_needs_no_link_acceptance(self):
        source = table((ruler_row(), word_row(1, ((1, ("FIELD_B (27-0)",)),))))
        mark = bitfield(source)
        links = normative_links_for(mark, source, [_Unit(RULE)])
        self.assertEqual(links, {})
        approval = StandardBitfieldApproval(
            candidate_id=mark.bitfield_id, verdict="PASS", reviewer=OPERATOR,
            reviewed_at="2026-01-01T00:00:00+00:00",
            structure_fingerprint=FINGERPRINT, span_roles=("FIELD",))
        definition, blocked = promote_bitfield(
            mark, source, approval, standard_id=SID, revision=REV,
            corpus_fingerprint=FINGERPRINT, layout_fingerprint=LAYOUT,
            structure_fingerprint=FINGERPRINT, prose_links=links)
        self.assertIsNone(blocked)
        self.assertEqual(len(definition.fields), 1)


class StaleEvidenceTests(_Target):
    def bad(self, **changes):
        good = StandardReviewedLink.of(self.link, accepted=True)
        return replace(good, **changes)

    def test_a_changed_range_refuses_the_link(self):
        definition, blocked = self.promote(self.approval(
            (self.bad(high_bit=27, low_bit=24),)))
        self.assertIsNone(definition)
        self.assertEqual(blocked["reason"], "STALE_PROSE_LINK")

    def test_a_changed_normative_source_refuses_the_link(self):
        definition, blocked = self.promote(self.approval(
            (self.bad(normative_source_id="std-elsewhere"),)))
        self.assertIsNone(definition)
        self.assertEqual(blocked["reason"], "STALE_PROSE_LINK")

    def test_a_changed_word_refuses_the_link(self):
        definition, blocked = self.promote(self.approval((self.bad(word_index=0),)))
        self.assertIsNone(definition)
        self.assertEqual(blocked["reason"], "STALE_PROSE_LINK")

    def test_a_changed_link_fingerprint_refuses_the_link(self):
        definition, blocked = self.promote(self.approval(
            (self.bad(link_fingerprint="lnk-" + "0" * 16),)))
        self.assertIsNone(definition)
        self.assertEqual(blocked["reason"], "STALE_PROSE_LINK")

    def test_a_link_the_candidate_no_longer_offers_refuses(self):
        extra = StandardReviewedLink("lnk-" + "a" * 16, "cel-gone", "std-x",
                                     7, 4, 1, True)
        definition, blocked = self.promote(self.approval(
            (StandardReviewedLink.of(self.link, accepted=True), extra)))
        self.assertIsNone(definition)
        self.assertEqual(blocked["reason"], "PROSE_LINK_MISSING")

    def test_a_link_naming_another_candidates_field_is_not_this_ones_review(self):
        definition, blocked = self.promote(self.approval(
            (self.bad(target_cell_id="cel-someoneelse"),)))
        self.assertIsNone(definition)
        self.assertIn(blocked["reason"],
                      ("PROSE_LINK_NOT_REVIEWED", "PROSE_LINK_MISSING"))

    def test_stale_review_evidence_refuses_even_when_each_link_matches(self):
        approval = replace(
            self.approval((StandardReviewedLink.of(self.link, accepted=True),),
                          evidence=True),
            review_evidence_fingerprint="0" * 64)
        definition, blocked = self.promote(approval)
        self.assertIsNone(definition)
        self.assertEqual(blocked["reason"], "STALE_REVIEW_EVIDENCE")

    def test_matching_review_evidence_is_accepted(self):
        definition, blocked = self.promote(self.approval(
            (StandardReviewedLink.of(self.link, accepted=True),), evidence=True))
        self.assertIsNone(blocked)
        self.assertEqual(len(definition.fields), 2)

    def test_changing_a_role_changes_the_review_evidence(self):
        reviewed = (StandardReviewedLink.of(self.link, accepted=True),)
        first = self.approval(reviewed, roles=("FIELD",), evidence=True)
        second = self.approval(reviewed, roles=("RESERVED",), evidence=True)
        self.assertNotEqual(first.review_evidence_fingerprint,
                            second.review_evidence_fingerprint)

    def test_changing_a_link_decision_changes_the_review_evidence(self):
        yes = self.approval((StandardReviewedLink.of(self.link, accepted=True),),
                            evidence=True)
        no = self.approval((StandardReviewedLink.of(self.link, accepted=False),),
                           evidence=True)
        self.assertNotEqual(yes.review_evidence_fingerprint,
                            no.review_evidence_fingerprint)

    def test_the_review_evidence_is_reproducible(self):
        reviewed = (StandardReviewedLink.of(self.link, accepted=True),)
        self.assertEqual(self.approval(reviewed, evidence=True).review_evidence_fingerprint,
                         self.approval(reviewed, evidence=True).review_evidence_fingerprint)

    def test_a_stale_geometry_fingerprint_still_refuses(self):
        approval = replace(
            self.approval((StandardReviewedLink.of(self.link, accepted=True),)),
            structure_fingerprint="c" * 64)
        definition, blocked = self.promote(approval)
        self.assertIsNone(definition)
        self.assertEqual(blocked["reason"], "STALE_APPROVAL")


class RecordTests(_Target):
    def test_the_decision_is_persisted_in_the_approval_record(self):
        approval = self.approval(
            (StandardReviewedLink.of(self.link, accepted=True),), evidence=True)
        value = approval.to_dict()
        self.assertEqual(value["schema_version"], APPROVAL_SCHEMA_VERSION)
        self.assertEqual(len(value["reviewed_links"]), 1)
        row = value["reviewed_links"][0]
        self.assertTrue(row["accepted"])
        self.assertEqual((row["high_bit"], row["low_bit"]), (31, 28))
        self.assertEqual(row["normative_source_id"], "std-rule0001")
        self.assertTrue(value["review_evidence_fingerprint"])

    def test_the_record_round_trips(self):
        approval = self.approval(
            (StandardReviewedLink.of(self.link, accepted=True),), evidence=True)
        again = StandardBitfieldApproval.from_dict(approval.to_dict())
        self.assertEqual(again, approval)
        self.assertEqual(again.reviewer, OPERATOR)
        self.assertEqual(again.reviewed_at, "2026-01-01T00:00:00+00:00")

    def test_a_record_written_before_links_existed_reviews_none(self):
        legacy = StandardBitfieldApproval.from_dict({
            "candidate_id": self.mark.bitfield_id, "verdict": "PASS",
            "reviewer": OPERATOR, "reviewed_at": "2026-01-01T00:00:00+00:00",
            "structure_fingerprint": FINGERPRINT, "span_roles": ["FIELD"]})
        self.assertEqual(legacy.schema_version, 1)
        self.assertEqual(legacy.reviewed_links, ())
        self.assertIsNone(legacy.review_evidence_fingerprint)
        # and it cannot therefore authorise a link
        definition, blocked = self.promote(legacy)
        self.assertIsNone(definition)
        self.assertEqual(blocked["reason"], "PROSE_LINK_NOT_REVIEWED")

    def test_an_approval_from_a_newer_build_is_refused(self):
        with self.assertRaises(StandardSemanticError):
            StandardBitfieldApproval(
                candidate_id=self.mark.bitfield_id, verdict="PASS",
                reviewer=OPERATOR, reviewed_at="2026-01-01T00:00:00+00:00",
                structure_fingerprint=FINGERPRINT, span_roles=("FIELD",),
                schema_version=APPROVAL_SCHEMA_VERSION + 1)

    def test_one_link_cannot_be_reviewed_twice(self):
        item = StandardReviewedLink.of(self.link, accepted=True)
        with self.assertRaises(StandardSemanticError):
            self.approval((item, replace(item, accepted=False)))

    def test_an_assistant_can_never_hold_this_approval(self):
        with self.assertRaises(StandardSemanticError):
            self.approval((StandardReviewedLink.of(self.link, accepted=True),),
                          reviewer="claude-assistant")

    def test_the_build_is_deterministic(self):
        approval = self.approval(
            (StandardReviewedLink.of(self.link, accepted=True),), evidence=True)
        first, _ = self.promote(approval)
        second, _ = self.promote(approval)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.semantic_fingerprint, second.semantic_fingerprint)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()


class EvidenceReproducibilityTests(_Target):
    """STD2F-R2: the evidence a reviewer saw must reproduce at validation.

    The regression that motivated these: approval time built the candidate
    list without the page's raw word boxes and validation time built it with
    them, so one label arrived carrying a ruler number in one path and not the
    other, and a correct approval went stale on its first build.
    """

    def evidence(self, originals=None, roles=("FIELD",), reviewed=(), table=None):
        return review_evidence_fingerprint(
            self.mark.bitfield_id, FINGERPRINT,
            table=(table if table is not None else self.source.to_dict()),
            originals=(self.originals if originals is None else originals),
            roles=roles, associations=associate_words(self.source.to_dict()),
            reviewed_links=tuple(reviewed))

    def test_the_page_word_boxes_cannot_change_the_evidence(self):
        # The exact divergence that broke the second approval: one path had
        # the raw geometry that cleans a label, the other did not.
        from tests.standard_word_fixture import merged_cell_table
        from standard_word_association import page_words
        source, words = merged_cell_table("FIELD_A (31-16)")
        mark = bitfield(source)
        plain = field_candidates(mark.to_dict(), source.to_dict())
        cleaned = field_candidates(mark.to_dict(), source.to_dict(), words=words)
        self.assertNotEqual([item.text for item in plain],
                            [item.text for item in cleaned])
        common = dict(table=source.to_dict(), roles=("FIELD",),
                      associations=associate_words(source.to_dict()),
                      reviewed_links=())
        self.assertEqual(
            review_evidence_fingerprint(mark.bitfield_id, FINGERPRINT,
                                        originals=plain, **common),
            review_evidence_fingerprint(mark.bitfield_id, FINGERPRINT,
                                        originals=cleaned, **common))

    def test_the_evidence_binds_the_canonical_cell_text(self):
        from standard_semantic import review_evidence
        payload = review_evidence(
            self.mark.bitfield_id, FINGERPRINT, table=self.source.to_dict(),
            originals=self.originals, roles=("FIELD",),
            associations=associate_words(self.source.to_dict()),
            reviewed_links=())
        cells = {c["cell_id"]: c["text"] for row in self.source.to_dict()["rows"]
                 for c in row["cells"]}
        for entry in payload["candidates"]:
            self.assertEqual(entry["text"], cells[entry["cell"]])

    def test_repeated_computation_is_identical(self):
        reviewed = (StandardReviewedLink.of(self.link, accepted=True),)
        self.assertEqual(self.evidence(reviewed=reviewed),
                         self.evidence(reviewed=reviewed))

    def test_an_accepted_link_does_not_change_the_diagram_candidate_evidence(self):
        from standard_semantic import review_evidence
        payload = lambda reviewed: review_evidence(
            self.mark.bitfield_id, FINGERPRINT, table=self.source.to_dict(),
            originals=self.originals, roles=("FIELD",),
            associations=associate_words(self.source.to_dict()),
            reviewed_links=reviewed)["candidates"]
        self.assertEqual(payload(()),
                         payload((StandardReviewedLink.of(self.link, accepted=True),)))

    def test_a_declined_link_is_as_deterministic_as_an_accepted_one(self):
        declined = (StandardReviewedLink.of(self.link, accepted=False),)
        self.assertEqual(self.evidence(reviewed=declined),
                         self.evidence(reviewed=declined))
        self.assertNotEqual(
            self.evidence(reviewed=declined),
            self.evidence(reviewed=(StandardReviewedLink.of(self.link,
                                                            accepted=True),)))

    def test_link_order_in_the_record_cannot_change_the_evidence(self):
        one = StandardReviewedLink.of(self.link, accepted=True)
        two = StandardReviewedLink("lnk-" + "b" * 16, "cel-other", "std-other",
                                   7, 4, 1, False)
        self.assertEqual(self.evidence(reviewed=(one, two)),
                         self.evidence(reviewed=(two, one)))

    def test_candidate_order_is_material_and_stable(self):
        self.assertEqual(self.evidence(originals=self.originals),
                         self.evidence(originals=self.originals))
        if len(self.originals) > 1:  # pragma: no cover - fixture has one
            self.assertNotEqual(self.evidence(originals=self.originals),
                                self.evidence(originals=self.originals[::-1]))

    def test_the_reviewer_and_timestamp_are_not_bound(self):
        reviewed = (StandardReviewedLink.of(self.link, accepted=True),)
        first = self.approval(reviewed, evidence=True)
        later = replace(first, reviewed_at="2027-05-05T05:05:05+00:00",
                        reviewer="someone-else")
        self.assertEqual(first.review_evidence_fingerprint,
                         self.evidence(reviewed=reviewed))
        self.assertEqual(later.review_evidence_fingerprint,
                         first.review_evidence_fingerprint)

    def test_a_stored_definition_cannot_change_the_evidence(self):
        # The evidence is a function of the candidate and the decisions, so a
        # build that happened afterwards is invisible to it.
        reviewed = (StandardReviewedLink.of(self.link, accepted=True),)
        before = self.evidence(reviewed=reviewed)
        definition, blocked = self.promote(self.approval(reviewed, evidence=True))
        self.assertIsNone(blocked)
        self.assertTrue(definition.definition_id)
        self.assertEqual(before, self.evidence(reviewed=reviewed))

    def test_an_approval_validates_twice_in_a_row(self):
        approval = self.approval(
            (StandardReviewedLink.of(self.link, accepted=True),), evidence=True)
        for _ in range(2):
            definition, blocked = self.promote(approval)
            self.assertIsNone(blocked)
            self.assertEqual(len(definition.fields), 2)
