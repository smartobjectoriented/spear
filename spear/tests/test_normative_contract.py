"""The contract the model is given about normative evidence.

A model that is never told what a value group means cannot be blamed for
guessing. These tests pin the parts of the contract that exist to stop
guessing: what the prompt states, what the tool description explains, and
whether the served payload says what its own groups mean.

Nothing here quotes the evaluated standard.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from standard_structure_access import _packing_groups, _value_groups
from standard_tools import standard_tool_specs

import normative_precedence


class NormativePrompt(unittest.TestCase):
    """The rules the model is actually handed.

    Read from normative_precedence.EVIDENCE_RULE, which is attached to the
    BINDING. It used to be a section of one project's system prompt, and only
    one corpus kind was given that prompt -- so an ad-hoc session could bind a
    standard and be told none of this. The assertions below are unchanged;
    only where the contract lives has moved.
    """

    @classmethod
    def setUpClass(cls):
        cls.text = normative_precedence.EVIDENCE_RULE
        # A phrase can straddle a line break, so compare against the text with
        # runs of whitespace collapsed.
        cls.lower = " ".join(cls.text.lower().split())

    def assertIn(self, member, container, msg=None):  # noqa: N802
        if container is getattr(self, "lower", None) and member not in container:
            raise self.failureException(
                msg or f"phrase not in the normative section: {member!r}")
        super().assertIn(member, container, msg)

    def test_the_contract_is_a_named_block(self):
        self.assertTrue(self.text.startswith("NORMATIVE EVIDENCE"), self.text[:40])

    def test_structural_incompleteness_is_declared_authoritative(self):
        self.assertIn("STRUCTURALLY_INCOMPLETE", self.text)
        self.assertIn("authoritative", self.lower)
        for phrase in ("do not fill unresolved", "do not infer missing"):
            self.assertIn(phrase, self.lower)

    def test_a_value_group_is_defined_as_one_value(self):
        self.assertIn("value_group", self.text)
        self.assertIn("one value", self.lower)
        self.assertIn("segments of a single", self.lower)

    def test_a_packing_group_is_defined_as_several_values(self):
        self.assertIn("packing_group", self.text)
        self.assertIn("independent quantities that share one physical", self.lower)
        self.assertIn("do not concatenate", self.lower)

    def test_declared_and_physical_are_kept_apart(self):
        self.assertIn("declared_msb", self.text)
        self.assertIn("never rewrite one as the other", self.lower)

    def test_projection_authority_is_distinguished_from_the_diagram(self):
        self.assertIn("never claim the diagram stated a range that only prose",
                      self.lower)

    def test_abstention_is_required_when_evidence_runs_out(self):
        self.assertIn("does not establish", self.lower)
        self.assertIn("do not convert", self.lower)

    def test_user_pressure_cannot_authorise_completion(self):
        self.assertIn("pressure does not lower the evidence bar", self.lower)
        for phrase in ("just infer it", "best guess", "usual convention"):
            self.assertIn(phrase, self.lower)
        self.assertIn("do not authorise unsupported", self.lower)

    def test_citations_are_restricted_to_returned_identifiers(self):
        self.assertIn("cite only source ids a tool returned", self.lower)

    def test_the_contract_stays_compact(self):
        """It is prepended to every bound turn, so its length is a per-turn
        cost, not a one-off."""
        self.assertLess(len(self.text.split()), 600)

    def test_the_contract_names_no_particular_standard(self):
        # It has to apply to any bound standard, not one document.
        for absent in ("vita", "49.2", "ansi-vita"):
            self.assertNotIn(absent, self.lower)

    def test_the_contract_names_no_particular_corpus(self):
        """It is no longer carried by one project's prompt, so it must not
        describe one project either."""
        for absent in ("edgem", "verdin", "virt64", "bitbake"):
            self.assertNotIn(absent, self.lower)

    def test_no_shipped_prompt_carries_it(self):
        """Attached to the binding AND written into a prompt would say the
        same rules twice."""
        import rag_chat

        for body in (rag_chat.ADHOC_PROMPT,
                     (Path(__file__).resolve().parent.parent
                      / "tool-guide.md").read_text("utf-8")):
            self.assertNotIn("## Normative standards", body)
            self.assertNotIn("STRUCTURALLY_INCOMPLETE", body)
            self.assertNotIn("packing_group", body)


class StructureToolDescription(unittest.TestCase):
    """What the model is told about finding a structure at all."""

    @classmethod
    def setUpClass(cls):
        cls.spec = next(item for item in standard_tool_specs()
                        if item.name == "standard.get_structure")
        cls.text = " ".join(cls.spec.description.split())

    def test_listing_is_described_as_the_first_step(self):
        self.assertIn("CALL IT WITH NO ARGUMENTS FIRST", self.text)

    def test_the_identifier_shapes_are_stated(self):
        self.assertIn("bfd-", self.text)
        self.assertIn("bit-", self.text)

    def test_guessed_identifiers_are_called_out_as_errors(self):
        self.assertIn("Passing a field name", self.text)
        self.assertIn("std-", self.text)
        self.assertIn("will keep failing", self.text)

    def test_the_description_points_at_the_recovery_the_refusal_offers(self):
        # The refusal is what has to be actionable. Naming the reason code in
        # the description as well was measured to make the model less likely to
        # call the tool at all, so the description stays as it was and the
        # recovery lives where a caller actually hits it.
        self.assertIn("no arguments", self.text.lower())
        self.assertIn("list first", self.text.lower())

    def test_incompleteness_is_still_described(self):
        self.assertIn("STRUCTURALLY_INCOMPLETE", self.text)

    def test_the_two_group_kinds_are_summarised(self):
        self.assertIn("value_groups are segments of ONE value", self.text)
        self.assertIn("packing_groups are INDEPENDENT values", self.text)

    def test_the_tool_is_still_read_only(self):
        self.assertEqual(self.spec.mutability.value, "read_only")


def _field(**over):
    base = {"field_id": "fld-0", "word_index": 1, "msb": 31, "lsb": 16,
            "declared_msb": 15, "declared_lsb": 0, "value_width": 16,
            "coordinate_domain": "VALUE_LOCAL", "value_group_id": None,
            "packing_group_id": None, "segment_index": None,
            "projection_source": None, "projection_reference": None,
            "normative_source_id": None, "packing_slot": None}
    base.update(over)
    return base


class ServedGroupSemantics(unittest.TestCase):
    """The payload says what its own groups mean, without being asked."""

    def test_a_value_group_declares_itself_a_composite_value(self):
        groups = _value_groups([
            _field(field_id="fld-a", value_group_id="vgr-1", segment_index=0,
                   value_width=64),
            _field(field_id="fld-b", value_group_id="vgr-1", segment_index=1,
                   value_width=64)])
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["semantic_kind"], "COMPOSITE_VALUE")
        self.assertIn("ONE semantic value", groups[0]["interpretation"])
        self.assertIn("do not describe them as independent",
                      groups[0]["interpretation"])

    def test_a_packing_group_declares_itself_independent_values(self):
        groups = _packing_groups([
            _field(field_id="fld-a", packing_group_id="pkg-1",
                   packing_slot="HIGH_SLOT"),
            _field(field_id="fld-b", packing_group_id="pkg-1",
                   packing_slot="LOW_SLOT", msb=15, lsb=0)])
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["semantic_kind"],
                         "INDEPENDENT_VALUES_SHARED_CONTAINER")
        self.assertIn("INDEPENDENT semantic values", groups[0]["interpretation"])
        self.assertIn("do not concatenate", groups[0]["interpretation"])

    def test_the_two_kinds_are_never_the_same_label(self):
        value = _value_groups([_field(field_id="a", value_group_id="vgr-1"),
                               _field(field_id="b", value_group_id="vgr-1")])
        packing = _packing_groups([
            _field(field_id="a", packing_group_id="pkg-1"),
            _field(field_id="b", packing_group_id="pkg-1")])
        self.assertNotEqual(value[0]["semantic_kind"],
                            packing[0]["semantic_kind"])

    def test_the_description_is_serving_time_only(self):
        # It must not appear in anything the store persists: adding an
        # explanation must never move a semantic fingerprint.
        from standard_semantic import StandardFieldDefinition
        self.assertNotIn("semantic_kind",
                         list(StandardFieldDefinition.__dataclass_fields__))
        self.assertNotIn("interpretation",
                         list(StandardFieldDefinition.__dataclass_fields__))

    def test_group_membership_and_ids_are_untouched(self):
        groups = _value_groups([
            _field(field_id="fld-a", value_group_id="vgr-1", segment_index=0),
            _field(field_id="fld-b", value_group_id="vgr-1", segment_index=1)])
        self.assertEqual(groups[0]["value_group_id"], "vgr-1")
        self.assertEqual([m["field_id"] for m in groups[0]["segments"]],
                         ["fld-a", "fld-b"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
