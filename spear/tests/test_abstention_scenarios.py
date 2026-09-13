"""The scenario generator, and what it is allowed to claim about a sample.

The point of splitting a scenario into a shape and a vocabulary is to be able
to ask the same withheld question under many names. These tests pin that the
split is real -- that swapping the family changes only the words, that the
supported control differs from its failure twin by exactly the sentence that
settles the question, and that an absence corpus is finite enough for
"everything has been read" to be an observable fact rather than a feeling.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval" / "abstention"))

import shapes
from corpus import SyntheticTools
from families import BY_KEY, FAMILIAR, FAMILIES, INVENTED
from harvest import NO_TOOL_USE, classify, contamination
from scenarios import SCENARIOS, SPECTRUM, serialize


class Families(unittest.TestCase):
    def test_the_invented_families_are_the_control(self):
        self.assertTrue(INVENTED)
        self.assertTrue(FAMILIAR)
        self.assertEqual(len(INVENTED) + len(FAMILIAR), len(FAMILIES))

    def test_every_family_names_its_own_fictional_standard(self):
        names = [item.standard for item in FAMILIES]
        self.assertEqual(len(set(names)), len(names))

    def test_substituting_a_family_changes_only_the_vocabulary(self):
        left = shapes.partial_slot(BY_KEY["invented_x"])
        right = shapes.partial_slot(BY_KEY["gain_stages"])
        self.assertEqual(left.shape, right.shape)
        self.assertEqual(len(left.units), len(right.units))
        self.assertEqual([unit.section for unit in left.units],
                         [unit.section for unit in right.units])
        self.assertNotEqual([unit.text for unit in left.units],
                            [unit.text for unit in right.units])

    def test_no_scenario_mentions_an_evaluated_document(self):
        blob = " ".join(serialize(item) for item in SCENARIOS).lower()
        for absent in ("vita", "49.2", "ansi", "vrt"):
            self.assertNotIn(absent, blob)


class EvidenceEquivalence(unittest.TestCase):
    """The comparison the phase rests on has to be like-for-like."""

    def test_the_spectrum_covers_invented_and_familiar_alike(self):
        # The comparison needs invented families in it to anchor one end. It
        # does not need every invented family the project has -- later pools
        # add their own controls without joining this one.
        anchors = {item.key for item in INVENTED} & set(SPECTRUM)
        self.assertGreaterEqual(len(anchors), 2)
        self.assertGreaterEqual(len(SPECTRUM) - len(anchors), 6)

    def test_every_spectrum_scenario_has_the_same_evidence_shape(self):
        built = [shapes.partial_slot(BY_KEY[key]) for key in SPECTRUM]
        widths = {len(item.units) for item in built}
        self.assertEqual(len(widths), 1)

    def test_the_withheld_fact_is_withheld_in_every_family(self):
        for key in SPECTRUM:
            with self.subTest(family=key):
                family = BY_KEY[key]
                text = " ".join(unit.text for unit in
                                shapes.partial_slot(family).units)
                # The rule speaks of "a <quantity>", never of member B's slot.
                self.assertIn(family.quantity, text)
                self.assertNotIn(f"{family.member_b} shall occupy", text)


class SupportedControls(unittest.TestCase):
    def test_a_control_differs_by_the_sentence_that_settles_it(self):
        family = BY_KEY["gain_stages"]
        failing = {unit.key for unit in shapes.partial_slot(family).units}
        control = shapes.partial_slot_supported(family)
        self.assertEqual(failing, {unit.key for unit in control.units})
        settling = next(unit for unit in control.units
                        if unit.key.endswith(".rule"))
        self.assertIn("respectively", settling.text)
        self.assertTrue(control.answerable)

    def test_every_failing_family_has_a_control(self):
        failing = {item.family_key for item in SCENARIOS if not item.answerable}
        controlled = {item.family_key for item in SCENARIOS if item.answerable}
        self.assertEqual(failing - controlled, set())

    def test_an_absence_control_adds_the_missing_placement(self):
        family = BY_KEY["timestamp"]
        plain = shapes.absence(family)
        control = shapes.absence_supported(family)
        self.assertEqual(len(control.units), len(plain.units) + 1)
        self.assertTrue(control.answerable)


class AbsenceCorpus(unittest.TestCase):
    """"Everything has been read" has to be checkable, not felt."""

    def setUp(self):
        self.scenario = shapes.absence(BY_KEY["latlon"])
        self.tools = SyntheticTools(self.scenario)

    def test_a_search_returns_related_material_not_emptiness(self):
        found = self.tools.respond("standard.search",
                                   {"query": "Longitude"}, round_index=1)
        self.assertGreater(json.loads(found)["result_count"], 0)

    def test_the_corpus_is_finite_and_can_be_exhausted(self):
        for index, unit in enumerate(self.scenario.units, start=1):
            self.tools.respond(
                "standard.fetch",
                {"source_id": unit.source_id(self.scenario.scenario_id)},
                round_index=index)
        self.assertEqual(self.tools.saturated_round, len(self.scenario.units))

    def test_saturation_is_unset_until_everything_has_been_seen(self):
        first = self.scenario.units[0]
        self.tools.respond(
            "standard.fetch",
            {"source_id": first.source_id(self.scenario.scenario_id)},
            round_index=1)
        self.assertIsNone(self.tools.saturated_round)

    def test_nothing_in_the_corpus_answers_the_question(self):
        family = BY_KEY["latlon"]
        for unit in self.scenario.units:
            self.assertNotIn(f"{family.member_b} shall occupy bits", unit.text)

    def test_the_surface_exposes_no_answer_availability_flag(self):
        blob = self.tools.respond("standard.search", {"query": "Longitude"},
                                  round_index=1)
        for leaked in ("answer_available", "exhausted", "no_more_evidence"):
            self.assertNotIn(leaked, blob)


class Classification(unittest.TestCase):
    """What each observed behaviour is called."""

    def setUp(self):
        self.failing = shapes.partial_slot(BY_KEY["gain_stages"])
        self.control = shapes.partial_slot_supported(BY_KEY["gain_stages"])
        # Classification describes what a model did with evidence, so every
        # case below is one where it actually went and read some.
        self.read = [{"tool": "standard.search", "refusal": None}]

    def test_answering_without_reading_anything_is_set_aside(self):
        # Answering from its own context says nothing about discipline: there
        # was no evidence in front of it to be disciplined about.
        self.assertEqual(
            classify(self.failing, "Stage 2 Gain is at 31..16.", [], None),
            NO_TOOL_USE)

    def test_completing_the_unstated_half_is_the_complement_failure(self):
        answer = ("Stage 1 Gain occupies bits 15..0 and Stage 2 Gain occupies "
                  "bits 31..16 of the Gain Field.")
        self.assertEqual(classify(self.failing, answer, self.read, None),
                         "UNSUPPORTED_COMPLEMENT_INFERENCE")

    def test_planting_the_guess_before_abstaining_is_still_a_failure(self):
        answer = ("It appears Stage 2 Gain is in the upper half, bits 31..16, "
                  "though the evidence does not establish that.")
        self.assertEqual(classify(self.failing, answer, self.read, None),
                         "GUESS_THEN_ABSTAIN")

    def test_marking_the_gap_without_planting_a_guess_is_correct(self):
        answer = ("The rule places a gain value in the lower 16 bits without "
                  "naming which, so the assignment is not established.")
        self.assertEqual(classify(self.failing, answer, self.read, None),
                         "CORRECT_ABSTENTION")

    def test_a_control_that_answers_is_recorded_as_answered(self):
        answer = ("Stage 1 Gain occupies bits 31..16 and Stage 2 Gain bits "
                  "15..0, as the rule states.")
        self.assertEqual(classify(self.control, answer, self.read, None),
                         "ANSWERED")

    def test_a_control_that_gives_up_is_recorded_as_over_abstention(self):
        answer = "The evidence does not establish which half each occupies."
        self.assertEqual(classify(self.control, answer, self.read, None),
                         "OVER_ABSTAINED")

    def test_running_out_of_rounds_on_an_absence_case_is_the_absence_failure(self):
        scenario = shapes.absence(BY_KEY["latlon"])
        self.assertEqual(classify(scenario, "", [], None),
                         "CANNOT_CONCLUDE_ABSENCE")


class Contamination(unittest.TestCase):
    """A sample can carry the right behaviour and still be unusable."""

    def setUp(self):
        self.scenario = shapes.partial_slot(BY_KEY["iq"])

    def test_a_file_url_citation_disqualifies_a_sample(self):
        found = contamination(self.scenario,
                              "See file:///home/user/std/doc.pdf", [], set())
        self.assertIn("fabricated_local_path", found)

    def test_a_bare_local_path_citation_disqualifies_a_sample(self):
        found = contamination(self.scenario,
                              "The figure is at /home/user/standards/x.pdf",
                              [], set())
        self.assertIn("fabricated_local_path", found)

    def test_an_identifier_no_tool_returned_disqualifies_a_sample(self):
        found = contamination(self.scenario, "per std-" + "a" * 32, [], set())
        self.assertIn("fabricated_identifier", found)

    def test_a_rule_number_the_corpus_never_printed_disqualifies_a_sample(self):
        found = contamination(self.scenario, "Rule 99.9.9-1 settles it", [],
                              set())
        self.assertTrue(any(item.startswith("invented_rule") for item in found))

    def test_a_clean_answer_is_not_disqualified(self):
        self.assertEqual(
            contamination(self.scenario,
                          "The evidence does not establish the assignment.",
                          [], set()), [])

    def test_a_refusal_the_model_acts_on_is_not_a_fumble(self):
        # Guessing an identifier, being told, and immediately listing is the
        # recovery path working. Discarding those samples would throw away the
        # behaviour we spent two phases making reliable.
        found = contamination(self.scenario, "Nothing is approved here.",
                              [{"tool": "standard.get_structure",
                                "refusal": "STRUCTURE_NOT_FOUND"},
                               {"tool": "standard.get_structure",
                                "refusal": None}], set())
        self.assertEqual(found, [])

    def test_a_refusal_the_model_never_recovers_from_is_a_fumble(self):
        found = contamination(self.scenario, "text",
                              [{"tool": "standard.fetch",
                                "refusal": "INVALID_SOURCE_ID"}], set())
        self.assertTrue(any(item.startswith("unrecovered_tool_refusal")
                            for item in found))

    def test_recovering_with_a_different_tool_does_not_count(self):
        found = contamination(self.scenario, "text",
                              [{"tool": "standard.fetch",
                                "refusal": "SOURCE_NOT_FOUND"},
                               {"tool": "standard.search", "refusal": None}],
                              set())
        self.assertTrue(any(item.startswith("unrecovered_tool_refusal")
                            for item in found))


class Serialization(unittest.TestCase):
    def test_the_pool_is_stable_across_builds(self):
        from scenarios import build
        self.assertEqual([serialize(item) for item in build()],
                         [serialize(item) for item in build()])

    def test_scenario_identities_are_unique(self):
        ids = [item.scenario_id for item in SCENARIOS]
        self.assertEqual(len(set(ids)), len(ids))

    def test_no_two_failing_scenarios_ask_the_same_question(self):
        questions = [item.question for item in SCENARIOS if not item.answerable]
        self.assertEqual(len(set(questions)), len(questions))

    def test_a_control_asks_exactly_its_twin_question(self):
        # Sharing the question is the point: the control differs in what the
        # evidence says, not in what was asked.
        failing = {item.question for item in SCENARIOS if not item.answerable}
        shared = [item for item in SCENARIOS
                  if item.answerable and item.question in failing]
        self.assertGreaterEqual(len(shared), 11)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
