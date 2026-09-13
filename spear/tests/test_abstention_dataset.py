"""The audit that stands between a preference file and a fine-tune.

A preference pair is only worth training on if the two sides differ in the one
way we mean. These tests check that the auditor actually refuses the ways that
stop being true: a winning answer that cites something no tool returned, two
sides that differ in nothing, a counterexample whose winning answer gives up,
and an example that quietly restates a held-out evaluation prompt.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval" / "abstention"))

from validate import ANSWERING_CLASS, audit

TOOLS = [{"type": "function",
          "function": {"name": "standard.search", "description": "search",
                       "parameters": {"type": "object",
                                      "properties": {"query": {"type": "string"}},
                                      "required": ["query"]}}}]
GOOD_SOURCE = "std-" + "a" * 32


def _record(**over):
    base = {
        "pair_id": "p1", "class": "UNSUPPORTED_COMPLEMENT_INFERENCE",
        # Provenance a losing side has to carry: observed, seen more than once,
        # and produced without the tool path getting in the way.
        "rejected_origin": "sampled", "reproductions": 3,
        "tool_path_clean": True,
        # The audit that stands between a correct deduction and a poisoned
        # pair, and the provenance a reviewer needs to trace the sample.
        "entailment_checked": True, "entailment_result": "NOT_ENTAILED",
        "semantic_family": "iq", "structural_shape": "partial_slot",
        "source_phase": "FT0.1", "train_or_eval": "train",
        "licensed_content": False,
        "prompt": [
            {"role": "system", "content": "be careful"},
            {"role": "user", "content": "which half does A occupy?"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "standard.search",
                                          "arguments": '{"query": "A"}'}}]},
            {"role": "tool", "tool_call_id": "c1",
             "content": '{"results": [{"source_id": "' + GOOD_SOURCE + '"}]}'},
        ],
        "chosen": [{"role": "assistant", "content":
                    "The rule places one value in the lower half but does not "
                    "name which, so the assignment is not established by this "
                    "evidence and I would be guessing to complete it."}],
        "rejected": [{"role": "assistant", "content":
                      "By the usual convention the first-named field takes the "
                      "upper half."}],
        "tools": TOOLS,
    }
    base.update(over)
    return base


class Auditor(unittest.TestCase):
    def test_a_clean_pair_passes(self):
        self.assertEqual(audit([_record()]), [])

    def test_a_fabricated_identifier_in_the_winning_answer_is_caught(self):
        found = audit([_record(chosen=[{"role": "assistant", "content":
                                        "Not established by this evidence, see "
                                        "std-" + "b" * 32}])])
        self.assertTrue(any("cites identifiers" in item for item in found))

    def test_two_identical_sides_are_caught(self):
        same = [{"role": "assistant", "content": "not established, unresolved"}]
        found = audit([_record(chosen=same, rejected=same)])
        self.assertTrue(any("identical" in item for item in found))

    def test_an_empty_side_is_caught(self):
        found = audit([_record(rejected=[{"role": "assistant", "content": ""}])])
        self.assertTrue(any("empty" in item for item in found))

    def test_a_counterexample_that_abstains_on_the_winning_side_is_caught(self):
        found = audit([_record(
            **{"class": ANSWERING_CLASS}, rejected_origin="authored",
            reproductions=0,
            chosen=[{"role": "assistant", "content":
                     "The evidence does not establish which half."}],
            rejected=[{"role": "assistant", "content": "A is at 31..16."}])])
        self.assertTrue(any("chosen abstains" in item for item in found))
        self.assertTrue(any("over-abstention" in item for item in found))

    def test_a_winning_answer_reasoning_from_convention_is_caught(self):
        found = audit([_record(chosen=[{"role": "assistant", "content":
                                        "Not established, though by the usual "
                                        "convention it would be the upper half "
                                        "and remains unresolved otherwise."}])])
        self.assertTrue(any("convention/layout" in item for item in found))

    def test_a_duplicate_pair_is_caught(self):
        found = audit([_record(), _record(pair_id="p2")])
        self.assertTrue(any("duplicate" in item for item in found))

    def test_a_prompt_that_leaks_a_held_out_case_is_caught(self):
        found = audit([_record()],
                      eval_prompts=("which half does A occupy?",),
                      duplicate_threshold=0.05)
        self.assertTrue(any("leaks a held-out" in item for item in found))

    def test_a_thin_winning_answer_is_caught(self):
        found = audit([_record(chosen=[{"role": "assistant",
                                        "content": "Not established."}])])
        self.assertTrue(any("too thin" in item for item in found))

    def test_a_malformed_message_list_is_caught(self):
        found = audit([_record(prompt=[{"role": "user", "content": "hi"},
                                       {"role": "tool", "tool_call_id": "nope",
                                        "content": "{}"}])])
        self.assertTrue(any("not a valid target" in item for item in found))


class Provenance(unittest.TestCase):
    """What a pair has to say about where its losing side came from."""

    def test_a_label_nobody_defined_is_caught(self):
        found = audit([_record(**{"class": "VIBES"})])
        self.assertTrue(any("unknown failure class" in item for item in found))

    def test_an_authored_losing_side_on_a_failure_class_is_caught(self):
        # Writing the failure is exactly what this phase exists to avoid.
        found = audit([_record(rejected_origin="authored", reproductions=0)])
        self.assertTrue(any("authored rather than observed" in item
                            for item in found))

    def test_an_authored_losing_side_is_allowed_on_a_control(self):
        # The model does not over-abstain on evidence it can read, so the
        # losing side of a control has to be written. That one is legitimate.
        found = audit([_record(
            **{"class": ANSWERING_CLASS}, rejected_origin="authored",
            reproductions=0,
            chosen=[{"role": "assistant", "content":
                     "A occupies the upper half, bits 31..16, and B the lower "
                     "half, bits 15..0, as the rule states directly."}],
            rejected=[{"role": "assistant", "content":
                       "The evidence does not establish which half."}])])
        self.assertEqual(found, [])

    def test_a_failure_seen_only_once_is_caught(self):
        found = audit([_record(reproductions=1)])
        self.assertTrue(any("reproduced 1 times" in item for item in found))

    def test_a_dirty_tool_path_is_caught(self):
        found = audit([_record(tool_path_clean=False)])
        self.assertTrue(any("tool path was not recorded as clean"
                            in item for item in found))


class FabricatedCitations(unittest.TestCase):
    def test_a_file_url_is_caught(self):
        found = audit([_record(rejected=[{"role": "assistant", "content":
                                          "See file:///home/user/std/doc.pdf"}])])
        self.assertTrue(any("local filesystem path" in item for item in found))

    def test_a_bare_local_path_is_caught(self):
        found = audit([_record(chosen=[{"role": "assistant", "content":
                                        "Not established; see "
                                        "/home/user/standards/doc.pdf for the "
                                        "figure that would settle it."}])])
        self.assertTrue(any("local filesystem path" in item for item in found))

    def test_an_ordinary_answer_is_not_flagged_as_a_path(self):
        self.assertEqual(audit([_record()]), [])


class EntailmentMetadata(unittest.TestCase):
    """A negative pair has to carry the audit that cleared it."""

    def test_an_unaudited_pair_is_refused(self):
        found = audit([_record(entailment_checked=False)])
        self.assertTrue(any("never audited" in item for item in found))

    def test_an_entailed_negative_is_refused(self):
        # The deduction the model made was correct; training against it is the
        # one mistake this whole audit exists to prevent.
        found = audit([_record(entailment_result="ENTAILED")])
        self.assertTrue(any("not NOT_ENTAILED" in item for item in found))

    def test_an_ambiguous_negative_is_refused(self):
        found = audit([_record(entailment_result="AMBIGUOUS")])
        self.assertTrue(any("not NOT_ENTAILED" in item for item in found))

    def test_missing_provenance_is_refused(self):
        record = _record()
        del record["source_phase"]
        self.assertTrue(any("missing metadata" in item
                            for item in audit([record])))

    def test_a_pair_marked_as_licensed_is_refused(self):
        found = audit([_record(licensed_content=True)])
        self.assertTrue(any("licensed content" in item for item in found))

    def test_a_gap_marked_across_a_line_break_still_counts(self):
        # An answer wrapped for reading splits the phrase; a matcher that
        # misses it reports good answers as bad.
        found = audit([_record(chosen=[{"role": "assistant", "content":
                                        "The rule places one value in the "
                                        "lower half of the container. Which "
                                        "of the two members that is, the "
                                        "evidence does\nnot establish, so I "
                                        "am not going to assign either one to "
                                        "a physical half here."}])])
        self.assertEqual(found, [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
