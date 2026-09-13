"""How a tool call is written depends on who is going to read it.

The evaluation harness speaks the OpenAI wire form, where a tool call's
arguments are a JSON string. The Qwen chat template iterates them as a mapping
and raises on a string, which is why every one of the frozen training records
failed to render. The two representations are both correct for their own
consumer, so the conversion belongs at the boundary between them rather than in
the harness or in the audited pair files.

These tests pin that the conversion happens there, that it loses nothing, and
that it refuses rather than repairs when an argument string is not valid JSON.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval" / "abstention"))

from build_pairs import TRAINER_FIELDS, trainer_arguments, trainer_record


def _call(arguments, name="standard.search", call_id="call_1"):
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": arguments}}


def _record(arguments='{"query": "gain"}'):
    return {
        "pair_id": "p1", "class": "UNSUPPORTED_COMPLEMENT_INFERENCE",
        "semantic_family": "iq", "structural_shape": "partial_slot",
        "source_phase": "FT0.1", "train_or_eval": "train",
        "prompt": [
            {"role": "system", "content": "be careful"},
            {"role": "user", "content": "which half?"},
            {"role": "assistant", "content": "", "tool_calls": [_call(arguments)]},
            {"role": "tool", "tool_call_id": "call_1", "content": '{"ok": true}'},
        ],
        "chosen": [{"role": "assistant", "content": "one half is stated"}],
        "rejected": [{"role": "assistant", "content": "both halves assigned"}],
        "tools": [{"type": "function", "function": {"name": "standard.search"}}],
    }


class Conversion(unittest.TestCase):
    def test_a_string_argument_becomes_a_mapping(self):
        found = trainer_arguments('{"definition_id": "bfd-123"}')
        self.assertEqual(found, {"definition_id": "bfd-123"})

    def test_a_mapping_argument_is_left_alone(self):
        found = trainer_arguments({"definition_id": "bfd-123"})
        self.assertEqual(found, {"definition_id": "bfd-123"})

    def test_an_empty_argument_string_becomes_an_empty_mapping(self):
        # A tool called with no arguments is how listing is requested.
        self.assertEqual(trainer_arguments(""), {})
        self.assertEqual(trainer_arguments("{}"), {})

    def test_the_conversion_is_canonically_lossless(self):
        original = '{"b": 2, "a": {"deep": [1, 2, {"x": null}]}}'
        found = trainer_arguments(original)
        self.assertEqual(json.dumps(json.loads(original), sort_keys=True),
                         json.dumps(found, sort_keys=True))

    def test_nested_objects_survive(self):
        found = trainer_arguments('{"outer": {"inner": {"leaf": "v"}}}')
        self.assertEqual(found["outer"]["inner"]["leaf"], "v")

    def test_arrays_survive(self):
        found = trainer_arguments('{"ids": ["a", "b", "c"], "n": [1, 2]}')
        self.assertEqual(found["ids"], ["a", "b", "c"])
        self.assertEqual(found["n"], [1, 2])

    def test_scalars_keep_their_types(self):
        found = trainer_arguments(
            '{"flag": true, "off": false, "nothing": null, "n": 5, "f": 1.5}')
        self.assertIs(found["flag"], True)
        self.assertIs(found["off"], False)
        self.assertIsNone(found["nothing"])
        self.assertEqual(found["n"], 5)
        self.assertEqual(found["f"], 1.5)

    def test_malformed_json_refuses_rather_than_repairs(self):
        # Guessing what a broken argument meant would put invented evidence in
        # the training data.
        with self.assertRaises(ValueError):
            trainer_arguments('{"query": "unterminated')

    def test_a_json_scalar_is_not_an_argument_object(self):
        with self.assertRaises(ValueError):
            trainer_arguments('"just a string"')


class RecordShape(unittest.TestCase):
    def setUp(self):
        self.source = _record()
        self.trainer = trainer_record(self.source)

    def test_the_trainer_view_carries_only_the_four_trainer_fields(self):
        self.assertEqual(set(self.trainer), set(TRAINER_FIELDS))

    def test_arguments_are_a_mapping_in_the_trainer_view(self):
        call = self.trainer["prompt"][2]["tool_calls"][0]
        self.assertIsInstance(call["function"]["arguments"], dict)

    def test_the_audited_record_keeps_the_wire_form(self):
        # The evaluation side reads these back; converting them there would
        # change an artifact that is correct for its own consumer.
        call = self.source["prompt"][2]["tool_calls"][0]
        self.assertIsInstance(call["function"]["arguments"], str)

    def test_the_tool_call_id_is_untouched(self):
        self.assertEqual(self.trainer["prompt"][2]["tool_calls"][0]["id"],
                         "call_1")

    def test_the_tool_name_is_untouched(self):
        self.assertEqual(
            self.trainer["prompt"][2]["tool_calls"][0]["function"]["name"],
            "standard.search")

    def test_message_order_and_roles_are_untouched(self):
        self.assertEqual([m["role"] for m in self.trainer["prompt"]],
                         ["system", "user", "assistant", "tool"])

    def test_tool_results_are_untouched(self):
        self.assertEqual(self.trainer["prompt"][3]["content"], '{"ok": true}')
        self.assertEqual(self.trainer["prompt"][3]["tool_call_id"], "call_1")

    def test_the_chosen_answer_is_untouched(self):
        self.assertEqual(self.trainer["chosen"], self.source["chosen"])

    def test_the_rejected_answer_is_untouched(self):
        self.assertEqual(self.trainer["rejected"], self.source["rejected"])

    def test_the_tool_view_is_untouched(self):
        self.assertEqual(self.trainer["tools"], self.source["tools"])

    def test_converting_twice_changes_nothing_further(self):
        self.assertEqual(trainer_record(self.trainer), self.trainer)

    def test_the_source_record_is_not_mutated(self):
        before = json.dumps(_record(), sort_keys=True)
        trainer_record(json.loads(before))
        self.assertEqual(json.dumps(_record(), sort_keys=True), before)


class SemanticEquivalence(unittest.TestCase):
    """Everything but the representation of one field has to match."""

    @staticmethod
    def _canonical(record):
        found = json.loads(json.dumps(
            {k: record[k] for k in TRAINER_FIELDS if k in record}))
        for message in found["prompt"]:
            for call in message.get("tool_calls") or []:
                args = call["function"].get("arguments")
                if isinstance(args, str):
                    call["function"]["arguments"] = json.loads(args or "{}")
        return json.dumps(found, sort_keys=True)

    def test_old_and_new_forms_are_the_same_conversation(self):
        source = _record()
        self.assertEqual(self._canonical(source),
                         self._canonical(trainer_record(source)))

    def test_a_changed_argument_value_would_be_caught(self):
        # The comparison has to be able to fail, or it proves nothing.
        source = _record()
        altered = trainer_record(_record('{"query": "different"}'))
        self.assertNotEqual(self._canonical(source), self._canonical(altered))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()


class _StubTokenizer:
    """Enough of a chat template to test the decomposition, not the model.

    Mimics the Qwen shape: turns wrapped in im_start/im_end, an optional
    generation marker, and tools folded into the system turn.
    """

    def apply_chat_template(self, messages, tools=None,
                            add_generation_prompt=False, tokenize=False):
        out = []

        if tools:
            out.append(f"<|tools|>{len(tools)}<|/tools|>\n")

        for message in messages:
            role = message.get("role")
            body = message.get("content") or ""

            for call in message.get("tool_calls") or ():
                body += (f"<tool_call>{call['function']['name']}"
                         f"{json.dumps(call['function']['arguments'], sort_keys=True)}"
                         f"</tool_call>")

            out.append(f"<|im_start|>{role}\n{body}<|im_end|>\n")

        if add_generation_prompt:
            out.append("<|im_start|>assistant\n")

        return "".join(out)


def _conversational(**over):
    from build_pairs import trainer_record
    return trainer_record(_record(**over))


class AxolotlRendering(unittest.TestCase):
    """Three strings, decomposed the way the installed strategy expects."""

    def setUp(self):
        from build_pairs import render_dpo_strings
        self.tok = _StubTokenizer()
        self.source = _conversational()
        self.rendered = render_dpo_strings(self.source, self.tok)

    def test_all_three_fields_are_strings(self):
        for key in ("prompt", "chosen", "rejected"):
            with self.subTest(field=key):
                self.assertIsInstance(self.rendered[key], str)
                self.assertTrue(self.rendered[key].strip())

    def test_the_audited_record_stays_conversational(self):
        # The human-facing artifact is not what changed.
        self.assertIsInstance(self.source["prompt"], list)
        self.assertIsInstance(self.source["prompt"][0], dict)

    def test_the_prompt_ends_at_the_assistant_boundary(self):
        self.assertTrue(self.rendered["prompt"].endswith("<|im_start|>assistant\n"))

    def test_the_prompt_carries_neither_answer(self):
        for key in ("chosen", "rejected"):
            with self.subTest(answer=key):
                text = self.source[key][0]["content"]
                self.assertNotIn(text, self.rendered["prompt"])

    def test_a_completion_does_not_repeat_the_prompt(self):
        for key in ("chosen", "rejected"):
            with self.subTest(answer=key):
                self.assertNotIn("<|im_start|>system", self.rendered[key])
                self.assertNotIn("<|im_start|>user", self.rendered[key])
                self.assertNotIn("[[dummy_message]]", self.rendered[key])

    def test_a_completion_carries_no_second_generation_marker(self):
        for key in ("chosen", "rejected"):
            with self.subTest(answer=key):
                self.assertNotIn("<|im_start|>assistant", self.rendered[key])

    def test_each_completion_preserves_its_answer_text(self):
        for key in ("chosen", "rejected"):
            with self.subTest(answer=key):
                self.assertTrue(
                    self.rendered[key].startswith(self.source[key][0]["content"]))

    def test_each_completion_ends_with_exactly_one_terminator(self):
        for key in ("chosen", "rejected"):
            with self.subTest(answer=key):
                self.assertEqual(self.rendered[key].count("<|im_end|>"), 1)
                self.assertTrue(self.rendered[key].endswith("<|im_end|>"))

    def test_prompt_plus_completion_reproduces_the_conversation(self):
        # The property the whole repair rests on: what the trainer sees is what
        # the model would have seen, up to the trailing whitespace the strategy
        # strips on purpose.
        from build_pairs import conversational_reference
        for key in ("chosen", "rejected"):
            with self.subTest(answer=key):
                built = self.rendered["prompt"] + self.rendered[key]
                reference = conversational_reference(self.source, key, self.tok)
                self.assertEqual(built, reference.rstrip())

    def test_tool_calls_and_results_appear_exactly_once(self):
        whole = self.rendered["prompt"] + self.rendered["chosen"]
        self.assertEqual(whole.count("<tool_call>"), 1)
        self.assertEqual(whole.count('<|im_start|>tool'), 1)

    def test_tools_are_rendered_into_the_prompt(self):
        self.assertIn("<|tools|>1<|/tools|>", self.rendered["prompt"])


class RenderingRefusals(unittest.TestCase):
    """What the renderer refuses rather than guesses at."""

    def setUp(self):
        from build_pairs import render_dpo_strings
        self.render = render_dpo_strings
        self.tok = _StubTokenizer()

    def test_two_final_targets_are_refused(self):
        record = _conversational()
        record["chosen"] = [{"role": "assistant", "content": "one"},
                            {"role": "assistant", "content": "two"}]
        with self.assertRaises(ValueError):
            self.render(record, self.tok)

    def test_a_missing_target_is_refused(self):
        record = _conversational()
        record["rejected"] = []
        with self.assertRaises(ValueError):
            self.render(record, self.tok)

    def test_an_empty_answer_is_refused(self):
        record = _conversational()
        record["chosen"] = [{"role": "assistant", "content": "   "}]
        with self.assertRaises(ValueError):
            self.render(record, self.tok)

    def test_a_non_assistant_target_is_refused(self):
        record = _conversational()
        record["chosen"] = [{"role": "user", "content": "not an answer"}]
        with self.assertRaises(ValueError):
            self.render(record, self.tok)

    def test_a_prefix_that_already_answers_is_refused(self):
        record = _conversational()
        record["prompt"] = list(record["prompt"]) + [
            {"role": "assistant", "content": "already answered"}]
        with self.assertRaises(ValueError):
            self.render(record, self.tok)
