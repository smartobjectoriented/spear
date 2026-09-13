"""A task file with lettered sections is a list, and the turn is judged by it.

Handed the seven-section VITA 49.2 acknowledge work order, the model changed
the signature section A asks for, propagated it until the tree compiled, and
answered "The changes have been made successfully". C, D and E were untouched
— twice, at 60 rounds and at 120, so the budget was not the reason.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import work_order
from model_backend import ConversationMessage, TextBlock

ORDER = """# Task: VITA 49.2 Acknowledge handling

## Scope

Change only these files: src/command/command_wire.h, src/command/cmdlink.c

## A. command_wire.h — the subtype becomes a parameter

    typedef enum { ... } CmdAckKind;

## B. command_wire.c, cmdWireEncodeAck

Select it from `kind` instead.

## C. command_wire.c, cmdWireDecodeAck

Read bits 20 and 19.

## D. cmdlink.c, handle() — the behaviour change

Extract sendAck(link, &rep, kind, from).

## G. tests/test_command.c

Update the existing call sites.

## Style

Follow the file you are editing.
"""

READ = "[tool result]\n[file: doc/ack-task.md]\n" + ORDER


def conversation(*texts):
    return [ConversationMessage("user", (TextBlock(t),)) for t in texts]


class Parsing(unittest.TestCase):
    def test_only_lettered_sections_count(self):
        labels = [item["label"] for item in work_order.sections(ORDER)]

        self.assertEqual(labels, ["A", "B", "C", "D", "G"])

    def test_each_section_carries_the_files_it_names(self):
        found = {item["label"]: item["files"]
                 for item in work_order.sections(ORDER)}

        self.assertEqual(found["A"], ["command_wire.h"])
        self.assertEqual(found["D"], ["cmdlink.c"])
        self.assertEqual(found["G"], ["test_command.c"])

    def test_a_reference_file_is_not_a_work_order(self):
        self.assertEqual(work_order.find(conversation(
            "[tool result]\n[file: src/x.c]\nint main(void) { return 0; }")), "")

    def test_the_read_block_is_found_in_the_conversation(self):
        self.assertIn("## D. cmdlink.c",
                      work_order.find(conversation("hello", READ, "do it")))


class Tally(unittest.TestCase):
    # What both runs actually produced: A, B and G written, C and D not.
    CHANGED = ["src/command/command_wire.h", "src/command/command_wire.c",
               "tests/test_command.c"]

    def test_the_runs_that_started_this(self):
        missing = work_order.unaddressed(ORDER, self.CHANGED)

        self.assertEqual([item["label"] for item in missing], ["D"])

    def test_a_section_sharing_a_file_with_a_written_one_counts_as_touched(self):
        """B and C both name command_wire.c: a write to it clears both. The
        tally is about files, and says so rather than pretending otherwise."""
        missing = [item["label"]
                   for item in work_order.unaddressed(ORDER, self.CHANGED)]

        self.assertNotIn("C", missing)

    def test_the_demand_names_the_sections_and_refuses_the_shortcut(self):
        demand = work_order.demand(work_order.unaddressed(ORDER, self.CHANGED))

        self.assertIn("D (cmdlink.c)", demand)
        self.assertIn("section A's work, not the task", demand)
        # No checks in this order, so no instruction to run one.
        self.assertNotIn("RUN THAT COMMAND FIRST", demand)

    def test_the_record_is_silent_when_every_section_was_written_to(self):
        every = self.CHANGED + ["src/command/cmdlink.c"]

        self.assertEqual(work_order.record(ORDER, every), "")

    def test_the_record_names_what_was_skipped(self):
        text = work_order.record(ORDER, self.CHANGED)

        self.assertIn("WORK ORDER TALLY", text)
        self.assertIn("Sections with a write this turn: A, B, C, G", text)
        self.assertIn("never written to: D (cmdlink.c)", text)

    def test_a_turn_with_no_work_order_is_untouched(self):
        self.assertEqual(work_order.record("", ["src/x.c"]), "")


ACCEPTED = ORDER + """
## Acceptance

A: grep -q CMD_ACK_KIND_VALIDATION src/command/command_wire.h
B: grep -q CMD_ACK_KIND_VALIDATION src/command/command_wire.c
D: grep -q 'static void sendAck' src/command/cmdlink.c

## Style

Follow the file you are editing.
"""


class AcceptanceChecks(unittest.TestCase):
    """File granularity marked C done because B shares its file, and D done
    because a new parameter was threaded through cmdlink.c. A section knows
    what finishing it looks like; the work order is where that belongs."""

    def test_the_checks_are_parsed_and_the_style_section_is_not(self):
        found = work_order.acceptance(ACCEPTED)

        self.assertEqual([item["label"] for item in found], ["A", "B", "D"])
        self.assertTrue(found[2]["check"].startswith("grep -q 'static void"))

    def test_an_order_without_them_yields_none(self):
        self.assertEqual(work_order.acceptance(ORDER), [])

    def test_unmet_runs_each_check(self):
        seen = []

        def run(command):
            seen.append(command)
            return "command_wire.h" in command

        missing = work_order.unmet(ACCEPTED, run)

        self.assertEqual(len(seen), 3)
        self.assertEqual([item["label"] for item in missing], ["B", "D"])

    def test_the_demand_quotes_the_check_that_fails(self):
        missing = work_order.unmet(ACCEPTED, lambda c: False)

        demand = work_order.demand(missing)

        self.assertIn("sendAck", demand)
        self.assertIn("RUN THAT COMMAND FIRST", demand)
        self.assertIn("redefinitions", demand)

    def test_the_record_reports_checks_when_the_order_supplies_them(self):
        missing = work_order.unmet(ACCEPTED, lambda c: "command_wire" in c)
        text = work_order.record(ACCEPTED, [], missing)

        self.assertIn("running the order's own acceptance checks", text)
        self.assertIn("Sections whose check passes: A, B", text)
        self.assertIn("Sections whose check fails: D", text)

    def test_the_record_is_silent_when_every_check_passes(self):
        self.assertEqual(
            work_order.record(ACCEPTED, [], work_order.unmet(
                ACCEPTED, lambda c: True)), "")


if __name__ == "__main__":
    unittest.main()
