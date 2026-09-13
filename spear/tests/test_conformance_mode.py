"""A turn that asks whether code conforms is shaped, not merely filtered.

The failing control listed the structure registry, read nine unrelated
structures, and rendered them as "Validated Structures (PASS/ACCEPTABLE)"
without opening a source file. No sentence in it was a verdict, so no guard
touched it, and it read as an assessment. The record appended here is what
says, deterministically, what the turn did and did not look at.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import conformance_mode
import standard_answer_policy as sap
from conformance_guard import ClauseLedger
from conformance_mode import CodeReadLedger, is_conformance_turn, record, render

SID, REV = "ANSI-VITA-49.2", "2017-R2024"
FETCH = {"unit": {"section": "8.4.1.1", "text": "Rule 8.4.1.1-2: only one bit"}}


def clauses(*payloads):
    found = ClauseLedger()

    for payload in payloads:
        found.observe(payload)

    return found


class Detection(unittest.TestCase):
    def test_the_two_questions_that_started_this(self):
        for question in ("can you validate the implementation ?",
                         "can you validate the ACK implementation against "
                         "VITA49.2 ?"):
            self.assertTrue(is_conformance_turn(question), question)

    def test_a_question_about_the_standard_alone_is_not_one(self):
        for question in ("How should the ACK be managed when using VITA49.2 "
                         "Commands packets?",
                         "what does Rule 8.4.1.1-3 require?",
                         "validate the config file"):
            self.assertFalse(is_conformance_turn(question), question)

    def test_french(self):
        self.assertTrue(is_conformance_turn("valide l'implémentation"))
        self.assertTrue(is_conformance_turn("le code est-il conforme ?"))


class Ledger(unittest.TestCase):
    def test_files_come_from_the_command_and_only_from_it(self):
        """A path printed by a search is a path the turn has seen the name
        of. Crediting it made "source files read" list the whole tree."""
        found = CodeReadLedger()
        found.observe("bash", {"command": "sed -n '592,670p' src/command/"
                                          "command_wire.c"},
                      "CmdWireStatus cmdWireEncodeAck(...)")
        found.observe("bash", {"command": "grep -rn EncodeAck src"},
                      "src/command/cmdlink.c:272: cmdWireEncodeAck(&rep")

        self.assertEqual(found.files, {"src/command/command_wire.c"})
        self.assertTrue(found.read_anything())

    def test_standard_tools_are_not_code_reads(self):
        found = CodeReadLedger()
        found.observe("standard.fetch", {"source_id": "std-1"}, "src/x.c")

        self.assertFalse(found.read_anything())


class Record(unittest.TestCase):
    def test_no_code_read_is_the_first_line(self):
        text = render("## Validated Structures\n1. bfd-04f0… PASS",
                      clauses(FETCH), CodeReadLedger(), standard_id=SID,
                      revision=REV)

        self.assertTrue(text.startswith("No source file was read this turn"))
        self.assertIn("Verdict: none can be issued — no source file was read",
                      text)
        self.assertIn("Clauses read this turn: §8.4.1.1 (1)", text)

    def test_cited_but_never_read_is_named(self):
        code = CodeReadLedger().observe("bash", {"command": "cat src/a.c"})
        text = record("The encoder satisfies Rule 8.3.1.5-4 and Rule "
                      "8.4.1.1-2.", clauses(FETCH), code)

        self.assertIn("cited but never read: §8.3.1.5", text)
        self.assertNotIn("§8.4.1.1 —", text)

    def test_read_but_not_assessed_is_named(self):
        code = CodeReadLedger().observe("bash", {"command": "cat src/a.c"})
        text = record("Nothing cited.", clauses(FETCH, {"section": "8.2"}),
                      code)

        self.assertIn("read but not assessed: §8.2, §8.4.1.1", text)

    def test_the_record_carries_no_provenance_to_fabricate(self):
        text = record("x", clauses(FETCH), CodeReadLedger())

        self.assertNotIn("std-", text)
        self.assertNotIn("://", text)


class Policy(unittest.TestCase):
    """The record reaches the reader through finalize, and only on a
    conformance turn."""

    def run_turn(self, question, *, code_call=None):
        policy = sap.policy_for({"standard_id": SID, "revision": REV},
                                question)
        policy.observe_tool_result(
            "standard.fetch",
            '{"unit": {"section": "8.4.1.1", "text": "Rule 8.4.1.1-2"}}')

        if code_call:
            policy.observe_code_read("bash", {"command": code_call},
                                     "int main(void)")

        return policy, policy.finalize("Rule 8.4.1.1-2 is satisfied.",
                                       rounds=3)

    def test_a_conformance_turn_gets_the_record(self):
        policy, text = self.run_turn("validate the implementation",
                                     code_call="cat src/command/cmdlink.c")

        self.assertTrue(policy.conformance_turn)
        self.assertIn("ASSESSMENT RECORD", text)
        self.assertIn("Source files read: src/command/cmdlink.c (1)", text)
        self.assertTrue(text.startswith("Rule 8.4.1.1-2 is satisfied."))

    def test_a_conformance_turn_that_read_no_code_says_so_first(self):
        policy, text = self.run_turn("is the encoder compliant?")

        self.assertTrue(text.startswith("No source file was read this turn"))
        self.assertEqual(policy.trace()["code_read"], [])

    def test_any_other_bound_turn_is_untouched(self):
        policy, text = self.run_turn("How should the ACK be managed when "
                                     "using VITA49.2 Commands packets?",
                                     code_call="cat src/x.c")

        self.assertFalse(policy.conformance_turn)
        self.assertEqual(text, "Rule 8.4.1.1-2 is satisfied.")

    def test_the_format_rule_is_a_request_not_a_dependency(self):
        # Whatever the model does with FORMAT_RULE, the record is the same.
        self.assertIn("one row per clause", conformance_mode.FORMAT_RULE)


class PathsAreTakenWhole(unittest.TestCase):
    """"Source files read" listed a mangled path, two files the turn had only
    seen the NAME of, and none of the files it had actually read."""

    def files(self, command):
        return CodeReadLedger().observe("bash", {"command": command}).files

    def test_an_absolute_path_is_taken_whole(self):
        self.assertEqual(
            self.files("sed -n '175,200p' /home/operator/work/customer/"
                       "acme/proj492/src/command/command.h"),
            {"/home/operator/work/customer/acme/proj492/src/command/"
             "command.h"})

    def test_a_dot_in_a_directory_name_does_not_cut_the_path(self):
        found = self.files("cat build/CMakeFiles/3.28.3/CompilerIdC/x.c")

        self.assertEqual(found, {"build/CMakeFiles/3.28.3/CompilerIdC/x.c"})

    def test_find_and_ls_credit_nothing(self):
        for command in ("find . -name '*.c' -o -name '*.h'",
                        "ls -la src/command/",
                        "find . -name '*.c' | head -20"):
            with self.subTest(command=command):
                self.assertEqual(self.files(command), set(), command)

    def test_a_pipeline_credits_the_stage_that_reads(self):
        self.assertEqual(
            self.files("sed -n '1,50p' src/a.c | grep -n Ack"),
            {"src/a.c"})

    def test_a_result_never_credits_a_file(self):
        """grep -rn prints the tree; being printed is not being read."""
        found = CodeReadLedger().observe(
            "bash", {"command": "grep -rln Ack src"},
            "src/command/command_wire.c\nsrc/apps/v492c-cmd/cmd.c").files

        self.assertEqual(found, set())


if __name__ == "__main__":
    unittest.main()


class ClauseAccounting(unittest.TestCase):
    """Naming a clause is not addressing it.

    The first version accepted the clause appearing anywhere in the answer,
    and passed a turn that had restated twelve rules and implemented one --
    restating them is precisely what the model does.
    """

    CARRIED = ["8.2.1", "8.3.1.4", "8.4.1.1"]

    def test_a_clause_is_addressed_when_the_answer_says_where(self):
        answer = ("Rule 8.4.1.1-2 is now satisfied in command_wire.c:648. "
                  "Rule 8.3.1.4-1 is handled by cmdlink.c line 277.")

        self.assertEqual(
            conformance_mode.unaddressed_clauses(answer, self.CARRIED),
            ["8.2.1"])

    def test_restating_the_rules_addresses_none_of_them(self):
        answer = ("Rule 8.4.1.1-2 requires exactly one bit. Rule 8.3.1.4-1 "
                  "describes NACK-only. §8.2.1 lists three variants.")

        self.assertEqual(
            conformance_mode.unaddressed_clauses(answer, self.CARRIED),
            self.CARRIED)

    def test_a_file_in_another_sentence_does_not_count(self):
        answer = ("Rule 8.4.1.1-2 requires exactly one bit.\n"
                  "Separately, command_wire.c was reformatted.")

        self.assertIn("8.4.1.1",
                      conformance_mode.unaddressed_clauses(answer, self.CARRIED))

    def test_nothing_carried_asks_nothing(self):
        self.assertEqual(conformance_mode.unaddressed_clauses("x", []), [])

    def test_the_demand_counts_both_sides_and_forbids_restating(self):
        demand = conformance_mode.clause_demand(["8.2.1"], self.CARRIED)

        self.assertIn("accounts for 2 of them", demand)
        self.assertIn("§8.2.1", demand)
        self.assertIn("Do not restate the rule", demand)


class ARuleIsNotItsSection(unittest.TestCase):
    """The same eight of eleven came back every time.

    The guard that reads citations normalises "8.4.1.1-2" to its section, so
    a carried RULE could never be matched by anything — and a turn that had
    written "Rule 8.4.1.1-2 is now met at command_wire.c:638" was told the
    clause was unaddressed, and sent to the standard to look up what it had
    just satisfied.
    """

    def test_a_cited_rule_accounts_for_itself(self):
        self.assertEqual(conformance_mode.unaddressed_clauses(
            "Rule 8.4.1.1-2 is now met at src/command/command_wire.c:638.",
            ["8.4.1.1-2"]), [])

    def test_one_rule_does_not_speak_for_its_neighbour(self):
        """Otherwise a turn that did step one would escape being asked for
        the rest — the very failure this check exists to catch."""

        self.assertEqual(conformance_mode.unaddressed_clauses(
            "Rule 8.4.1.1-2 is met at src/a.c:12.",
            ["8.4.1.1-2", "8.4.1.1-3"]), ["8.4.1.1-3"])

    def test_a_section_without_a_suffix_accounts_for_its_rules(self):
        self.assertEqual(conformance_mode.unaddressed_clauses(
            "src/a.c:12 satisfies §8.4.1.1 in full.",
            ["8.4.1.1-2", "8.4.1.1-3"]), [])

    def test_a_number_that_is_not_a_citation_counts_for_nothing(self):
        for text in ("version 8.4.1.1-2 of src/a.c",
                     "I read the standard and it says a lot about src/a.c"):
            with self.subTest(text=text):
                self.assertEqual(
                    conformance_mode.unaddressed_clauses(text, ["8.4.1.1-2"]),
                    ["8.4.1.1-2"])
