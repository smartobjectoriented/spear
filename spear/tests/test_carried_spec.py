"""The previous answer, carried into the turn that changes the code.

Two prompts is the normal shape here: "how should X work?", then "change the
code". The first answer is the specification -- written by the model from the
bound standard -- and the second turn was starting from nothing, reading
whatever a fresh search surfaced: glossary entries, context fields, never the
clause that governs the subject.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent_runtime import AgentContext, is_write_request

# Synthetic: what a previous turn concluded, in the words of a standard that
# does not exist. The shape is what matters -- a rule cited, a second fact
# beside it -- not which document it came from.
ANSWER = ("Exactly one of SelA/SelB/SelC shall be set (Rule 4.2.1.1-2). "
          "The deferral bit controls when a reply is produced at all.")


class TheRequestThatCarriesIt(unittest.TestCase):
    def test_the_users_own_second_prompt(self):
        self.assertTrue(is_write_request(
            "can you validate and make changes in the code accordingly"))

    def test_the_first_prompt_carries_nothing(self):
        self.assertFalse(is_write_request(
            "How should the ACK be managed when using VITA49.2 Commands packets?"))


class TheContextCarriesIt(unittest.TestCase):
    def test_the_field_exists_and_defaults_empty(self):
        self.assertIn("prior_answer", AgentContext.__dataclass_fields__)
        self.assertIn("prior_clauses", AgentContext.__dataclass_fields__)

    def test_the_rule_is_built_from_it(self):
        """The text goes into the system rules, which the terminal never
        echoes -- so this is the only place the wiring is visible at all."""
        import task_controller
        import inspect

        source = inspect.getsource(task_controller.TaskController.run)

        self.assertIn("WHAT THIS SESSION ALREADY ESTABLISHED", source)
        self.assertIn('getattr(context, "prior_answer"', source)
        self.assertIn("is_write_request(asked)", source)

    def test_it_is_not_carried_into_a_question(self):
        import task_controller
        import inspect

        source = inspect.getsource(task_controller.TaskController.run)
        head = source[source.index("WHAT THIS SESSION") - 400:
                      source.index("WHAT THIS SESSION")]

        self.assertIn("if answered and is_write_request(asked):", head)


if __name__ == "__main__":
    unittest.main()
