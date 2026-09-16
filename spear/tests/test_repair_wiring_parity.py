"""Both callers of the normative policy must supply the repair hook.

The hook used to be an attribute set after construction. The evaluation
harness set it; the interactive runtime never did. So every measurement ran
a repair path that no real session had, and the divergence was invisible
because both paths built the same object and only one finished configuring
it.

Making it a construction parameter is what stops that recurring: a caller
that omits it now says so, and these tests hold the two call sites together.
"""

from __future__ import annotations

import inspect
import unittest

import agent_runtime
import standard_answer_policy


BINDING = {"standard_id": "ACME-1", "revision": "R1",
           "corpus_manifest_sha256": "0" * 64,
           "index_fingerprint": "1" * 64,
           "retrieval_fingerprint": "2" * 64}


class TheHookIsPartOfConstruction(unittest.TestCase):

    def test_policy_for_accepts_it(self):
        signature = inspect.signature(standard_answer_policy.policy_for)

        self.assertIn("repair_ask", signature.parameters)

    def test_supplying_it_arms_the_repair(self):
        policy = standard_answer_policy.policy_for(
            BINDING, "does it apply?", repair_ask=lambda text: "rewritten")

        self.assertIsNotNone(policy.repair_ask)

    def test_omitting_it_still_means_no_repair(self):
        """A caller with no model to ask is a real case; it simply has to be
        said now rather than happening by default."""
        policy = standard_answer_policy.policy_for(BINDING, "does it apply?")

        self.assertIsNone(policy.repair_ask)


class BothCallSitesSupplyIt(unittest.TestCase):

    def test_the_evaluation_harness_supplies_it(self):
        import pathlib

        source = (pathlib.Path(agent_runtime.__file__).parent
                  / "eval" / "modeluse" / "harness.py").read_text()

        self.assertIn("repair_ask=repair_ask", source)

    def test_the_interactive_runtime_supplies_it(self):
        source = inspect.getsource(agent_runtime)

        self.assertIn("repair_ask=lambda text: self._repair_ask", source)

    def test_the_runtime_asks_without_tools(self):
        """A repair that could reach a tool would be a second turn wearing
        the first one's clothes."""
        source = inspect.getsource(agent_runtime.AgentRuntime._repair_ask)

        self.assertIn("use_tools=False", source)

    def test_a_failed_ask_withholds_rather_than_raising(self):
        source = inspect.getsource(agent_runtime.AgentRuntime._repair_ask)

        self.assertIn("except Exception", source)
        self.assertIn('return ""', source)


class TheRepairStaysOneShot(unittest.TestCase):

    def test_a_refused_repair_is_not_retried(self):
        import answer_repair
        import normative_claims

        asked = []

        def ask(text):
            asked.append(text)
            return "The provisions do not settle the question."

        found = normative_claims.NormativeEvidence()
        found.observe({"section": "5.2", "source_id": "std-" + "0" * 32,
                       "modality": "SHALL", "content_type": "REQUIREMENT",
                       "text": "Rule 5.2-1: A Widget shall report."})
        answer_repair.attempt(
            "does it apply?", "Rule 5.2-1 applies.", found,
            [{"kind": normative_claims.UNGROUNDED_IDENTIFIER,
              "identifier": "SelD"}], ask=ask)

        self.assertEqual(len(asked), 1)


if __name__ == "__main__":
    unittest.main()
