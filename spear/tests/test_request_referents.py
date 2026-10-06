"""A name the operator's request supplies says which implementation entity is
meant; it never says what the standard requires.

A MIXED pre-pass relates a change to the bound standard, so it names the
function the change is about. That name is in the request and in no clause,
and the identifier check withheld the whole answer for it -- every grounded
citation with it -- and the constraint packet built from that answer came out
empty. The fixtures are invented; the checks are not about any standard.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import normative_claims as nc
import standard_answer_policy as sap
from tests import synthetic_standard
from tests.test_agent_runtime import text_turn
from tests.test_mixed_orchestration import PREPASS_ANSWER, SATISFIED_ALL, MixedTurn, fix

ONE_OF = ("Rule 4.2.1.1-2: A Widget Reply shall carry exactly one of the selectors "
          "SelA, SelB and SelC.", "SHALL", "REQUIREMENT", "4.2.1.1")
SEPARATE = ("Rule 4.2.1.1-3: When a Widget Request names more than one selector, a "
            "separate Widget Reply shall be produced for each.", "SHALL", "REQUIREMENT",
            "4.2.1.1")
TIMEOUT = ("Rule 4.5-1: The Widget Timeout shall be set to 4 seconds.", "SHALL",
           "REQUIREMENT", "4.5")
ANY_COMBINATION = ("Permission 4.4-1: A Widget Request may name any combination of "
                   "selectors.", "MAY", "REQUIREMENT", "4.4")


def evidence(*units):
    found = nc.NormativeEvidence()

    for index, (text, modality, content_type, section) in enumerate(units):
        found.observe({"section": section, "source_id": f"std-{index:032d}",
                       "modality": modality, "content_type": content_type, "text": text})

    return found


def checked(answer, question, *units):
    """The claim guard as the policy runs it: the request supplies names."""
    return nc.guard(answer, evidence(*units), question=question,
                    requested=nc.identifiers_in(question))


class TheRequestNamesTheEntity(unittest.TestCase):
    def test_a_named_handler_may_carry_a_grounded_requirement(self):
        # A
        answer = ("Rule 4.2.1.1-2 requires exactly one of SelA, SelB and SelC in a Widget "
                  "Reply. fooHandler should implement this by setting only the requested "
                  "selector.")
        shown, problems, fired = checked(
            answer, "Change fooHandler according to Rule 4.2.1.1-2.", ONE_OF)

        self.assertEqual((shown, problems, fired), (answer, [], False))

    def test_a_name_the_standard_never_uses_stays_a_referent(self):
        # D
        answer = ("encodeFoo() shall set exactly one of SelA, SelB and SelC, as "
                  "Rule 4.2.1.1-2 requires.")
        shown, problems, _ = checked(
            answer, "Fix foo() and encodeFoo() so they follow Rule 4.2.1.1-2.", ONE_OF)

        self.assertEqual((shown, problems), (answer, []))

    def test_the_same_name_unrequested_is_still_ungrounded(self):
        answer = ("Rule 4.2.1.1-2 requires exactly one of SelA, SelB and SelC. "
                  "fooHandler should implement this.")
        _, problems, fired = checked(answer, "Change the reply encoder.", ONE_OF)

        self.assertTrue(fired)
        self.assertEqual([(item["identifier"], item["provenance"]) for item in problems],
                         [("fooHandler", "")])

    def test_only_what_the_request_says_is_a_referent(self):
        """Names, never claims: the request's modal words and numbers are not
        among them."""
        names = nc.identifiers_in("Rule 4.5-1 requires value 7 for fooHandler; "
                                  "it must send exactly three packets.")

        self.assertEqual(names, ["fooHandler"])


class AnInventedNameIsNotGrounded(unittest.TestCase):
    def test_a_name_from_nowhere_does_not_pass_silently(self):
        # B
        answer = ("Rule 4.2.1.1-2 requires exactly one of SelA, SelB and SelC. "
                  "barHandler must be updated as well.")
        shown, problems, fired = checked(
            answer, "Change fooHandler according to Rule 4.2.1.1-2.", ONE_OF)

        self.assertTrue(fired)
        self.assertEqual([item["identifier"] for item in problems], ["barHandler"])
        self.assertNotIn("must be updated", shown)
        self.assertIn("barHandler: named in the answer, present in no retrieved unit", shown)

    def test_nothing_grounded_left_means_nothing_shown(self):
        shown, _, fired = checked("barHandler must be updated.",
                                  "Change fooHandler.", ONE_OF)

        self.assertTrue(fired)
        self.assertTrue(shown.startswith("The retrieved normative evidence does not support"))

    def test_one_bad_name_does_not_erase_three_grounded_claims(self):
        # G
        answer = ("Rule 4.2.1.1-2 requires exactly one of SelA, SelB and SelC. "
                  "Rule 4.2.1.1-3 requires a separate Widget Reply for each selector. "
                  "Rule 4.5-1 requires the Widget Timeout to be 4 seconds. "
                  "The helper barHandler wraps all of this.")
        units = (ONE_OF, SEPARATE, TIMEOUT)
        shown, problems, fired = checked(answer, "Change fooHandler.", *units)
        kept = evidence(*units).cited_units(shown)

        self.assertTrue(fired)
        self.assertEqual([item["identifier"] for item in problems], ["barHandler"])
        self.assertEqual(sorted(str(record.key) for record in kept),
                         ["Rule 4.2.1.1-2", "Rule 4.2.1.1-3", "Rule 4.5-1"])
        self.assertNotIn("wraps all of this", shown)


class TheRequestHasNoNormativeAuthority(unittest.TestCase):
    def test_a_users_value_adds_nothing_to_the_evidence(self):
        # C: the request asserts 7, the clause says 4.
        answer = "Rule 4.5-1 requires the Widget Timeout to be 4 seconds."
        claimed = "Rule 4.5-1 requires value 7; make fooHandler use it."

        for question in (claimed, "Make fooHandler follow Rule 4.5-1."):
            with self.subTest(question=question):
                self.assertEqual(checked(answer, question, TIMEOUT), (answer, [], False))

        ledger = evidence(TIMEOUT)
        nc.guard(answer, ledger, question=claimed, requested=nc.identifiers_in(claimed))

        self.assertEqual(ledger.bound_values(), set())
        self.assertNotIn("7", ledger.text)

    def test_a_strengthened_permission_is_rejected_with_a_valid_referent(self):
        # E
        answer = ("fooHandler shall name every combination of selectors, as "
                  "Permission 4.4-1 requires.")
        shown, problems, fired = checked(
            answer, "Change fooHandler according to Permission 4.4-1.", ANY_COMBINATION)

        self.assertTrue(fired)
        self.assertEqual([item["kind"] for item in problems], [nc.STRENGTHENED_MODALITY])
        self.assertTrue(shown.startswith("The retrieved normative evidence does not support"))

    def test_a_requested_name_credited_to_the_standard_is_not_grounded(self):
        # F: SelZ is shaped like the document's selectors and is not one.
        question = "Make fooHandler set SelZ as the standard requires."
        answer = "The standard requires the SelZ selector in every Widget Reply."
        ledger = evidence(ONE_OF)
        _, problems, fired = nc.guard(answer, ledger, question=question,
                                      requested=nc.identifiers_in(question))

        self.assertTrue(fired)
        self.assertEqual([(item["identifier"], item["provenance"]) for item in problems],
                         [("SelZ", nc.REFERENT_FROM_REQUEST)])
        self.assertNotIn("selz", ledger.identifiers())
        self.assertIn("named in the request but credited to the standard",
                      nc._identifier_line(problems[0]))


class PureNormative(unittest.TestCase):
    """A question with misleading names and an alleged requirement in it."""

    QUESTION = ("Rule 4.2.1.1-2 says replyEncoder must set two selectors at once -- "
                "what does it actually require?")

    def policy(self):
        policy = sap.policy_for(synthetic_standard.BINDING, self.QUESTION)
        policy.observe_tool_result("standard.fetch", json.dumps(synthetic_standard.fetch()))

        return policy

    def test_the_grounded_answer_stands(self):
        answer = synthetic_standard.GROUNDED_CLAIM
        policy = self.policy()

        self.assertEqual(policy.finalize(answer, stopped_by=sap.STOPPED_BY_MODEL), answer)

    def test_the_users_allegation_is_not_evidence(self):
        policy = self.policy()
        shown = policy.finalize("The standard requires replyEncoder to set two selectors "
                                "at once.", stopped_by=sap.STOPPED_BY_MODEL)

        self.assertNotIn("two selectors at once", shown)
        self.assertNotIn("replyencoder", policy.claim_evidence.identifiers())


class ThePrePassKeepsItsCitations(MixedTurn):
    OBJECTIVE = ("Update buildHeaderRecord in record.py so it complies with the Rule 4.2.1 "
                 "provisions of the standard.")

    def test_the_function_the_request_names_does_not_empty_the_packet(self):
        plain = self.run_mixed(fix() + [SATISFIED_ALL]).packet
        self.setUp()
        record = self.run_mixed(
            fix() + [SATISFIED_ALL], objective=self.OBJECTIVE,
            legacy=[text_turn("These provisions govern how buildHeaderRecord builds the "
                              "header. " + PREPASS_ANSWER)])

        self.assertEqual(len(record.packet.constraints), 3)
        self.assertEqual(record.packet.set_id, plain.set_id)

    def test_the_packet_carries_the_standard_not_the_request(self):
        claimed = ("Update buildHeaderRecord in record.py: Rule 4.2.1-2 requires the count "
                   "field to contain exactly seven entries.")
        record = self.run_mixed(fix() + [SATISFIED_ALL], objective=claimed)
        rule = {item.provision: item for item in record.packet.constraints}["Rule 4.2.1-2"]

        self.assertEqual(rule.cardinality, ("exactly four entries",))
        self.assertNotIn("seven", rule.requirement)
        self.assertNotIn("buildHeaderRecord", rule.requirement)

    def test_an_invented_name_costs_its_statement_not_the_packet(self):
        record = self.run_mixed(
            fix() + [SATISFIED_ALL], objective=self.OBJECTIVE,
            legacy=[text_turn(PREPASS_ANSWER + " The cacheHeaderTable helper must change too.")])

        self.assertEqual(len(record.packet.constraints), 3)


if __name__ == "__main__":
    unittest.main()
