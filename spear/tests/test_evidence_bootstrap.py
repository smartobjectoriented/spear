"""When the orchestration layer fetches evidence the model declined to fetch.

Two supported controls answer a bound normative question at round one with no
tool call at all, identically across three server processes, while the evidence
they need is present and reachable. Instructing the model to look has already
been measured as ineffective, so the harness makes one deterministic call and
resumes. These tests pin when that fires, what it calls, and that it cannot
become a retry loop or a second guess at the user's question.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import evidence_bootstrap as bootstrap


class TriggerPredicate(unittest.TestCase):
    def test_a_bound_normative_zero_tool_answer_triggers(self):
        self.assertTrue(bootstrap.should_bootstrap(
            "List the word number and bit range for Coarse Time", [],
            answer="I cannot find that standard."))

    def test_existing_model_tool_use_prevents_it(self):
        self.assertFalse(bootstrap.should_bootstrap(
            "List the word number and bit range for Coarse Time",
            [{"tool": "standard.search"}], answer="anything"))

    def test_any_standard_tool_counts_as_having_looked(self):
        for name in bootstrap.STANDARD_TOOLS:
            self.assertFalse(bootstrap.should_bootstrap(
                "bit range of the Status Word", [{"tool": name}],
                answer="anything"), name)

    def test_it_fires_at_most_once(self):
        self.assertFalse(bootstrap.should_bootstrap(
            "bit range of the Status Word", [], already_fired=True,
            answer="still nothing"))

    def test_the_binding_decides_what_is_normative_now(self):
        """The filtering moved upstream: standard_scope.engages() decides
        whether a turn carries the binding at all, so a turn that HAS one asks
        something the standard settles — by construction.

        Deciding again here could only disagree by being narrower, and it did:
        bound to NIST-RS274NGC, "in RS274NGC, which G codes are in modal group
        1?" matched no word in the list, no retrieval was forced, and the
        answer came out of interp_array.cc instead of the specification."""
        self.assertTrue(bootstrap.is_normative_turn(
            "in RS274NGC, which G codes are in modal group 1?"))
        self.assertTrue(bootstrap.is_normative_turn("Hello, how are you?"))
        # …and with no binding, the vocabulary is all there is to go on.
        self.assertFalse(bootstrap.is_normative_turn(
            "Hello, how are you today?", bound=False))
        self.assertTrue(bootstrap.is_normative_turn(
            "bit range of the Status Word", bound=False))

    def test_an_unbound_session_does_not_trigger(self):
        self.assertFalse(bootstrap.should_bootstrap(
            "bit range of the Status Word", [], bound=False, answer="x"))

    def test_an_empty_answer_does_not_trigger(self):
        # Producing nothing is a different fault and must stay visible.
        self.assertFalse(bootstrap.should_bootstrap(
            "bit range of the Status Word", [], answer="   "))


class DeterministicRouting(unittest.TestCase):
    def test_an_explicit_structure_id_routes_to_get_structure(self):
        name, arguments = bootstrap.route(
            "Generate the complete struct for bfd-0ccd61a481e290d5 now")
        self.assertEqual(name, "standard.get_structure")
        self.assertEqual(arguments, {"definition_id": "bfd-0ccd61a481e290d5"})

    def test_a_prose_normative_query_routes_to_search(self):
        question = "Give the octet offset and length of every component."
        name, arguments = bootstrap.route(question)
        self.assertEqual(name, "standard.search")
        self.assertEqual(arguments, {"query": question})

    def test_the_search_query_is_the_user_question_verbatim(self):
        question = "  List the word number and bit range for Coarse Time.  "
        _, arguments = bootstrap.route(question)
        self.assertEqual(arguments["query"], question.strip())

    def test_routing_is_deterministic(self):
        question = "Give the octet offset of the Command Word"
        self.assertEqual(bootstrap.route(question), bootstrap.route(question))

    def test_an_invalid_looking_identifier_is_not_repaired(self):
        # A malformed id is left to the existing get_structure contract rather
        # than swapped for a guess.
        name, arguments = bootstrap.route("show me bfd-notavalidid please")
        self.assertEqual(name, "standard.search")

    def test_other_identifier_families_route_to_get_structure(self):
        for prefix in ("bfd", "fld", "pkg", "vgr", "bit"):
            name, arguments = bootstrap.route(f"describe {prefix}-1234abcd5678")
            self.assertEqual(name, "standard.get_structure", prefix)


class PolicyIsSeparateFromTheGuard(unittest.TestCase):
    def test_the_bootstrap_module_does_not_import_the_guard(self):
        source = (ROOT / "evidence_bootstrap.py").read_text()
        self.assertNotIn("evidence_guard", source)

    def test_the_guard_module_does_not_import_the_bootstrap(self):
        source = (ROOT / "evidence_guard.py").read_text()
        self.assertNotIn("evidence_bootstrap", source)

    def test_no_standard_tool_semantics_are_referenced(self):
        # The policy chooses a call; it never reimplements one.
        source = (ROOT / "evidence_bootstrap.py").read_text()
        self.assertNotIn("StandardToolService", source)
        self.assertNotIn("STANDARD_STORE", source)


if __name__ == "__main__":
    unittest.main()
