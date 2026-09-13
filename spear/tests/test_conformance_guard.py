"""What a bound answer may say about compliance, checked against what it read.

The failing control read the code, read nothing of the standard on its own,
and declared the implementation "correct and compliant with ANSI/VITA
49.2-2017 (R2024)". It is not. The sentence binds no range and cites nothing
fabricated, so every earlier guard let it through. The rule here is
entitlement, not truth: an affirmative verdict survives only when it names a
clause the turn actually read and does not quantify over the whole standard.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import conformance_guard
from conformance_guard import (
    UNREAD_CLAUSE, UNSCOPED_VERDICT, ClauseLedger, clauses_in, findings, guard,
)
from standard_schema import StandardBinding

import synthetic_standard

SID, REV = synthetic_standard.SID, synthetic_standard.REV

# What standard.fetch hands back, for a standard that does not exist: the
# clause text of one that does is not ours to carry. The shape is the real
# tool's and the semantics are the ones under test -- an exactly-one-of-three
# rule with a sibling that says what to do when more than one is asked for.
FETCH = synthetic_standard.fetch()
SEARCH = synthetic_standard.search()

THE_SENTENCE = (f"The implementation is correct and compliant with "
                f"{SID}-{REV}.")


def ledger(*payloads):
    found = ClauseLedger()

    for payload in payloads:
        found.observe(payload)

    return found


class Ledger(unittest.TestCase):
    """Exactly the clauses the tools put in front of the model."""

    def test_sections_come_from_the_payload_and_from_rule_identifiers(self):
        found = ledger(FETCH)

        self.assertIn("4.2.1.1", found.sections)

    def test_a_search_records_its_sections_and_its_snippets(self):
        found = ledger(SEARCH)

        self.assertIn("4.2", found.sections)

    def test_a_refusal_records_nothing(self):
        found = ledger({"error": "STRUCTURE_NOT_FOUND",
                        "detail": "see Rule 8.4-2"})

        # A refusal's detail is advice, not a clause the model read: the
        # ledger keys on the fields a retrieval carries text in, and "detail"
        # is not one of them. The policy never feeds a refusal in anyway.
        self.assertEqual(found.sections, set())

    def test_a_child_read_covers_its_parent_but_not_the_reverse(self):
        found = ledger(FETCH)

        self.assertTrue(found.covers("4.2.1.1"))
        self.assertTrue(found.covers("4.2"))
        self.assertFalse(found.covers("4.2.1.1.9"))
        self.assertFalse(found.covers("8.3"))

    def test_clause_forms(self):
        self.assertEqual(clauses_in("per Rule 8.4.1-2 and Observation 8.2.1-3, "
                                    "see §8.3 and Table 8.4.1-1"),
                         {"8.4.1", "8.2.1", "8.3"})


class Verdicts(unittest.TestCase):
    def setUp(self):
        self.ledger = ledger(FETCH, SEARCH)

    def test_the_sentence_that_started_this(self):
        found = findings(THE_SENTENCE, self.ledger)

        self.assertEqual([item["kind"] for item in found], [UNSCOPED_VERDICT])

    def test_the_other_form_it_took(self):
        text = ("The implementation correctly follows the VITA 49.2 "
                "specification for ACK management, properly echoing CAM "
                "control bits.")

        self.assertEqual(findings(text, self.ledger)[0]["kind"],
                         UNSCOPED_VERDICT)

    def test_a_scoped_verdict_on_a_clause_that_was_read_survives(self):
        text = ("Rule 4.2.1.1-2 is satisfied: widgetEncodeReply sets exactly "
                "one of the three bits.")

        self.assertEqual(findings(text, self.ledger), [])

    def test_a_scoped_verdict_on_a_clause_never_read_is_unentitled(self):
        text = "The encoder correctly implements Rule 8.3.1.5-4."
        found = findings(text, self.ledger)

        self.assertEqual(found[0]["kind"], UNREAD_CLAUSE)
        self.assertEqual(found[0]["clauses"], ["8.3.1.5"])

    def test_a_global_subject_is_unscoped_whatever_it_cites(self):
        text = ("The implementation is compliant, see Rule 4.2.1.1-2 for the "
                "one-bit rule.")

        self.assertEqual(findings(text, self.ledger)[0]["kind"],
                         UNSCOPED_VERDICT)

    def test_a_negative_verdict_is_left_alone(self):
        for text in (f"The implementation is not compliant with {SID}.",
                     "This does not conform to the specification.",
                     "Rule 4.2.1.1-3 is violated: only one reply is sent."):
            self.assertEqual(findings(text, self.ledger), [], text)

    def test_an_answer_with_no_verdict_is_untouched(self):
        text = ("There are three variants of Widget Reply: SelA, SelB "
                "and SelC (§4.1.1). A request names them through the "
                "selector bits.")

        self.assertEqual(guard(text, self.ledger), (text, [], [], False))


class Replacement(unittest.TestCase):
    def test_the_verdict_is_replaced_with_the_record_of_what_was_read(self):
        answer = ("SelA and SelB are distinct selectors. " + THE_SENTENCE
                  + " See §4.2.1.1 for the one-selector rule.")
        cleaned, problems, replaced, fired = guard(
            answer, ledger(FETCH, SEARCH), standard_id=SID, revision=REV)

        self.assertTrue(fired)
        self.assertEqual(replaced, [THE_SENTENCE])
        self.assertNotIn("compliant", cleaned)
        self.assertIn("No compliance verdict is issued.", cleaned)
        self.assertIn("§4.2", cleaned)
        self.assertIn("§4.2.1.1", cleaned)
        self.assertIn(f"Conformance with {SID} {REV} as a whole was not "
                      f"established", cleaned)
        # The substance around it is kept, byte for byte.
        self.assertTrue(cleaned.startswith("SelA and SelB are distinct "
                                           "selectors."))
        self.assertTrue(cleaned.endswith("See §4.2.1.1 for the one-selector rule."))

    def test_nothing_read_says_so(self):
        cleaned, _, _, fired = guard(THE_SENTENCE, ClauseLedger(),
                                     standard_id=SID, revision=REV)

        self.assertTrue(fired)
        self.assertIn(f"no clause of {SID} {REV} was read",
                      cleaned)

    def test_a_second_unentitled_sentence_is_cut_not_repeated(self):
        answer = (THE_SENTENCE + " The code adheres to the standard "
                  "throughout.")
        cleaned, _, replaced, _ = guard(answer, ledger(FETCH))

        self.assertEqual(len(replaced), 2)
        self.assertEqual(cleaned.count("No compliance verdict"), 1)
        self.assertNotIn("adheres", cleaned)

    def test_the_replacement_carries_no_provenance_to_fabricate(self):
        cleaned, _, _, _ = guard(THE_SENTENCE, ledger(FETCH))

        self.assertNotIn("std-", cleaned)
        self.assertNotIn("://", cleaned)



class BoundStandardUrls(unittest.TestCase):
    """A web page naming the bound standard is not the bound standard.

    An unbound turn fetched file:///.../ref-vita492-standard.md, then
    https://standards.example/doc/vita492-standard/, and edited five source
    files against what it found.
    """

    BOUND = "https://standards.example/doc/vita492-standard/"

    def setUp(self):
        import rag_chat
        self.rag = rag_chat
        self.addCleanup(setattr, rag_chat, "STANDARD_ENGAGED_BEFORE",
                        rag_chat.STANDARD_ENGAGED_BEFORE)
        rag_chat.STANDARD_ENGAGED_BEFORE = True

        # The binding comes from the test, not from the operator's private
        # standards store. Reading that store made every assertion here depend
        # on a licensed PDF having been ingested on this machine: with an
        # empty store `active_binding()` is None, the refusal short-circuits,
        # and all three tests report on nothing -- two of them by passing.

        self.bind(StandardBinding("ANSI-VITA-49.2", "2017-R2024",
                                  "a" * 64, "b" * 64, "c" * 64, "extractor-1"))

    def bind(self, binding):
        patcher = patch.object(self.rag.STANDARD_OPERATOR, "active_binding",
                               return_value=binding)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_the_two_urls_that_started_this(self):
        for url in (self.BOUND,
                    "file:///opt/llm/spear/claude/ref-vita492-standard.md"):
            found = self.rag.standard_url_refusal(url)
            self.assertIn("standard.search", found, url)

    def test_the_refusal_names_the_standard_it_is_bound_to(self):
        found = self.rag.standard_url_refusal(self.BOUND)

        self.assertIn("ANSI-VITA-49.2 2017-R2024", found)
        self.assertIn("sha256", found)

    def test_an_unrelated_url_is_untouched(self):
        for url in ("https://cmake.org/cmake/help/latest/",
                    "https://man7.org/linux/man-pages/man2/sendto.2.html"):
            self.assertEqual(self.rag.standard_url_refusal(url), "", url)

    def test_nothing_is_refused_before_the_session_engages(self):
        self.rag.STANDARD_ENGAGED_BEFORE = False
        self.assertEqual(self.rag.standard_url_refusal(self.BOUND), "")

    def test_an_unbound_session_refuses_nothing(self):
        """No standard pinned is the normal state of a fresh checkout."""

        self.bind(None)
        self.assertEqual(self.rag.standard_url_refusal(self.BOUND), "")

class TheClearanceThatPassed(unittest.TestCase):
    """Verbatim from a run that "verified" the implementation without ever
    opening cmdWireEncodeAck, and cleared it."""

    def setUp(self):
        self.ledger = ClauseLedger()
        self.ledger.sections = {"8.3.1", "8.3.1.3", "3.3", "3.7.1"}

    def test_no_adaptations_are_needed(self):
        text = ("No adaptations are needed — the current implementation is "
                "compliant with ANSI-VITA-49.2.")

        self.assertEqual(findings(text, self.ledger)[0]["kind"],
                         UNSCOPED_VERDICT)

    def test_any_adjective_before_the_noun(self):
        for subject in ("the current implementation", "the whole codebase",
                        "this new converter", "our existing design",
                        "the present controllee"):
            text = f"{subject} is compliant."
            with self.subTest(subject=subject):
                self.assertEqual(findings(text, self.ledger)[0]["kind"],
                                 UNSCOPED_VERDICT)

    def test_the_standard_under_its_canonical_name(self):
        for name in ("VITA 49.2", "VITA49.2", "ANSI-VITA-49.2",
                     "ANSI/VITA 49.2", "ANSI-VITA-49.2-2017"):
            text = f"The encoder conforms to {name}."
            with self.subTest(name=name):
                self.assertTrue(findings(text, self.ledger), name)

    def test_the_adverb_after_the_verb(self):
        text = ("The codebase already implements the VITA49.2 ACK management "
                "correctly.")

        self.assertTrue(findings(text, self.ledger))

    def test_a_scoped_verdict_still_survives_all_of_this(self):
        led = ClauseLedger(); led.sections = {"4.2.1.1"}

        self.assertEqual(findings(
            "Rule 4.2.1.1-2 is satisfied: exactly one selector is set.",
            led), [])


class TheStandardsOwnName(unittest.TestCase):
    """Which standard counts comes from the binding, not from a pattern.

    The first version spelled VITA 49.2 into the regex, which made a guard
    that exists to catch an unscoped claim work for exactly one standard and
    pass the same sentence about any other.
    """

    VITA = {"standard_id": "ANSI-VITA-49.2", "revision": "2017-R2024"}
    ISO = {"standard_id": "ISO-26262", "revision": "2018"}

    def test_each_binding_recognises_only_its_own(self):
        vita = conformance_guard.bare_standard(self.VITA)
        iso = conformance_guard.bare_standard(self.ISO)

        self.assertTrue(vita.search("compliant with ANSI-VITA-49.2"))
        self.assertFalse(vita.search("compliant with ISO 26262"))
        self.assertTrue(iso.search("compliant with ISO 26262"))
        self.assertFalse(iso.search("compliant with ANSI-VITA-49.2"))

    def test_however_the_writer_punctuated_it(self):
        vita = conformance_guard.bare_standard(self.VITA)

        for spelling in ("ANSI-VITA-49.2", "ANSI/VITA 49.2", "VITA 49.2",
                         "vita-49.2"):
            with self.subTest(spelling=spelling):
                self.assertTrue(vita.search(f"conforms to {spelling}"),
                                spelling)

    def test_with_no_binding_no_name_can_match(self):
        none = conformance_guard.bare_standard(None)

        self.assertFalse(none.search("compliant with ANSI-VITA-49.2"))
        self.assertFalse(none.search("compliant with anything at all"))

    def test_a_binding_is_accepted_as_a_dict_or_an_object(self):
        """It travels as a dict through the runtime and as a StandardBinding
        in the operator plane; read as attributes only, a dict yielded
        nothing and the pattern matched nothing at all."""
        class Bound:
            standard_id = "ANSI-VITA-49.2"
            revision = "2017-R2024"

        self.assertTrue(conformance_guard.bare_standard(Bound()).search(
            "compliant with ANSI-VITA-49.2"))
        self.assertTrue(conformance_guard.bare_standard(self.VITA).search(
            "compliant with ANSI-VITA-49.2"))

    def test_a_short_initial_cannot_bind_on_noise(self):
        """Terms of three characters or fewer are initials; matching them
        would flag any sentence containing them."""
        vita = conformance_guard.bare_standard(self.VITA)

        self.assertFalse(vita.search("compliant with is"))


class ObjectOrAttribute(unittest.TestCase):
    """Naming the standard is a claim on the whole of it -- unless the name
    is there only to say which rule is meant."""

    BOUND = {"standard_id": "ANSI-VITA-49.2", "revision": "2017-R2024"}

    def setUp(self):
        self.ledger = ClauseLedger()
        self.ledger.sections = {"8.4.1.1", "8.2"}

    def kinds(self, sentence):
        return [item["kind"]
                for item in findings(sentence, self.ledger, self.BOUND)]

    def test_the_standard_as_the_object_of_the_verdict(self):
        """A cited rule does not narrow a claim made on the document."""
        self.assertEqual(
            self.kinds("The encoder conforms to ANSI-VITA-49.2, see "
                       "Rule 8.4.1.1-2."),
            [UNSCOPED_VERDICT])

    def test_the_standard_as_the_attribute_of_a_clause(self):
        for sentence in ("Rule 8.4.1.1-2 of ANSI-VITA-49.2 is satisfied by "
                         "cmdWireEncodeAck.",
                         "§8.4.1.1 of VITA 49.2 is satisfied.",
                         "Section 8.2 in ANSI-VITA-49.2 is satisfied."):
            with self.subTest(sentence=sentence):
                self.assertEqual(self.kinds(sentence), [], sentence)

    def test_a_scoped_verdict_naming_no_standard_is_untouched(self):
        self.assertEqual(
            self.kinds("Rule 8.4.1.1-2 is satisfied: exactly one bit is set."),
            [])

    def test_naming_it_with_no_clause_at_all_is_still_unscoped(self):
        self.assertEqual(
            self.kinds("The encoder is compliant with ANSI-VITA-49.2."),
            [UNSCOPED_VERDICT])

    def test_an_unread_clause_is_still_reported_as_such(self):
        self.assertEqual(
            self.kinds("Rule 9.9.9-1 of ANSI-VITA-49.2 is satisfied."),
            [UNREAD_CLAUSE])


if __name__ == "__main__":
    unittest.main()
