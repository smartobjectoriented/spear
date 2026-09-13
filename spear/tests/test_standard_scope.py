"""Which turns carry the bound standard.

The binding attached to every prompt of every session. Asked to download the
RS274/NGC G-code specification, the assistant spent eleven turns explaining
that the system is bound to ANSI-VITA-49.2 and not RS274/NGC. The word that
put it there was "standard", in "inject it in our RAG as a standard".
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import standard_scope                                          # noqa: E402
from standard_schema import StandardBinding                    # noqa: E402


def binding(standard_id="ANSI-VITA-49.2", revision="2017-R2024"):
    return StandardBinding(standard_id, revision, "a" * 64, "b" * 64,
                           "c" * 64, "extractor-1")


class EngagementTests(unittest.TestCase):
    def setUp(self):
        self.binding = binding()

    def assertEngages(self, question):
        self.assertTrue(standard_scope.engages(self.binding, question), question)

    def assertFree(self, question):
        self.assertFalse(standard_scope.engages(self.binding, question), question)

    def test_the_question_that_started_this_does_not_bind(self):
        self.assertFree("please get the complete Code-G pdf so that we can "
                        "inject it in our RAG as a standard.")

    def test_generic_english_does_not_bind(self):
        """standard, revision, rule, section are ordinary words. They are in
        evidence_bootstrap's list, which may stay wide because it only runs
        once a binding is in force; here they would BE the binding."""
        for question in ("index this pdf as a standard",
                         "which revision of buildroot are we on?",
                         "what is the rule for copyright headers?",
                         "move that section of the README",
                         "commit and push"):
            self.assertFree(question)

    def test_naming_the_standard_binds(self):
        for question in ("summarise the ANSI-VITA-49.2 conformance rules",
                         "what does VITA 49.2 say about it?",
                         "in vita-49.2, is that mandatory?",
                         "does 2017-R2024 change that?"):
            self.assertEngages(question)

    def test_the_subject_matter_binds_without_the_name(self):
        """A normative question rarely repeats the identifier. Missing one of
        these costs grounding on a licensed source, so they stay."""
        for question in ("what is the width of the reserved field?",
                         "how many octets does the packet header take?",
                         "which bits are reserved in that structure?",
                         "what is the encoding of that enum?"):
            self.assertEngages(question)

    def test_a_structure_identifier_binds_on_its_own(self):
        self.assertEngages("describe bfd-0123abcd89")
        self.assertEngages("what is in vgr-deadbeef12?")

    def test_no_binding_means_no_engagement(self):
        self.assertFalse(standard_scope.engages(None, "what about VITA 49.2?"))

    def test_an_empty_question_does_not_bind(self):
        self.assertFree("")
        self.assertFree("   ")


class IdentityTermTests(unittest.TestCase):
    def test_terms_come_from_the_binding_not_from_a_list(self):
        terms = standard_scope.identity_terms(binding("IEEE-1588", "2019"))
        self.assertIn("ieee-1588", terms)
        self.assertIn("1588", terms)
        self.assertIn("ieee 1588", terms)
        self.assertIn("2019", terms)

    def test_short_letter_only_parts_are_dropped(self):
        """Matching two or three bare letters would bind on noise."""
        terms = standard_scope.identity_terms(binding("ANSI-X-3", "R1"))
        self.assertNotIn("x", terms)
        self.assertIn("3", terms)          # a digit is specific enough

    def test_a_number_inside_a_longer_number_does_not_match(self):
        self.assertFalse(standard_scope.engages(
            binding(), "the offset is 149.25 metres".replace("offset", "value")))

    def test_another_standard_named_in_the_question_does_not_bind_this_one(self):
        self.assertFalse(standard_scope.engages(
            binding(), "get the RS274/NGC specification from NIST"))


if __name__ == "__main__":
    unittest.main()


class SessionStickinessTests(unittest.TestCase):
    """Once a session engages the standard, it stays engaged until /clear.

    "can you validate the implementation ?" ran free and called the code
    compliant on the strength of twelve greps. A stray "/quit" reached the
    model as a prompt and the unbound turn that followed fetched the
    specification from the public web and edited five source files.
    """

    def setUp(self):
        self.binding = binding()

    def engages(self, question, engaged_before):
        return standard_scope.engages(self.binding, question,
                                      engaged_before=engaged_before)

    def test_the_two_turns_that_started_this(self):
        for question in ("can you validate the implementation ?", "/quit"):
            self.assertFalse(self.engages(question, False), question)
            self.assertTrue(self.engages(question, True), question)

    def test_anything_at_all_stays_bound(self):
        for question in ("commit and push", "run the build", "et ça ?",
                         "rename the README"):
            self.assertTrue(self.engages(question, True), question)

    def test_stickiness_follows_a_binding_and_cannot_create_one(self):
        """The RS274/NGC failure stays fixed: engaged_before is set only by a
        turn that engaged on its own, and /clear puts it back."""
        for question in ("please get the complete Code-G pdf so that we can "
                         "inject it in our RAG as a standard.",
                         "commit and push", "/quit"):
            self.assertFalse(self.engages(question, False), question)

    def test_no_binding_is_still_no_engagement(self):
        self.assertFalse(standard_scope.engages(None, "anything",
                                                engaged_before=True))

    def test_an_empty_question_never_engages(self):
        self.assertFalse(self.engages("   ", True))


class ReadFileContextTests(unittest.TestCase):
    """A task file the operator !read names the standard; the typed line
    does not. "carry out the task in doc/ack-task.md" ran with no standard
    tools, on a task whose first rule was to ground every claim in them."""

    def setUp(self):
        self.binding = binding()

    def test_the_line_that_started_this(self):
        question = "carry out the task in doc/ack-task.md"
        task = ("[file: doc/ack-task.md]\n# Task\nThe bound standard is "
                "ANSI-VITA-49.2 2017-R2024. Every normative claim ...")

        self.assertFalse(standard_scope.engages(self.binding, question))
        self.assertTrue(standard_scope.engages(self.binding, question,
                                               context=task))

    def test_only_the_name_counts_in_context(self):
        c_file = ("[file: src/command/command_wire.c]\n"
                  "/* bit 26, Acknowledge packet; packet size in words */")

        self.assertFalse(standard_scope.engages(
            self.binding, "fix the typo in the header comment", context=c_file))

    def test_context_never_binds_without_a_binding(self):
        self.assertFalse(standard_scope.engages(
            None, "do it", context="ANSI-VITA-49.2 2017-R2024"))
