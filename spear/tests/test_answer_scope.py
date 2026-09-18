"""The current message decides what to go and look at.

Nothing here names a real standard or a real codebase: what is being tested is
that a conversation's subject does not become a question's subject.
"""

import unittest

import answer_scope as scope
import standard_query_expansion as expansion
from tool_registry import ToolCategory


class CurrentTurn(unittest.TestCase):

    def test_a_question_about_the_document_is_normative(self):
        self.assertEqual(scope.NORMATIVE,
                         scope.of("What does the standard require for XYZ?"))

    def test_a_bound_session_reads_a_bare_question_as_normative(self):
        # It names no subject at all. With a document bound, that document is
        # what the session is for, which is how the normative policy already
        # decides it is active.
        self.assertEqual(
            scope.NORMATIVE,
            scope.of("How should XYZ be managed?", standard_bound=True))

    def test_the_same_bare_question_is_general_with_nothing_bound(self):
        self.assertEqual(scope.GENERAL, scope.of("How should XYZ be managed?"))

    def test_naming_the_local_work_is_an_implementation_turn(self):
        self.assertEqual(scope.IMPLEMENTATION,
                         scope.of("How does our code do this?"))

    def test_asking_both_is_mixed(self):
        self.assertEqual(
            scope.MIXED,
            scope.of("Does our implementation satisfy the standard?"))

    def test_asking_for_a_change_is_implementation_work(self):
        self.assertEqual(scope.IMPLEMENTATION,
                         scope.of("Update the parser to handle this."))


class HistoryInterpretsButDoesNotWiden(unittest.TestCase):

    def test_implementation_history_does_not_widen_a_normative_question(self):
        self.assertEqual(
            scope.NORMATIVE,
            scope.of("What does the standard require in general?",
                     prior=scope.IMPLEMENTATION, standard_bound=True))

    def test_a_self_contained_question_ignores_the_conversation(self):
        self.assertEqual(
            scope.NORMATIVE,
            scope.of("How should XYZ be managed?", prior=scope.IMPLEMENTATION,
                     standard_bound=True))

    def test_a_message_that_points_back_is_about_what_it_points_at(self):
        self.assertEqual(
            scope.MIXED,
            scope.of("Does that comply with the standard?",
                     prior=scope.IMPLEMENTATION, standard_bound=True))

    def test_a_referent_with_no_document_named_inherits_outright(self):
        self.assertEqual(
            scope.IMPLEMENTATION,
            scope.of("What about its error handling?",
                     prior=scope.IMPLEMENTATION, standard_bound=True))

    def test_a_referent_inherits_nothing_when_nothing_preceded_it(self):
        self.assertEqual(scope.NORMATIVE,
                         scope.of("What about that?", standard_bound=True))


class WhatIsOffered(unittest.TestCase):

    def test_a_normative_turn_is_not_offered_the_local_tools(self):
        self.assertTrue(scope.withholds_local_tools(
            scope.NORMATIVE, standard_bound=True))

    def test_every_other_scope_keeps_them(self):
        for other in (scope.IMPLEMENTATION, scope.MIXED, scope.GENERAL):
            with self.subTest(scope=other):
                self.assertFalse(scope.withholds_local_tools(
                    other, standard_bound=True))

    def test_nothing_is_withheld_when_no_document_is_bound(self):
        self.assertFalse(scope.withholds_local_tools(
            scope.NORMATIVE, standard_bound=False))

    def test_the_withheld_tools_are_named_by_category(self):
        # By what a tool IS, so one added later is covered without being
        # remembered here.
        self.assertIn(ToolCategory.COMMAND, scope.LOCAL_CATEGORIES)
        self.assertIn(ToolCategory.FILE_WRITE, scope.LOCAL_CATEGORIES)
        self.assertNotIn(ToolCategory.RETRIEVAL, scope.LOCAL_CATEGORIES)


class QueryExpansionReadsThisTurnOnly(unittest.TestCase):
    """The boundary with corpus-native expansion."""

    VOCABULARY = {"xyz": 8, "xyzv": 20, "xyzx": 18}

    def test_a_shorthand_only_in_the_conversation_does_not_expand(self):
        # The stem is in the history, not in the question. Expanding on it
        # would let an earlier subject steer this turn's retrieval.
        terms, why = expansion.expand(
            ("what", "does", "the", "standard", "require"),
            self.VOCABULARY, self.VOCABULARY, 500,
            query="What does the standard require?")

        self.assertEqual({}, why)
        self.assertNotIn("xyzv", terms)

    def test_a_shorthand_in_this_question_still_expands(self):
        terms, why = expansion.expand(
            ("what", "about", "xyz"), self.VOCABULARY, self.VOCABULARY, 500,
            query="What about XYZ?")

        self.assertEqual(("xyzv", "xyzx"), why["xyz"])
        self.assertIn("xyzv", terms)


if __name__ == "__main__":
    unittest.main()
