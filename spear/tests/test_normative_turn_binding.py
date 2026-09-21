"""A question the bound document settles has to reach the bound document.

Measured failure, from one real interactive session: a standard was bound, and
the operator asked how a structure the document defines works and how it
should be parsed. The turn made twenty-two tool calls, every one of them a
shell command against the local source tree. It called no normative tool. It
took the rules from the implementation's comments, and presented them as the
standard's.

Nothing downstream was broken. The opening retrieval, the withholding of local
tools, the evidence ledgers and the claim guards were all in place and all
switched off, because the turn never bound the standard at all: the question
carried no identity term and none of the subject nouns, so it ran as if no
document were bound.

So this is about the one decision upstream of all of them -- does THIS turn
carry the bound document -- and about the ordering it exists to enforce:

    authoritative evidence before local implementation evidence can support
    a normative claim.

Synthetic throughout. The structure names below are invented; a rule that
needed to know one document's vocabulary would be the same bug in a new place.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import answer_scope
import code_evidence
import evidence_bootstrap
import normative_claims
import standard_scope
from tool_registry import ToolCategory

#: A term this document defines and English does not.
TERM = "XCW"


def binding():
    return SimpleNamespace(standard_id="ACME-1234.5", revision="2019-R2023",
                           handle="std-0a1b2c3d4e5f")


def bound(question, **kw):
    return standard_scope.engages(binding(), question, **kw)


def scope(question, **kw):
    engaged = bound(question)

    return answer_scope.of(question, standard_bound=engaged, **kw), engaged


def tool(name, category):
    return SimpleNamespace(name=name, category=category)

TOOLS = (tool("standard.search", ToolCategory.RETRIEVAL),
         tool("standard.fetch", ToolCategory.RETRIEVAL),
         tool("bash", ToolCategory.COMMAND),
         tool("edit_file", ToolCategory.FILE_WRITE))


def offered(question):
    answer, engaged = scope(question)

    return {t.name for t in
            answer_scope.offered(TOOLS, answer, standard_bound=engaged)}


class AnExplanationOfAStructureGoesToTheDocument(unittest.TestCase):
    """CASE 1 -- the failure itself. A bound document and "explain how X
    works": the authoritative source is consulted before anything is said,
    and reading the source tree is not what the turn owes."""

    QUESTION = f"Explain how the different {TERM} work and how it should be parsed"

    def test_the_turn_binds_the_document(self):
        self.assertTrue(bound(self.QUESTION))

    def test_it_is_a_normative_question(self):
        self.assertEqual(scope(self.QUESTION)[0], answer_scope.NORMATIVE)

    def test_the_first_call_is_an_authoritative_one(self):
        name, _ = evidence_bootstrap.route(self.QUESTION)

        self.assertTrue(name.startswith("standard."))

    def test_the_local_tools_are_not_on_the_table(self):
        self.assertEqual(offered(self.QUESTION), {"standard.search",
                                                  "standard.fetch"})

    def test_an_answer_with_no_normative_call_is_sent_back_first(self):
        self.assertTrue(evidence_bootstrap.should_bootstrap(
            self.QUESTION,
            [{"tool": "bash", "arguments": {"command": f"grep -rn {TERM} src/"}}],
            answer=f"Each {TERM} is parsed in order."))

    def test_reading_no_code_at_all_is_a_complete_turn(self):
        """Code inspection is not required for this question. The turn is
        judged on whether it reached the document, not on how much of the
        repository it opened."""
        self.assertFalse(evidence_bootstrap.should_bootstrap(
            self.QUESTION,
            [{"tool": "standard.search", "arguments": {"query": TERM}}],
            answer=f"Each {TERM} is parsed in order."))

    def test_an_explanatory_question_is_recognised_as_such(self):
        self.assertTrue(
            standard_scope.asks_what_the_document_settles(self.QUESTION))


class AskingForSomethingToBeDoneBindsNothing(unittest.TestCase):
    """The boundary the case above must not cross. These are the shapes that
    bound a whole session to a document once before, over eleven turns of
    trying to download a PDF -- an action requested, not a meaning asked."""

    def test_fetching_a_document_is_not_a_question_about_it(self):
        for question in ("how do I download the ACME-1234 pdf",
                         "please get the complete ACME-1234.5 document",
                         "inject it in our corpus as a standard",
                         "add a --verbose flag to the CLI",
                         "fix the crash in handshake_accept"):
            with self.subTest(question=question):
                self.assertFalse(
                    standard_scope.asks_what_the_document_settles(question))


class HistoryDoesNotDecideAFreshQuestion(unittest.TestCase):
    """CASE 2 -- a self-contained question about the document, asked after a
    long stretch of implementation work. What was being done before is not
    what is being asked now."""

    QUESTION = "Explain what the standard says about record framing"

    def implementation_history(self):
        return answer_scope.of("fix the framing bug in our encoder",
                               standard_bound=True)

    def test_the_prior_turn_really_was_implementation_work(self):
        self.assertEqual(self.implementation_history(),
                         answer_scope.IMPLEMENTATION)

    def test_the_question_still_binds_the_document(self):
        self.assertTrue(bound(self.QUESTION))

    def test_and_is_still_normative(self):
        self.assertEqual(
            answer_scope.of(self.QUESTION, prior=self.implementation_history(),
                            standard_bound=True),
            answer_scope.NORMATIVE)

    def test_so_the_code_tools_stay_withheld(self):
        self.assertEqual(
            {t.name for t in answer_scope.offered(
                TOOLS,
                answer_scope.of(self.QUESTION,
                                prior=self.implementation_history(),
                                standard_bound=True),
                standard_bound=True)},
            {"standard.search", "standard.fetch"})


class AQuestionAboutOurOwnCodeIsAnsweredFromOurOwnCode(unittest.TestCase):
    """CASE 3 -- the symmetric mistake. "How does OUR parser handle X" is a
    question about this repository; forcing a normative retrieval on it would
    be the same overreach in the other direction."""

    QUESTION = f"How does our parser handle {TERM}?"

    def test_it_is_implementation_scope(self):
        self.assertEqual(scope(self.QUESTION)[0], answer_scope.IMPLEMENTATION)

    def test_the_code_tools_are_available(self):
        self.assertIn("bash", offered(self.QUESTION))

    def test_no_normative_retrieval_is_forced(self):
        self.assertFalse(evidence_bootstrap.should_bootstrap(
            self.QUESTION,
            [{"tool": "bash", "arguments": {"command": f"grep -rn {TERM} src/"}}],
            bound=False, answer="It reads them in order."))


class AComparisonNeedsBothSides(unittest.TestCase):
    """CASE 4 -- "compare ours with what the standard requires" asks for the
    document AND the repository, and must be given both."""

    QUESTION = "Compare our parser with what the standard requires"

    def test_it_binds_the_document(self):
        self.assertTrue(bound(self.QUESTION))

    def test_it_is_mixed_scope(self):
        self.assertEqual(scope(self.QUESTION)[0], answer_scope.MIXED)

    def test_both_kinds_of_tool_are_offered(self):
        self.assertEqual(offered(self.QUESTION),
                         {"standard.search", "standard.fetch", "bash",
                          "edit_file"})


class ACommentIsNotAClause(unittest.TestCase):
    """CASE 5 -- the substitution that made the answer wrong. A comment in the
    source states a rule; the document was never opened. What the comment
    establishes is that a name exists, and nothing else."""

    COMMENT = ("src/codec/frame.c:12:  /* the XcwCount shall be present in "
               "every record */")

    def read_the_comment(self):
        found = code_evidence.CodeEvidenceLedger()
        found.observe("bash", self.COMMENT)

        return found

    def test_the_comment_grounds_that_the_name_exists(self):
        self.assertTrue(self.read_the_comment().knows("XcwCount"))

    def test_but_the_normative_ledger_still_knows_nothing(self):
        self.assertFalse(normative_claims.NormativeEvidence().knows_anything())

    def test_and_a_code_tool_is_not_an_authoritative_one(self):
        self.assertNotIn("standard.search", code_evidence.GROUNDING_TOOLS)
        self.assertNotIn("standard.fetch", code_evidence.GROUNDING_TOOLS)

    def test_so_the_turn_is_sent_to_the_document_before_it_may_answer(self):
        question = f"Explain how the different {TERM} work"
        calls = [{"tool": "bash",
                  "arguments": {"command": f"grep -rn {TERM} src/codec/"}}]

        self.assertTrue(evidence_bootstrap.should_bootstrap(
            question, calls,
            answer="The XcwCount shall be present in every record."))

    def test_on_a_normative_turn_the_comment_cannot_even_be_reached(self):
        self.assertNotIn(
            "bash", offered(f"Explain how the different {TERM} work"))


class AnImplementationSubsetIsNotTheSpecification(unittest.TestCase):
    """CASE 6 -- the repository handles two of the forms the document defines.
    An answer that reports two is reporting the implementation, and a count
    is a normative claim like any other."""

    def evidence(self):
        return normative_claims.NormativeEvidence()

    def test_a_count_taken_from_the_code_is_unsupported(self):
        problems = normative_claims.cardinality_findings(
            f"There are exactly two {TERM} forms.", self.evidence(),
            question=f"how many {TERM} forms are there?")

        self.assertEqual([p["kind"] for p in problems],
                         [normative_claims.UNSUPPORTED_CARDINALITY])

    def test_naming_it_as_the_implementation_is_not_a_claim_about_the_document(self):
        found = code_evidence.CodeEvidenceLedger()
        found.observe("bash", f"src/codec/frame.c:40:  case {TERM}_A: case {TERM}_B:")

        self.assertTrue(found.files())

    def test_the_document_is_still_the_only_source_of_force(self):
        """Existence, and only existence -- the code ledger carries no
        modality, so it can never raise a claim's supported level."""
        self.assertEqual(self.evidence().level(), normative_claims.INFORMATIVE)


if __name__ == "__main__":
    unittest.main()
