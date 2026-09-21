"""What a self-contained normative turn may be answered OUT OF.

The turn before this work stopped a normative question from going and reading
a working tree: the local tools came off the table, and none was called. It
did not stop the question being answered from a working tree the session had
already been shown. The conversation is re-injected whole each turn, and one
earlier turn of implementation work carries, in its own answer, every file
path and function name it found.

Measured on the session this exists for: a self-contained normative question,
asked after a turn of code changes, produced a draft naming a function that
occurs seven times in that earlier answer and nowhere in the retrieved
document. Zero tool calls were needed for that and zero were made. The
identifier guard caught it and withheld the whole answer, which is the guard
working and the turn still wasted.

So the distinction `answer_scope` already draws, applied to the generation
context rather than to the tool table:

    history for interpretation is not history for evidence

A turn that stands on its own words and asks about the document keeps the
shape of the conversation and what was said in it about the document, and
does not keep the project's own material. A turn that points back at the last
one keeps everything, because it is a question about the last one.

Synthetic throughout: the symbols below are invented, and a rule that had to
know one project's vocabulary would be the same defect in a new place.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import answer_scope
import code_evidence
import normative_claims

#: Two names that exist only in this project's source.
IMPL = "FooImpl"
FIELD = "cache_state"

#: A term the bound document defines.
TERM = "ABC"

#: A turn of implementation work, as the CLI records one: what the operator
#: typed, the transcript of what the tools returned appended to it, and the
#: model's running account of the code it read.
IMPLEMENTATION_HISTORY = [
    {"role": "user", "content":
        "check the code base and make the necessary changes\n\n"
        "[Tools executed during this turn — results:\n"
        'bash {"command": "grep -rn parse src/"}\n'
        f"src/codec/frame.c:41:  {IMPL}_dispatch(&{FIELD});\n]"},
    {"role": "assistant", "content":
        f"I read src/codec/frame.c. The {IMPL} helper walks the words in "
        f"order and keeps what it has seen in {FIELD}, which is reset "
        f"between records. I changed {IMPL}_dispatch accordingly."},
]

#: A turn about the document, with nothing of the project in it.
NORMATIVE_HISTORY = [
    {"role": "user", "content":
        f"How should the {TERM} indicator be handled?\n\n"
        "[Tools executed during this turn — results:\n"
        'standard.fetch {"source_id": "[a source read in an earlier turn]"}\n]'},
    {"role": "assistant", "content":
        f"Per Rule 9.1-1, the {TERM} indicator shall appear once per record "
        "and is read before the words it selects."},
]


def carry(history, question, *, prior=None, standard_bound=True):
    """The conversation this turn would be generated from."""
    scope = answer_scope.of(question, prior=prior,
                            standard_bound=standard_bound)

    return scope, answer_scope.carried(
        history, scope, standard_bound=standard_bound,
        self_contained=answer_scope.self_contained(question))


def text_of(history):
    return "\n".join(message["content"] for message in history)


class ASelfContainedExplanationCarriesNoneOfTheProject(unittest.TestCase):
    """CASE 1 -- the failure itself."""

    QUESTION = f"Explain how {TERM} works."

    def carried(self):
        return carry(IMPLEMENTATION_HISTORY, self.QUESTION,
                     prior=answer_scope.IMPLEMENTATION)

    def test_the_scope_is_normative(self):
        self.assertEqual(self.carried()[0], answer_scope.NORMATIVE)

    def test_the_question_stands_on_its_own_words(self):
        self.assertTrue(answer_scope.self_contained(self.QUESTION))

    def test_neither_symbol_survives_into_the_context(self):
        carried = text_of(self.carried()[1])

        self.assertNotIn(IMPL, carried)
        self.assertNotIn(FIELD, carried)

    def test_nor_does_the_tool_transcript(self):
        self.assertNotIn("Tools executed", text_of(self.carried()[1]))

    def test_the_exchange_is_still_there_as_an_exchange(self):
        """Continuity is kept: two turns happened, and the model is told what
        kind of thing they were. What it is not given is their content."""
        carried = self.carried()[1]

        self.assertEqual([m["role"] for m in carried],
                         [m["role"] for m in IMPLEMENTATION_HISTORY])
        self.assertIn(answer_scope.ELIDED_ASK, carried[0]["content"])
        self.assertIn("not evidence", carried[1]["content"])

    def test_the_local_tools_are_withheld_as_before(self):
        self.assertTrue(answer_scope.withholds_local_tools(
            self.carried()[0], standard_bound=True))


class SoDoesAQuestionAboutWhatTheDocumentRequires(unittest.TestCase):
    """CASE 2 -- the same isolation, asked the other way round."""

    QUESTION = f"What does the standard require for {TERM}?"

    def carried(self):
        return carry(IMPLEMENTATION_HISTORY, self.QUESTION,
                     prior=answer_scope.IMPLEMENTATION)

    def test_the_scope_is_normative(self):
        self.assertEqual(self.carried()[0], answer_scope.NORMATIVE)

    def test_neither_symbol_survives_into_the_context(self):
        carried = text_of(self.carried()[1])

        self.assertNotIn(IMPL, carried)
        self.assertNotIn(FIELD, carried)


class AQuestionAboutThatImplementationKeepsIt(unittest.TestCase):
    """CASE 3 -- the turn this must not apply to. "Does that comply" is a
    question about the last turn, and cutting it out of the conversation
    leaves nothing for "that" to mean."""

    QUESTION = f"Does that implementation comply with {TERM}?"

    def carried(self):
        return carry(IMPLEMENTATION_HISTORY, self.QUESTION,
                     prior=answer_scope.IMPLEMENTATION)

    def test_it_is_mixed_scope(self):
        self.assertEqual(self.carried()[0], answer_scope.MIXED)

    def test_it_does_not_stand_on_its_own_words(self):
        self.assertFalse(answer_scope.self_contained(self.QUESTION))

    def test_the_implementation_context_is_retained(self):
        carried = text_of(self.carried()[1])

        self.assertIn(IMPL, carried)
        self.assertIn(FIELD, carried)

    def test_the_payload_is_not_withheld(self):
        self.assertFalse(answer_scope.withholds_history_payload(
            self.carried()[0], standard_bound=True, self_contained=False))


class AQuestionAboutThatFunctionKeepsItToo(unittest.TestCase):
    """CASE 4 -- naming the document does not make a question about a
    particular function into a question about the document alone."""

    QUESTION = "What does that function do according to the standard?"

    def carried(self):
        return carry(IMPLEMENTATION_HISTORY, self.QUESTION,
                     prior=answer_scope.IMPLEMENTATION)

    def test_it_is_mixed_scope(self):
        self.assertEqual(self.carried()[0], answer_scope.MIXED)

    def test_the_referent_is_still_in_the_conversation(self):
        self.assertIn(IMPL, text_of(self.carried()[1]))


class NormativeContinuityIsKept(unittest.TestCase):
    """CASE 5 -- what was said about the document stays said. A self-contained
    normative follow-up, so the withholding is fully switched on, and what it
    withholds is the project's material rather than the conversation."""

    QUESTION = f"What ordering does the standard give for {TERM} words?"

    def carried(self):
        return carry(NORMATIVE_HISTORY, self.QUESTION,
                     prior=answer_scope.NORMATIVE)

    def test_the_earlier_normative_answer_is_carried_whole(self):
        carried = text_of(self.carried()[1])

        self.assertIn("Rule 9.1-1", carried)
        self.assertIn("shall appear once per record", carried)

    def test_the_operators_own_question_is_carried(self):
        self.assertIn(f"How should the {TERM} indicator be handled?",
                      text_of(self.carried()[1]))

    def test_but_the_tool_transcript_is_not(self):
        """An earlier turn's retrieval is not this turn's evidence: its
        handles are already redacted where it is stored, and a clause this
        turn needs is a clause this turn fetches."""
        self.assertNotIn("Tools executed", text_of(self.carried()[1]))

    def test_an_unbound_session_is_left_alone(self):
        self.assertEqual(
            answer_scope.carried(NORMATIVE_HISTORY, answer_scope.NORMATIVE,
                                 standard_bound=False),
            list(NORMATIVE_HISTORY))


class HistoryCannotGroundAName(unittest.TestCase):
    """CASE 6 -- and if one got through anyway, it is still not grounded.
    The guard is unchanged; what it reads is unchanged. A name seen in a
    conversation was never evidence, and this states it."""

    def evidence(self):
        found = normative_claims.NormativeEvidence()
        found.observe({
            "citation": {"section": "9.1", "source_id": "std-1",
                         "standard_id": "ACME-1234.5", "revision": "2019-R2023"},
            "neighbors": [{"section": "9.1", "source_id": "std-1",
                           "modality": "SHALL", "content_type": "Rule",
                           "text": f"The {TERM}Word shall be present in every "
                                   "record."}]})

        return found

    def test_a_name_only_the_conversation_has_is_ungrounded(self):
        problems = normative_claims.identifier_findings(
            f"The {IMPL} helper reads the words in order.", self.evidence())

        self.assertEqual([p["identifier"] for p in problems], [IMPL])
        self.assertFalse(problems[0]["read_from_code"])

    def test_a_turn_that_read_no_code_permits_nothing(self):
        """`permitted` is what the tools returned THIS turn. A normative turn
        is offered no code tools, so it is empty, and history does not fill
        it."""
        ledger = code_evidence.CodeEvidenceLedger()
        ledger.observe("bash", text_of(IMPLEMENTATION_HISTORY))

        self.assertNotIn("standard.search", code_evidence.GROUNDING_TOOLS)
        self.assertEqual(code_evidence.CodeEvidenceLedger().identifiers(), set())

    def test_the_document_still_grounds_its_own_names(self):
        self.assertEqual(
            normative_claims.identifier_findings(
                f"The {TERM}Word is read first.", self.evidence()), [])


if __name__ == "__main__":
    unittest.main()
