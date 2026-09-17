"""A retrieval handle must not outlive the turn that obtained it.

The failure these pin down is not a wrong answer but a wrongly SOURCED one: a
turn that answered a fresh question out of the previous question's clauses,
because the previous turn's transcript -- persisted so a follow-up could refer
back to it -- still carried the source ids, and fetching one is all it takes.
"""

import unittest

import evidence_handles


class Redaction(unittest.TestCase):

    def test_a_handle_does_not_survive_persistence(self):
        transcript = ('standard.fetch {"source_id": "std-'
                      '0123456789abcdef0123456789abcdef"}')

        self.assertTrue(evidence_handles.handles_in(transcript))
        self.assertFalse(
            evidence_handles.handles_in(evidence_handles.redact(transcript)))

    def test_every_handle_goes_not_merely_the_first(self):
        transcript = ("read std-0123456789abcdef0123456789abcdef then "
                      "std-fedcba9876543210fedcba9876543210")

        self.assertEqual(
            [], evidence_handles.handles_in(evidence_handles.redact(transcript)))

    def test_the_words_around_it_survive(self):
        transcript = ('standard.search {"query": "acknowledgement subtypes"} '
                      '-> std-0123456789abcdef0123456789abcdef, Rule 4.2-6, '
                      'page 11')
        kept = evidence_handles.redact(transcript)

        # What a follow-up needs in order to refer back -- the tool, the
        # question asked, the clause, the page -- is prose and stays.
        for word in ("standard.search", "acknowledgement subtypes",
                     "Rule 4.2-6", "page 11"):
            self.assertIn(word, kept)

    def test_what_replaces_it_says_what_happened(self):
        kept = evidence_handles.redact("std-0123456789abcdef0123456789abcdef")

        self.assertEqual(evidence_handles.REDACTED, kept)

    def test_a_clause_number_is_not_a_handle(self):
        # Only the handle is machine-actionable. Citations are how a reader
        # and a later turn SHOULD refer to evidence, and they are untouched.
        transcript = "Rule 4.2-6, §4.2, p.11"

        self.assertEqual(transcript, evidence_handles.redact(transcript))

    def test_text_without_handles_is_returned_unchanged(self):
        transcript = "bash: grep -n Ack src/command/command_wire.c"

        self.assertEqual(transcript, evidence_handles.redact(transcript))

    def test_empty_and_absent_text(self):
        self.assertEqual("", evidence_handles.redact(""))
        self.assertEqual("", evidence_handles.redact(None))
        self.assertEqual([], evidence_handles.handles_in(None))

    def test_a_shape_that_only_looks_like_a_handle_is_left_alone(self):
        # Too short, too long, and not hexadecimal: none of these can be
        # fetched, so none of them is worth rewriting.
        for text in ("std-abc", "std-" + "0" * 40, "std-" + "z" * 32):
            self.assertEqual(text, evidence_handles.redact(text))


class PersistedTranscript(unittest.TestCase):
    """The shape the conversation actually stores."""

    def test_a_turns_transcript_cannot_be_replayed_as_a_handle(self):
        tool_log = [
            'standard.search {"query": "how should the ACK be managed"}',
            'standard.fetch {"source_id": "std-'
            '89abcdef89abcdef89abcdef89abcdef"} {"citation": {"page": 11}}',
        ]
        persisted = evidence_handles.redact("\n\n".join(tool_log)[:2000])

        self.assertEqual([], evidence_handles.handles_in(persisted))
        self.assertIn("how should the ACK be managed", persisted)
        self.assertIn("page", persisted)


if __name__ == "__main__":
    unittest.main()


class Ownership(unittest.TestCase):
    """A handle is good for the turn that retrieved it, and no other."""

    HANDLE = "std-" + "0123456789abcdef" * 2

    def test_a_turn_owns_what_its_own_result_showed(self):
        turn = {}
        evidence_handles.issue(turn, f'results: [{{"source_id": "{self.HANDLE}"}}]')

        self.assertTrue(evidence_handles.owns(turn, self.HANDLE))

    def test_the_next_turn_owns_nothing_by_inheritance(self):
        first = {}
        evidence_handles.issue(first, self.HANDLE)

        self.assertFalse(evidence_handles.owns({}, self.HANDLE))

    def test_a_handle_quoted_in_conversation_is_not_issued(self):
        # The shape the real failure took: the id was in the previous turn's
        # persisted transcript, and reading it is not retrieving it.
        turn = {}
        history = f'[Tools executed during this turn: fetch "{self.HANDLE}"]'

        self.assertNotIn(history, (None,))
        self.assertFalse(evidence_handles.owns(turn, self.HANDLE))

    def test_a_handle_remembered_in_memory_is_not_issued(self):
        turn = {}
        memory = f"- [2026-01-01] the useful unit is {self.HANDLE}"

        self.assertIn(self.HANDLE, memory)
        self.assertFalse(evidence_handles.owns(turn, self.HANDLE))

    def test_an_invented_handle_of_the_right_shape_is_not_issued(self):
        self.assertFalse(evidence_handles.owns({}, "std-" + "a" * 32))

    def test_re_issuing_the_same_handle_makes_it_usable_again(self):
        # Retrieval is deterministic, so a fresh search for the same question
        # returns the same ids. Owning them again is the point: they were
        # retrieved again, for this turn.
        turn = {}
        self.assertFalse(evidence_handles.owns(turn, self.HANDLE))
        evidence_handles.issue(turn, self.HANDLE)
        self.assertTrue(evidence_handles.owns(turn, self.HANDLE))

    def test_every_handle_in_one_result_is_issued(self):
        other = "std-" + "fedcba9876543210" * 2
        turn = {}
        evidence_handles.issue(turn, f"{self.HANDLE} and {other}")

        self.assertTrue(evidence_handles.owns(turn, self.HANDLE))
        self.assertTrue(evidence_handles.owns(turn, other))

    def test_one_issued_does_not_carry_an_unissued_neighbour(self):
        turn = {}
        evidence_handles.issue(turn, self.HANDLE)

        self.assertTrue(evidence_handles.owns(turn, self.HANDLE))
        self.assertFalse(evidence_handles.owns(turn, "std-" + "b" * 32))

    def test_issuance_reads_the_result_the_model_saw(self):
        # Not a list the caller assembles: a parent or a companion carried
        # alongside a fetched unit is a handle the model can see and use, and
        # reading the text is what keeps those in scope.
        parent = "std-" + "abcdefabcdefabcd" * 2
        turn = {}
        issued = evidence_handles.issue(
            turn, f'{{"unit": {{"source_id": "{self.HANDLE}"}}, '
                  f'"parent": {{"source_id": "{parent}"}}}}')

        self.assertEqual({self.HANDLE, parent}, issued)
        self.assertTrue(evidence_handles.owns(turn, parent))

    def test_a_turn_without_a_cache_owns_nothing_and_does_not_crash(self):
        self.assertEqual(set(), evidence_handles.issued_this_turn(None))
        self.assertEqual(set(), evidence_handles.issue(None, self.HANDLE))
