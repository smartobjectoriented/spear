"""A change asked for in a standard's terms is MIXED, whatever the standard.

Two real turns asked for a change "the way the Arm architecture specification
requires" and to "make timer_init() ... set up" a register, and both ran as
plain implementation turns: the request named the document as its authority
outside a question, its subject was a function and a file rather than a word
like "code", and "make X do Y" was not read as asking for a change. Each
cause is syntax, not vocabulary, and is tested here against more than one
standard -- and against the requests that must stay as they were.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import answer_scope
import standard_scope
from agent_runtime import is_write_request

ARM = SimpleNamespace(standard_id="ARM-DDI0487", revision="M.c")
VITA = SimpleNamespace(standard_id="ANSI-VITA-49.2", revision="2017-R2024")
SYNTH = SimpleNamespace(standard_id="SYNTH-MIXED", revision="M-1")


def routed(binding, text):
    """(engages, scope, write) as the MIXED gate reads a request."""
    return (standard_scope.engages(binding, text),
            answer_scope.of(text, standard_bound=True), is_write_request(text))


class Recall(unittest.TestCase):
    def test_a_change_in_the_documents_terms_is_mixed(self):
        for binding, text in (
                (ARM, "Make timer_init() in timer.c set up the system counter frequency "
                      "the way the Arm architecture specification requires."),
                (VITA, "Make the encoder in src/wire.c emit one acknowledgement packet per "
                       "requested type, as the standard requires."),
                (SYNTH, "Make build_header() in record.py fill the count field the way the "
                        "specification defines it."),
                (VITA, "Update cmdWireEncodeAck() so it marks the acknowledgement type as "
                       "the specification mandates.")):
            with self.subTest(text=text):
                self.assertEqual(routed(binding, text), (True, answer_scope.MIXED, True))


class Precision(unittest.TestCase):
    def test_a_question_naming_a_revision_is_not_about_a_file(self):
        self.assertEqual(answer_scope.of("What does ARM-DDI0487 M.c say about CNTFRQ?",
                                         standard_bound=True), answer_scope.NORMATIVE)

    def test_a_requirements_file_is_not_a_normative_requirement(self):
        self.assertEqual(answer_scope.of("Add numpy to requirements.txt.", standard_bound=True),
                         answer_scope.IMPLEMENTATION)

    def test_plain_changes_stay_implementation(self):
        for text in ("Fix the off-by-one in ring.c.", "Rename parse_frame() to read_frame().",
                     "Make the build pass on arm64."):
            with self.subTest(text=text):
                self.assertFalse(standard_scope.engages(VITA, text))
                self.assertEqual(answer_scope.of(text, standard_bound=True),
                                 answer_scope.IMPLEMENTATION)

    def test_make_that_names_nothing_in_the_tree_is_no_write(self):
        for text in ("Make sure the build passes.",
                     "Make a list of the files that use cmdWireEncodeAck()."):
            with self.subTest(text=text):
                self.assertFalse(is_write_request(text))

    def test_the_standard_library_is_not_the_bound_standard(self):
        self.assertFalse(standard_scope.engages(
            VITA, "Why does the standard library require an allocator in ring.c?"))


if __name__ == "__main__":
    unittest.main()
