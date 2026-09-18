"""A question written in shorthand, and the names a standard actually uses.

Nothing here names a real standard: the vocabularies are invented, and what is
being tested is the rule -- that a corpus may lend a question its own terms,
and only its own, and only where the question asked in shorthand.
"""

import unittest

import standard_query_expansion as expansion

# A technical family, an ordinary word that merely begins alike, and some
# filler so document frequencies mean something.
VOCABULARY = {
    "xyz": 12, "xyzv": 30, "xyzx": 28, "xyzs": 24, "xyz-p": 6,
    "state": 400, "stated": 90, "states": 120,
    "packet": 900, "packets": 950,
    "how": 70, "however": 40,
    "the": 2800, "their": 300, "them": 280,
}
COUNT = 6000


def expand(terms, query, vocabulary=None):
    vocabulary = VOCABULARY if vocabulary is None else vocabulary
    return expansion.expand(terms, vocabulary, vocabulary, COUNT, query=query)


class Shorthand(unittest.TestCase):

    def test_a_shorthand_brings_in_the_family_the_corpus_spells(self):
        terms, why = expand(("how", "xyz", "packets"), "How is XYZ handled?")

        for member in ("xyzv", "xyzx", "xyzs", "xyz-p"):
            self.assertIn(member, terms)
        self.assertEqual(("xyz-p", "xyzs", "xyzv", "xyzx"), why["xyz"])

    def test_the_question_keeps_every_word_it_came_with(self):
        original = ("how", "xyz", "packets")
        terms, _ = expand(original, "How is XYZ handled?")

        self.assertEqual(original, terms[:len(original)])

    def test_a_question_that_already_names_a_member_is_left_alone(self):
        # Asking about XYZV is a narrower question than asking about XYZ, and
        # widening it back out would answer the one that was not asked.
        terms, why = expand(("xyzv", "packets"), "Does XYZV apply?")

        self.assertEqual({}, why)
        self.assertEqual(("xyzv", "packets"), terms)

    def test_lower_case_words_are_not_shorthand(self):
        # `state` has a family here. It is also just a word, and the writer
        # said so by not capitalising it.
        terms, why = expand(("the", "state", "of", "packets"),
                            "the state of packets")

        self.assertEqual({}, why)
        self.assertNotIn("stated", terms)

    def test_one_capital_at_the_start_of_a_sentence_is_not_shorthand(self):
        terms, why = expand(("how", "state"), "How state is reported")

        self.assertEqual({}, why)
        self.assertNotIn("however", terms)

    def test_nothing_is_added_when_the_corpus_has_no_such_family(self):
        terms, why = expand(("abc", "packets"), "What is ABC?")

        self.assertEqual({}, why)
        self.assertEqual(("abc", "packets"), terms)

    def test_a_term_this_corpus_uses_everywhere_is_not_expanded(self):
        # Ordinary language here, whatever it is elsewhere.
        vocabulary = dict(VOCABULARY, the=2800, thex=5)
        terms, why = expand(("the",), "THE thing", vocabulary)

        self.assertEqual({}, why)
        self.assertNotIn("thex", terms)


class Bounds(unittest.TestCase):

    def test_a_family_is_taken_whole_or_not_at_all(self):
        # Half a family drops whichever member sorts last, which is arbitrary
        # and invisible in the answer.
        big = {f"abc{index:02d}": 3 for index in range(expansion.MAX_ADDED + 4)}
        big["abc"] = 5
        terms, why = expand(("abc",), "ABC", big)

        self.assertEqual({}, why)
        self.assertEqual(("abc",), terms)

    def test_a_stem_that_prefixes_too_much_is_a_prefix_not_a_name(self):
        wide = {f"ab{chr(letter)}": 4 for letter in range(97, 97 + 20)}
        wide["ab"] = 3

        self.assertEqual((), expansion.family_for("ab", wide))

    def test_a_long_suffix_is_a_different_word(self):
        vocabulary = {"xyz": 5, "xyzabcd": 5}

        self.assertEqual((), expansion.family_for("xyz", vocabulary))

    def test_a_stem_too_short_to_mean_anything_is_refused(self):
        self.assertEqual((), expansion.family_for("xy", {"xy": 1, "xyz": 1}))

    def test_the_same_question_always_expands_the_same_way(self):
        first, why_first = expand(("xyz",), "XYZ")
        second, why_second = expand(("xyz",), "XYZ")

        self.assertEqual(first, second)
        self.assertEqual(why_first, why_second)

    def test_terms_come_only_from_the_bound_corpus(self):
        terms, _ = expand(("xyz",), "XYZ")

        for term in terms[1:]:
            self.assertIn(term, VOCABULARY)

    def test_an_empty_corpus_changes_nothing(self):
        terms, why = expansion.expand(("xyz",), {}, {}, 0, query="XYZ")

        self.assertEqual(("xyz",), terms)
        self.assertEqual({}, why)


if __name__ == "__main__":
    unittest.main()
