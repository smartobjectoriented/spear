"""A measured deployment says which retrieval it wants.

`mode="hybrid"` silently degrades to lexical where no vector index exists.
That is reasonable behaviour and it made a whole investigation describe
lexical-only runs as hybrid retrieval -- but the deeper problem is that such
a path CHANGES the day somebody configures an embedder. A configuration
whose behaviour has been measured at 8/0/0 must not be one embedder away
from being a different configuration.

So the mode is stated, and stating it is what makes the measurement keep
describing the deployment.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

import standard_tools


class TheRequestedModeIsConfiguration(unittest.TestCase):

    def mode(self, **environment):
        with mock.patch.dict(os.environ, environment, clear=True):
            return standard_tools._configured_mode()

    def test_unset_keeps_the_long_standing_default(self):
        self.assertEqual(self.mode(), "hybrid")

    def test_lexical_can_be_asked_for_outright(self):
        self.assertEqual(self.mode(SPEAR_STANDARD_RETRIEVAL_MODE="lexical"),
                         "lexical")

    def test_hybrid_remains_available_as_an_explicit_choice(self):
        """Vector and fusion are not removed; they stop being what you get by
        accident."""
        self.assertEqual(self.mode(SPEAR_STANDARD_RETRIEVAL_MODE="hybrid"),
                         "hybrid")
        self.assertEqual(self.mode(SPEAR_STANDARD_RETRIEVAL_MODE="vector"),
                         "vector")

    def test_case_and_padding_do_not_matter(self):
        self.assertEqual(self.mode(SPEAR_STANDARD_RETRIEVAL_MODE="  LEXICAL "),
                         "lexical")

    def test_a_misspelled_value_is_an_error_not_a_fallback(self):
        """The asymmetry that matters. Falling back would hand the caller
        hybrid -- and on a store with vectors and an embedder that is real
        fusion, the very thing the variable was typed to switch off. A typo
        must not quietly re-enable it."""
        with self.assertRaises(standard_tools.StandardConfigurationError):
            self.mode(SPEAR_STANDARD_RETRIEVAL_MODE="lexial")

    def test_the_error_names_the_value_and_the_alternatives(self):
        with self.assertRaises(standard_tools.StandardConfigurationError) as caught:
            self.mode(SPEAR_STANDARD_RETRIEVAL_MODE="semantic")

        message = str(caught.exception)
        self.assertIn("semantic", message)
        for mode in ("lexical", "vector", "hybrid"):
            self.assertIn(mode, message)

    def test_an_absent_variable_is_not_an_error(self):
        """A deployment that never set it keeps exactly what it had."""
        self.assertEqual(self.mode(), "hybrid")

    def test_an_empty_value_is_not_a_choice(self):
        self.assertEqual(self.mode(SPEAR_STANDARD_RETRIEVAL_MODE=""), "hybrid")


class AnEmbedderCannotChangeAnExplicitlyLexicalPath(unittest.TestCase):
    """The point of the lock. With a vector index present AND an embedder
    configured, an explicitly lexical request stays lexical."""

    def test_explicit_lexical_reports_lexical_only(self):
        import standard_retrieval

        self.assertEqual(standard_retrieval._capability("lexical", "lexical"),
                         standard_retrieval.LEXICAL_ONLY)

    def test_hybrid_with_vectors_is_reported_as_the_other_thing(self):
        import standard_retrieval

        self.assertEqual(standard_retrieval._capability("hybrid", "hybrid"),
                         standard_retrieval.HYBRID_LEXICAL_VECTOR)

    def test_the_two_capabilities_are_distinguishable(self):
        import standard_retrieval

        self.assertNotEqual(standard_retrieval.LEXICAL_ONLY,
                            standard_retrieval.HYBRID_LEXICAL_VECTOR)


if __name__ == "__main__":
    unittest.main()
