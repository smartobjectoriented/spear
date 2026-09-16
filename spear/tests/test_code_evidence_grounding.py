"""An identifier the model READ is not an identifier the model invented.

Every case here is synthetic. The names are made up, the clauses are made up,
and nothing in this file knows what standard or which repository provoked it:
what is being tested is the rule, which is that content a code-reading tool
returned this turn grounds the EXISTENCE of a name and nothing else.
"""

import unittest

import code_evidence
import normative_claims
import standard_answer_policy


CLAUSE = {"units": [{
    "section": "4.2", "source_id": "u-1", "modality": "SHALL",
    "content_type": "requirement",
    "text": "A Widget packet shall carry exactly one FlagA indicator."}]}


def evidence():
    ledger = normative_claims.NormativeEvidence()
    ledger.observe(CLAUSE)
    return ledger


def ungrounded(answer, ledger=None, **kwargs):
    found = normative_claims.identifier_findings(
        answer, ledger or evidence(), **kwargs)
    return [item["identifier"] for item in found]


class CodeContentGrounds(unittest.TestCase):
    """Content a tool returned may ground a name."""

    def test_file_read_result(self):
        ledger = code_evidence.CodeEvidenceLedger()
        ledger.observe("read_file", "static int widgetEncodeFlag(void) {\n")

        self.assertEqual(
            [], ungrounded("The code calls widgetEncodeFlag here.",
                           permitted=ledger.identifiers()))

    def test_grep_output(self):
        ledger = code_evidence.CodeEvidenceLedger()
        ledger.observe("bash", "src/widget.c:41:  WidgetReport rep = {0};\n")

        self.assertEqual(
            [], ungrounded("The code fills a WidgetReport.",
                           permitted=ledger.identifiers()))

    def test_corpus_search_snippet(self):
        ledger = code_evidence.CodeEvidenceLedger()
        ledger.observe("search_corpus", "  bool widgetIsFlagged(const W *w);")

        self.assertTrue(ledger.knows("widgetIsFlagged"))

    def test_spelling_is_normalised_like_everywhere_else(self):
        ledger = code_evidence.CodeEvidenceLedger()
        ledger.observe("bash", "src/widget.c:7:  int Widget_Flag;\n")

        # `Widget-Flag`, `Widget_Flag` and `WidgetFlag` are one name to the
        # evidence ledger, and must be one name to this one too. What counts
        # as an identifier at all is decided in exactly one place, and this
        # ledger asks there rather than deciding again.
        self.assertTrue(ledger.knows("WidgetFlag"))
        self.assertTrue(ledger.knows("Widget-Flag"))


class OnlyReturNedContentGrounds(unittest.TestCase):
    """Asking for a name, or listing a file, is not reading one."""

    def test_grep_pattern_alone_does_not_ground(self):
        ledger = code_evidence.CodeEvidenceLedger()
        # The model grepped for a name it believed in, and found nothing.
        ledger.observe("bash", "")

        self.assertEqual(
            ["widgetProcessFlag"],
            ungrounded("The code calls widgetProcessFlag.",
                       permitted=ledger.identifiers()))

    def test_tool_arguments_are_never_read(self):
        ledger = code_evidence.CodeEvidenceLedger()

        # The signature takes no arguments at all: there is nowhere for a
        # pattern to enter from.
        with self.assertRaises(TypeError):
            ledger.observe("bash", {"command": "grep widgetProcessFlag ."},
                           "")

    def test_filename_alone_does_not_ground(self):
        ledger = code_evidence.CodeEvidenceLedger()
        ledger.observe("bash", "./src/WidgetEncoder.c\n./src/w.h\n")

        self.assertEqual(
            ["WidgetEncoder"],
            ungrounded("The code calls WidgetEncoder.",
                       permitted=ledger.identifiers()))

    def test_a_lone_name_is_content_not_a_listing(self):
        ledger = code_evidence.CodeEvidenceLedger()
        # An enum member as a header writes it: on its own line, and the only
        # place the name is ever spelled.
        ledger.observe("read_file", "enum {\n    WidgetFlagA,\n};\n")

        self.assertTrue(ledger.knows("WidgetFlagA"))

    def test_directory_listing_does_not_ground(self):
        ledger = code_evidence.CodeEvidenceLedger()
        ledger.observe("bash", "src/WidgetPart/\nsrc/other/\n")

        self.assertFalse(ledger.knows("WidgetPart"))

    def test_path_inside_a_content_line_does_not_ground(self):
        ledger = code_evidence.CodeEvidenceLedger()
        ledger.observe("bash", "  reading src/WidgetEncoder.c now\n")

        self.assertFalse(ledger.knows("WidgetEncoder"))

    def test_unread_repository_file_does_not_ground(self):
        # Nothing was returned at all: the name exists somewhere, and the
        # model never saw it.
        ledger = code_evidence.CodeEvidenceLedger()

        self.assertEqual(
            ["WidgetReport"],
            ungrounded("The code fills a WidgetReport.",
                       permitted=ledger.identifiers()))

    def test_model_written_edit_does_not_ground(self):
        ledger = code_evidence.CodeEvidenceLedger()
        ledger.observe("edit_file", "+  widgetProcessFlag(w);\n")

        self.assertFalse(ledger.knows("widgetProcessFlag"))

    def test_invented_name_still_fires(self):
        ledger = code_evidence.CodeEvidenceLedger()
        ledger.observe("bash", "src/widget.c:41:  widgetEncodeFlag(w);\n")

        self.assertEqual(
            ["widgetProcessFlag"],
            ungrounded("The code calls widgetEncodeFlag from "
                       "widgetProcessFlag.",
                       permitted=ledger.identifiers()))


class CodeIsNotNormative(unittest.TestCase):
    """Existence is all it grants."""

    def test_code_grounding_does_not_license_a_requirement(self):
        ledger = code_evidence.CodeEvidenceLedger()
        ledger.observe("bash", "src/widget.c:9:  /* a WidgetToken is optional */")

        clause = {"units": [{
            "section": "4.3", "source_id": "u-2", "modality": "MAY",
            "content_type": "permission",
            "text": "A Widget packet may carry a WidgetToken."}]}
        spoken = normative_claims.NormativeEvidence()
        spoken.observe(clause)

        answer = "A Widget packet shall carry a WidgetToken."

        # The name is grounded, so it is not reported as invented...
        self.assertEqual([], ungrounded(answer, spoken,
                                        permitted=ledger.identifiers()))

        # ...and the claim is still too strong for the clause behind it.
        kinds = {item["kind"] for item in normative_claims.findings(
            answer, spoken, permitted=ledger.identifiers())}
        self.assertIn(normative_claims.STRENGTHENED_MODALITY, kinds)

    def test_a_code_name_may_describe_the_implementation(self):
        ledger = code_evidence.CodeEvidenceLedger()
        ledger.observe("bash", "src/widget.c:9:  /* a WidgetToken is set */")

        self.assertEqual(
            [], ungrounded("The code sets a WidgetToken before sending.",
                           permitted=ledger.identifiers()))

    def test_a_code_name_may_not_be_credited_to_the_standard(self):
        ledger = code_evidence.CodeEvidenceLedger()
        ledger.observe("bash", "src/widget.c:9:  /* a WidgetToken is set */")

        # Same name, same grounding, and now the sentence says the DOCUMENT
        # requires it. A comment in a source file is not the document.
        found = normative_claims.identifier_findings(
            "The standard requires a WidgetToken in every packet.",
            evidence(), permitted=ledger.identifiers())

        self.assertEqual(["WidgetToken"],
                         [item["identifier"] for item in found])
        self.assertTrue(found[0]["read_from_code"])

    def test_a_documented_name_is_not_touched_by_the_sentence_it_sits_in(self):
        # FlagA is in the clause, so it is grounded wherever it appears --
        # including in a statement of what the standard requires.
        self.assertEqual(
            [], ungrounded("The standard requires exactly one FlagA."))

    def test_code_content_is_not_added_to_the_normative_ledger(self):
        policy = standard_answer_policy.StandardAnswerPolicy(question="q?")
        policy.observe_code_read(
            "bash", {"command": "grep -rn Widget src"},
            "src/widget.c:41:  WidgetReport rep;\n")

        self.assertTrue(policy.code_evidence.knows("WidgetReport"))
        self.assertFalse(policy.claim_evidence.knows_anything())
        self.assertEqual(normative_claims.INFORMATIVE,
                         policy.claim_evidence.level())


class StandardGroundingUnchanged(unittest.TestCase):

    def test_standard_identifier_needs_no_permission(self):
        self.assertEqual([], ungrounded("A FlagA indicator is carried."))

    def test_absent_identifier_still_fires_without_code_reads(self):
        self.assertEqual(["FlagB"],
                         ungrounded("A FlagB indicator is carried."))

    def test_shared_name_is_reported_once_from_both_sides(self):
        ledger = code_evidence.CodeEvidenceLedger()
        ledger.observe("bash", "src/widget.c:3:  int FlagA;\n")

        # Named by the clause AND by the code: no finding, and the code side
        # still records where it saw it.
        self.assertEqual([], ungrounded("A FlagA indicator is carried.",
                                        permitted=ledger.identifiers()))
        self.assertEqual([{"identifier": "FlagA", "files": ["src/widget.c"],
                           "tools": ["bash"]}],
                         ledger.provenance())


class Provenance(unittest.TestCase):

    def test_records_file_and_tool(self):
        ledger = code_evidence.CodeEvidenceLedger()
        ledger.observe("bash", "src/a.c:1:  WidgetReport x;\n"
                               "src/b.c:2:  WidgetReport y;\n")

        self.assertEqual([{"identifier": "WidgetReport",
                           "files": ["src/a.c", "src/b.c"],
                           "tools": ["bash"]}], ledger.provenance())

    def test_file_is_absent_when_the_result_did_not_name_one(self):
        ledger = code_evidence.CodeEvidenceLedger()
        ledger.observe("read_file", "  WidgetReport x;\n")

        self.assertEqual([{"identifier": "WidgetReport", "files": [],
                           "tools": ["read_file"]}], ledger.provenance())

    def test_order_is_stable(self):
        first = code_evidence.CodeEvidenceLedger()
        first.observe("bash", "a.c:1: WidgetB b; WidgetA a;\n")
        second = code_evidence.CodeEvidenceLedger()
        second.observe("bash", "a.c:1: WidgetA a; WidgetB b;\n")

        self.assertEqual([item["identifier"] for item in first.provenance()],
                         [item["identifier"] for item in second.provenance()])


class RepairCannotEnlargeTheSet(unittest.TestCase):
    """A rewrite runs no tools, so it may not ground a new name."""

    def test_repair_is_judged_against_the_same_permitted_set(self):
        seen = []

        def ask(text):
            seen.append(text)
            # The rewrite swaps the invented name for another invented name.
            return ("The code calls widgetHandleFlag.\n"
                    "It is described in section 4.2.")

        policy = standard_answer_policy.StandardAnswerPolicy(
            question="Does the code conform?", repair_ask=ask)
        policy.observe_code_read(
            "bash", {"command": "grep -rn Widget src"},
            "src/widget.c:41:  widgetEncodeFlag(w);\n")
        policy.claim_evidence.observe(CLAUSE)
        policy.clauses.observe(CLAUSE, "")

        policy.finalize("The code calls widgetProcessFlag.")

        self.assertTrue(seen, "the repair was never attempted")
        self.assertTrue(policy.claims_fired)
        self.assertIn("widgetHandleFlag",
                      {item.get("identifier")
                       for item in policy.claim_findings})
        self.assertFalse(policy.code_evidence.knows("widgetHandleFlag"))


if __name__ == "__main__":
    unittest.main()
