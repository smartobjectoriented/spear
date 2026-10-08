"""The public documentation names the interfaces the code actually has.

Each test reads a name from the code -- a tool, a status, a predicate, a flag
-- and checks that the page documenting it spells it. The prose is free to
change; a renamed status or a tool added to the coding core is not, without
the page that describes it changing too.
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DOCS = ROOT.parent / "doc" / "source"


def page(name: str) -> str:
    return (DOCS / name).read_text(encoding="utf-8")


def literal(name: str) -> str:
    return f"``{name}``"


class CodingCore(unittest.TestCase):
    def test_the_documented_toolset_is_the_coding_view(self):
        from harness.tool_exposure import ToolExposurePolicy
        from harness.tool_registry import coding_schemas

        names = set(ToolExposurePolicy._CODING_READ + ToolExposurePolicy._CODING_WRITE)
        text = page("reasoning/implementation.rst")
        table = re.findall(r"^   \* - ``([a-z_]+)``$", text, re.M)

        self.assertEqual(set(table) & {"read_file", "search_files", "patch", "write_file",
                                       "delete_file", "terminal", "bash", "edit_file",
                                       "append_file", "search_corpus"}, names)

        for argument in coding_schemas()["terminal"]["parameters"]["properties"]:
            self.assertIn(literal(argument), text)


class RequestClasses(unittest.TestCase):
    def test_every_class_is_documented(self):
        from context import answer_scope

        text = page("overview/architecture.rst")

        for name in (answer_scope.GENERAL, answer_scope.IMPLEMENTATION,
                     answer_scope.NORMATIVE, answer_scope.MIXED):
            self.assertIn(literal(name), text)


class Verdicts(unittest.TestCase):
    def test_every_implementation_state_is_documented(self):
        from evidence import completion
        from evidence.completion import Evidence

        states = {
            completion.decide([]).state,
            completion.decide([Evidence("patch", ("a.c",))]).state,
            completion.decide([Evidence("patch", ("a.c",)),
                               Evidence("terminal", (), "make", 0)]).state,
        }
        text = page("reasoning/evidence.rst")

        self.assertEqual(states, {"NO_CHANGE", "UNVERIFIED", "VERIFIED"})
        for state in states:
            self.assertIn(literal(state), text)

    def test_every_normative_status_and_composite_is_documented(self):
        from normative import normative_constraints as nc

        text = page("reasoning/evidence.rst")

        for status in set(nc.STATUSES) - {nc.NOT_APPLICABLE}:
            self.assertIn(literal(status), text)

        for implementation in ("VERIFIED", "UNVERIFIED", "NO_CHANGE"):
            for normative in nc.STATUSES:
                self.assertIn(literal(nc.composite(implementation, normative)), text)


class MixedMode(unittest.TestCase):
    def test_predicates_relations_and_applicability_are_documented(self):
        from normative import normative_coverage as cov
        from normative import normative_predicates as np

        text = page("reasoning/mixed.rst")

        for name in (np.EXACT_COUNT, np.VALUE_EQUALS, np.CONDITIONAL_VALUE,
                     cov.SAME_NORMATIVE_LIST, cov.SAME_TABLE, cov.EXPLICIT_REFERENCE,
                     cov.APPLICABLE, cov.NOT_APPLICABLE, cov.UNRESOLVED):
            self.assertIn(literal(name), text)

    def test_conformance_checks_are_documented_where_they_are_configured(self):
        from normative import normative_evidence as ne

        for name, text in (("using/projects.rst", page("using/projects.rst")),
                           ("reasoning/mixed.rst", page("reasoning/mixed.rst"))):
            for value in ne.SEMANTICS + (ne.PASSED, ne.FAILED, ne.NOT_RUN, ne.ERROR):
                with self.subTest(page=name, value=value):
                    self.assertIn(literal(value), text)

        projects = page("using/projects.rst")

        for key in ("normative_checks", "normative_applicability", "provisions",
                    "command", "evidence.semantics", "enabled", "applicability", "reason"):
            self.assertIn(literal(key), projects)


class CommandLine(unittest.TestCase):
    def test_every_permission_and_corpus_flag_is_documented(self):
        source = (ROOT / "cli/rag_chat.py").read_text(encoding="utf-8")
        start, end = source.index("PERMISSIONS  ("), source.index("SESSION SETTINGS  (")
        flags = set(re.findall(r"(?<![\w-])(--[a-z][a-z-]+)", source[start:end]))
        text = page("using/usage.rst")

        self.assertIn("--auto", flags)
        for flag in flags:
            with self.subTest(flag=flag):
                self.assertIn(flag, text)


if __name__ == "__main__":
    unittest.main()
