"""A turn sees its own workspace's context, what its class calls for, and
whatever is explicitly generic -- nothing else.

Workspace isolation, generic and project-specific rules, path scope, task
classes and MIXED phases, the budget, standard selection, tool families and
the tool contract, the audit trail -- and the contamination that motivated
all of it: one project's build-system instructions reaching another.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from context import context_selection as cs
from context import context_sources as src
from harness import tool_selection as ts
from context.context_selection import Candidate, SourceType
from context.workspace_context import from_session

RULE_A = "---\nscope: corpus alpha\n---\nAlpha builds with `make alpha-fw`."
RULE_B = "---\nscope: corpus beta\n---\nBeta builds with `ninja -C out beta`."
GENERIC = "---\nscope: global\n---\nNever rewrite an existing copyright header."
UNSCOPED = "Python: a blank line after a multi-line comment block."
DOCS = "---\nscope: global\npaths: doc/**\n---\nBuild the docs with `make -C doc html`."
NORMATIVE_RULE = "---\nscope: global\ntasks: normative\n---\nCite the provision by label."
GENERAL_RULE = "---\nscope: global\ntasks: general\n---\nAnswer briefly."

# The historical contamination: one product's build instructions -- its
# boards, its capsule flag, its patch-regeneration script -- reaching an
# unrelated tree.
FOREIGN = ("---\nscope: corpus verdin, virt64\n---\nInfrastructure built with bitbake; "
           "platforms virt64 and verdin-imx8mp; `build.sh -a bsp-torizon -e1c dev`; "
           "after changing patched sources run `updiff.sh`.")
FOREIGN_UNSCOPED = ("Boards: verdin-imx8mp. Torizon images use `-e1c fc`. "
                    "Regenerate patches with updiff.sh.")
FOREIGN_WORDS = ("verdin", "torizon", "-e1c", "updiff.sh")


def write_rules(directory, **files):
    for name, text in files.items():
        Path(directory, f"{name}.md").write_text(text)


def workspace(name="alpha", registered=True, root=None, **spec):
    root = root or tempfile.mkdtemp()
    return from_session(project=name, spec=spec, registered=registered, project_root=root,
                        corpus_root=root)


class Selection(unittest.TestCase):
    def setUp(self):
        self.rules = tempfile.mkdtemp()
        write_rules(self.rules, **{"10-alpha": RULE_A, "11-beta": RULE_B,
                                   "20-generic": GENERIC, "30-unscoped": UNSCOPED,
                                   "40-docs": DOCS, "50-normative": NORMATIVE_RULE,
                                   "60-general": GENERAL_RULE})
        self.selector = cs.DeterministicContextSelector()

    def select(self, ws, phase=cs.IMPLEMENTATION, request="Fix the parser.", extra=()):
        return self.selector.select(ws, phase, src.rule_candidates(self.rules) + list(extra),
                                    request=request)

    def reasons(self, selection):
        return {item.item_id: item.reason for item in selection.rejected}


class WorkspaceIsolation(Selection):
    def test_each_workspace_gets_its_own_rules_and_the_generic_ones(self):
        for name, own, foreign in (("alpha", "rule:alpha", "rule:beta"),
                                   ("beta", "rule:beta", "rule:alpha")):
            with self.subTest(workspace=name):
                selection = self.select(workspace(name))

                self.assertIn(own, selection.ids())
                self.assertIn("rule:generic", selection.ids())
                self.assertNotIn(foreign, selection.ids())
                self.assertIn("wrong workspace", self.reasons(selection)[foreign])

    def test_a_rule_nobody_declared_generic_applies_nowhere(self):
        selection = self.select(workspace("alpha"))

        self.assertNotIn("rule:unscoped", selection.ids())
        self.assertIn("not declared generic", self.reasons(selection)["rule:unscoped"])

    def test_an_unregistered_tree_inherits_no_project(self):
        root = tempfile.mkdtemp()
        ws = workspace("alpha", registered=False, root=root)
        selection = self.select(ws)

        self.assertEqual(ws.workspace_id, f"adhoc:{os.path.realpath(root)}")
        self.assertEqual(ws.names, frozenset())
        self.assertEqual(sorted(selection.ids()), ["rule:generic"])

    def test_a_tree_s_own_map_belongs_to_it_registered_or_not(self):
        own = src.corpus_rule_candidates([("repo", "Sources are in src/.", (cs.WORKSPACE,))])
        selection = self.select(workspace("x", registered=False), extra=own)

        self.assertIn("rule:corpus:repo", selection.ids())

    def test_the_model_facing_prompt_carries_only_the_workspace_s_rules(self):
        from agent import prompt as core_prompt

        for name, present, absent in (("alpha", "make alpha-fw", "ninja -C out beta"),
                                      ("beta", "ninja -C out beta", "make alpha-fw")):
            with self.subTest(workspace=name):
                selection = self.select(workspace(name))
                system = core_prompt.build(cwd="/tmp", tool_names=ts.CODING_TOOLS, model="",
                                           project_rules=selection.text("global_rules"))

                self.assertIn(present, system)
                self.assertNotIn(absent, system)
                self.assertNotIn("blank line after a multi-line comment", system)


class HistoricalContamination(unittest.TestCase):
    """Permanent: a product's build instructions must not reach another tree."""

    def test_foreign_build_instructions_never_reach_an_unrelated_project(self):
        rules = tempfile.mkdtemp()
        write_rules(rules, **{"25-product-build": FOREIGN, "26-boards": FOREIGN_UNSCOPED,
                              "20-generic": GENERIC})
        skills = src.skill_candidates([
            ("add-rootfs-package", "Run ./scripts/updiff.sh on verdin.", ("verdin", "virt64")),
            ("debug-build", "torizon build: use -e1c.", ())])

        for registered in (True, False):
            ws = workspace("pos_sol", registered=registered)

            for phase in (cs.IMPLEMENTATION, cs.GENERAL, cs.MIXED_IMPLEMENTATION):
                with self.subTest(registered=registered, phase=phase):
                    selection = cs.DeterministicContextSelector().select(
                        ws, phase, src.rule_candidates(rules) + skills,
                        request="Create a symlink in linux/rootfs/images to initrd.cpio.")
                    shown = "".join(item.text for item in selection.selected)

                    for word in FOREIGN_WORDS:
                        self.assertNotIn(word, shown)

    def test_the_same_rule_does_reach_the_project_it_names(self):
        rules = tempfile.mkdtemp()
        write_rules(rules, **{"25-product-build": FOREIGN})
        selection = cs.DeterministicContextSelector().select(
            workspace("verdin"), cs.IMPLEMENTATION, src.rule_candidates(rules))

        self.assertIn("updiff.sh", selection.text("global_rules"))


class PathAndTaskScope(Selection):
    def test_a_path_scoped_rule_needs_the_request_to_name_such_a_path(self):
        named = self.select(workspace(), request="Fix the warning in doc/source/index.rst.")
        unnamed = self.select(workspace(), request="Fix the parser in src/parse.c.")

        self.assertIn("rule:docs", named.ids())
        self.assertNotIn("rule:docs", unnamed.ids())
        self.assertIn("path scope mismatch", self.reasons(unnamed)["rule:docs"])

    def test_each_phase_gets_the_rules_written_for_it(self):
        expected = {
            cs.IMPLEMENTATION: {"rule:alpha", "rule:generic"},
            cs.MIXED_IMPLEMENTATION: {"rule:alpha", "rule:generic"},
            cs.NORMATIVE: {"rule:normative"},
            cs.MIXED_PREPASS: {"rule:normative"},
            cs.GENERAL: {"rule:general"},
            cs.MIXED_POSTCHECK: set(),
        }

        for phase, ids in expected.items():
            with self.subTest(phase=phase):
                self.assertEqual(set(self.select(workspace("alpha"), phase).ids()), ids)

    def test_normative_turns_get_no_implementation_context(self):
        extra = [Candidate("memory:project", SourceType.PROJECT_MEMORY, "m", "x"),
                 Candidate("skill:s", SourceType.PROJECT_SKILL, "s", "y",
                           scope=(cs.PROJECTS, "alpha")),
                 Candidate("retrieval:corpus", SourceType.RETRIEVED_CORPUS, "c", "z"),
                 Candidate("project:build", SourceType.BUILD_METADATA, "p", "b")]
        selection = self.select(workspace("alpha"), cs.NORMATIVE, extra=extra)

        for item_id in ("memory:project", "skill:s", "retrieval:corpus", "project:build"):
            self.assertNotIn(item_id, selection.ids())
            self.assertIn("wrong task class", self.reasons(selection)[item_id])

    def test_the_constraint_packet_belongs_to_the_implementation_pass_only(self):
        packet = [Candidate("normative:packet", SourceType.NORMATIVE_CONSTRAINT_SET,
                            "ncs-1", "C1 …", rendered_by_runtime=True)]

        for phase, kept in ((cs.MIXED_IMPLEMENTATION, True), (cs.MIXED_POSTCHECK, True),
                            (cs.MIXED_PREPASS, False), (cs.IMPLEMENTATION, False)):
            with self.subTest(phase=phase):
                ids = self.select(workspace(), phase, extra=packet).ids()
                self.assertEqual("normative:packet" in ids, kept)


class Budget(Selection):
    def test_optional_context_goes_first_and_mandatory_never(self):
        items = [Candidate("request", SourceType.USER_REQUEST, "u", "q" * 300, mandatory=True),
                 Candidate("memory:project", SourceType.PROJECT_MEMORY, "m", "m" * 3000),
                 Candidate("rule:big", SourceType.PROJECT_RULE, "r", "r" * 3000,
                           scope=(cs.GENERIC,)),
                 Candidate("project:build", SourceType.BUILD_METADATA, "b", "b" * 60)]
        budget = cs.Budget(window=2200, output_reservation=100, mandatory_tokens=101)
        selection = self.selector.select(workspace(), cs.IMPLEMENTATION, items, budget=budget)

        self.assertIn("request", selection.ids())
        self.assertIn("project:build", selection.ids())
        self.assertEqual(selection.dropped_for_budget, ["memory:project"])
        self.assertIn("budget", self.reasons(selection)["memory:project"])

    def test_nothing_is_dropped_when_it_fits(self):
        budget = cs.Budget(window=200000, output_reservation=65536, mandatory_tokens=5000)
        selection = self.selector.select(workspace("alpha"), cs.IMPLEMENTATION,
                                         src.rule_candidates(self.rules), budget=budget)

        self.assertEqual(selection.dropped_for_budget, [])


class Standards(unittest.TestCase):
    def ws(self, bound=None, **spec):
        from types import SimpleNamespace

        root = tempfile.mkdtemp()
        return from_session(project="alpha", spec=spec, registered=True, project_root=root,
                            corpus_root=root,
                            binding=SimpleNamespace(standard_id=bound[0], revision=bound[1])
                            if bound else None)

    TWO = {"standards": [{"id": "ACME-FRAME", "revision": "2024"},
                         {"id": "ACME-BUS", "revision": "3"}]}

    def test_a_named_standard_wins(self):
        choice = cs.select_standard(self.ws(**self.TWO), "What does ACME-BUS 3 require?",
                                    phase=cs.NORMATIVE)

        self.assertEqual(choice.standard, ("ACME-BUS", "3"))

    def test_two_named_standards_are_ambiguous(self):
        choice = cs.select_standard(self.ws(**self.TWO),
                                    "Compare ACME-BUS and ACME-FRAME.", phase=cs.NORMATIVE)

        self.assertTrue(choice.ambiguous)
        self.assertIsNone(choice.standard)

    def test_otherwise_the_binding_then_the_only_declared_standard(self):
        bound = cs.select_standard(self.ws(bound=("ACME-FRAME", "2024"), **self.TWO),
                                   "What does Rule 7.1-3 require?", phase=cs.NORMATIVE)
        several = cs.select_standard(self.ws(**self.TWO), "What does Rule 7.1-3 require?",
                                     phase=cs.NORMATIVE)
        one = cs.select_standard(self.ws(standards=[{"id": "ACME-BUS", "revision": "3"}]),
                                 "What does Rule 7.1-3 require?", phase=cs.NORMATIVE)

        self.assertEqual(bound.reason, "the established binding")
        self.assertTrue(several.ambiguous)
        self.assertEqual(one.standard, ("ACME-BUS", "3"))

    def test_a_turn_that_needs_no_standard_selects_none(self):
        self.assertIsNone(cs.select_standard(self.ws(**self.TWO), "ACME-BUS",
                                             phase=cs.GENERAL).standard)


class Phases(unittest.TestCase):
    def test_classes_map_to_phases(self):
        self.assertEqual(cs.phases("IMPLEMENTATION", bound=False, write=True),
                         (cs.IMPLEMENTATION,))
        self.assertEqual(cs.phases("GENERAL", bound=False, write=False), (cs.GENERAL,))
        self.assertEqual(cs.phases("NORMATIVE", bound=True, write=False), (cs.NORMATIVE,))
        self.assertEqual(cs.phases("MIXED", bound=True, write=True),
                         (cs.MIXED_QUESTION, cs.MIXED_PREPASS, cs.MIXED_IMPLEMENTATION))
        self.assertEqual(cs.phases("MIXED", bound=True, write=False), (cs.MIXED_QUESTION,))
        self.assertEqual(cs.phases("MIXED", bound=False, write=True), (cs.IMPLEMENTATION,))

    def test_the_paths_that_run_on_the_coding_core(self):
        self.assertTrue(cs.runs_on_coding_core(cs.IMPLEMENTATION, bound=False))
        self.assertTrue(cs.runs_on_coding_core(cs.MIXED_IMPLEMENTATION, bound=True))
        self.assertFalse(cs.runs_on_coding_core(cs.NORMATIVE, bound=True))
        self.assertFalse(cs.runs_on_coding_core(cs.MIXED_PREPASS, bound=True))


class ToolFamilies(unittest.TestCase):
    def test_the_coding_family_is_the_six_tools(self):
        selection = ts.DeterministicToolSelector().select(cs.IMPLEMENTATION, ts.CODING_TOOLS,
                                                          coding=True)

        self.assertEqual((selection.family, selection.tools), ("CODING", ts.CODING_TOOLS))

    def test_a_tool_outside_the_family_is_refused(self):
        with self.assertRaises(ts.ToolContractError):
            ts.DeterministicToolSelector().select(cs.IMPLEMENTATION,
                                                  ts.CODING_TOOLS + ("bash",), coding=True)

    def test_normative_and_post_check_families(self):
        selector = ts.DeterministicToolSelector()

        self.assertEqual(selector.select(cs.MIXED_PREPASS, ("standard.search",),
                                         coding=False).family, "NORMATIVE")
        post = selector.select(cs.MIXED_POSTCHECK, (), coding=False)
        self.assertEqual((post.family, post.tools), (ts.EVIDENCE_PROVIDERS, ()))

    def test_the_contract_holds_for_the_coding_view_and_refuses_an_unread_argument(self):
        from harness.tool_registry import ToolDefinition, coding_schemas

        schemas = coding_schemas()
        view = [ToolDefinition(name, "", dict(schemas[name]["parameters"]))
                for name in ("read_file", "search_files", "patch", "write_file", "terminal")]
        ts.check_contract(view, coding=True)

        bad = dict(schemas["terminal"]["parameters"])
        bad["properties"] = {**bad["properties"], "background": {"type": "boolean"}}

        with self.assertRaises(ts.ToolContractError):
            ts.check_contract([ToolDefinition("terminal", "", bad)], coding=True)


class Boundary(unittest.TestCase):
    def test_nothing_is_read_that_was_not_handed_over(self):
        self.assertEqual(src.rule_candidates(""), [])
        self.assertEqual(src.rule_candidates("/nonexistent/rules.d"), [])
        self.assertEqual(src.learned_candidate("/nonexistent/rules-learned.md"), [])

    def test_a_skill_must_say_where_it_applies(self):
        skills = src.skill_candidates([("a", "x", ()), ("b", "y", ("any",)),
                                       ("c", "z", ("alpha",))])
        selection = cs.DeterministicContextSelector().select(
            workspace("other"), cs.IMPLEMENTATION, skills)

        self.assertEqual(selection.ids(), ["skill:b"])


if __name__ == "__main__":
    unittest.main()


class ThroughTheOrchestration(unittest.TestCase):
    """The passes of a MIXED turn run on their own selections, and say so."""

    def run_turn(self):
        from context.context_engine import ContextItem, ContextLayer, Freshness
        from tests.test_mixed_orchestration import SATISFIED_ALL, MixedTurn, fix

        class Turn(MixedTurn):
            def runTest(self):
                pass

        turn = Turn()
        turn.setUp()
        self.addCleanup(turn.tearDown)
        normative = ContextItem(
            item_id="system:global-rules", layer=ContextLayer.SYSTEM_RULES, source="rules.d",
            content="\n\nNORMATIVE-ONLY: cite provisions by label.", priority=100,
            protected=True, freshness=Freshness.CURRENT, inclusion_reason="test")
        turn.run_mixed(fix() + [SATISFIED_ALL], phase_contexts={
            cs.MIXED_PREPASS: {"context_items": (normative,)},
            cs.MIXED_IMPLEMENTATION: {"coding_context": "\n\nIMPLEMENTATION-ONLY: tabs."}})

        return turn

    def test_each_pass_is_given_its_own_selection(self):
        turn = self.run_turn()
        prepass = turn.backend.calls[0]["system"]
        implementation = turn.backend.raw_calls[0]["messages"][0]["content"]

        self.assertIn("NORMATIVE-ONLY", prepass)
        self.assertNotIn("IMPLEMENTATION-ONLY", prepass)
        self.assertIn("IMPLEMENTATION-ONLY", implementation)
        self.assertNotIn("NORMATIVE-ONLY", implementation)

    def test_each_pass_records_its_tool_family_and_the_packet(self):
        from runtime.tracing import EventType

        turn = self.run_turn()
        families = {event.metadata["phase"]: event.metadata
                    for event in turn.events(EventType.TOOLSET_SELECTED)}
        packet = [event.metadata for event in turn.events(EventType.CONTEXT_SELECTED)
                  if event.metadata.get("type") == "NORMATIVE_CONSTRAINT_SET"]

        self.assertEqual(families[cs.MIXED_PREPASS]["family"], "NORMATIVE")
        self.assertTrue(all(name.startswith("standard.")
                            for name in families[cs.MIXED_PREPASS]["tools"]))
        self.assertEqual(families[cs.MIXED_IMPLEMENTATION]["family"], "CODING")
        self.assertEqual(sorted(families[cs.MIXED_IMPLEMENTATION]["tools"]),
                         sorted(ts.CODING_TOOLS))
        self.assertEqual(families[cs.MIXED_POSTCHECK]["tools"], [])
        self.assertEqual(packet[0]["source"], turn.context.mixed_record.packet.set_id)

    def test_audit_never_reaches_the_model(self):
        import json

        turn = self.run_turn()
        seen = json.dumps([call["messages"] for call in turn.backend.raw_calls]
                          + [str(call["system"]) for call in turn.backend.calls])

        for word in ("toolset_selected", "context_selected", "ORCHESTRATOR_EVIDENCE"):
            self.assertNotIn(word, seen)
