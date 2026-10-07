"""A bound change runs as three passes: normative, implementation, check.

End to end through TaskController, on a real store built from a synthetic
standard (tests/mixed_standard_fixture.py) and a real working tree:

  - the normative runtime, read-only, cites the provisions; the constraint
    packet is built from the provision records, not from its prose;
  - the frozen coding core sees the six accepted coding tools and a brief --
    no standard tool, no corpus text beyond the provisions themselves;
  - a tool-less check of the final source proposes candidate findings; a
    constraint is decided only by a deterministic predicate over the final
    source, never by the model's word or a second reading of it;
  - one repair for an established violation of a requirement, none for a
    model-only finding or an ambiguity;
  - the verdict keeps implementation evidence and normative status apart,
    and qualifies a compliance claim the check did not establish.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import normative_constraints as nc
from agent_runtime import AgentRuntime
from result_store import ResultStore
from standard_tools import StandardToolService
from task_controller import TaskController, TaskRequest
from tests.mixed_standard_fixture import REVISION, STANDARD_ID, build_store
from tests.test_agent_loop import Host, call, turn
from tests.test_agent_runtime import ScriptedBackend, make_context, text_turn
from tests.test_core_runtime import _Recording
from tool_registry import ToolRegistry, coding_tool_specs, native_tool_specs
from tool_router import ToolExecutionContext, ToolRouter
from tracing import EventType, TraceEmitter

OBJECTIVE = ("Update the header implementation in record.py so it complies with "
             "the Rule 4.2.1 provisions of the standard.")

RECORD = '''def build_header(flag_x, metadata=None):
    mode = 1
    count = [0, 0, 0]
    return {"mode": mode, "count": count, "metadata": metadata}
'''

FIXED = '''def build_header(flag_x, metadata=None):
    mode = 2 if flag_x else 1
    count = [0, 0, 0, 0]
    return {"mode": mode, "count": count, "metadata": metadata}
'''

# Compliant too, in shapes no predicate reads: only a model could judge it.
OPAQUE = '''def build_header(flag_x, metadata=None):
    mode = 1
    if flag_x:
        mode = 2
    count = [0] * 4
    return {"mode": mode, "count": count, "metadata": metadata}
'''

PREPASS_ANSWER = (
    "Rule 4.2.1-1 requires the mode field to be 2 when flag X is set. "
    "Rule 4.2.1-2 requires the count field to contain exactly four entries. "
    "Permission 4.2.1-3 allows the metadata field to be absent.")

def _unused(*args, **kwargs):
    raise AssertionError("the coding core runs its own tools")


class Recorder:
    def __init__(self):
        self.events = []

    def record(self, event):
        self.events.append(event)


CODING = ("read_file", "search_files", "patch", "write_file", "delete_file", "terminal")


class Backend(ScriptedBackend):
    """Legacy turns for the normative pass; raw turns for the core and the check."""

    def __init__(self, legacy, raw):
        super().__init__(legacy)
        self.raw, self.raw_calls = list(raw), []
        self.model = "qwen3"

    def complete_messages(self, messages, tools, *, max_tokens, on_token=None):
        self.raw_calls.append({"messages": messages, "tools": tools})
        return self.raw.pop(0)


def verdicts(*items):
    """A check's JSON answer: (id, status, line, text) per constraint."""
    return turn(json.dumps({"constraints": [
        {"id": cid, "status": status, "reason": f"{cid} {status.lower()}",
         "evidence": [{"file": "record.py", "line": line, "text": text}] if text else []}
        for cid, status, line, text in items]}))


CONFIRMED = turn('{"verdict": "CONFIRMED", "reason": "the code contradicts it"}')
REFUTED = turn('{"verdict": "REFUTED", "reason": "the constraint does not name that"}')

SATISFIED_ALL = verdicts(("C1", "SATISFIED", 2, "mode = 2 if flag_x else 1"),
                         ("C2", "SATISFIED", 3, "count = [0, 0, 0, 0]"),
                         ("C3", "SATISFIED", 4, '"metadata": metadata'))


def fix(answer="Done. The header is now compliant with SYNTH-MIXED.", content=FIXED):
    return [turn("", call("p1", "write_file", path="record.py", content=content)),
            turn(answer)]


def opaque(*items):
    """A check's answer about OPAQUE, quoting its lines."""
    lines = {"C1": (4, "mode = 2"), "C2": (5, "count = [0] * 4"),
             "C3": (6, '"metadata": metadata')}

    return verdicts(*((cid, status) + lines[cid] for cid, status in items))


class MixedTurn(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp()
        base = Path(self.directory)
        self.store = build_store(base)
        self.repo = base / "repo"
        self.repo.mkdir()
        (self.repo / "record.py").write_text(RECORD)
        (self.repo / "Makefile").write_text("all:\n\t@python3 -c 'import record'\n")
        for argv in (["git", "init", "-q"], ["git", "add", "-A"],
                     ["git", "-c", "user.email=a@b", "-c", "user.name=a", "commit", "-qm", "i"]):
            subprocess.run(argv, cwd=self.repo, check=True)

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def run_mixed(self, raw, *, objective=OBJECTIVE, legacy=None, project=None, runner=None,
                  phase_contexts=None):
        registry = ToolRegistry()
        StandardToolService(self.store).register(registry)

        for spec in native_tool_specs() + coding_tool_specs():
            try:
                registry.get(spec.name)
            except KeyError:
                registry.register(spec, _unused)

        router, results = ToolRouter(registry), ResultStore(Path(self.directory) / "results")

        def execute(agent, call_id, name, arguments, cache):
            return router.execute(ToolExecutionContext(
                agent.task_id, agent.trace, cache, result_store=results,
                metadata={"standard_binding": agent.standard_binding}), call_id, name, arguments)

        self.backend = Backend(legacy or [text_turn(PREPASS_ANSWER)], raw)
        self.recorder = Recorder()
        context = make_context(self.backend, executor=execute, rounds=12, actions=12,
                               context_limit=200_000, trace=TraceEmitter(self.recorder),
                               conversation=[])
        from model_backend import ConversationMessage, TextBlock
        context.conversation = [ConversationMessage("user", (TextBlock(objective),))]
        context.standard_binding = self.store.binding(STANDARD_ID, REVISION).to_dict()
        context.project_root = str(self.repo)
        context.normative_project = project
        context.normative_authority = "projects.json:test"
        context.normative_check_runner = runner
        context.phase_contexts = phase_contexts
        host = Host(str(self.repo))
        context.coding_host = lambda ctx, cache, record: _Recording(host, record)
        self.context = context
        self.result = TaskController(AgentRuntime(), registry, tool_executor=execute,
                                     standard_store=self.store).run(
            TaskRequest(objective, context, (str(self.repo),), enable_planning=False,
                        coding_core=True))

        return getattr(self.context, "mixed_record", None)

    def events(self, event_type):
        return [event for event in self.recorder.events if event.event_type == event_type]


class TheThreePasses(MixedTurn):
    def test_constraints_reach_the_core_and_the_check_judges_the_final_source(self):
        record = self.run_mixed(fix() + [SATISFIED_ALL])
        packet = record.packet

        self.assertEqual([(c.provision, c.modality) for c in packet.constraints],
                         [("Rule 4.2.1-1", "SHALL"), ("Rule 4.2.1-2", "SHALL"),
                          ("Permission 4.2.1-3", "MAY")])
        self.assertEqual(packet.constraints[0].condition, "when flag X is set")
        self.assertEqual(packet.constraints[1].cardinality, ("exactly four entries",))
        self.assertTrue(all(c.instance_id and c.source_id for c in packet.constraints))
        self.assertEqual(record.normative, nc.SATISFIED)
        self.assertEqual(record.implementation, "UNVERIFIED")
        self.assertEqual(record.verdict, "COMPLIANT BUT IMPLEMENTATION UNVERIFIED")
        self.assertFalse(record.repaired)
        self.assertEqual((self.repo / "record.py").read_text(), FIXED)

    def test_the_core_sees_the_coding_tools_and_the_brief_only(self):
        self.run_mixed(fix() + [SATISFIED_ALL])
        first = self.backend.raw_calls[0]
        names = [tool["function"]["name"] for tool in first["tools"]]
        request = first["messages"][-1]["content"]

        self.assertEqual(sorted(names), sorted(CODING))
        self.assertFalse(any(name.startswith("standard.") for name in names))
        self.assertIn("C2 [Rule 4.2.1-2, SHALL] Rule 4.2.1-2: The count field shall contain "
                      "exactly four entries.", request)
        self.assertNotIn("Older readers", request)            # informative: not a constraint
        self.assertNotIn("Synthetic Record Format", request)  # no corpus dump

    def test_the_check_is_tool_less_and_reads_code_not_prose(self):
        self.run_mixed(fix("Done. Everything complies, trust me.") + [SATISFIED_ALL])
        check = self.backend.raw_calls[-1]

        self.assertEqual(check["tools"], [])
        self.assertIn("mode = 2 if flag_x else 1", check["messages"][-1]["content"])
        self.assertNotIn("trust me", check["messages"][-1]["content"])

    def test_the_orchestration_is_audited(self):
        self.run_mixed(fix() + [SATISFIED_ALL])

        for event_type in (EventType.NORMATIVE_PREPASS_STARTED,
                           EventType.NORMATIVE_CONSTRAINT_SET_CREATED,
                           EventType.IMPLEMENTATION_STARTED, EventType.IMPLEMENTATION_FINISHED,
                           EventType.NORMATIVE_POSTCHECK_STARTED,
                           EventType.NORMATIVE_CONSTRAINT_STATUS,
                           EventType.FINAL_MIXED_VERDICT):
            with self.subTest(event=event_type.value):
                self.assertTrue(self.events(event_type))

        verdict = self.events(EventType.FINAL_MIXED_VERDICT)[0].metadata
        self.assertEqual(verdict["normative"], nc.SATISFIED)

    def test_a_verified_build_and_satisfied_constraints(self):
        record = self.run_mixed([
            turn("", call("p1", "write_file", path="record.py", content=FIXED)),
            turn("", call("t1", "terminal", command="make")),
            turn("Done. It builds."), SATISFIED_ALL])

        self.assertEqual(record.verdict, "VERIFIED + COMPLIANT")


class TheEvidenceBoundary(MixedTurn):
    """Implementation VERIFIED means the configured validation passed on the
    final source. It does not mean a property no check reads was proven: that
    is what the normative status is for, and it stays NOT_DEMONSTRATED."""

    WRONG = '''def build_header(flag_x, metadata=None):
    mode = 1
    if not flag_x:
        mode = 2
    count = [0] * 4
    return {"mode": mode, "count": count, "metadata": metadata}
'''

    def test_passing_project_tests_do_not_prove_a_hidden_property(self):
        record = self.run_mixed([
            turn("", call("p1", "write_file", path="record.py", content=self.WRONG)),
            turn("", call("t1", "terminal", command="make")),
            turn("Done. It builds, and the header is compliant with the standard."),
            verdicts(("C1", "SATISFIED", 3, "if not flag_x:"),
                     ("C2", "SATISFIED", 5, "count = [0] * 4"))])

        # The configured validation ran on the final source and passed.
        self.assertEqual(record.implementation, "VERIFIED")

        # No predicate reads this shape, and the model's word is not evidence.
        self.assertEqual(record.normative, nc.NOT_DEMONSTRATED)
        self.assertNotIn("COMPLIANT", record.verdict.replace("NOT COMPLIANT", ""))
        self.assertIn("*(compliance not established)*", self.result.final_response)

        # The property itself is false: an oracle outside SPEAR finds it.
        hidden = subprocess.run(
            [sys.executable, "-c", "import record; assert record.build_header(True)['mode'] == 2"],
            cwd=self.repo, capture_output=True)
        self.assertNotEqual(hidden.returncode, 0)


class Violations(MixedTurn):
    def test_one_repair_then_a_satisfied_recheck(self):
        record = self.run_mixed([
            turn("", call("p1", "write_file", path="record.py",
                          content=FIXED.replace("[0, 0, 0, 0]", "[0, 0, 0, 0, 0]"))),
            turn("Done."),
            verdicts(("C1", "SATISFIED", 2, "mode = 2 if flag_x else 1"),
                     ("C2", "VIOLATED", 3, "count = [0, 0, 0, 0, 0]"),
                     ("C3", "SATISFIED", 4, '"metadata": metadata')),
            CONFIRMED,
            turn("", call("p2", "write_file", path="record.py", content=FIXED)),
            turn("Repaired."),
            SATISFIED_ALL])

        self.assertTrue(record.repaired)
        self.assertEqual(len(record.checks), 2)
        self.assertEqual(record.normative, nc.SATISFIED)
        repair = self.backend.raw_calls[4]["messages"][-1]["content"]
        self.assertIn("C2 [Rule 4.2.1-2, SHALL]", repair)
        self.assertNotIn("C1 [", repair)
        self.assertTrue(self.events(EventType.REPAIR_STARTED))

    def test_there_is_only_one_repair(self):
        violated = verdicts(("C1", "SATISFIED", 2, "mode = 2 if flag_x else 1"),
                            ("C2", "VIOLATED", 3, "count = [0, 0, 0]"),
                            ("C3", "SATISFIED", 4, '"metadata": metadata'))
        record = self.run_mixed([
            turn("", call("p1", "write_file", path="record.py",
                          content=FIXED.replace("[0, 0, 0, 0]", "[0, 0, 0]"))),
            turn("Done."), violated, CONFIRMED, turn("I could not change it."), violated,
            CONFIRMED])

        self.assertEqual(record.verdict, "NOT COMPLIANT")
        self.assertEqual(len(record.checks), 2)
        self.assertEqual(self.backend.raw, [])

    def test_a_passing_build_does_not_make_a_violation_compliant(self):
        record = self.run_mixed([
            turn("", call("p1", "write_file", path="record.py",
                          content=FIXED.replace("[0, 0, 0, 0]", "[0, 0, 0]"))),
            turn("", call("t1", "terminal", command="make")),
            turn("Done. It builds and complies."),
            verdicts(("C2", "VIOLATED", 3, "count = [0, 0, 0]")), CONFIRMED,
            turn("Checked again: the header complies with the standard."),
            verdicts(("C2", "VIOLATED", 3, "count = [0, 0, 0]")), CONFIRMED])

        self.assertEqual(record.implementation, "VERIFIED")
        self.assertEqual(record.normative, nc.VIOLATED)
        self.assertEqual(record.verdict, "NOT COMPLIANT")
        self.assertIn("*(compliance not established)*", self.result.final_response)

    def test_a_models_ambiguity_is_a_candidate_and_repairs_nothing(self):
        record = self.run_mixed(fix(content=OPAQUE) + [
            opaque(("C1", "AMBIGUOUS"), ("C2", "VIOLATED"), ("C3", "SATISFIED")), CONFIRMED])
        statuses = {item.constraint_id: item for item in record.statuses}

        self.assertFalse(record.repaired)
        self.assertEqual(statuses["C1"].status, nc.NOT_DEMONSTRATED)
        self.assertEqual(statuses["C1"].candidate.candidate, nc.POSSIBLE_AMBIGUITY)
        self.assertEqual(record.verdict, "COMPLIANCE NOT DEMONSTRATED")
        self.assertEqual(len(record.checks), 1)


class EvidenceDiscipline(MixedTurn):
    def test_a_refuted_violation_the_source_contradicts_is_satisfied(self):
        record = self.run_mixed(fix() + [
            verdicts(("C1", "SATISFIED", 2, "mode = 2 if flag_x else 1"),
                     ("C2", "VIOLATED", 3, "count = [0, 0, 0, 0]"),
                     ("C3", "SATISFIED", 4, '"metadata": metadata')), REFUTED])
        statuses = {item.constraint_id: item for item in record.statuses}

        self.assertFalse(record.repaired)
        self.assertEqual(statuses["C2"].status, nc.SATISFIED)
        self.assertEqual(statuses["C2"].authority, "predicate:EXACT_COUNT")
        self.assertEqual(statuses["C2"].candidate.candidate, nc.POSSIBLE_VIOLATION)
        self.assertEqual(statuses["C2"].candidate.advisory, "REFUTED")

    def test_a_ticked_constraint_is_a_compliance_claim(self):
        self.assertIn("*(compliance not established)*",
                      nc.qualify("- C1: ✓ only one subtype is set", nc.NOT_DEMONSTRATED))

    def test_a_quote_the_file_lacks_is_not_a_source_fact(self):
        record = self.run_mixed(fix(content=OPAQUE) + [verdicts(
            ("C1", "SATISFIED", 4, "mode = 2"),
            ("C2", "SATISFIED", 5, "count = [0, 0, 0, 0]"),
            ("C3", "SATISFIED", 6, '"metadata": metadata'))])
        statuses = {item.constraint_id: item for item in record.statuses}

        self.assertEqual(statuses["C2"].candidate.facts, ())
        self.assertEqual(statuses["C2"].candidate.ungrounded, 1)
        self.assertEqual(statuses["C2"].status, nc.NOT_DEMONSTRATED)
        self.assertEqual(record.verdict, "COMPLIANCE NOT DEMONSTRATED")

    def test_an_unsupported_compliance_claim_is_qualified(self):
        self.run_mixed(fix("Done. The header now fully complies with SYNTH-MIXED.",
                           content=OPAQUE) + [opaque(("C1", "SATISFIED"))])

        self.assertIn("COMPLIANCE NOT DEMONSTRATED", self.result.final_response)
        self.assertIn("fully complies with SYNTH-MIXED. *(compliance not established)*",
                      self.result.final_response)

    def test_a_check_of_an_earlier_tree_is_stale(self):
        from unittest.mock import patch

        prints = iter(["before", "checked", "changed-after"])

        with patch.object(nc, "source_fingerprint", lambda root: next(prints)):
            record = self.run_mixed(fix() + [SATISFIED_ALL])

        self.assertEqual(record.normative, nc.NOT_DEMONSTRATED)


class NormativeAuthority(MixedTurn):
    """The check proposes; only the final source decides."""

    def test_a_confirmed_false_finding_repairs_nothing(self):
        record = self.run_mixed(fix(content=OPAQUE) + [
            opaque(("C1", "SATISFIED"), ("C2", "VIOLATED"), ("C3", "SATISFIED")), CONFIRMED])
        statuses = {item.constraint_id: item for item in record.statuses}
        audited = {event.metadata["constraint"]: event.metadata
                   for event in self.events(EventType.NORMATIVE_CONSTRAINT_STATUS)}

        self.assertFalse(record.repaired)
        self.assertEqual(self.backend.raw, [])
        self.assertEqual(statuses["C2"].status, nc.NOT_DEMONSTRATED)
        self.assertEqual(statuses["C2"].candidate.advisory, "CONFIRMED")
        self.assertEqual(statuses["C1"].status, nc.NOT_DEMONSTRATED)
        self.assertEqual(record.verdict, "COMPLIANCE NOT DEMONSTRATED")
        self.assertEqual((audited["C2"]["candidate"], audited["C2"]["authority"],
                          audited["C2"]["advisory"]), (nc.POSSIBLE_VIOLATION, "none", "CONFIRMED"))
        self.assertEqual(audited["C2"]["candidate_facts"][0]["excerpt"], "count = [0] * 4")
        self.assertFalse(self.events(EventType.REPAIR_STARTED))

    def test_a_plausible_violation_of_compliant_source_is_overruled_by_it(self):
        record = self.run_mixed(fix() + [
            verdicts(("C1", "VIOLATED", 2, "mode = 2 if flag_x else 1"),
                     ("C2", "SATISFIED", 3, "count = [0, 0, 0, 0]")), CONFIRMED])
        statuses = {item.constraint_id: item for item in record.statuses}

        self.assertFalse(record.repaired)
        self.assertEqual(statuses["C1"].status, nc.SATISFIED)
        self.assertEqual(statuses["C1"].authority, "predicate:CONDITIONAL_VALUE")
        self.assertEqual(record.normative, nc.SATISFIED)

    def test_a_may_claimed_violated_repairs_nothing(self):
        record = self.run_mixed(fix() + [
            verdicts(("C3", "VIOLATED", 4, '"metadata": metadata')), CONFIRMED])

        self.assertFalse(record.repaired)
        self.assertEqual(record.normative, nc.SATISFIED)

    def test_a_deterministic_violation_earns_one_repair_with_its_contradiction(self):
        record = self.run_mixed([
            turn("", call("p1", "write_file", path="record.py",
                          content=FIXED.replace("[0, 0, 0, 0]", "[0, 0, 0]"))),
            turn("Done."),
            turn('{"constraints": []}'),
            turn("", call("p2", "write_file", path="record.py", content=FIXED)),
            turn("Repaired."),
            turn('{"constraints": []}')])
        repair = self.backend.raw_calls[3]
        brief = repair["messages"][-1]["content"]
        first = [item["statuses"] for item in record.checks]

        self.assertTrue(record.repaired)
        self.assertIn("C2 [Rule 4.2.1-2, SHALL]", brief)
        self.assertIn("EXACT_COUNT: the provision states 4, the final source gives 3", brief)
        self.assertNotIn("C1 [", brief)
        for word in ("POSSIBLE_", "candidate", "advisory", "predicate:"):
            self.assertNotIn(word, json.dumps(repair["messages"]))
        self.assertEqual(sorted(tool["function"]["name"] for tool in repair["tools"]),
                         sorted(CODING))
        self.assertEqual({item.constraint_id: item.status for item in first[0]}["C2"],
                         nc.VIOLATED)
        self.assertEqual(record.normative, nc.SATISFIED)
        self.assertEqual(record.implementation, "UNVERIFIED")

    def test_a_failed_project_build_never_leaves_the_implementation_verified(self):
        from project_build import ProjectCommands

        def build(context):
            context.project_commands = ProjectCommands(build="make", source="make")
            context.project_verifier = lambda command: ("failed", "record.py:1: error")

        original = MixedTurn.run_mixed

        def run_mixed(test, raw, **kwargs):
            from unittest.mock import patch
            import agent_runtime

            real = agent_runtime.project_build_runs

            def with_harness(context, tool_log):
                build(context)
                return real(context, tool_log)

            with patch.object(agent_runtime, "project_build_runs", with_harness):
                return original(test, raw, **kwargs)

        record = run_mixed(self, [
            turn("", call("p1", "write_file", path="record.py", content=FIXED)),
            turn("", call("t1", "terminal", command="make")),
            turn("Done. It builds."), SATISFIED_ALL])

        self.assertEqual(record.implementation, "UNVERIFIED")
        self.assertEqual(record.verdict, "COMPLIANT BUT IMPLEMENTATION UNVERIFIED")


class CoverageInTheTurn(MixedTurn):
    def test_a_rule_the_pass_did_not_cite_is_closed_in_from_its_list(self):
        record = self.run_mixed(fix() + [SATISFIED_ALL], legacy=[text_turn(
            "Rule 4.2.1-2 requires the count field to contain exactly four entries.")])
        packet = {item.provision: item for item in record.packet.constraints}
        expanded = self.events(EventType.CONSTRAINT_COVERAGE_EXPANDED)[0].metadata
        decided = self.events(EventType.CONSTRAINT_APPLICABILITY_DECIDED)

        self.assertEqual(sorted(packet), ["Rule 4.2.1-1", "Rule 4.2.1-2"])
        self.assertEqual(packet["Rule 4.2.1-1"].origin, "CLOSURE")
        self.assertEqual(packet["Rule 4.2.1-1"].source_relation, "SAME_NORMATIVE_LIST")
        self.assertEqual(packet["Rule 4.2.1-1"].originating, packet["Rule 4.2.1-2"].instance_id)
        self.assertEqual(expanded["added"][0]["relation"], "SAME_NORMATIVE_LIST")
        self.assertEqual({event.metadata["applicability"] for event in decided}, {"APPLICABLE"})
        self.assertIn("Rule 4.2.1-1: The mode field", self.backend.raw_calls[0]["messages"][-1]["content"])
        self.assertEqual(record.report["closure_added"], 1)
        self.assertEqual(record.verdict, "COMPLIANT BUT IMPLEMENTATION UNVERIFIED")

    def test_an_objective_that_names_no_provision_leaves_applicability_unresolved(self):
        record = self.run_mixed(fix() + [SATISFIED_ALL], objective=(
            "Update the header implementation in record.py so it complies with the "
            "header field provisions of the standard."))

        self.assertEqual({item.applicability for item in record.packet.constraints},
                         {"UNRESOLVED"})
        self.assertEqual(record.normative, nc.NOT_DEMONSTRATED)
        self.assertEqual(record.report["unresolved_applicability"], 2)
        self.assertIn("applicability unresolved 2", self.result.final_response)


class TheOtherPathsAreUntouched(MixedTurn):
    def test_a_normative_question_stays_on_the_normative_runtime(self):
        self.run_mixed([], objective="What does Rule 4.2.1-2 require of the count field?",
                       legacy=[text_turn("Rule 4.2.1-2 requires exactly four entries.")])

        self.assertIsNone(getattr(self.context, "mixed_record", None))
        self.assertEqual(self.backend.raw_calls, [])

    def test_an_unbound_change_runs_on_the_core_without_a_packet(self):
        from unittest.mock import patch

        with patch.object(self.store, "binding", lambda *a, **k: None):
            pass

        backend = Backend([], [turn("", call("p1", "write_file", path="record.py",
                                             content=FIXED)), turn("Done.")])
        registry = ToolRegistry()

        for spec in native_tool_specs() + coding_tool_specs():
            registry.register(spec, _unused)

        context = make_context(backend, rounds=6, actions=6, context_limit=200_000)
        from model_backend import ConversationMessage, TextBlock
        context.conversation = [ConversationMessage("user", (TextBlock(
            "Update record.py so the count has four entries."),))]
        context.project_root = str(self.repo)
        host = Host(str(self.repo))
        context.coding_host = lambda ctx, cache, record: _Recording(host, record)
        TaskController(AgentRuntime(), registry, tool_executor=lambda *a: None).run(
            TaskRequest("Update record.py so the count has four entries.", context,
                        (str(self.repo),), enable_planning=False, coding_core=True))

        self.assertIsNone(getattr(context, "mixed_record", None))
        names = [tool["function"]["name"] for tool in backend.raw_calls[0]["tools"]]
        self.assertEqual(sorted(names), sorted(CODING))
        self.assertNotIn("Normative constraints", backend.raw_calls[0]["messages"][-1]["content"])


if __name__ == "__main__":
    unittest.main()


class TheNormativePassReadsOnlyTheBoundStandard(MixedTurn):
    def test_the_pass_is_read_only_with_the_standard_tools(self):
        self.run_mixed(fix() + [SATISFIED_ALL])
        names = [tool.name for tool in self.backend.calls[0]["tools"]]

        self.assertIn("standard.search", names)
        self.assertFalse(set(names) & {"patch", "write_file", "delete_file", "edit_file"})

    def test_another_standard_in_the_store_contributes_nothing(self):
        from standard_ingest import ingest_pdf
        from standard_retrieval import rebuild_lexical_index
        from tests.standard_fixture import synthetic_pdf_bytes

        other = Path(self.directory) / "other.pdf"
        other.write_bytes(synthetic_pdf_bytes())
        ingest_pdf(self.store, other, standard_id="TEST-STD", revision="TEST-1",
                   source_origin="TEST_FIXTURE")
        rebuild_lexical_index(self.store, "TEST-STD", "TEST-1")
        bound = {unit.source_id for unit in self.store.load_units(STANDARD_ID, REVISION)}
        foreign = {unit.source_id for unit in self.store.load_units("TEST-STD", "TEST-1")}
        record = self.run_mixed(fix() + [SATISFIED_ALL])

        self.assertTrue(record.packet.constraints)
        self.assertEqual(record.packet.standard_id, STANDARD_ID)
        self.assertTrue(all(item.source_id in bound for item in record.packet.constraints))
        self.assertFalse(any(item.source_id in foreign for item in record.packet.constraints))


class TheNormativePassIsIsolated(MixedTurn):
    def test_a_failed_pass_leaves_the_implementation_a_live_state(self):
        from agent_runtime import AgentRuntime
        from unittest.mock import patch

        original = AgentRuntime.run

        def failing_prepass(runtime, context, **kwargs):
            result = original(runtime, context, **kwargs)

            if context.execution_core == "legacy":
                from working_state import StateEventType
                context.apply_state_event(StateEventType.TASK_FAILED, summary="guard")

            return result

        with patch.object(AgentRuntime, "run", failing_prepass):
            record = self.run_mixed(fix() + [SATISFIED_ALL])

        self.assertEqual(record.normative, nc.SATISFIED)
        self.assertEqual((self.repo / "record.py").read_text(), FIXED)

    def test_a_tool_written_as_text_does_not_run_in_the_pass(self):
        from tests.test_agent_runtime import tool_turn

        self.run_mixed(fix() + [SATISFIED_ALL], legacy=[
            tool_turn("b1", "bash", command="cat record.py"), text_turn(PREPASS_ANSWER)])
        names = [tool.name for tool in self.backend.calls[0]["tools"]]

        self.assertTrue(all(name.startswith("standard.") for name in names))
        later = json.dumps([str(call) for call in self.backend.calls[1:]])
        self.assertIn("is not available here", later)
        self.assertNotIn("def build_header", later)


class ThePacketHoldsWhatTheAnswerCites(unittest.TestCase):
    def records(self):
        import provision_identity

        with tempfile.TemporaryDirectory() as directory:
            store = build_store(Path(directory))
            units = store.load_units(STANDARD_ID, REVISION)

        return {str(record.key): record for record in provision_identity.records_for_units(
            [unit.to_dict() if hasattr(unit, "to_dict") else unit for unit in units])}

    def test_a_source_cited_by_id_is_selected_and_nothing_else(self):
        records = self.records()
        rule = records["Rule 4.2.1-2"]
        packet = nc.build(records, f"The count is governed by [§4.2, source {rule.source_id}].",
                          standard_id=STANDARD_ID, revision=REVISION, objective="o")

        self.assertEqual([item.provision for item in packet.constraints], ["Rule 4.2.1-2"])

    def test_an_answer_that_cites_nothing_yields_no_constraint(self):
        packet = nc.build(self.records(), "The standard has rules about headers.",
                          standard_id=STANDARD_ID, revision=REVISION, objective="o")

        self.assertEqual(packet.constraints, ())


class ACitedSourceTheLedgerMissedComesFromTheStore(MixedTurn):
    def test_the_unit_is_read_from_the_bound_store(self):
        import mixed_orchestration
        from types import SimpleNamespace

        unit = next(item for item in self.store.load_units(STANDARD_ID, REVISION)
                    if item.text.startswith("Rule 4.2.1-2"))
        orchestrator = mixed_orchestration.MixedOrchestrator(
            SimpleNamespace(standard_store=self.store))
        found = orchestrator._cited_sources(
            SimpleNamespace(standard_id=STANDARD_ID, revision=REVISION),
            f"Governed by [§4.2, source {unit.source_id}].", {})
        packet = nc.build(found, f"source {unit.source_id}", standard_id=STANDARD_ID,
                          revision=REVISION, objective="o")

        self.assertEqual([item.provision for item in packet.constraints], ["Rule 4.2.1-2"])
        self.assertEqual(packet.constraints[0].source_id, unit.source_id)
