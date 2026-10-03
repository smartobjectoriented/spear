"""SPEAR's evidence plane for coding turns, and the routing around the core."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import completion
import project_build
import rag_chat
import session_replay
from agent.host import ToolRecord
from agent_roles import AgentRole
from model_backend import OpenAICompatibleBackend
from tool_exposure import ToolExposurePolicy
from tool_registry import CODING_TOOL_NAMES


def record(name, arguments, result, **kwargs):
    kwargs.setdefault("ok", True)
    return ToolRecord("c", name, arguments, result, **kwargs)


def log(*records):
    return [completion.canonical_entry(item) for item in records]


PATCH = record("patch", {"path": "src/a.sh"}, json.dumps({"success": True}),
               changed_paths=("src/a.sh",))


def terminal_record(command, output, code, **kwargs):
    return record("terminal", {"command": command},
                  json.dumps({"output": output, "exit_code": code, "error": None}),
                  ok=code == 0, exit_code=code, **kwargs)


class Verdict(unittest.TestCase):
    def test_no_change(self):
        self.assertEqual(completion.decide(log(terminal_record("ls", "x", 0))).state, "NO_CHANGE")

    def test_a_change_with_nothing_run_is_unverified(self):
        self.assertEqual(completion.decide(log(PATCH)).state, "UNVERIFIED")

    def test_a_passing_build_after_the_change_verifies(self):
        verdict = completion.decide(log(PATCH, terminal_record("make", "[100%] Built target a", 0)))

        self.assertEqual(verdict.state, "VERIFIED")

    def test_a_failing_build_does_not(self):
        self.assertEqual(completion.decide(log(PATCH, terminal_record("make", "error", 2))).state,
                         "UNVERIFIED")

    def test_a_timeout_is_not_a_pass(self):
        timed = terminal_record("make", "partial", 124, timed_out=True)

        self.assertEqual(completion.decide(log(PATCH, timed)).state, "UNVERIFIED")

    def test_a_refused_command_proves_nothing(self):
        refused = record("terminal", {"command": "python3 -m py_compile src/a.sh"},
                         json.dumps({"error": "refused: not allowed"}), ok=False, refused=True)

        self.assertEqual(completion.decide(log(PATCH, refused)).state, "UNVERIFIED")


class Qualify(unittest.TestCase):
    VERDICT = completion.Verdict("UNVERIFIED", ("src/a.sh",), "nothing ran afterwards")

    def test_unconditional_claims_are_marked_where_they_stand(self):
        answer = ("I changed the recipe. The implementation is complete and correct.\n"
                  "1. The link is created on every build ✓\n"
                  "It should survive a clean, but I could not run bitbake.")
        text = completion.qualify(answer, self.VERDICT)

        self.assertTrue(text.startswith("**UNVERIFIED**"))
        self.assertIn("complete and correct. *(not verified)*", text)
        self.assertIn("every build *(not verified)*", text)
        self.assertNotIn("✓", text)
        self.assertIn("It should survive a clean, but I could not run bitbake.", text)

    def test_a_verified_answer_is_untouched(self):
        verdict = completion.Verdict("VERIFIED", ("src/a.sh",))

        self.assertEqual(completion.qualify("It works.", verdict), "It works.")


class TheToolsetFollowsTheTaskType(unittest.TestCase):
    REQUEST = ("Can you create a symlink in out/links to data.bin and make "
               "sure it's always there after a clean and build")

    def select(self, objective, **kwargs):
        return ToolExposurePolicy().select(rag_chat.TOOL_REGISTRY, AgentRole.MAIN,
                                           objective=objective, **kwargs).names

    def test_implementation_gets_the_coding_vocabulary(self):
        names = self.select(self.REQUEST, toolset="coding", retrieval_available=False)

        self.assertEqual(set(names), {"read_file", "search_files", "patch",
                                      "terminal", "write_file", "delete_file"})

    def test_a_read_only_request_keeps_reading_only(self):
        names = self.select("which file defines add, without editing files",
                            toolset="coding", retrieval_available=False)

        self.assertEqual(set(names), {"read_file", "search_files", "terminal"})

    def test_an_index_adds_corpus_search(self):
        self.assertIn("search_corpus", self.select(self.REQUEST, toolset="coding"))

    def test_other_views_never_see_the_coding_tools(self):
        names = self.select(self.REQUEST)

        self.assertFalse(set(names) & set(CODING_TOOL_NAMES))
        self.assertIn("bash", names)



class TheCoreStatesNoSampling(unittest.TestCase):
    """The agent core's request is Hermes': no temperature, top_p or penalties."""

    def test_the_raw_request(self):
        from model_backend import complete_raw_messages

        sent = {}

        class Completions:
            def create(self, **kwargs):
                sent.update(kwargs)
                return iter(())

        client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
        complete_raw_messages(client, model="qwen3", messages=[{"role": "user", "content": "x"}],
                              tools=[], max_tokens=65536)

        self.assertEqual(set(sent), {"model", "messages", "max_tokens", "stream",
                                     "stream_options", "timeout"})
        self.assertEqual(sent["max_tokens"], 65536)
        self.assertEqual(sent["timeout"], 1800.0)

class AProbedMakefileIsNotAPromiseOfTests(unittest.TestCase):
    def probe(self, makefile):
        root = tempfile.mkdtemp()
        Path(root, "Makefile").write_text(makefile)

        return project_build.probe(root)

    def test_no_test_target_no_test_command(self):
        self.assertEqual(self.probe("all:\n\ttrue\n").test, "")

    def test_a_declared_target_is_used(self):
        self.assertEqual(self.probe("all:\n\ttrue\ntest:\n\ttrue\n").test, "make test")
        self.assertEqual(self.probe("check: all\n").test, "make check")




if __name__ == "__main__":
    unittest.main()
