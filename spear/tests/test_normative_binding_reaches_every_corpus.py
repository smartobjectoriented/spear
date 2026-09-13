"""A bound standard is told the rules, whatever corpus it was bound from.

The defect this pins down: the normative evidence contract -- what a
value_group means, that STRUCTURALLY_INCOMPLETE is authoritative, that
pressure does not lower the evidence bar -- lived in one project's system
prompt. Which system prompt a session runs on is chosen by corpus KIND
(`rag_chat.main` chose it by corpus kind; a corpus now names one through
prompt_file, and a corpus that names none runs on ADHOC_PROMPT, 435
characters that say nothing about standards).

So a session on any other corpus could bind a standard, call the standard.*
tools, receive a value_group, and never have been told what one is. The tools
answer the question; the contract that says how to read the answer arrived
only for one kind of corpus.

A binding is a property of the TURN. The corpus it was bound from is not, and
must not decide whether the rules are present. These tests drive a real bound
turn through TaskController and read the system string the model is actually
given -- not the source, not a prompt file.
"""

from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

import normative_precedence
from agent_runtime import AgentRuntime
from session_store import (FileSessionStore, SessionConfiguration,
                           SessionHandle, new_session_id)
from standard_ingest import ingest_pdf
from standard_retrieval import StandardRetrieval, rebuild_lexical_index
from standard_crossrefs import rebuild_cross_reference_index
from standard_store import StandardStore
from standard_tools import StandardToolService
from task_controller import TaskController, TaskRequest
from tests.standard_fixture import synthetic_pdf_bytes
from tests.test_agent_runtime import ScriptedBackend, make_context, text_turn
from tool_registry import ToolRegistry
from tool_router import ToolRouter

ROOT = Path(__file__).resolve().parent.parent

#: A phrase from the evidence contract that appears nowhere else, so finding
#: it in a system string means the contract itself arrived.
MARKER = "A packing_group is SEVERAL values"


class ABoundTurnCarriesTheRules(unittest.TestCase):
    """Driven end to end: what reaches the model, not what the source says."""

    def system_string(self, *, bind):
        """Run one turn and return the system prompt the backend received."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = root / "fixture.pdf"
            pdf.write_bytes(synthetic_pdf_bytes())
            standards = StandardStore(root / "standards")
            ingest_pdf(standards, pdf, standard_id="TEST-STD",
                       revision="TEST-1", source_origin="TEST_FIXTURE")
            rebuild_lexical_index(standards, "TEST-STD", "TEST-1")
            rebuild_cross_reference_index(standards, "TEST-STD", "TEST-1")

            backend = ScriptedBackend([text_turn("An answer.")])
            registry = ToolRegistry()
            StandardToolService(standards).register(registry)
            context = make_context(backend, rounds=4, actions=4)

            if bind:
                context.standard_binding = standards.binding(
                    "TEST-STD", "TEST-1").to_dict()

            sessions = FileSessionStore(root / "sessions")
            context.session = SessionHandle(
                sessions, new_session_id(),
                SessionConfiguration("/workspace", "project", "safe"))

            router = ToolRouter(registry)
            TaskController(AgentRuntime(), registry,
                           tool_executor=router.execute,
                           standard_store=standards).run(
                TaskRequest("What does it require?", context, ("/workspace",),
                            enable_planning=False))

            return backend.calls[0]["system"]

    # ── the fix ──────────────────────────────────────────────────────

    def test_a_bound_turn_is_given_the_evidence_contract(self):
        self.assertIn(MARKER, self.system_string(bind=True))

    def test_it_arrives_exactly_once(self):
        """Attached to the binding AND left in a prompt would say it twice."""
        self.assertEqual(self.system_string(bind=True).count(MARKER), 1)

    def test_an_unbound_turn_is_not_given_it(self):
        """No standard, no rules about standards: the contract is about
        reading retrieved clauses, and a session that retrieves none has no
        use for several hundred words saying how."""
        system = self.system_string(bind=False)

        self.assertNotIn(MARKER, system)
        self.assertNotIn("BOUND STANDARD", system)
        self.assertNotIn(normative_precedence.PRECEDENCE_RULE, system)

    def test_both_halves_of_the_contract_arrive_together(self):
        system = self.system_string(bind=True)

        self.assertIn(normative_precedence.PRECEDENCE_RULE, system)
        self.assertIn(normative_precedence.EVIDENCE_RULE, system)


class CorpusKindDoesNotDecide(unittest.TestCase):
    """The structural claim, checked where the structure is."""

    def test_the_controller_cannot_see_the_corpus_kind(self):
        """The strongest form of the guarantee: the code that attaches the
        rules has no access to the thing that used to gate them."""
        source = inspect.getsource(TaskController)

        for name in ("PROJECT_KIND", "edgem1", "prompt_file", "ADHOC_PROMPT"):
            with self.subTest(name=name):
                self.assertNotIn(name, source)

    def test_the_shipped_prompt_does_not_carry_the_contract(self):
        """It comes from the binding, so no prompt needs to carry it -- and a
        prompt that did would say it twice."""
        import rag_chat

        body = rag_chat.ADHOC_PROMPT

        self.assertNotIn("packing_group", body)
        self.assertNotIn("STRUCTURALLY_INCOMPLETE", body)
        self.assertNotIn("## Normative standards", body)

    def test_the_contract_lives_where_the_binding_is_announced(self):
        source = inspect.getsource(TaskController.run)

        self.assertIn("normative_precedence.EVIDENCE_RULE", source)
        self.assertIn("BOUND STANDARD", source)

    def test_the_contract_describes_no_corpus(self):
        """It is handed to every bound turn now, so it may not describe the
        one project whose prompt used to carry it."""
        lower = normative_precedence.EVIDENCE_RULE.lower()

        for absent in ("edgem", "verdin", "virt64", "bitbake", "vita"):
            with self.subTest(word=absent):
                self.assertNotIn(absent, lower)


if __name__ == "__main__":
    unittest.main()
