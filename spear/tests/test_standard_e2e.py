import tempfile
import unittest
from pathlib import Path

from agent_runtime import AgentRuntime
from result_store import ResultStore
from session_store import FileSessionStore, SessionConfiguration, SessionHandle, new_session_id
from standard_ingest import ingest_pdf
from standard_retrieval import StandardRetrieval, rebuild_lexical_index
from standard_crossrefs import rebuild_cross_reference_index
from standard_vector_index import rebuild_vector_index
from standard_store import StandardStore
from standard_tools import StandardToolService
from task_controller import TaskController, TaskRequest, TaskStatus
from tests.standard_fixture import synthetic_pdf_bytes
from tests.test_standard_hybrid_retrieval import FixtureSemanticEmbedder
from tests.test_agent_runtime import ScriptedBackend, make_context, text_turn, tool_turn
from tool_registry import ToolRegistry
from tool_router import ToolExecutionContext, ToolRouter


class StandardEndToEndTests(unittest.TestCase):
    def test_fake_model_search_fetch_cite_and_grounded_session(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); pdf = root / "fixture.pdf"
            pdf.write_bytes(synthetic_pdf_bytes())
            standards = StandardStore(root / "standards")
            ingest_pdf(standards, pdf, standard_id="TEST-STD", revision="TEST-1",
                       source_origin="TEST_FIXTURE")
            rebuild_lexical_index(standards, "TEST-STD", "TEST-1")
            rebuild_cross_reference_index(standards, "TEST-STD", "TEST-1")
            embedder = FixtureSemanticEmbedder()
            rebuild_vector_index(standards, "TEST-STD", "TEST-1", embedder)
            binding = standards.binding("TEST-STD", "TEST-1")
            source = StandardRetrieval(standards, embedder).search(
                "TEST-STD", "TEST-1", "shall reject reserved value")[0]
            citation = StandardRetrieval(standards).cite(
                "TEST-STD", "TEST-1", source.source_id).render()
            backend = ScriptedBackend([
                tool_turn("search_1", "standard.search", query="reserved value"),
                tool_turn("fetch_1", "standard.fetch", source_id=source.source_id),
                tool_turn("cite_1", "standard.cite", source_id=source.source_id),
                text_turn("Reserved value 0xF must be rejected. " + citation),
            ])
            registry = ToolRegistry(); StandardToolService(
                standards, embedder=embedder).register(registry)
            router = ToolRouter(registry); results = ResultStore(root / "results")
            tool_metadata = []

            def execute(agent, call_id, name, arguments, cache):
                envelope = router.execute(ToolExecutionContext(
                    agent.task_id, agent.trace, cache, result_store=results,
                    metadata={"standard_binding": agent.standard_binding},
                ), call_id, name, arguments)
                agent.standard_source_ids_used.update(
                    envelope.metadata.get("standard_source_ids", ()))
                tool_metadata.append(envelope.metadata)
                return envelope

            context = make_context(backend, executor=execute, rounds=8, actions=8)
            context.standard_binding = binding.to_dict()
            sessions = FileSessionStore(root / "sessions")
            context.session = SessionHandle(
                sessions, new_session_id(),
                SessionConfiguration("/workspace", "project", "safe"),
            )
            result = TaskController(
                AgentRuntime(), registry, tool_executor=execute,
                standard_store=standards,
            ).run(TaskRequest(
                "What is required for the reserved value?", context,
                ("/workspace",), enable_planning=False,
            ))
            self.assertEqual(result.status, TaskStatus.COMPLETED)
            self.assertIn(citation, result.final_response)
            self.assertIn(source.source_id, context.standard_source_ids_used)
            snapshot = sessions.load_snapshot(context.session_id)
            self.assertEqual(snapshot.standard_binding["revision"], "TEST-1")
            self.assertIn(source.source_id, snapshot.standard_source_ids_used)
            self.assertEqual(tool_metadata[0]["retrieval_mode_used"], "hybrid")
            self.assertNotIn("query", tool_metadata[0])
            system = backend.calls[0]["system"]
            self.assertIn("BOUND STANDARD", system)
            self.assertIn("Normative claims must use retrieved", system)
            self.assertNotIn("Implementations shall reject reserved value", system)

    def test_agent_runtime_has_no_standard_or_vita_orchestration_branch(self):
        runtime = Path(__file__).parents[1] / "agent_runtime.py"
        text = runtime.read_text("utf-8")
        self.assertNotIn("VITA", text)
        self.assertNotIn("standard.search", text)
        self.assertNotIn("standard.fetch", text)
