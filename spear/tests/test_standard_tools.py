import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from agent_roles import AgentRole
from result_store import ResultStore
from standard_ingest import ingest_pdf
from standard_retrieval import rebuild_lexical_index
from standard_store import StandardStore
from standard_tools import (
    INVALID_SOURCE_ID, STANDARD_TOOL_NAMES, StandardToolService,
    _FETCH_RESULT_POLICY, render_fetch_for_model, standard_tool_specs,
)
from tests.standard_fixture import synthetic_pdf_bytes
from tool_exposure import ToolExposurePolicy
from tool_registry import ToolRegistry, ToolResultPolicy
from tool_router import (
    ToolExecutionContext, ToolHandlerResult, ToolResultStatus, ToolRouter,
)
from tracing import NullTraceRecorder, TraceEmitter


class StandardToolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); root = Path(self.temp.name)
        pdf = root / "fixture.pdf"; pdf.write_bytes(synthetic_pdf_bytes())
        self.store = StandardStore(root / "standards")
        ingest_pdf(self.store, pdf, standard_id="TEST-STD", revision="TEST-1",
                   source_origin="TEST_FIXTURE")
        rebuild_lexical_index(self.store, "TEST-STD", "TEST-1")
        self.binding = self.store.binding("TEST-STD", "TEST-1").to_dict()
        self.registry = ToolRegistry(); StandardToolService(self.store).register(self.registry)
        self.router = ToolRouter(self.registry)
        self.context = ToolExecutionContext(
            "task_12345678", TraceEmitter(NullTraceRecorder()), {},
            result_store=ResultStore(root / "results"),
            metadata={"standard_binding": self.binding},
        )

    def tearDown(self): self.temp.cleanup()

    def execute(self, name, arguments):
        return self.router.execute(self.context, "call_1", name, arguments)

    def test_search_fetch_and_cite_are_grounded_bounded_and_read_only(self):
        search = self.execute("standard.search", {"query": "reserved value"})
        self.assertTrue(search.success)
        result = json.loads(search.text)["results"][0]
        self.assertLessEqual(len(result["snippet"]), 500)
        fetched = self.execute("standard.fetch", {"source_id": result["source_id"]})
        payload = json.loads(fetched.text)
        self.assertEqual(payload["unit"]["source_id"], result["source_id"])
        cited = self.execute("standard.cite", {"source_id": result["source_id"]})
        self.assertEqual(json.loads(cited.text)["rendered"],
                         payload["citation"]["rendered"])
        self.assertTrue(all(self.registry.get(name).mutability.value == "read_only"
                            for name in STANDARD_TOOL_NAMES))

    def test_binding_is_required_and_model_cannot_switch_revision(self):
        denied = self.execute("standard.search", {
            "query": "reserved", "revision": "OTHER"})
        self.assertFalse(denied.success)
        self.assertIn("active StandardBinding", denied.text)
        no_binding = ToolExecutionContext(
            "task_12345678", TraceEmitter(NullTraceRecorder()), {}, metadata={})
        result = self.router.execute(no_binding, "call_2", "standard.search",
                                     {"query": "reserved"})
        self.assertFalse(result.success)

    def test_tools_hidden_without_binding_visible_with_binding_and_operator_absent(self):
        policy = ToolExposurePolicy()
        hidden = policy.select(self.registry, AgentRole.MAIN, objective="explain reserved")
        visible = policy.select(self.registry, AgentRole.MAIN, objective="explain reserved",
                                standard_bound=True)
        self.assertFalse(STANDARD_TOOL_NAMES & set(hidden.names))
        self.assertEqual(STANDARD_TOOL_NAMES, STANDARD_TOOL_NAMES & set(visible.names))
        self.assertNotIn("standard.ingest", {item.name for item in self.registry.list_specs()})
        self.assertNotIn("standard.use", {item.name for item in self.registry.list_specs()})

    def test_unknown_source_id_injection_fails_without_path_access(self):
        # The refusal is now data the caller can act on rather than an opaque
        # failure, so assert the property that matters directly: a traversal
        # attempt resolves nothing, reads nothing and touches nothing.
        result = self.execute("standard.fetch", {"source_id": "../../etc/passwd"})
        payload = json.loads(result.text)
        self.assertEqual(payload["error"], INVALID_SOURCE_ID)
        self.assertNotIn("unit", payload)
        self.assertNotIn("root:", result.text)
        self.assertEqual(result.affected_paths, ())
        self.assertIs(result.mutation, False)

    def test_large_fetch_uses_existing_result_store_reference(self):
        source = json.loads(self.execute(
            "standard.search", {"query": "shall reject"}).text)["results"][0]["source_id"]
        tiny = ToolRegistry()
        original = self.registry.get("standard.fetch")
        tiny.register(replace(
            original, result_policy=ToolResultPolicy(
                model_context_chars=40, store_large_results=True, preview_chars=20)),
            self.registry.handler("standard.fetch"))
        envelope = ToolRouter(tiny).execute(
            self.context, "call_large", "standard.fetch", {"source_id": source})
        self.assertTrue(envelope.truncated_for_model)
        self.assertIsNotNone(envelope.result_reference)
        self.assertTrue(self.context.result_store.exists(envelope.result_reference))

class FetchRenderingTests(unittest.TestCase):
    """fetch -> rendering -> ToolResultPolicy -> what the model is sent.

    The clause the call asked for used to arrive last. Serialised with sorted
    keys, `citation`, `neighbors` and `parent` came before `unit`, the size
    cap kept the first 1800 characters, and six fetches out of nine in one
    turn showed the model no unit text at all -- one of them the clause that
    contradicted the answer the turn then gave.
    """

    RULES = (
        "This clause states the following. "
        "Rule 9.9.9-1: the first requirement, stated in full. "
        "Rule 9.9.9-2: the second requirement, which a reader stopping here "
        "would take for the whole of it. "
        "Rule 9.9.9-3: the third requirement, which qualifies the second and "
        "is the one a truncated read loses. "
        "Observation 9.9.9-1: the closing observation, last in the unit and "
        "first to be cut."
    )

    def unit(self, source, text, *, section="9.9.9", page=42):
        return {
            "source_id": source, "section": section, "page": page,
            "standard_id": "TEST-STD", "revision": "TEST-1", "text": text,
            "citation_rendered": f"[TEST-STD TEST-1 §{section}, p.{page}, source {source}]",
            "source_content_sha256": "f" * 64, "source_pdf_sha256": "e" * 64,
            "unit_position": 1777, "warnings": [], "cross_references": [],
            "heading_path": ["9 Section", "9.9 Subsection", "9.9.9 Clause"],
            "extractor_version": "poppler-structure-v2", "layout_kind": "BODY",
        }

    def fetched(self, text=None, neighbours=6):
        source = "std-" + "a" * 32
        return {
            "unit": self.unit(source, text or self.RULES),
            "citation": {"standard_id": "TEST-STD", "revision": "TEST-1",
                         "section": "9.9.9", "page": 42, "source_id": source,
                         "rendered": f"[TEST-STD TEST-1 §9.9.9, p.42, source {source}]"},
            "parent": self.unit("std-" + "b" * 32, "Parent heading text."),
            "neighbors": [self.unit("std-" + f"{index:032x}", "Neighbour text. " * 40)
                          for index in range(neighbours)],
            "resolved_cross_references": [], "unresolved_cross_references": [],
        }

    def sent_to_model(self, fetched, **kwargs):
        """Everything between the handler and the conversation."""

        policy = _FETCH_RESULT_POLICY
        rendered = render_fetch_for_model(
            fetched, budget=policy.model_context_chars, **kwargs)
        stored = json.dumps(fetched, ensure_ascii=False, sort_keys=True)

        if len(rendered) > policy.model_context_chars:
            return (rendered[:policy.preview_chars]
                    + "\n\n[tool output limited for model context]"), stored

        return rendered, stored

    def test_the_requested_text_arrives_whole_and_first(self):
        shown, stored = self.sent_to_model(self.fetched())

        self.assertIn(self.RULES, shown, "the clause must arrive complete")
        self.assertIn("Rule 9.9.9-3", shown)
        self.assertIn("Observation 9.9.9-1", shown)
        self.assertLess(len(shown), _FETCH_RESULT_POLICY.model_context_chars)

        # First, so no cap can reach it before the metadata.

        self.assertLess(shown.index("Rule 9.9.9-1"), shown.index("CITATION:"))
        self.assertLess(shown.index("Observation 9.9.9-1"),
                        shown.index("NEIGHBOURS"))

        # And the structured form is untouched: it is what gets stored.

        self.assertIn('"unit"', stored)
        self.assertIn('"neighbors"', stored)

    def test_neighbours_and_metadata_never_displace_the_text(self):
        crowded = self.fetched(neighbours=40)
        shown, _ = self.sent_to_model(crowded)

        self.assertIn(self.RULES, shown)
        self.assertIn("further entries omitted", shown)

        # The checksums and positions of the neighbours are not what the call
        # asked for, and they are nowhere near the text.

        self.assertNotIn("source_content_sha256", shown)
        self.assertNotIn("unit_position", shown)

    def test_a_unit_too_long_says_so_and_offers_a_usable_continuation(self):
        long_text = "".join(f"Rule 9.9.9-{index}: requirement number {index}. "
                            for index in range(400))
        shown, _ = self.sent_to_model(self.fetched(text=long_text))

        self.assertIn("INCOMPLETE", shown)
        self.assertIn("text_offset=", shown)
        self.assertIn("must not be", shown)
        self.assertLessEqual(len(shown), _FETCH_RESULT_POLICY.model_context_chars)

        # The continuation is a call the model can actually make, and it
        # resumes where the first one stopped -- not a storage reference it
        # has no way to read.

        offset = int(shown.split("text_offset=")[1].split("]")[0])
        self.assertGreater(offset, 0)
        resumed, _ = self.sent_to_model(self.fetched(text=long_text),
                                        text_offset=offset)
        self.assertIn(f"continued from character {offset}", resumed)
        self.assertIn(long_text[offset:offset + 200], resumed)

    def test_the_router_shows_the_rendering_and_stores_the_structure(self):
        """The whole path, through the real router and the real policy."""

        registry = ToolRegistry()
        fetched = self.fetched()
        registry.register(
            replace(next(item for item in standard_tool_specs()
                         if item.name == "standard.fetch"),
                    handler_key="probe"),
            lambda context, arguments: ToolHandlerResult(
                json.dumps(fetched, ensure_ascii=False, sort_keys=True),
                model_text=render_fetch_for_model(
                    fetched, budget=_FETCH_RESULT_POLICY.model_context_chars)))

        with tempfile.TemporaryDirectory() as directory:
            context = ToolExecutionContext(
                task_id="task_render", trace=TraceEmitter(NullTraceRecorder()),
                cache={}, result_store=ResultStore(directory))
            envelope = registry and ToolRouter(registry).execute(
                context, "call", "standard.fetch",
                {"source_id": "std-" + "a" * 32})

        self.assertEqual(envelope.status, ToolResultStatus.OK)

        # What the model is sent.

        self.assertIn(self.RULES, envelope.model_content)
        self.assertIn("Observation 9.9.9-1", envelope.model_content)
        self.assertFalse(envelope.truncated_for_model)
        self.assertNotIn("source_content_sha256", envelope.model_content)

        # What is kept: the structured result, whole, and referenced when the
        # size policy says so.

        self.assertIn('"neighbors"', envelope.text)
        self.assertIn("source_content_sha256", envelope.text)
        self.assertGreater(len(envelope.text), len(envelope.model_content))
        self.assertIsNotNone(envelope.result_reference)

    def test_the_rendering_is_what_the_handler_hands_over(self):
        """The structured result is stored; the rendering is what is shown."""

        result = ToolHandlerResult(
            json.dumps(self.fetched(), sort_keys=True),
            model_text=render_fetch_for_model(
                self.fetched(), budget=_FETCH_RESULT_POLICY.model_context_chars))

        self.assertIsNotNone(result.model_text)
        self.assertIn("Observation 9.9.9-1", result.model_text)
        self.assertNotIn("Observation 9.9.9-1"[:0] + "sha256", result.model_text)


class StandardToolDescriptionTests(unittest.TestCase):
    """What the model is told about the two tools, before it picks one.

    A turn spent eighteen standard.cite calls, seventeen of them on units a
    fetch had just returned with their citations attached. The descriptions
    said nothing about that, and nothing about cite returning no text at all.
    """

    def spec(self, name):
        return next(item for item in standard_tool_specs() if item.name == name)

    def test_fetch_says_its_citations_are_ready_to_use(self):
        description = self.spec("standard.fetch").description

        self.assertIn("citation_rendered", description)
        self.assertIn("neighbour", description)
        self.assertIn("do NOT call standard.cite", description)

    def test_cite_says_what_it_is_not(self):
        description = self.spec("standard.cite").description

        self.assertIn("ONLY", description)
        self.assertIn("returns no", description)
        self.assertIn("NOT a reading of the standard", description)
        self.assertIn("already carries its citation", description)

    def test_both_descriptions_reach_the_model(self):
        """They are the spec text the registry hands to the provider."""

        registry = ToolRegistry()

        for item in standard_tool_specs():
            registry.register(item, lambda *args, **kwargs: "OK")

        offered = {item.name: item.description
                   for item in registry.definitions_for_model(role="main")}

        self.assertIn("do NOT call standard.cite", offered["standard.fetch"])
        self.assertIn("NOT a reading of the standard", offered["standard.cite"])
