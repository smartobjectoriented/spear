import unittest

from agent_roles import AgentRole, explorer_role, planning_role, reviewer_role
from tool_exposure import ToolExposurePolicy
from tool_registry import ToolRegistry, native_tool_specs
from agent_runtime import AgentRuntime


def registry():
    value = ToolRegistry()
    for spec in native_tool_specs():
        value.register(spec, None if spec.name == "bash" else lambda *_: None)
    return value


class ToolExposureTests(unittest.TestCase):
    def setUp(self):
        self.registry = registry()
        self.policy = ToolExposurePolicy()

    def test_main_implementation_gets_mutation_but_not_unrequested_web_memory(self):
        view = self.policy.select(self.registry, AgentRole.MAIN,
                                  objective="implement and test the fix")
        self.assertIn("edit_file", view.names)
        self.assertNotIn("search_internet", view.names)
        self.assertNotIn("remember", view.names)

    def test_main_information_view_is_lazy_and_ordered(self):
        """Lazy about the specialised tools, never about the workspace floor.

        A local project turn always keeps a way to read, search, edit and
        run. The mutation heuristic read "correct main.py" as read-only --
        "correct" was not in its verb list -- and handed the turn four
        schemas, none of which could change a file.
        """

        view = self.policy.select(self.registry, AgentRole.MAIN,
                                  objective="explain the architecture")
        self.assertEqual(view.names,
                         ("bash", "edit_file", "write_file", "search_corpus"))
        self.assertGreater(view.schema_token_estimate, 0)
        self.assertTrue(view.approximate)
        full_cost = AgentRuntime._tool_schema_tokens(
            self.registry.definitions_for_model(), True,
        )
        self.assertEqual(len(self.registry.definitions_for_model()), 10)
        self.assertLess(view.schema_token_estimate, full_cost)

    def test_read_only_role_views_match_trusted_specs(self):
        for role in (planning_role(), explorer_role(), reviewer_role()):
            view = self.policy.select(self.registry, role, objective="inspect repository")
            self.assertEqual(view.names, ("bash", "search_corpus"))
            self.assertNotIn("edit_file", view.names)
            self.assertNotIn("search_internet", view.names)
            self.assertNotIn("remember", view.names)

    def test_hidden_forbidden_tool_cannot_be_requested_through_role_view(self):
        view = self.policy.select(self.registry, reviewer_role(),
                                  objective="edit and remember then use web")
        self.assertEqual(view.names, ("bash", "search_corpus"))
        reviewer_role().validate_registry(self.registry)

    def test_memory_requires_both_policy_and_explicit_intent(self):
        view = self.policy.select(
            self.registry, AgentRole.MAIN,
            objective="search the latest information online and remember it",
            web_enabled=True, memory_write_enabled=True,
        )
        self.assertIn("search_internet", view.names)
        # fetch_url is WEB too: finding a page and reading it are exposed alike.
        self.assertIn("fetch_url", view.names)
        self.assertIn("remember", view.names)

    def test_search_internet_never_depends_on_the_wording(self):
        """Guessing web intent from vocabulary withheld the capability whenever
        the phrasing missed. "please get the complete Code-G pdf" matched no
        keyword, so the model spent eleven turns narrating "let me fetch the
        specification from the web" with no tool to call — and even a bare
        `fetch https://...` matched nothing. Search is how a network intent is
        DISCOVERED, so it is never withheld on wording."""
        for objective in ("please get the complete Code-G pdf so that we can "
                          "inject it in our RAG as a standard.",
                          "fetch https://nvlpubs.nist.gov/nistpubs/IR/x.pdf",
                          "read this page and summarise it",
                          "explain the architecture",
                          "correct main.py"):
            view = self.policy.select(self.registry, AgentRole.MAIN,
                                      objective=objective, web_enabled=True)
            self.assertIn("search_internet", view.names, objective)

    def test_fetch_url_needs_an_address_or_a_network_intent(self):
        """It retrieves ONE named address, so a task naming none has nothing
        for it to do. Offered to "correct main.py" — which had no edit_file —
        a run tried fetch_url(save_as="main.py") to write a local file."""

        for objective in ("please get the complete Code-G pdf so that we can "
                          "inject it in our RAG as a standard.",
                          "fetch https://nvlpubs.nist.gov/nistpubs/IR/x.pdf",
                          "read this page and summarise it",
                          "download the upstream changelog"):
            view = self.policy.select(self.registry, AgentRole.MAIN,
                                      objective=objective, web_enabled=True)
            self.assertIn("fetch_url", view.names, objective)

        for objective in ("correct main.py",
                          "Find the failing assertion in the long evidence "
                          "file and correct main.py.",
                          "explain the architecture"):
            view = self.policy.select(self.registry, AgentRole.MAIN,
                                      objective=objective, web_enabled=True)
            self.assertNotIn("fetch_url", view.names, objective)

    def test_a_local_task_keeps_the_tools_it_needs(self):
        """The reported failure, as a contract.

        "Find the failing assertion … and correct main.py" came back with
        four schemas and no edit_file, and the model reached for the network
        tool it had been given instead.
        """

        view = self.policy.select(
            self.registry, AgentRole.MAIN, web_enabled=True,
            objective="Find the failing assertion in the long evidence file "
                      "and correct main.py.")

        for name in ("bash", "edit_file", "write_file", "search_corpus"):
            self.assertIn(name, view.names)

        self.assertNotIn("fetch_url", view.names)

    def test_an_explicit_prohibition_takes_the_write_tools_away(self):
        """The observed task, verbatim.

        "without editing files" contains the word "editing", which _MUTATION
        matches, so the mutation heuristic concluded the opposite of what the
        request said. The prohibition is read FIRST, and it wins: the floor
        guarantees a way to work, never a way the request refused.
        """

        view = self.policy.select(
            self.registry, AgentRole.MAIN, web_enabled=True,
            objective="Answer which file defines add without editing files.")

        self.assertTrue(view.read_only)
        self.assertIn("bash", view.names)
        self.assertIn("search_corpus", view.names)

        for name in ("edit_file", "write_file", "append_file", "delete_file",
                     "remember"):
            self.assertNotIn(name, view.names)

    def test_a_task_that_asks_for_a_change_keeps_its_tools(self):
        """The other half: nothing here forbids anything."""

        view = self.policy.select(
            self.registry, AgentRole.MAIN, web_enabled=True,
            objective="Find the failing assertion in the long evidence file "
                      "and correct main.py.")

        self.assertFalse(view.read_only)

        for name in ("bash", "edit_file", "write_file", "search_corpus"):
            self.assertIn(name, view.names)

    def test_a_prohibition_scoped_to_one_file_is_not_a_read_only_task(self):
        """Two real objectives that name something to leave alone.

        Reading either as read-only would take away the tools the task needs
        to do the very thing it asks for.
        """

        for objective in (
            "Fix the greeting function and do not edit the unrelated data file.",
            "Rename the configuration key host to endpoint in config.py and "
            "its test, without changing unrelated.py.",
        ):
            with self.subTest(objective=objective):
                view = self.policy.select(self.registry, AgentRole.MAIN,
                                          objective=objective)
                self.assertFalse(view.read_only)
                self.assertIn("edit_file", view.names)

    def test_the_prohibition_beats_an_explicit_mutation_expectation(self):
        """A caller saying "this one writes" does not override the request."""

        view = self.policy.select(
            self.registry, AgentRole.MAIN, mutation_expected=True,
            objective="review src/ read-only and report what you find")

        self.assertTrue(view.read_only)
        self.assertNotIn("edit_file", view.names)

    def test_read_only_wording_that_must_and_must_not_trigger(self):
        for objective in ("without editing files", "without modifying anything",
                          "without changing the code", "do not edit files",
                          "do not modify the repository", "don't change anything",
                          "read-only", "a read only pass over the tree"):
            with self.subTest(objective=objective):
                self.assertTrue(self.policy.read_only_intent(objective))

        for objective in ("edit main.py", "do not mention the unrelated tests",
                          "do not edit unrelated.py", "without changing utils.py",
                          "correct main.py"):
            with self.subTest(objective=objective):
                self.assertFalse(self.policy.read_only_intent(objective))

    def test_the_floor_does_not_cross_the_role_boundary(self):
        """A guaranteed set is not a way around a read-only role."""

        for role in (planning_role(), explorer_role(), reviewer_role()):
            view = self.policy.select(self.registry, role,
                                      objective="correct main.py",
                                      web_enabled=True)
            self.assertEqual(view.names, ("bash", "search_corpus"))

    def test_web_enabled_is_still_the_switch(self):
        """Withholding them is a decision (--no-network), not an inference."""
        view = self.policy.select(self.registry, AgentRole.MAIN,
                                  objective="fetch https://example.com/x.pdf",
                                  web_enabled=False)
        self.assertNotIn("fetch_url", view.names)
        self.assertNotIn("search_internet", view.names)
        self.assertIn("bash", view.names)            # the rest is untouched


if __name__ == "__main__":
    unittest.main()
