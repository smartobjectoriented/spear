import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ArchitectureBoundaryTests(unittest.TestCase):
    def test_runtime_layers_do_not_import_cli(self):
        lower_layers = (
            "agent_runtime.py", "task_controller.py", "context_engine.py",
            "compaction.py", "tool_registry.py", "tool_router.py",
            "session_store.py", "memory_store.py", "verification.py",
            "checkpoint.py", "orchestration.py", "reviewer.py",
        )
        for name in lower_layers:
            tree = ast.parse((ROOT / name).read_text(encoding="utf-8"))
            imports = {
                node.module for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
            }
            direct = {
                alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
                for alias in node.names
            }
            self.assertNotIn("rag_chat", imports | direct, name)

    def test_task_controller_has_no_terminal_or_provider_construction(self):
        tree = ast.parse((ROOT / "task_controller.py").read_text(encoding="utf-8"))
        calls = {
            node.func.id for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertFalse({"input", "print"} & calls)
        imports = {
            node.module for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        self.assertNotIn("model_backend.OpenAICompatibleBackend", imports)
        self.assertNotIn("rag_chat", imports)

    def test_prompts_keep_guidance_not_obsolete_enforcement_claims(self):
        """The prompts the platform SHIPS.

        It used to ship two. The second was a domain prompt for one customer's
        build system, now named through prompt_file by whoever deploys it and
        no longer in this tree -- so the two assertions that quoted its wording
        are gone rather than relocated to a file that never said it.
        """
        import rag_chat

        guide = (ROOT / "tool-guide.md").read_text(encoding="utf-8")

        self.assertNotIn("The user will be asked for confirmation",
                         rag_chat.ADHOC_PROMPT)
        self.assertNotIn("<tool_call><function=edit_file>", guide)
        self.assertIn("Use the native tools exposed for the current role", guide)


if __name__ == "__main__":
    unittest.main()
