import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def imported_modules(tree):
    """Every module a tree imports, and each package it imports from."""
    names = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            names |= {node.module.split(".")[0], node.module}
        elif isinstance(node, ast.Import):
            names |= {alias.name.split(".")[0] for alias in node.names}

    return names


class ArchitectureBoundaryTests(unittest.TestCase):
    def test_runtime_layers_do_not_import_cli(self):
        lower_layers = (
            "runtime/agent_runtime.py", "runtime/task_controller.py",
            "context/context_engine.py", "runtime/compaction.py",
            "harness/tool_registry.py", "harness/tool_router.py",
            "runtime/session_store.py", "context/memory_store.py",
            "evidence/verification.py", "harness/checkpoint.py",
            "runtime/orchestration.py", "runtime/reviewer.py",
        )
        for name in lower_layers:
            tree = ast.parse((ROOT / name).read_text(encoding="utf-8"))
            self.assertNotIn("cli", imported_modules(tree), name)

    def test_task_controller_has_no_terminal_or_provider_construction(self):
        tree = ast.parse((ROOT / "runtime/task_controller.py").read_text(encoding="utf-8"))
        calls = {
            node.func.id for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertFalse({"input", "print"} & calls)
        imported = {alias.name for node in ast.walk(tree)
                    if isinstance(node, ast.ImportFrom) for alias in node.names}
        self.assertNotIn("OpenAICompatibleBackend", imported)
        self.assertNotIn("cli", imported_modules(tree))

    def test_prompts_keep_guidance_not_obsolete_enforcement_claims(self):
        """The prompts the platform SHIPS.

        It used to ship two. The second was a domain prompt for one customer's
        build system, now named through prompt_file by whoever deploys it and
        no longer in this tree -- so the two assertions that quoted its wording
        are gone rather than relocated to a file that never said it.
        """
        from cli import rag_chat

        guide = (ROOT / "tool-guide.md").read_text(encoding="utf-8")

        self.assertNotIn("The user will be asked for confirmation",
                         rag_chat.ADHOC_PROMPT)
        self.assertNotIn("<tool_call><function=edit_file>", guide)
        self.assertIn("Use the native tools exposed for the current role", guide)


if __name__ == "__main__":
    unittest.main()
