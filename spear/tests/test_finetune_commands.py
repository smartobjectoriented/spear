import tempfile
import unittest
from pathlib import Path

import rag_chat
from finetune_commands import (
    FinetuneCommandError, handle_finetune_command, parse_finetune_command,
)
from tests.test_training_controller import FakeTrainingLauncher
from training_controller import TrainingController
from tests.deployment_fixture import deployment
from training_launcher import TrainingExecutionConfiguration
from training_store import TrainingStore


class FinetuneCommandTests(unittest.TestCase):
    def test_strict_command_grammar_and_defaults(self):
        self.assertEqual(parse_finetune_command("/finetune start").method, "sft")
        self.assertTrue(parse_finetune_command("/finetune start dpo --force").force)
        self.assertEqual(parse_finetune_command("/finetune logs 12").lines, 12)
        for value in ("/finetune start sft --shell=x", "/finetune inspect",
                      "/finetune status extra", "/finetune unknown"):
            with self.assertRaises(FinetuneCommandError): parse_finetune_command(value)

    def test_empty_status_start_and_list_render_without_model(self):
        with tempfile.TemporaryDirectory() as directory:
            launcher = FakeTrainingLauncher()
            control = TrainingController(TrainingStore(directory + "/source"),
                directory + "/training",
                execution=deployment(source_model_revision="d" * 40),
                launcher=launcher)
            self.assertIn("COLLECTING", handle_finetune_command("/finetune status", control))
            self.assertIn("not started", handle_finetune_command("/finetune start", control))
            self.assertIn("No fine-tuning jobs", handle_finetune_command("/finetune list", control))
            self.assertEqual(launcher.start_calls, 0)

    def test_doctor_local_and_remote_flag_and_status_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            control = TrainingController(TrainingStore(directory + "/source"),
                directory + "/training",
                execution=deployment(source_model_revision="d" * 40),
                launcher=FakeTrainingLauncher())
            local = handle_finetune_command("/finetune doctor", control)
            self.assertIn("REMOTE", local); self.assertIn("NOT CHECKED", local)
            with self.assertRaises(Exception): parse_finetune_command("/finetune doctor nope")

    def test_finetune_is_not_a_model_tool(self):
        names = {spec.name for spec in rag_chat.TOOL_REGISTRY.list_specs()}
        self.assertNotIn("finetune", names)
        self.assertFalse(any("finetune" in str(item).lower()
                             for item in rag_chat.CANONICAL_TOOLS))

    def test_runtime_cannot_import_or_dispatch_training_control(self):
        root = Path(rag_chat.__file__).parent
        runtime = (root / "agent_runtime.py").read_text()
        registry = (root / "tool_registry.py").read_text()
        self.assertNotIn("training_controller", runtime)
        self.assertNotIn("finetune", runtime.lower())
        self.assertNotIn("finetune", registry.lower())
        chat = (root / "rag_chat.py").read_text()
        self.assertLess(chat.index('user_input == "/finetune"'),
                        chat.index("working_state = WorkingState.start"))


if __name__ == "__main__": unittest.main()
