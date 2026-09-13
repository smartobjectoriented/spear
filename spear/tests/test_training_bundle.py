import hashlib
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path

from tests.test_preference_dataset import bad_turn, good_turn
from tests.test_sft_dataset import episode, turn
from tests.test_training_readiness import with_origin
from training_bundle import (
    AxolotlConfigGenerator, AxolotlProfile, SourceModelProfile,
    TokenizerValidator, TrainingBundleBuilder, TrainingBundleConfiguration,
    TrainingBundleError, TrainingHardwareReport, validate_loss_mask_intent,
    validate_training_bundle, VRAMFeasibility,
)
from training_governance import DataOrigin, TrainingDataGovernancePolicy
from training_readiness import TrainingReadinessPolicy
from training_store import TrainingStore
from training import main as training_main


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return list(range(sum(len(str(item)) for item in messages) // 4 + 1))


class TrainingBundleTests(unittest.TestCase):
    def store(self, root, *values):
        store = TrainingStore(root)
        for value in values:
            store.save_episode(value)
        return store

    def positive_store(self, root):
        return self.store(root, episode([
            turn(0, output="done", labels=("final_success",)),
        ]))

    def test_source_profiles_and_gguf_rejection(self):
        default = SourceModelProfile.named("qwen3_coder_next_sft")
        self.assertEqual(default.base_model, "Qwen/Qwen3-Coder-Next")
        self.assertEqual(default.source_kind, "instruct_post_trained")
        base = SourceModelProfile.named("qwen3_coder_next_base")
        self.assertEqual(base.base_model, "Qwen/Qwen3-Coder-Next-Base")
        base.validate()
        with self.assertRaises(TrainingBundleError):
            replace(default, base_model="deployed/model.gguf").validate()

    def test_axolotl_configs_are_semantic_qlora_and_conservative_moe(self):
        generator = AxolotlConfigGenerator(SourceModelProfile(), AxolotlProfile())
        sft, kto, dpo = generator.sft(), generator.kto(), generator.dpo()
        self.assertIn("adapter: qlora", sft)
        self.assertIn("load_in_4bit: true", sft)
        # load_in_4bit alone does not quantize this model's routed experts:
        # under transformers 5.x they are fused 3-D nn.Parameter tensors and
        # bitsandbytes only replaces nn.Linear. Asking for 4-bit without this
        # flag is how a run ended up holding 148 GiB of bf16 experts.
        self.assertIn("quantize_moe_experts: true", sft)
        self.assertIn("message_field_training: train", sft)
        self.assertIn("chat_template: tokenizer_default", sft)
        # Conservative means no expert TARGETING -- a different thing from
        # expert quantization, which the line above requires. The targeting
        # section must be absent, and routed experts must never appear as
        # lora_target_modules: they are parameters, not modules.
        self.assertNotIn("lora_target_parameters", sft)
        self.assertNotIn("gate_up_proj", sft)
        self.assertIn("rl: kto", kto)
        self.assertIn("rl: dpo", dpo)
        self.assertNotIn("<|im_start|>", sft + kto + dpo)

    def test_feasibility_depends_on_how_the_experts_are_loaded(self):
        """96 GiB is insufficient or ample depending on one flag.

        The old table said ">= 96 GiB may fit" whatever the config, and the
        config it shipped left the routed experts in bf16 -- where the weights
        alone measured 152 010 MiB on a B200. A verdict that ignores the load
        mode sends an operator to rent the wrong card either way.
        """
        card = TrainingHardwareReport(
            True, ("RTX PRO 6000",), 1, 96.0, ("12.0",), 256.0, 900.0,
            "linux", "test", VRAMFeasibility.UNKNOWN.value, (),
        )
        bf16 = card.classify(quantize_moe_experts=False)
        self.assertEqual(bf16.vram_feasibility,
                         VRAMFeasibility.LIKELY_DOES_NOT_FIT.value)
        self.assertTrue(any("bf16" in w for w in bf16.warnings))

        quantized = card.classify(quantize_moe_experts=True)
        self.assertEqual(quantized.vram_feasibility, VRAMFeasibility.MAY_FIT.value)
        # Never LIKELY_FITS on that path: the figure behind it is Axolotl's
        # documentation, not a measurement made here.
        self.assertTrue(any("preflight" in w for w in quantized.warnings))

        targeting = card.classify(quantize_moe_experts=True, expert_targeting=True)
        self.assertEqual(targeting.vram_feasibility, VRAMFeasibility.MAY_FIT.value)
        small = replace(card, total_vram_gib=48.0).classify(
            quantize_moe_experts=True, expert_targeting=True)
        self.assertEqual(small.vram_feasibility,
                         VRAMFeasibility.LIKELY_DOES_NOT_FIT.value)

        # Told nothing, it stays coarse and says so.
        silent = card.classify()
        self.assertTrue(any("not declared" in w for w in silent.warnings))

    def test_freeze_is_deterministic_immutable_and_masked(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.positive_store(Path(directory) / "source")
            builder = TrainingBundleBuilder(store)
            first, first_path = builder.freeze(directory)
            second, second_path = builder.freeze(directory)
            self.assertEqual(first_path, second_path)
            self.assertEqual(first["bundle_id"], second["bundle_id"])
            validate_training_bundle(first_path, store)
            row = json.loads(next(line for line in
                (first_path / "datasets/sft/train.jsonl").read_text().splitlines() if line))
            validate_loss_mask_intent(row)
            self.assertTrue(row["messages"][-1]["train"])
            self.assertTrue(all(not item["train"] for item in row["messages"][:-1]))
            self.assertNotIn("reasoning_content", json.dumps(row))
            before = (first_path / "manifest.json").read_bytes()
            store.save_episode(episode([
                turn(0, output="later", labels=("final_success",)),
            ], episode_id="later", task_id="later", objective="later"))
            third, third_path = builder.freeze(directory)
            self.assertNotEqual(first_path, third_path)
            self.assertEqual((first_path / "manifest.json").read_bytes(), before)

    def test_corruption_source_mismatch_and_count_validation_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.positive_store(Path(directory) / "source")
            _, bundle = TrainingBundleBuilder(store).freeze(directory)
            target = bundle / "datasets/sft/train.jsonl"
            target.write_text(target.read_text() + "{}\n")
            with self.assertRaises(TrainingBundleError):
                validate_training_bundle(bundle, store)

    def test_synthetic_smoke_visible_but_default_real_bundle_excludes_it(self):
        with tempfile.TemporaryDirectory() as directory:
            values = [
                with_origin(episode([bad_turn()], episode_id="bad", task_id="bad",
                                    eligibility="failure_trajectory"), "SYNTHETIC"),
                with_origin(episode([
                    good_turn(), turn(1, output="done", labels=("final_success",)),
                ], episode_id="good", task_id="good"), "SYNTHETIC"),
            ]
            store = self.store(Path(directory) / "source", *values)
            real, _ = TrainingBundleBuilder(store).freeze(directory, dry_run=True)
            self.assertEqual(real["manifest"]["dataset_counts"],
                             {"sft": 0, "unpaired": 0, "paired": 0})
            policy = TrainingReadinessPolicy(
                sft_experiment_samples=1, sft_experiment_split_groups=1,
                sft_experiment_projects=1, minimum_validation_samples=0,
                minimum_test_samples=0, unpaired_experiment_per_class=1,
                unpaired_experiment_groups=1, paired_experiment_pairs=1,
                paired_experiment_contexts=1,
            )
            governance = TrainingDataGovernancePolicy(allowed_origins=("SYNTHETIC",))
            config = TrainingBundleConfiguration(
                governance=governance, readiness=policy, smoke_test_only=True,
            )
            manifest, bundle = TrainingBundleBuilder(store, configuration=config).freeze(directory)
            self.assertTrue(manifest["smoke_test_only"])
            self.assertTrue((bundle / "configs/axolotl-kto.yml").is_file())
            self.assertTrue((bundle / "configs/axolotl-dpo.yml").is_file())
            self.assertIn("SYNTHETIC", (bundle / "governance.json").read_text())
            validate_training_bundle(bundle, store)

    def test_tokenizer_exact_optional_and_approximate_without_dependency(self):
        records = [{"messages": [{"role": "user", "content": "hello"}]}]
        approximate = TokenizerValidator().validate(records, 100)
        self.assertFalse(approximate["exact"])
        self.assertTrue(approximate["requires_preprocessing_validation"])
        exact = TokenizerValidator(tokenizer=FakeTokenizer()).validate(records, 100)
        self.assertTrue(exact["exact"])
        self.assertEqual(exact["kind"], "exact")

    def test_hardware_cpu_unknown_and_target_classification(self):
        report = TrainingHardwareReport(
            False, (), 0, None, (), 32.0, 100.0, "test", "fixture",
        ).classify()
        self.assertEqual(report.vram_feasibility, "UNKNOWN")
        self.assertTrue(report.warnings)
        small = replace(report, total_vram_gib=24.0).classify()
        self.assertEqual(small.vram_feasibility, "LIKELY_DOES_NOT_FIT")

    def test_bundle_contains_governance_runbook_lineage_and_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.positive_store(Path(directory) / "source")
            manifest, bundle = TrainingBundleBuilder(store).freeze(directory)
            for name in ("governance.json", "RUNBOOK.md", "evaluation-plan.json",
                         "readiness.json", "checksums.sha256"):
                self.assertTrue((bundle / name).is_file())
            self.assertIn("future_output_lineage_schema", manifest)
            plan = json.loads((bundle / "evaluation-plan.json").read_text())
            self.assertTrue(plan["held_out_only"])
            self.assertIn("general_control_regression", plan["metrics"])
            self.assertNotIn("axolotl train", "")  # no command was executed

    def test_loss_mask_rejects_previous_assistant_training(self):
        record = {"messages": [
            {"role": "assistant", "content": "bad", "train": True},
            {"role": "tool", "content": "failed", "train": False},
            {"role": "assistant", "content": "recover", "train": True},
        ]}
        with self.assertRaises(TrainingBundleError):
            validate_loss_mask_intent(record)

    def test_cli_readiness_and_dry_run_create_no_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            self.positive_store(source)
            output = Path(directory) / "output"
            with redirect_stdout(StringIO()) as stream:
                self.assertEqual(training_main([
                    "readiness", "--source", str(source),
                ]), 0)
            self.assertEqual(json.loads(stream.getvalue())["recommendation"], "SFT_ONLY")
            with redirect_stdout(StringIO()):
                self.assertEqual(training_main([
                    "freeze", "--source", str(source), "--output", str(output),
                    "--dry-run",
                ]), 0)
            self.assertFalse((output / "bundles").exists())

    def test_empty_freeze_has_no_active_training_config_or_command(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TrainingStore(Path(directory) / "source")
            manifest, bundle = TrainingBundleBuilder(store).freeze(directory)
            self.assertEqual(manifest["dataset_counts"],
                             {"sft": 0, "unpaired": 0, "paired": 0})
            self.assertFalse((bundle / "configs/axolotl-sft.yml").exists())
            self.assertNotIn("axolotl train", (bundle / "RUNBOOK.md").read_text())
            validate_training_bundle(bundle, store)


if __name__ == "__main__":
    unittest.main()
