import unittest

from training_data import TRAINING_SCHEMA_VERSION, TrainingEpisode
from training_governance import DataOrigin, TrainingDataGovernancePolicy


def episode(origin):
    return TrainingEpisode.from_dict({
        "schema_version": TRAINING_SCHEMA_VERSION,
        "episode_id": "episode_12345678", "task_id": "task_12345678",
        "session_id": None, "timestamp": "2026-01-01T00:00:00Z",
        "provenance": {
            "project": "test", "data_origin": "NORMAL_USAGE",
            "standard_binding": {
                "standard_id": "TEST", "revision": "1",
                "pdf_sha256": "a" * 64, "corpus_manifest_sha256": "b" * 64,
                "index_fingerprint": "c" * 64, "extractor_version": "test-v1",
                "corpus_schema_version": 1, "bound_at": None,
                "data_origin": origin,
            },
            "standard_source_ids_used": ["std-" + "d" * 32],
        },
        "task": {}, "turns": [], "tool_views": {}, "execution": {},
        "outcome": {}, "training_metadata": {
            "standard_export_prohibited": True,
            "requires_governance_review": True,
        },
    })


class StandardTrainingGovernanceTests(unittest.TestCase):
    def test_licensed_standard_provenance_is_retained_but_default_export_denied(self):
        value = episode(DataOrigin.LICENSED_STANDARD.value)
        decision = TrainingDataGovernancePolicy().assess(value)
        self.assertFalse(decision.allowed)
        self.assertIn("licensed_standard_review_required", decision.reasons)
        self.assertEqual(value.provenance["standard_source_ids_used"],
                         ["std-" + "d" * 32])
        self.assertNotIn("normative text", str(value.provenance))

    def test_synthetic_test_fixture_standard_remains_training_prohibited(self):
        value = episode(DataOrigin.TEST_FIXTURE.value)
        decision = TrainingDataGovernancePolicy().assess(value)
        self.assertFalse(decision.allowed)
        self.assertIn("test_fixture_excluded", decision.reasons)
