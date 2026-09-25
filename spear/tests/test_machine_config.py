"""spear-configure: machine.env from what the machine has, never a guess."""

import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import machine_config
from machine_config import ConfigureError, LOCAL_MARKER, main, render


class FakeTty(io.StringIO):
    def isatty(self):
        return True


class MachineConfigTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.private = root / "spear-private"
        for name in ("rules.d", "skills", "benches"):
            (self.private / name).mkdir(parents=True)
        (self.private / "engagement.env").write_text("")
        self.out = root / "machine.env"
        self.embedding = mock.patch.object(
            machine_config, "detect_embedding",
            side_effect=lambda remote=None: {"model": "BAAI/bge-m3",
                                             "revision": "abc123",
                                             "target": "gpu-host",
                                             "verified": True})
        self.embedding.start()

    def tearDown(self):
        self.embedding.stop()
        self.temp.cleanup()

    def run_main(self, *args, stdin=None):
        with mock.patch("sys.stdout", io.StringIO()):
            return main(["--output", str(self.out), "--private",
                         str(self.private), *args], stdin=stdin)

    def test_it_writes_every_setting_it_found(self):
        self.assertEqual(self.run_main(), 0)
        text = self.out.read_text()

        self.assertIn(f"SPEAR_RULES_DIR={self.private.resolve()}/rules.d", text)
        self.assertIn("SPEAR_BENCH_DIR=", text)
        self.assertIn("engagement.env", text)
        self.assertIn("SPEAR_STANDARD_EMBED_REVISION=abc123", text)
        self.assertIn("SPEAR_STANDARD_EMBED_REMOTE=gpu-host", text)
        self.assertNotIn("SPEAR_STANDARD_RETRIEVAL_MODE=", text)

    def test_check_passes_on_a_file_it_wrote_whatever_its_date(self):
        self.run_main()
        self.assertEqual(self.run_main("--check"), 0)

    def test_check_fails_and_writes_nothing_when_it_differs(self):
        self.out.write_text("export SPEAR_STATE_DIR=/elsewhere\n")
        self.assertEqual(self.run_main("--check"), 1)
        self.assertEqual(self.out.read_text(), "export SPEAR_STATE_DIR=/elsewhere\n")

    def test_local_additions_survive_regeneration(self):
        self.run_main()
        self.out.write_text(self.out.read_text() + "export MY_OWN=1\n")
        self.run_main("--state", "/other", "--yes")

        text = self.out.read_text()
        self.assertIn('SPEAR_STATE_DIR="/other"', text)
        self.assertIn(LOCAL_MARKER + "\nexport MY_OWN=1", text)

    def test_a_differing_file_is_replaced_only_on_consent_and_backed_up(self):
        self.out.write_text("hand made\n")

        self.assertEqual(self.run_main(stdin=FakeTty("n\n")), 1)
        self.assertEqual(self.out.read_text(), "hand made\n")

        self.assertEqual(self.run_main(stdin=FakeTty("y\n")), 0)
        self.assertEqual((self.out.parent / "machine.env.bak").read_text(),
                         "hand made\n")

    def test_without_a_terminal_it_needs_yes(self):
        self.out.write_text("hand made\n")
        self.assertEqual(self.run_main(stdin=io.StringIO("")), 1)
        self.assertEqual(self.out.read_text(), "hand made\n")

    def test_no_private_tree_means_in_tree_content(self):
        text = render("/s", None, None)
        self.assertNotIn("SPEAR_RULES_DIR", text)
        self.assertIn("No private tree", text)
        self.assertIn("No embedding host", text)


class UnreachableHostTests(unittest.TestCase):
    """--check still checks everything else; writing refuses."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.out = Path(self.temp.name) / "machine.env"

    def tearDown(self):
        self.temp.cleanup()

    def run_main(self, *args):
        def detect(remote=None, *, verify=True):
            if verify:
                raise machine_config.HostUnreachable("cannot reach gpu-host")
            return {"model": "m", "revision": "r1", "target": "gpu-host",
                    "verified": False}

        with mock.patch.object(machine_config, "detect_embedding", side_effect=detect), \
                mock.patch("sys.stdout", io.StringIO()), \
                mock.patch("sys.stderr", io.StringIO()):
            return main(["--output", str(self.out), "--no-private", *args])

    def test_writing_refuses_a_revision_the_host_never_confirmed(self):
        self.assertEqual(self.run_main("--yes"), 2)
        self.assertFalse(self.out.exists())

    def test_check_compares_the_rest_against_this_machine(self):
        self.out.write_text(render("$HOME/.local/state/spear", None,
                                   {"model": "m", "revision": "r1",
                                    "target": "gpu-host"}))
        self.assertEqual(self.run_main("--check"), 0)


class EmbeddingDetectionTests(unittest.TestCase):

    def detect(self, here, there):
        with mock.patch("embedding.remote_target", return_value="gpu-host"), \
                mock.patch("embedding.active_model", return_value="BAAI/bge-m3"), \
                mock.patch.object(machine_config, "local_revision",
                                  return_value=here):
            return machine_config.detect_embedding(lambda model, target: there)

    def test_the_revision_both_ends_agree_on_is_pinned(self):
        self.assertEqual(self.detect("r1", "r1")["revision"], "r1")

    def test_a_disagreement_is_refused_not_written(self):
        with self.assertRaises(ConfigureError) as caught:
            self.detect("r1", "r2")
        self.assertIn("r2", str(caught.exception))

    def test_a_model_missing_here_is_refused(self):
        with self.assertRaises(ConfigureError):
            self.detect(None, "r1")

    def test_an_unreachable_host_is_not_a_missing_model(self):
        def unreachable(model, target):
            raise RuntimeError("cannot reach gpu-host: Connection timed out")

        with mock.patch("embedding.remote_target", return_value="gpu-host"), \
                mock.patch("embedding.active_model", return_value="BAAI/bge-m3"), \
                mock.patch.object(machine_config, "local_revision",
                                  return_value="r1"):
            with self.assertRaises(machine_config.HostUnreachable) as caught:
                machine_config.detect_embedding(unreachable)

        self.assertIn("Connection timed out", str(caught.exception))
        self.assertNotIn("resolves", str(caught.exception))

    def test_no_host_means_no_embedding(self):
        with mock.patch("embedding.remote_target", return_value=None):
            self.assertIsNone(machine_config.detect_embedding())


if __name__ == "__main__":
    unittest.main()
