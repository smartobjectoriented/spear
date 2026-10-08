"""A public image names only the corpora marked public.

The image registry is generated from the building machine's projects.json, a
map of its trees: names, paths, federations, collection names. Before the
profile reached the generator, a public image carried all of it -- private
product trees included -- in an image meant to be pulled by anyone.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT.parent / "scripts" / "docker" / "gen-registry.py"


def load_generator():
    spec = importlib.util.spec_from_file_location("gen_registry", GEN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PublicImageRegistry(unittest.TestCase):
    def generate(self, profile):
        with tempfile.TemporaryDirectory() as repo:
            (Path(repo) / "client").mkdir()
            (Path(repo) / "docker").mkdir()
            (Path(repo) / "client" / "projects.json").write_text(json.dumps({
                "open":   {"path": "corpora/open", "kind": "generic",
                           "public": True, "corpora": ["closed", "lib"]},
                "lib":    {"path": "corpora/lib", "kind": "generic",
                           "public": True},
                "closed": {"path": "corpora/closed", "kind": "generic"},
            }))
            gen = load_generator()
            gen.REPO = repo
            argv, sys.argv = sys.argv, ["gen-registry.py", "--profile", profile]
            try:
                gen.main()
            finally:
                sys.argv = argv

            return json.loads(
                (Path(repo) / "docker" / "projects.docker.json").read_text())

    def test_public_keeps_only_the_marked_corpora(self):
        self.assertEqual(sorted(self.generate("public")), ["lib", "open"])

    def test_a_federation_does_not_name_a_withheld_corpus(self):
        self.assertEqual(self.generate("public")["open"]["corpora"], ["lib"])

    def test_private_keeps_everything(self):
        self.assertEqual(sorted(self.generate("private")),
                         ["closed", "lib", "open"])

    def test_the_corpus_cli_can_mark_one_public(self):
        from cli import corpus_registry

        self.assertIn("--public", corpus_registry.CORPUS_USAGE)


if __name__ == "__main__":
    unittest.main()
