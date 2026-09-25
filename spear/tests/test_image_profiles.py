"""What an image is allowed to carry, decided from what a document declares.

The store records, per ingested document, `source_origin` and whether the
original file was retained beside the extracted corpus. An image built to be
handed over must take only what says PUBLIC -- and the consequence of getting
this wrong is not a broken build but a licensed document travelling inside a
container somebody passes on.

So the rule is tested where it is decided, on a synthetic store: a list of
"the public ones" kept in the repository would have to name a customer's
standard in order to exclude it, and would be one forgotten edit away from
shipping it.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAGER = ROOT.parent / "scripts" / "docker" / "stage-standards.py"

PUBLIC = ("ACME-PUB", "R1", {"standard_id": "ACME-PUB", "revision": "R1",
                             "source_origin": "PUBLIC",
                             "raw_pdf_retained": False})
LICENSED = ("ACME-LIC", "R2", {"standard_id": "ACME-LIC", "revision": "R2",
                               "source_origin": "LICENSED_STANDARD",
                               "raw_pdf_retained": True})
UNDECLARED = ("ACME-UNK", "R3", {"standard_id": "ACME-UNK", "revision": "R3"})


def store(tmp, documents, bound=None):
    """A store on disk, in the shape the real one has."""
    root = Path(tmp) / "standards"

    for standard_id, revision, manifest in documents:
        d = root / standard_id / revision
        (d / "source").mkdir(parents=True)
        (d / "manifest.json").write_text(json.dumps(manifest))
        (d / "corpus").mkdir()

        if manifest.get("raw_pdf_retained"):
            (d / "source" / "original.pdf").write_bytes(b"%PDF-1.4 not really")

    if bound:
        (root / ".active-binding.json").write_text(json.dumps(
            {"active": {"standard_id": bound[0], "revision": bound[1]},
             "schema_version": 1}))

    return root


def stage(root, profile):
    out = root.parent / f"staged-{profile}"
    result = subprocess.run(
        [sys.executable, str(STAGER), "--profile", profile,
         "--store", str(root), str(out)],
        capture_output=True, text=True, check=True)

    return out, result.stdout


def taken(out):
    return {f"{p.parent.name} {p.name}" for p in out.glob("*/*")
            if p.is_dir()}


class APublicImageTakesOnlyWhatSaysPublic(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = store(self.tmp.name, [PUBLIC, LICENSED, UNDECLARED])

    def test_the_public_document_travels(self):
        out, _ = stage(self.root, "public")

        self.assertIn("ACME-PUB R1", taken(out))

    def test_the_licensed_one_does_not(self):
        out, _ = stage(self.root, "public")

        self.assertNotIn("ACME-LIC R2", taken(out))

    def test_nor_does_one_that_declares_nothing(self):
        """Unknown provenance has not earned the benefit of the doubt in an
        image somebody is about to hand to someone else."""
        out, _ = stage(self.root, "public")

        self.assertNotIn("ACME-UNK R3", taken(out))

    def test_the_original_document_never_leaves_with_it(self):
        out, _ = stage(self.root, "public")

        self.assertEqual(list(out.rglob("*.pdf")), [])

    def test_the_build_says_what_it_left_out(self):
        _, said = stage(self.root, "public")

        self.assertIn("LEFT OUT of a public image", said)


class APrivateImageTakesEverythingAndSaysSo(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = store(self.tmp.name, [PUBLIC, LICENSED])

    def test_it_takes_both(self):
        out, _ = stage(self.root, "private")

        self.assertEqual(taken(out), {"ACME-PUB R1", "ACME-LIC R2"})

    def test_it_names_the_licensed_material_it_carries(self):
        _, said = stage(self.root, "private")

        self.assertIn("licensed normative material", said)
        self.assertIn("ACME-LIC R2", said)

    def test_and_says_when_the_document_itself_is_in_there(self):
        _, said = stage(self.root, "private")

        self.assertIn("original.pdf", said)


class TheBindingTravelsOnlyWithItsDocument(unittest.TestCase):
    """A binding pointing at an absent store is worse than none: the session
    opens looking bound and answers from nothing."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_a_binding_on_a_left_out_document_is_dropped(self):
        root = store(self.tmp.name, [PUBLIC, LICENSED], bound=LICENSED)
        out, said = stage(root, "public")

        self.assertFalse((out / ".active-binding.json").exists())
        self.assertIn("binding dropped", said)

    def test_a_binding_on_a_document_that_travels_is_kept(self):
        root = store(self.tmp.name, [PUBLIC, LICENSED], bound=PUBLIC)
        out, said = stage(root, "public")

        self.assertTrue((out / ".active-binding.json").exists())
        self.assertIn("bound on open", said)

    def test_an_empty_store_is_not_an_error(self):
        root = store(self.tmp.name, [])
        out, said = stage(root, "public")

        self.assertEqual(taken(out), set())


if __name__ == "__main__":
    unittest.main()
