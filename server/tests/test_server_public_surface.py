"""The server tree is public, and nothing in it names a machine.

Two properties, checked by reading the tree rather than by remembering.

NAMES. Hostnames, addresses, accounts, key paths, GPU identifiers and model
paths are deployment facts. They live in ~/spear-runtime/config/, which is
outside the checkout, so no `git add` can reach them. This tree was assembled
partly from scripts that ran on one particular GPU host, and those scripts did
carry such values -- which is exactly why the check exists rather than the
intention.

ASSETS. Weights, CUDA build trees and caches are not in Git and not under it.
A script that resolves them relative to its own location would quietly put
80 GB inside a repository meant to be cloned.
"""

from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path

SERVER = Path(__file__).resolve().parent.parent
REPO = SERVER.parent


def files():
    """Every file of the server tree: what Git has AND what is on disk.

    The union, not one or the other. Taking Git's listing alone would scan a
    partially staged tree -- which is the run where a private value is most
    likely to slip through -- and taking the filesystem alone would miss a
    file staged from elsewhere. Neither omission is acceptable in a check
    whose whole job is to be exhaustive.
    """
    tracked = subprocess.run(["git", "ls-files", "server"], cwd=REPO,
                             capture_output=True, text=True).stdout.split()
    paths = {REPO / name for name in tracked}
    paths |= {p for p in SERVER.rglob("*")}

    return sorted(p for p in paths
                  if p.is_file() and "__pycache__" not in p.parts)


def text_of(path):
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return ""


class NothingHereNamesAMachine(unittest.TestCase):
    #: Assembled from fragments: spelled out, they would appear in this file
    #: and the scan below would find its own evidence.
    PRIVATE = {
        "retired platform name":  ("edgem" + "-ai", "EDGEM" + "-AI"),
        "the organisation":       ("edgem" + "tech", "EDGEM" + "Tech"),
        "its product line":       ("edgem" + "1",),
        "its infrastructure":     ("Infra" + "base",),
        "a private network":      ("10." + "190.",),
        "a private account path": ("/home/re" + "ds-ml",),
        "a private host":         ("re" + "ds-ml@",),
        "a private key":          ("id_pod" + "_gpu",),
    }

    def test_no_private_value_reaches_the_public_tree(self):
        found = {}

        for path in files():
            body = text_of(path)

            for label, needles in self.PRIVATE.items():
                for needle in needles:
                    if needle in body:
                        found.setdefault(str(path.relative_to(REPO)),
                                         []).append(f"{needle} ({label})")

        self.assertEqual(found, {}, f"private values in server/: {found}")

    def test_no_gpu_identifier_is_a_real_one(self):
        """A UUID may appear as an example. A real card's may not."""
        real = re.compile(r"GPU-(?!0{8}-0{4}-0{4}-0{4}-0{12})[0-9a-f]{8}-[0-9a-f-]+")
        named = {str(p.relative_to(REPO)): real.findall(text_of(p))
                 for p in files() if real.search(text_of(p))}

        self.assertEqual(named, {})

    #: A four-part pinned version has the same SHAPE as an IPv4 address -- the
    #: nvidia wheels are pinned that way -- so this scan cannot tell them
    #: apart, and an example spelled out here would be found by the scan
    #: itself. Requirement files are therefore read by the NAME scan above and
    #: by their own tests, not by this one. The exemption is by file kind, not
    #: by file: a new requirements file inherits it, a new .conf does not.
    VERSION_BEARING = (".txt",)

    def test_no_address_or_bare_hostname_is_configured(self):
        """Only the loopback address, which is a policy and not a machine:
        the server binds it so that reaching it needs a tunnel."""
        address = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
        allowed = {"127.0.0.1", "0.0.0.0"}
        found = {}

        for path in files():
            if path.suffix in self.VERSION_BEARING:
                continue

            hits = {a for a in address.findall(text_of(path))} - allowed
            if hits:
                found[str(path.relative_to(REPO))] = sorted(hits)

        self.assertEqual(found, {})

    def test_the_exempted_files_are_still_scanned_for_real_addresses(self):
        """The exemption is about SHAPE, not about trust.

        A requirement file that named a host would still be caught -- by the
        needle scan above, which looks for the values themselves rather than
        for four dotted numbers.
        """
        exempt = [p for p in files() if p.suffix in self.VERSION_BEARING]

        self.assertTrue(exempt, "no version-bearing file found to check")

        for path in exempt:
            body = text_of(path)

            for label, needles in self.PRIVATE.items():
                for needle in needles:
                    with self.subTest(file=path.name, value=label):
                        self.assertNotIn(needle, body)


class AssetsLiveOutsideTheCheckout(unittest.TestCase):
    def scripts(self):
        return [p for p in files() if p.suffix == ".sh"]

    def test_no_script_resolves_an_asset_against_its_own_location(self):
        """`$(dirname $0)/../..` is how a build tree ends up inside a clone.

        Locating a SIBLING SCRIPT that way is fine and expected -- the tree
        knows its own shape. Locating weights or a build tree that way is not.
        """
        offenders = {}

        for path in self.scripts():
            for line in text_of(path).splitlines():
                if line.lstrip().startswith("#"):
                    continue
                if "BASH_SOURCE" not in line and "$HERE" not in line:
                    continue
                if any(word in line for word in ("gguf", "GGUF", "models",
                                                 "llama.cpp", "build/bin",
                                                 "hf", "cache")):
                    offenders.setdefault(str(path.relative_to(REPO)),
                                         []).append(line.strip())

        self.assertEqual(offenders, {})

    def test_every_script_is_syntactically_valid(self):
        for path in self.scripts():
            with self.subTest(script=path.name):
                proc = subprocess.run(["bash", "-n", str(path)],
                                      capture_output=True, text=True)
                self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_every_script_is_executable(self):
        for path in self.scripts():
            with self.subTest(script=path.name):
                self.assertTrue(path.stat().st_mode & 0o111, path)

    def test_the_examples_are_examples(self):
        """config/ ships *.example only: a real server.conf here would be a
        deployment committed by accident."""
        shipped = {p.name for p in files() if p.parent.name == "config"}

        self.assertEqual(shipped, {"server.conf.example", "gpu.conf.example"})


class TheEmbeddingWorkerStaysGeneric(unittest.TestCase):
    """server/embed/ carries a worker now. It must never carry a client.

    The coupling this directory exists to remove was an import: the worker it
    replaces pulled in the client's module on the GPU host, and with it the
    client's model registry. The scan is textual and blunt on purpose -- a
    comment that merely mentions the shape is caught too, because a scan that
    made an exception for prose would be no scan at all.
    """

    def modules(self):
        return [p for p in files()
                if p.parent.name == "embed" and p.suffix == ".py"]

    def test_the_directory_actually_holds_the_worker(self):
        """A scan over an empty list passes for the wrong reason."""
        self.assertEqual({p.name for p in self.modules()},
                         {"protocol.py", "worker.py"})

    def test_no_worker_imports_the_client(self):
        for path in self.modules():
            body = text_of(path)

            with self.subTest(module=path.name):
                self.assertNotIn("import embedding", body)
                self.assertNotIn("sys.path.insert", body)

    def test_the_protocol_needs_no_machine_learning_stack(self):
        """It must be readable where there is nothing installed."""
        body = text_of(SERVER / "embed" / "protocol.py")

        for heavy in ("torch", "sentence_transformers", "numpy",
                      "transformers"):
            with self.subTest(dependency=heavy):
                self.assertNotIn(heavy, body)

    def test_the_deployment_contract_is_written_down(self):
        readme = (SERVER / "embed" / "README.md").read_text(encoding="utf-8")

        for subject in ("protocol", "spear_embed_protocol", "requirements.txt",
                        "SPEAR_EMBED_REMOTE_CMD", "SPEAR_GPU_UUID"):
            with self.subTest(subject=subject):
                self.assertIn(subject, readme)


if __name__ == "__main__":
    unittest.main()
