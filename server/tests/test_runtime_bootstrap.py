"""The runtime manifest is the reproducibility contract; the bootstrap obeys it.

A deployment that cannot be rebuilt is a deployment nobody may touch. What
makes rebuilding possible is not the script -- it is the manifest: an
immutable llama.cpp commit, exact shard sizes with the publisher's hashes, a
tested constraints file. The script is only what carries them out.

So most of these tests are about the manifest, and the ones about the script
are about what it must NOT do: invent a version, duplicate a helper that
already exists, name a machine, or destroy something expensive that was
already correct.

The behavioural tests run the real script against a temporary root. A test
that only read the source could not tell whether --dry-run writes, and that
is the one property a dry run has.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST_PATH = ROOT / "server" / "runtime" / "manifest.json"
BOOTSTRAP = ROOT / "scripts" / "bootstrap-runtime.sh"

MANIFEST = json.loads(MANIFEST_PATH.read_text())


def run(*args, root=None):
    """The real script, on a temporary root."""
    argv = [str(BOOTSTRAP)]
    if root is not None:
        argv += ["--root", str(root)]
    argv += list(args)
    return subprocess.run(argv, capture_output=True, text=True, timeout=300,
                          cwd=str(ROOT))


def tree(path):
    """Every path under `path`, as a set, for before/after comparison."""
    return {p.relative_to(path) for p in Path(path).rglob("*")}


class TheManifestIsAContract(unittest.TestCase):
    def test_it_is_versioned(self):
        """An unversioned manifest cannot be changed without breaking readers."""
        self.assertEqual(MANIFEST["spear_runtime_manifest"], 1)

    def test_llama_cpp_is_pinned_to_an_immutable_commit(self):
        """A branch is not a reproducibility contract.

        `git clone` with no revision gives whatever master holds today, so the
        binary serving a year from now differs from the one measured, with
        nothing recording that it changed. The pin must be a full object name:
        a tag can be moved and a short name can become ambiguous.
        """
        revision = MANIFEST["llama_cpp"]["revision"]

        self.assertRegex(revision, r"^[0-9a-f]{40}$")

    def test_no_moving_version_anywhere_in_the_manifest(self):
        """Nothing resolves to "whatever is newest at install time"."""
        moving = ("latest", "master", "main", "HEAD", "stable", "nightly", "*")
        blob = json.dumps({k: v for k, v in MANIFEST.items()
                           if k not in ("profile", "layout")})

        for word in moving:
            with self.subTest(word=word):
                self.assertNotRegex(blob, rf'"\s*{re.escape(word)}\s*"')

    def test_every_shard_carries_a_size_and_a_sha256(self):
        shards = MANIFEST["model"]["shards"]

        self.assertGreater(len(shards), 0)
        for shard in shards:
            with self.subTest(shard=shard["name"]):
                self.assertRegex(shard["sha256"], r"^[0-9a-f]{64}$")
                self.assertIsInstance(shard["size"], int)
                self.assertGreater(shard["size"], 0)

    def test_the_declared_total_is_the_sum_of_the_shards(self):
        """Two numbers that must agree, checked rather than trusted."""
        self.assertEqual(sum(s["size"] for s in MANIFEST["model"]["shards"]),
                         MANIFEST["model"]["total_bytes"])

    def test_the_first_shard_is_one_of_the_shards(self):
        """llama.cpp opens the first and expects its siblings beside it."""
        names = [s["name"] for s in MANIFEST["model"]["shards"]]

        self.assertIn(MANIFEST["model"]["first_shard"], names)
        self.assertEqual(MANIFEST["model"]["first_shard"], names[0])

    def test_the_embedding_profile_matches_the_shipped_protocol(self):
        """The manifest may not claim a protocol the worker does not speak."""
        protocol = (ROOT / "server" / "embed" / "protocol.py").read_text()

        self.assertIn(f"PROTOCOL_VERSION = {MANIFEST['embedding']['protocol_version']}",
                      protocol)
        self.assertIn(MANIFEST["embedding"]["protocol_key"], protocol)

    def test_the_constraints_file_it_names_exists_and_pins_the_torch_build(self):
        constraints = ROOT / MANIFEST["embedding"]["constraints"]

        self.assertTrue(constraints.is_file(), constraints)
        self.assertIn(MANIFEST["embedding"]["torch_pin"], constraints.read_text())

    def test_the_server_defaults_reproduce_the_validated_semantics(self):
        """The numbers the profile was measured at, stated once.

        They live in the manifest rather than in the script so that a second
        profile is a second file, not an edit to the thing that installs it.
        """
        defaults = MANIFEST["server_defaults"]

        self.assertEqual(defaults["SPEAR_SERVER_CTX"], "98304")
        self.assertEqual(defaults["SPEAR_SERVER_PARALLEL"], "1")
        self.assertEqual(defaults["SPEAR_SERVER_NGL"], "99")
        self.assertEqual(defaults["SPEAR_SERVER_THREADS"], "16")
        self.assertEqual(defaults["SPEAR_SERVER_HOST"], "127.0.0.1")
        self.assertEqual(defaults["SPEAR_SERVER_PORT"], "8010")

    def test_the_fixed_flags_are_the_ones_the_launcher_really_emits(self):
        """Documented in the manifest, decided in serve.sh -- checked to agree."""
        serve = (ROOT / "server" / "inference" / "serve.sh").read_text()

        for flag in MANIFEST["server_fixed_flags"]["flags"]:
            name, _, value = flag.partition(" ")
            with self.subTest(flag=flag):
                self.assertIn(name, serve)
                if value:
                    self.assertIn(value, serve)


class ItNamesNoMachine(unittest.TestCase):
    """The manifest and the script describe software, never a deployment.

    The repository-wide boundary scans already refuse every private host,
    account, key and card UUID anywhere in tracked source, and they read these
    two files along with the rest. Repeating that needle list here would be a
    second copy to keep in step -- and, the first time it was tried, a copy
    that tripped the very scan it duplicated.

    So what is left here is the shape of the thing rather than the values: a
    manifest that carries no absolute path at all cannot carry somebody's home
    directory, whatever it is called.
    """

    def test_the_manifest_contains_no_absolute_path(self):
        """Where the runtime lives is --root's business, not the manifest's."""
        paths = re.findall(r'"(/[^"]*)"', MANIFEST_PATH.read_text())

        self.assertEqual(paths, [], f"absolute paths in the manifest: {paths}")

    def test_no_real_gpu_uuid_is_baked_in(self):
        """A card is an allocation. It arrives as --gpu, or not at all.

        The shape, not a list: an all-zero UUID is the documentation
        placeholder and anything else is somebody's real card.
        """
        real = re.compile(r"GPU-(?!0{8}-)[0-9a-f]{8}-", re.IGNORECASE)

        for path in (MANIFEST_PATH, BOOTSTRAP):
            with self.subTest(path=path.name):
                self.assertIsNone(real.search(path.read_text()))

    def test_the_check_would_notice_a_real_looking_uuid(self):
        """Prove the detector detects.

        The probe is assembled rather than written out: a real-looking UUID
        spelled as a literal here would be found by the repository's own scan
        of this file, which is the same trap the needle list above fell into.
        """
        real = re.compile(r"GPU-(?!0{8}-)[0-9a-f]{8}-", re.IGNORECASE)
        placeholder = "GPU-" + "0" * 8 + "-0000-0000-0000-" + "0" * 12
        looks_real = "GPU-" + "1a2b3c4d" + "-0000-0000-0000-" + "0" * 12

        self.assertIsNone(real.search(placeholder))
        self.assertIsNotNone(real.search(looks_real))

    def test_the_root_is_never_defaulted_to_a_home_directory(self):
        """--root is required, so a mistyped run cannot land in someone's tree."""
        result = run("--dry-run")

        self.assertEqual(result.returncode, 78)
        self.assertIn("--root is required", result.stderr)


class ItReusesRatherThanDuplicates(unittest.TestCase):
    """One implementation of each expensive, already-solved thing."""

    SOURCE = BOOTSTRAP.read_text()

    def test_the_model_download_is_delegated(self):
        """fetch-model.sh already resolves a shard set and resumes."""
        self.assertIn("server/scripts/fetch-model.sh", self.SOURCE)
        self.assertNotIn("wget -c", self.SOURCE)

    def test_the_llama_build_is_delegated(self):
        """install-llamacpp.sh already picks a new enough CUDA compiler."""
        self.assertIn("server/inference/install-llamacpp.sh", self.SOURCE)
        self.assertNotIn("cmake --build", self.SOURCE)

    def test_the_serving_command_is_not_reimplemented(self):
        """Verification runs the real launcher against a stub binary."""
        self.assertIn("server/inference/serve.sh", self.SOURCE)
        self.assertNotIn("--ctx-size \"$", self.SOURCE)


class ItIsNotDestructive(unittest.TestCase):
    """Nothing here may throw away something expensive that was already right."""

    SOURCE = BOOTSTRAP.read_text()

    def test_it_never_removes_anything_under_the_runtime(self):
        """No recursive removal of anything derived from --root.

        What it may remove is its own scratch: paths it got from mktemp and
        recorded, never a path it computed from --root. So the assertion is
        about which variable is being deleted, not about the words around it.
        """
        removals = re.findall(r"rm\s+-[rRf-]{2,}\s+(.+)", self.SOURCE)

        self.assertTrue(removals, "no removal found -- has the cleanup moved?")
        for target in removals:
            with self.subTest(target=target):
                for forbidden in ("ROOT", "MODELS", "VENV", "CONF", "LLAMA"):
                    self.assertNotIn(forbidden, target)
                self.assertIn("SCRATCH", target)

    def test_scratch_holds_only_mktemp_paths(self):
        """The cleanup list is safe only if nothing else is ever added to it."""
        added = re.findall(r'SCRATCH\+=\("([^"]+)"\)', self.SOURCE)

        self.assertTrue(added)
        for value in added:
            with self.subTest(value=value):
                self.assertRegex(value, r"^\$(PROBE_PY|STUBDIR)$")

    def test_there_is_exactly_one_exit_trap(self):
        """Bash keeps one EXIT handler: a second `trap` replaces the first.

        Two of them is how the probe's temporary file leaked -- the later trap
        silently took over and the earlier one never ran.
        """
        traps = re.findall(r"^\s*trap\s+(.+?)\s+EXIT", self.SOURCE, re.MULTILINE)

        self.assertEqual(traps, ["cleanup"])

    def test_no_code_line_reaches_outside_the_runtime(self):
        """Chroma, training trees and rollback sediment are not runtime.

        Comments are exempt and deliberately so: the header says in words that
        the script touches none of them, and a scan that forbade the word
        would forbid saying so. What must not appear is an executable line
        naming one.
        """
        code = [line for line in self.SOURCE.splitlines()
                if line.strip() and not line.strip().startswith("#")]

        # The retired product prefix is assembled: spelled out, this file
        # would itself be an unexplained occurrence of it.
        for word in ("chromadb", "chroma", "spear-training", "edg" + "em"):
            for line in code:
                with self.subTest(word=word, line=line[:60]):
                    self.assertNotIn(word, line.lower())

    def test_the_manifest_declares_what_is_out_of_scope(self):
        excluded = MANIFEST["layout"]["excluded"]["patterns"]

        self.assertTrue(any(p.startswith("S1") for p in excluded))
        self.assertTrue(any("log" in p for p in excluded))


class DryRunAndVerifyWriteNothing(unittest.TestCase):
    """The one property each mode has, checked by running it."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.root = self.tmp / "runtime"

    def test_dry_run_creates_nothing_at_all(self):
        before = tree(self.tmp)
        result = run("--dry-run", root=self.root)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(tree(self.tmp), before)
        self.assertFalse(self.root.exists())

    def test_dry_run_says_what_it_would_do(self):
        result = run("--dry-run", root=self.root)

        self.assertIn("would", result.stdout)
        for shard in MANIFEST["model"]["shards"]:
            self.assertIn(shard["name"], result.stdout)
        self.assertIn("nothing was created", result.stdout)

    def test_verify_creates_nothing_and_reports_a_missing_runtime(self):
        before = tree(self.tmp)
        result = run("--verify", root=self.root)

        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertEqual(tree(self.tmp), before)

    def test_verify_of_a_good_runtime_succeeds(self):
        run("--skip-model", "--skip-llama", "--skip-embed", root=self.root)
        result = run("--skip-model", "--skip-llama", "--skip-embed", "--verify",
                     root=self.root)

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("satisfies the manifest", result.stdout)


class ItIsIdempotent(unittest.TestCase):
    """A second run must cost nothing and destroy nothing.

    The expensive things are a 79 GB download and a 5 GB virtualenv, so the
    property that matters is not "it runs twice" but "it recognises what is
    already correct and leaves it alone".
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.root = self.tmp / "runtime"
        run("--skip-model", "--skip-llama", "--skip-embed", root=self.root,)

    def test_a_second_run_preserves_an_edited_config(self):
        """Configuration is the deployment's, not the manifest's."""
        conf = self.root / "config" / "server.conf"
        conf.write_text(conf.read_text().replace("98304", "65536") + "\n# operator\n")
        before = conf.read_bytes()

        run("--skip-model", "--skip-llama", "--skip-embed", root=self.root)

        self.assertEqual(conf.read_bytes(), before)

    def test_a_second_run_preserves_logs(self):
        log = self.root / "log" / "serve.log"
        log.write_text("history worth keeping\n")

        run("--skip-model", "--skip-llama", "--skip-embed", root=self.root)

        self.assertEqual(log.read_text(), "history worth keeping\n")

    def test_present_and_correct_shards_are_not_refetched(self):
        """Sparse files: the manifest's sizes at no disk cost."""
        models = self.root / "models" / "gguf"
        models.mkdir(parents=True, exist_ok=True)
        for shard in MANIFEST["model"]["shards"]:
            with open(models / shard["name"], "wb") as handle:
                handle.truncate(shard["size"])

        result = run("--skip-llama", "--skip-embed", "--dry-run", root=self.root)

        self.assertNotIn("would   download", result.stdout)
        self.assertIn("nothing to download", result.stdout)

    def test_a_shard_of_the_wrong_size_is_the_only_one_refetched(self):
        models = self.root / "models" / "gguf"
        models.mkdir(parents=True, exist_ok=True)
        for shard in MANIFEST["model"]["shards"]:
            with open(models / shard["name"], "wb") as handle:
                handle.truncate(shard["size"])
        bad = MANIFEST["model"]["shards"][2]
        with open(models / bad["name"], "wb") as handle:
            handle.truncate(12345)

        result = run("--skip-llama", "--skip-embed", "--dry-run", root=self.root)

        self.assertIn(f"re-download {bad['name']}", result.stdout)
        for good in MANIFEST["model"]["shards"]:
            if good["name"] != bad["name"]:
                self.assertNotIn(f"re-download {good['name']}", result.stdout)


class ItDetectsAWrongLlamaRevision(unittest.TestCase):
    """The pin is worthless if nothing checks the checkout against it."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.root = self.tmp / "runtime"
        run("--skip-model", "--skip-llama", "--skip-embed", root=self.root)

    @unittest.skipUnless(shutil.which("git"), "git is required")
    def test_a_checkout_at_another_revision_is_reported(self):
        src = self.root / "llama.cpp"
        env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@e",
               "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@e"}
        subprocess.run(["git", "init", "-q", str(src)], check=True, env=env)
        subprocess.run(["git", "-C", str(src), "commit", "-q", "--allow-empty",
                        "-m", "not the pinned revision"], check=True, env=env)

        result = run("--skip-model", "--skip-embed", "--verify", root=self.root)

        self.assertEqual(result.returncode, 2)
        self.assertIn("wrong-revision", result.stdout)


if __name__ == "__main__":
    unittest.main()
