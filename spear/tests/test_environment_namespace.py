"""The environment variables SPEAR owns are spelled SPEAR_.

The platform used to read an EDGEM_ namespace, and the rename is meant to be
total rather than additive: an old name is not an alias, it is nothing. A
release that quietly honoured both would keep the old contract alive for as
long as anyone's shell still exported it, which is the opposite of retiring
it.

Two kinds of check live here, and the distinction matters.

The inventory test reads the tracked tree and fails on any EDGEM_ variable at
all. It does not carry a copy of the 79 renamed names -- a list like that is
wrong the day someone adds a variable, and its failure teaches nothing. It
carries no exceptions either: there were two for a while, both pointing at
somebody else's build system, and they left with the code that read them. A
new EDGEM_ name has nowhere to be written down, which is the point.

The behavioural tests take representative variables from different subsystems
and prove the rename reached the code that reads them: the new name changes
what the module does, and the old one does not. They read modules rather than
running sessions, and they touch no persistent state.
"""

from __future__ import annotations

import importlib
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REPO = ROOT.parent

_EDGEM = re.compile(r"EDGEM_[A-Z0-9_]+")


def tracked_files():
    out = subprocess.run(["git", "ls-files"], capture_output=True, text=True,
                         cwd=REPO).stdout.split()

    return [REPO / name for name in out]


def survivors():
    """Every EDGEM_ variable in the tracked tree, with where it was found.

    Expected to be empty. It reports locations rather than a count so that a
    failure names the file to fix.

    This file is the one exception, and not a whitelisted one: it spells the
    old names only to assert their absence, so scanning it would make the
    check fail on its own evidence.
    """
    found = {}

    for path in tracked_files():
        if path == Path(__file__).resolve():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        for name in _EDGEM.findall(text):
            found.setdefault(name, set()).add(
                str(path.relative_to(REPO)))

    return found


class TheOldNamespaceIsGone(unittest.TestCase):
    """Inventory, introspected rather than transcribed."""

    def setUp(self):
        self.found = survivors()

    def test_no_variable_of_the_old_namespace_is_left(self):
        """No allowance, no whitelist: the next one to appear fails here."""
        self.assertEqual(sorted(self.found), [], "\n".join(
            f"{name} in {sorted(self.found[name])}"
            for name in sorted(self.found)))

    def test_the_platform_reads_a_spear_namespace(self):
        spear = re.compile(r"SPEAR_[A-Z0-9_]+")
        names = set()

        for path in tracked_files():
            if not str(path).endswith(".py"):
                continue
            try:
                names |= set(spear.findall(path.read_text(encoding="utf-8")))
            except (UnicodeDecodeError, OSError):
                continue

        self.assertGreater(len(names), 50, sorted(names)[:5])


class TheNewNameIsTheOneThatWorks(unittest.TestCase):
    """Representative variables from different subsystems.

    Each proves the same two things: the SPEAR_ name reaches the code, and
    the EDGEM_ name reaches nothing. Modules that read their configuration at
    import time are reloaded under a patched environment, which is why these
    touch no session and no persistent state.
    """

    #: Both spellings of every variable these tests touch. They are cleared
    #: before each import so a default is genuinely a default -- this machine
    #: exports SPEAR_STATE_DIR, and a test that inherited it would be
    #: measuring the shell rather than the code.
    NAMESPACE = tuple(
        f"{prefix}_{name}" for name in
        ("STATE_DIR", "API_BASE", "DB_PATH", "CORPUS_ROOT", "CLAUSE_REDIRECTS")
        for prefix in ("SPEAR", "EDGEM"))

    def reload(self, module, **overrides):
        """Import a module with exactly this overlay and nothing inherited.

        Everything unrelated -- PATH, HOME -- is kept, or the module under
        test would fail to import for reasons that have nothing to do with
        the namespace.
        """
        env = {key: value for key, value in os.environ.items()
               if key not in self.NAMESPACE}
        env.update(overrides)

        with patch.dict(os.environ, env, clear=True):
            return importlib.reload(importlib.import_module(module))

    def check(self, module, attribute, variable, value, expected):
        """The new name decides the attribute; the old name does not."""
        legacy = variable.replace("SPEAR_", "EDGEM_", 1)

        new = self.reload(module, **{variable: value})
        self.assertEqual(getattr(new, attribute), expected,
                         f"{variable} did not reach {module}.{attribute}")

        old = self.reload(module, **{legacy: value})
        self.assertNotEqual(getattr(old, attribute), expected,
                            f"{legacy} still reaches {module}.{attribute}")

        return new

    def test_state_dir(self):
        self.check("backend_select", "STATE_FILE", "SPEAR_STATE_DIR",
                   "/tmp/spear-probe-state",
                   "/tmp/spear-probe-state/active-backend.conf")

    def test_api_base(self):
        self.check("rag_chat", "LLAMA_SERVER_URL", "SPEAR_API_BASE",
                   "http://127.0.0.1:9/v1", "http://127.0.0.1:9/v1")

    def test_db_path(self):
        self.check("rag_chat", "DB_PATH", "SPEAR_DB_PATH",
                   "/tmp/spear-probe-db", "/tmp/spear-probe-db")

    def test_corpus_root(self):
        self.check("rag_chat", "CORPORA_ROOT", "SPEAR_CORPUS_ROOT",
                   "/tmp/spear-probe-corpora", "/tmp/spear-probe-corpora")

    def test_a_runtime_limit(self):
        self.check("agent_runtime", "CLAUSE_REDIRECT_LIMIT",
                   "SPEAR_CLAUSE_REDIRECTS", "9", 9)

    def tearDown(self):
        # Leave every module as the rest of the suite expects to find it.
        for module in ("backend_select", "rag_chat", "agent_runtime"):
            if module in sys.modules:
                importlib.reload(sys.modules[module])


class TheTraceSwitchMovedToo(unittest.TestCase):
    """SPEAR_TRACE and SPEAR_TRACE_FILE, read where tracing is set up."""

    def source(self):
        return (ROOT / "rag_chat.py").read_text(encoding="utf-8")

    def test_the_switch_is_read_under_the_new_name(self):
        self.assertIn('"SPEAR_TRACE"', self.source())
        self.assertNotIn('"EDGEM_TRACE"', self.source())

    def test_the_destination_too(self):
        self.assertIn("SPEAR_TRACE_FILE", self.source())
        self.assertNotIn("EDGEM_TRACE_FILE", self.source())


class DerivedAndPropagatedNames(unittest.TestCase):
    """Where a name is built or handed on, not merely read.

    A literal substitution misses these, and each one is a contract with
    something outside the process: a child benchmark run, a remote training
    host, a shell heredoc.
    """

    def test_the_benchmark_builds_child_names_from_the_new_prefix(self):
        source = (ROOT / "benchmarks" / "runner.py").read_text(encoding="utf-8")

        self.assertIn('f"SPEAR_BENCH_{key.upper()}"', source)
        self.assertNotIn("EDGEM_BENCH", source)

    def test_the_training_log_marker_matches_what_the_remote_emits(self):
        """Both halves, or the launcher parses lines nothing writes."""
        source = (ROOT / "training_launcher.py").read_text(encoding="utf-8")

        self.assertIn("sed 's/^/SPEAR_LOG:/'", source)
        self.assertIn('len("SPEAR_LOG:")', source)
        self.assertIn('startswith("SPEAR_LOG:")', source)
        self.assertNotIn("EDGEM_LOG", source)

    def test_the_remote_heredoc_delimiter_still_pairs(self):
        source = (ROOT / "training_launcher.py").read_text(encoding="utf-8")

        self.assertEqual(source.count("SPEAR_RUN"), 2)
        self.assertNotIn("EDGEM_RUN", source)

    def test_the_benchmark_passes_the_new_names_to_its_child(self):
        source = (ROOT / "benchmarks" / "runner.py").read_text(encoding="utf-8")

        for name in ("SPEAR_STATE_DIR", "SPEAR_DB_PATH",
                     "SPEAR_BENCH_ANSWER_FILE"):
            with self.subTest(variable=name):
                self.assertIn(name, source)


class RedsIsNotOurs(unittest.TestCase):
    """REDS names an institute's machine, not a namespace SPEAR owns."""

    def test_the_reds_variables_are_untouched(self):
        names = set()

        for path in tracked_files():
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            names |= set(re.findall(r"\bREDS_[A-Z_]+", text))

        self.assertTrue({"REDS_HOST", "REDS_PORT", "REDS_MODEL"} <= names)

    def test_no_spear_reds_variable_was_invented(self):
        """Except the one SPEAR already owned: the key path it reads."""
        names = set()

        for path in tracked_files():
            try:
                names |= set(re.findall(r"SPEAR_REDS_[A-Z_]+",
                                        path.read_text(encoding="utf-8")))
            except (UnicodeDecodeError, OSError):
                continue

        self.assertEqual(names, {"SPEAR_REDS_KEY"})


if __name__ == "__main__":
    unittest.main()
