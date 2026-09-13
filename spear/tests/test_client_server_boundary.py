"""The client runtime never depends on the server tree.

`spear/` and `server/` share a repository so that a protocol change is one
commit. They do not share a process. The client decides what an answer means
-- retrieval semantics, corpus identity, the vector store, the agent runtime
-- and the server decides what a machine does. A client that imported the
server would make the server's presence a requirement for running the client
at all, and would let a server-side concern reach into retrieval, which is the
inversion the split exists to prevent.

This is a static scan, not an import: it reads the client's source and looks
at what it imports. So it can live here, in the suite a client change is
actually run against, without importing anything it forbids.

Tests may cross the boundary -- validating a contract means looking at both
sides -- and that is why this file scans production modules and not itself.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REPO = ROOT.parent


def client_modules():
    """Production client modules: spear/**.py, excluding tests and the venv."""
    out = subprocess.run(["git", "ls-files", "spear"], cwd=REPO,
                         capture_output=True, text=True).stdout.split()

    return [REPO / name for name in out
            if name.endswith(".py") and not name.startswith("spear/tests/")]


def imported_names(path):
    """Every top-level name this module imports, however it spells it."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError, OSError):
        return set()

    names = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.add(node.module.split(".")[0])

    return names


class TheClientDoesNotImportTheServer(unittest.TestCase):
    def test_no_production_module_imports_server(self):
        offenders = {str(p.relative_to(REPO)): sorted(n for n in imported_names(p)
                                                      if n == "server")
                     for p in client_modules()
                     if "server" in imported_names(p)}

        self.assertEqual(offenders, {}, f"client imports server: {offenders}")

    def test_no_production_module_reaches_the_server_tree_by_path(self):
        """An import is not the only way to depend on it. A path literal is
        the other, and it survives a grep for `import server`."""
        offenders = {}

        for path in client_modules():
            try:
                body = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for needle in ('"server/', "'server/", "/server/inference",
                           "/server/embed", "/server/scripts"):
                if needle in body:
                    offenders.setdefault(str(path.relative_to(REPO)),
                                         []).append(needle)

        self.assertEqual(offenders, {})

    def test_the_scan_actually_sees_the_client(self):
        """A scan over an empty list passes for the wrong reason."""
        modules = client_modules()

        self.assertGreater(len(modules), 50, len(modules))
        self.assertIn("rag_chat.py", {p.name for p in modules})

    def test_the_scan_would_notice_a_violation(self):
        """Prove the detector detects, rather than trusting it to."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            probe = Path(tmp) / "probe.py"
            probe.write_text("from server.embed import protocol\n")
            self.assertIn("server", imported_names(probe))

            probe.write_text("import server.inference\n")
            self.assertIn("server", imported_names(probe))

            probe.write_text("import serverless_thing\n")
            self.assertNotIn("server", imported_names(probe))


class TheLauncherMayStartTheServer(unittest.TestCase):
    """The one permitted direction, and it is not an import.

    `spear-chat --local` starts a local inference server by executing
    server/inference/serve.sh. That is a shell launching a sibling program,
    not a module depending on another module: nothing of the server is loaded
    into the client process, and the client passes configuration in rather
    than reading the server's.

    It must still say so clearly when the server tree is absent, because a
    client-only checkout is a thing that can exist.
    """

    def launcher(self):
        return (ROOT / "spear-chat.sh").read_text(encoding="utf-8")

    def test_local_mode_runs_the_generic_server(self):
        self.assertIn("server/inference/serve.sh", self.launcher())

    def test_the_retired_launcher_is_gone(self):
        self.assertFalse((ROOT / "edgem-server.sh").exists())
        self.assertFalse((ROOT / "spear-server.sh").exists())
        self.assertNotIn("edgem-server.sh", self.launcher())

    def test_a_missing_server_tree_is_reported_not_guessed(self):
        self.assertIn("--local needs the server/ tree", self.launcher())

    def test_the_client_passes_configuration_in(self):
        """The server has no opinion about this checkout, so the client must
        tell it which model, which card and which context."""
        body = self.launcher()

        for name in ("SPEAR_SERVER_MODEL", "SPEAR_SERVER_CTX",
                     "SPEAR_SERVER_LLAMA_BIN", "SPEAR_SERVER_GPU_UUID"):
            with self.subTest(variable=name):
                self.assertIn(name, body)


class TheRemoteEmbeddingConfigurationIsUntouched(unittest.TestCase):
    """E.3 reorganised directories. It did not move the deployed worker.

    The worker on the GPU host still speaks the old contract and still lives
    at the old path, so the client must still invoke exactly that. The
    replacement lands with the deployment, not before it.
    """

    def test_the_legacy_worker_stays_where_the_client_expects_it(self):
        self.assertTrue((ROOT / "deploy" / "embed_worker.py").is_file())

    def test_no_worker_was_added_under_server(self):
        self.assertFalse((REPO / "server" / "embed" / "worker.py").exists())
        self.assertFalse((REPO / "server" / "embed" / "protocol.py").exists())

    def test_the_client_still_ships_no_remote_command(self):
        """The command remains deployment configuration, as E.0 made it."""
        import embedding

        source = (ROOT / "embedding.py").read_text(encoding="utf-8")
        self.assertIn("SPEAR_EMBED_REMOTE_CMD", source)
        self.assertNotIn("edgem-ai-rag", source)
        self.assertTrue(hasattr(embedding, "remote_command"))


if __name__ == "__main__":
    unittest.main()
