"""A configured remote embedder is the embedder, and the code does not know
where it lives.

Two independent properties, and the tests are separated accordingly.

WHERE. The worker's path on the remote host used to be written into
embedding.py. That made the public tree carry one deployment's directory
layout: correct on exactly one machine, wrong everywhere else, and impossible
to reorganise on that machine without editing tracked source. The path is now
deployment configuration, with no built-in default -- a default naming a real
host would be the same assumption wearing a different hat.

WHETHER. Falling back to local compute when the remote embedder fails used to
be treated as a kindness: a slower index beats a failed one. It is not. The
two devices load the same weights but do not produce the same vectors, so a
collection half-filled from each is quietly inconsistent, and no later query
reports it -- results just get worse. So:

    no destination configured   -> local embedding, a valid deployment
    destination configured      -> remote, or an error. Never a substitute.

The distinction is the point: this is not "always fail", it is "never answer
with something other than what was asked for".
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import embedding                                            # noqa: E402

REPO = ROOT.parent

#: A destination that cannot resolve and a worker that does not exist. Every
#: remote call in this file is intercepted before it reaches ssh; these exist
#: to be echoed back in assertions, never to be contacted.
TARGET = "operator@gpu-host.example"
WORKER = ("/srv/inference/embed/venv/bin/python",
          "/srv/inference/embed/worker.py")

VECTORS = [[0.5, -0.25, 0.125, 0.0]]


def reply(vectors):
    """A worker's answer: the header line, then raw little-endian float32."""
    import struct

    flat = [value for vector in vectors for value in vector]
    body = struct.pack(f"<{len(flat)}f", *flat)

    return f"{len(vectors)} {len(vectors[0])}\n".encode() + body


class Configured(unittest.TestCase):
    """Base: an isolated config directory and no inherited environment.

    embedding.py reads its configuration from files beside itself and from the
    environment. A test that inherited either would be measuring this machine.
    """

    ENVIRONMENT = ("SPEAR_EMBED_REMOTE", "SPEAR_EMBED_REMOTE_SSH_OPTS",
                   "SPEAR_EMBED_REMOTE_CMD", "SPEAR_EMBED_REMOTE_BATCH")

    def setUp(self):
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

        self._saved = (embedding.REMOTE_CONF, embedding.REMOTE_CMD_CONF,
                       {k: os.environ.get(k) for k in self.ENVIRONMENT})
        self.addCleanup(self._restore)

        embedding.REMOTE_CONF = os.path.join(self.tmp.name, "remote.conf")
        embedding.REMOTE_CMD_CONF = os.path.join(self.tmp.name, "cmd.conf")

        for name in self.ENVIRONMENT:
            os.environ.pop(name, None)

    def _restore(self):
        embedding.REMOTE_CONF, embedding.REMOTE_CMD_CONF = self._saved[:2]

        for name, value in self._saved[2].items():
            os.environ.pop(name, None)

            if value is not None:
                os.environ[name] = value

    def destination(self, target=TARGET):
        Path(embedding.REMOTE_CONF).write_text(target + "\n")

    def command(self, *argv):
        """Write a command LINE, the way a deployment does.

        The file holds a command line, not a list, so an argument containing a
        space is quoted there -- and shlex reads back exactly the words that
        were meant. Writing the arguments joined by bare spaces would be a
        different command, and the test would be measuring its own helper.
        """
        import shlex

        Path(embedding.REMOTE_CMD_CONF).write_text(
            " ".join(shlex.quote(word) for word in argv) + "\n")

    def local_encoder(self):
        """Stand in for SentenceTransformer, and record that it was reached."""
        class Encoder:
            calls = 0

            def encode(self, texts, **kw):
                Encoder.calls += 1
                import numpy

                return numpy.zeros((len(texts), 4), dtype="float32")

        return Encoder()


class TheCodeDoesNotKnowWhereTheWorkerLives(unittest.TestCase):
    """Property 1: no remote path survives in the tracked runtime.

    The needles are assembled from fragments for the same reason the fixture
    scan below does it: written out, this file would contain the very strings
    it forbids, and the two scans would each find the other's evidence.
    """

    RETIRED = ("edgem" + "-ai-rag", "/home/re" + "ds-ml", "~/edg" + "em-ai")

    def sources(self):
        out = subprocess.run(["git", "ls-files", "spear"], cwd=REPO,
                             capture_output=True, text=True).stdout.split()

        return [REPO / name for name in out
                if name.endswith((".py", ".sh"))
                and not name.startswith("spear/tests/")]

    def test_no_module_names_a_remote_worker_path(self):
        """Whatever a deployment's layout is, the code must not have an
        opinion about it."""
        named = {}

        for path in self.sources():
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue

            for needle in self.RETIRED:
                if needle in text:
                    named.setdefault(str(path.relative_to(REPO)),
                                     []).append(needle)

        self.assertEqual(named, {})

    def test_the_module_ships_no_remote_command(self):
        """Absent configuration there is no command at all -- not a default
        that happens to be somebody's."""
        import ast

        tree = ast.parse((ROOT / "embedding.py").read_text(encoding="utf-8"))
        literals = [node.value for node in ast.walk(tree)
                    if isinstance(node, ast.Constant)
                    and isinstance(node.value, str)]

        self.assertEqual([s for s in literals if "bin/python" in s
                          or "worker.py" in s], [])


class NoDestinationMeansLocal(Configured):
    """CASE A: local embedding is a deployment, not a degraded one."""

    def test_without_a_destination_the_vectors_are_computed_here(self):
        encoder = self.local_encoder()

        with patch.object(embedding, "_st", return_value=encoder), \
             patch.object(embedding, "active_model", return_value="BAAI/bge-m3"), \
             patch("subprocess.run", side_effect=AssertionError("ssh was run")):
            vectors = embedding.embed_documents(["one", "two"])

        self.assertEqual(len(vectors), 2)
        self.assertEqual(type(encoder).calls, 1)

    def test_a_command_alone_does_not_make_a_deployment_remote(self):
        """The destination decides. A command left behind from a previous
        deployment must not silently re-enable the offload."""
        self.command(*WORKER)
        encoder = self.local_encoder()

        with patch.object(embedding, "_st", return_value=encoder), \
             patch.object(embedding, "active_model", return_value="BAAI/bge-m3"), \
             patch("subprocess.run", side_effect=AssertionError("ssh was run")):
            embedding.embed_documents(["one"])

        self.assertEqual(type(encoder).calls, 1)


class ADestinationMakesTheRemoteRequired(Configured):
    """CASE B: remote, or an error."""

    def test_a_configured_pair_is_invoked(self):
        self.destination()
        self.command(*WORKER)

        with patch("subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess(
                [], 0, reply(VECTORS), b"")

            vectors = embedding.embed_documents(["one"], model="BAAI/bge-m3")

        self.assertEqual(vectors, VECTORS)
        argv = run.call_args[0][0]
        self.assertEqual(argv[0], "ssh")
        self.assertEqual(argv[-2], TARGET)
        self.assertEqual(argv[-1], " ".join(WORKER))

    def test_a_missing_command_is_a_clear_failure_naming_both_keys(self):
        self.destination()
        encoder = self.local_encoder()

        with patch.object(embedding, "_st", return_value=encoder), \
             self.assertRaises(embedding.RemoteEmbeddingError) as raised:
            embedding.embed_documents(["one"], model="BAAI/bge-m3")

        message = str(raised.exception)
        self.assertIn("SPEAR_EMBED_REMOTE_CMD", message)
        self.assertIn(embedding.REMOTE_CMD_CONF, message)
        self.assertIn(TARGET, message)
        self.assertEqual(type(encoder).calls, 0)

    def test_the_file_the_message_names_is_the_one_beside_the_code(self):
        """The message is only actionable if the path it prints is the path a
        deployment is meant to write."""
        self.assertEqual("active-embed-remote-cmd.conf",
                         os.path.basename(self._saved[1]))
        self.assertEqual(str(ROOT), os.path.dirname(self._saved[1]))

    def test_ssh_failing_to_run_does_not_become_a_local_answer(self):
        self.destination()
        self.command(*WORKER)
        encoder = self.local_encoder()

        with patch.object(embedding, "_st", return_value=encoder), \
             patch("subprocess.run", side_effect=OSError("no ssh")), \
             self.assertRaises(embedding.RemoteEmbeddingError):
            embedding.embed_documents(["one"], model="BAAI/bge-m3")

        self.assertEqual(type(encoder).calls, 0)

    def test_a_worker_exiting_non_zero_reports_its_own_diagnosis(self):
        self.destination()
        self.command(*WORKER)

        with patch("subprocess.run") as run, \
             self.assertRaises(embedding.RemoteEmbeddingError) as raised:
            run.return_value = subprocess.CompletedProcess(
                [], 2, b"", b"CUDA out of memory")
            embedding.embed_documents(["one"], model="BAAI/bge-m3")

        self.assertIn("CUDA out of memory", str(raised.exception))

    def test_a_command_that_is_not_a_worker_is_a_protocol_failure(self):
        """The configured command ran and said something. Saying something is
        not the same as being an embedding worker."""
        self.destination()
        self.command("/bin/echo", "hello")

        with patch("subprocess.run") as run, \
             self.assertRaises(embedding.RemoteEmbeddingError) as raised:
            run.return_value = subprocess.CompletedProcess([], 0, b"hello\n", b"")
            embedding.embed_documents(["one"], model="BAAI/bge-m3")

        self.assertIn("no header", str(raised.exception))

    def test_a_truncated_payload_is_a_protocol_failure(self):
        self.destination()
        self.command(*WORKER)

        with patch("subprocess.run") as run, \
             self.assertRaises(embedding.RemoteEmbeddingError) as raised:
            run.return_value = subprocess.CompletedProcess(
                [], 0, b"1 4\n" + b"\0" * 8, b"")
            embedding.embed_documents(["one"], model="BAAI/bge-m3")

        self.assertIn("out of protocol", str(raised.exception))

    def test_a_short_count_is_refused_rather_than_padded(self):
        """Two texts in, one vector back. Accepting that would misalign every
        chunk in the collection from there on."""
        self.destination()
        self.command(*WORKER)

        with patch("subprocess.run") as run, \
             self.assertRaises(embedding.RemoteEmbeddingError):
            run.return_value = subprocess.CompletedProcess(
                [], 0, reply(VECTORS), b"")
            embedding.embed_documents(["one", "two"], model="BAAI/bge-m3")

    def test_nothing_is_sent_when_there_is_nothing_to_embed(self):
        self.destination()

        with patch("subprocess.run", side_effect=AssertionError("ssh was run")):
            self.assertEqual([], embedding.embed_documents(
                [], model="BAAI/bge-m3"))


class WhereTheCommandComesFrom(Configured):
    """Property: precedence, and what counts as unset."""

    def test_nothing_configured_is_no_command(self):
        self.assertIsNone(embedding.remote_command())

    def test_the_file_is_read(self):
        self.command(*WORKER)
        self.assertEqual(list(WORKER), embedding.remote_command())

    def test_the_variable_wins_over_the_file(self):
        self.command(*WORKER)
        os.environ["SPEAR_EMBED_REMOTE_CMD"] = "/usr/bin/python3 /srv/w.py"
        self.assertEqual(["/usr/bin/python3", "/srv/w.py"],
                         embedding.remote_command())

    def test_an_empty_variable_means_unset_not_the_file(self):
        """Matching the destination: half-configuring is never how the offload
        is turned off."""
        self.command(*WORKER)
        os.environ["SPEAR_EMBED_REMOTE_CMD"] = ""
        self.assertIsNone(embedding.remote_command())

    def test_the_file_may_be_documented(self):
        Path(embedding.REMOTE_CMD_CONF).write_text(
            "# the worker on this deployment's GPU host\n"
            f"{WORKER[0]} {WORKER[1]}\n")
        self.assertEqual(list(WORKER), embedding.remote_command())

    def test_a_file_of_only_comments_is_no_command(self):
        Path(embedding.REMOTE_CMD_CONF).write_text("# not configured yet\n")
        self.assertIsNone(embedding.remote_command())


class TheRemoteShellReadsExactlyWhatWasConfigured(Configured):
    """ssh joins its arguments and hands them to a shell, so every word is
    quoted. A path is a path, not the beginning of a second command."""

    def remote_argument(self, *argv):
        self.destination()
        self.command(*argv)

        with patch("subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess(
                [], 0, reply(VECTORS), b"")
            embedding.embed_documents(["one"], model="BAAI/bge-m3")

        return run.call_args[0][0][-1]

    def test_an_ordinary_path_is_passed_through_unchanged(self):
        self.assertEqual(" ".join(WORKER), self.remote_argument(*WORKER))

    def test_a_path_with_spaces_stays_one_word(self):
        sent = self.remote_argument("/srv/my tools/python", "/srv/w.py")
        self.assertEqual("'/srv/my tools/python' /srv/w.py", sent)

    def test_a_semicolon_cannot_start_a_second_command(self):
        sent = self.remote_argument("/srv/python; rm -rf ~", "/srv/w.py")
        self.assertTrue(sent.startswith("'/srv/python; rm -rf ~'"), sent)
        self.assertNotIn("; rm", sent.replace("'/srv/python; rm -rf ~'", ""))

    def test_substitution_and_globbing_are_inert(self):
        for hostile in ("$(id)", "`id`", "/srv/*/python", "a&&b", "x|y"):
            with self.subTest(argument=hostile):
                sent = self.remote_argument(hostile, "/srv/w.py")
                self.assertIn("'", sent.split(" /srv/w.py")[0])

    def test_a_home_relative_path_still_expands_remotely(self):
        """A deployment names a path in the remote account without knowing its
        home directory. Quoted literally, the shell would not expand it."""
        sent = self.remote_argument("~/embed/venv/bin/python", "~/embed/worker.py")
        self.assertEqual('"$HOME"/embed/venv/bin/python "$HOME"/embed/worker.py',
                         sent)

    def test_a_home_relative_path_with_a_space_is_still_one_word(self):
        sent = self.remote_argument("~/my embed/python", "/srv/w.py")
        self.assertEqual(""""$HOME"/'my embed/python' /srv/w.py""", sent)


class TheWireIsUnchanged(Configured):
    """The worker deployed today is the one from before this change, so the
    request it receives must be the request it already understands.

    A new protocol belongs with the new worker, in one step, so that a client
    and a worker can never disagree about who applies the document prefix.
    """

    def sent_payload(self):
        self.destination()
        self.command(*WORKER)

        with patch("subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess(
                [], 0, reply(VECTORS), b"")
            embedding.embed_documents(["one"], model="BAAI/bge-m3",
                                      batch_size=8)

        return json.loads(run.call_args.kwargs["input"])

    def test_the_request_keys_are_the_ones_the_current_worker_reads(self):
        self.assertEqual({"model", "texts", "batch"}, set(self.sent_payload()))

    def test_the_values_are_the_ones_it_expects(self):
        payload = self.sent_payload()
        self.assertEqual("BAAI/bge-m3", payload["model"])
        self.assertEqual(["one"], payload["texts"])
        self.assertEqual(8, payload["batch"])

    def test_the_client_does_not_yet_send_retrieval_semantics(self):
        """Today's worker applies the prefix itself. Sending one as well would
        apply it twice."""
        self.assertNotIn("prefix", self.sent_payload())

    def test_ssh_is_never_interactive(self):
        """A wrong key must fail the index, not sit on a password prompt."""
        self.destination()
        self.command(*WORKER)

        with patch("subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess(
                [], 0, reply(VECTORS), b"")
            embedding.embed_documents(["one"], model="BAAI/bge-m3")

        self.assertIn("BatchMode=yes", run.call_args[0][0])


class NothingHereReachesAnybodysMachine(unittest.TestCase):
    """The fixtures name no real host, and no test in this file contacts one.

    Every remote call above is intercepted at subprocess.run. The destination
    is under .example, which RFC 2606 reserves precisely so that a test can
    name a host without one existing.
    """

    #: Spelled out, each of these would appear in this file and the scan
    #: below would find its own assertion list. Assembled, they do not.
    PRIVATE = ("reds" + "-ml", "10." + "190.", "edgem" + "-ai",
               "heig" + "-vd", "id_pod" + "_gpu")

    def test_the_fixtures_name_nothing_real(self):
        source = Path(__file__).read_text(encoding="utf-8")

        for private in self.PRIVATE:
            with self.subTest(value=private):
                self.assertNotIn(private, source)

    def test_the_destination_is_a_reserved_name(self):
        self.assertTrue(TARGET.endswith(".example"))


if __name__ == "__main__":
    unittest.main()
