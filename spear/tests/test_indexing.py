"""Indexing guard rails: chunk size and binary filtering.

Both defects covered here had been present from the start but were invisible:
the default ONNX embedder truncates at 256 tokens without a word, so a 1.2 MB
chunk of binary went unnoticed. The first GPU embedder OOM'd the card.
"""
import os
import sys
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import embedding
import index_corpus
import rag_chat


class ChunkCapTest(unittest.TestCase):

    def test_line_longer_than_chunk_size_is_split(self):
        """chunk_text splits on newlines. A binary or minified file has none:
        without a cap it comes out as a single chunk."""
        blob = "A" * 500_000                       # zero newline
        chunks = index_corpus.chunk_text(blob, "blob.bin")
        self.assertGreater(len(chunks), 1)
        for ch in chunks:
            self.assertLessEqual(len(ch["text"]), index_corpus.MAX_CHUNK_CHARS)

    def test_normal_file_keeps_its_line_numbers(self):
        text = "\n".join(f"ligne {i}" for i in range(400))
        chunks = index_corpus.chunk_text(text, "a.c")
        self.assertTrue(all(len(c["text"]) <= index_corpus.MAX_CHUNK_CHARS
                            for c in chunks))
        self.assertEqual(1, chunks[0]["start_line"])
        self.assertTrue(all(c["end_line"] >= c["start_line"] for c in chunks))

    def test_split_chunks_keep_the_source_line_range(self):
        chunks = index_corpus.chunk_text("B" * 40_000, "x.bin")
        self.assertEqual({(c["start_line"], c["end_line"]) for c in chunks},
                         {(chunks[0]["start_line"], chunks[0]["end_line"])})


class BinaryFilterTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _write(self, name, data):
        p = os.path.join(self.tmp.name, name)
        with open(p, "wb") as f:
            f.write(data)
        return p

    def test_elf_binary_is_rejected(self):
        p = self._write("uuu", b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 4096)
        self.assertFalse(index_corpus.is_indexable_text(p))

    def test_extensionless_shell_script_is_kept(self):
        """The filter must NOT require an extension: the helpers under
        scripts/ have none, and they are precisely why that directory is
        walked without an extension filter."""
        p = self._write("build", b"#!/bin/sh\nset -e\nexec make \"$@\"\n")
        self.assertTrue(index_corpus.is_indexable_text(p))

    def test_oversized_text_is_rejected(self):
        p = self._write("huge.sh", b"# comment\n" * 20_000)
        self.assertGreater(os.path.getsize(p), index_corpus.MAX_SCRIPT_BYTES)
        self.assertFalse(index_corpus.is_indexable_text(p))

    def test_missing_file_is_not_fatal(self):
        self.assertFalse(index_corpus.is_indexable_text(
            os.path.join(self.tmp.name, "absent")))


if __name__ == "__main__":
    unittest.main()


class StagedSwapTest(unittest.TestCase):
    """The live collection must never be destroyed before a complete
    replacement exists."""

    def setUp(self):
        import chromadb
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.client = chromadb.PersistentClient(path=self.tmp.name)

    def _names(self):
        return {c.name for c in self.client.list_collections()}

    def test_existing_collection_survives_a_failed_build(self):
        live = embedding.create_collection(self.client, "corpus")
        live.add(ids=["old"], documents=["ancien contenu"],
                 embeddings=[[0.1, 0.2]])

        staging = embedding.staged_collection(self.client, "corpus")
        staging.add(ids=["new"], documents=["partiel"], embeddings=[[0.3, 0.4]])
        # ... the build crashes here: no commit_staged

        survivor = self.client.get_collection("corpus")
        self.assertEqual(1, survivor.count())
        self.assertEqual(["old"], survivor.get()["ids"])

    def test_commit_replaces_and_leaves_no_staging_behind(self):
        live = embedding.create_collection(self.client, "corpus")
        live.add(ids=["old"], documents=["ancien"], embeddings=[[0.1, 0.2]])

        staging = embedding.staged_collection(self.client, "corpus")
        staging.add(ids=["a", "b"], documents=["x", "y"],
                    embeddings=[[0.3, 0.4], [0.5, 0.6]])
        embedding.commit_staged(self.client, "corpus")

        self.assertEqual({"corpus"}, self._names())
        self.assertEqual(2, self.client.get_collection("corpus").count())

    def test_stale_staging_from_an_interrupted_run_is_discarded(self):
        embedding.staged_collection(self.client, "corpus").add(
            ids=["moitie"], documents=["reste d'un run tue"],
            embeddings=[[0.1, 0.2]])
        fresh = embedding.staged_collection(self.client, "corpus")
        self.assertEqual(0, fresh.count())

    def test_first_indexing_needs_no_existing_collection(self):
        embedding.staged_collection(self.client, "corpus").add(
            ids=["a"], documents=["x"], embeddings=[[0.1, 0.2]])
        embedding.commit_staged(self.client, "corpus")
        self.assertEqual({"corpus"}, self._names())

    def test_commit_preserves_the_embedder_stamp(self):
        os.environ["SPEAR_EMBED_MODEL"] = "BAAI/bge-m3"
        self.addCleanup(os.environ.pop, "SPEAR_EMBED_MODEL", None)
        embedding.staged_collection(self.client, "corpus").add(
            ids=["a"], documents=["x"], embeddings=[[0.1, 0.2]])
        embedding.commit_staged(self.client, "corpus")
        self.assertEqual("BAAI/bge-m3", embedding.collection_model(
            self.client.get_collection("corpus")))


class HelpDoesNotIndexTest(unittest.TestCase):
    """`spear-index --help` used to run an indexation.

    --help matched no branch of the parser, fell through the
    `not a.startswith("-")` guard as if it were a flag, and the walk started on
    the current directory. A request for usage text must not write a
    collection — and on a large tree it does not even fail fast.
    """

    APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _usage(self, script, flag):
        return subprocess.run(
            [sys.executable, os.path.join(self.APP, script), flag],
            capture_output=True, text=True, timeout=60,
            env=dict(os.environ, SPEAR_DB_PATH="/nonexistent/must-not-be-written"))

    def test_index_dir_prints_usage_and_indexes_nothing(self):
        for flag in ("--help", "-h"):
            done = self._usage("index_dir.py", flag)
            self.assertEqual(0, done.returncode, done.stderr)
            self.assertIn("spear-index", done.stdout)
            self.assertNotIn("Indexing", done.stdout)

    def test_index_corpus_prints_usage_and_indexes_nothing(self):
        done = self._usage("index_corpus.py", "--help")
        self.assertEqual(0, done.returncode, done.stderr)
        self.assertNotIn("Indexing", done.stdout)


class ReindexOptionsTest(unittest.TestCase):
    """Indexing options must survive /reindex AND a save_projects(): otherwise
    the corpus re-pollutes itself."""

    def setUp(self):
        import rag_chat
        self.rag_chat = rag_chat
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self._file, self._proj = rag_chat.PROJECTS_FILE, rag_chat.PROJECT
        rag_chat.PROJECTS_FILE = os.path.join(self.tmp.name, "projects.json")
        self.addCleanup(setattr, rag_chat, "PROJECTS_FILE", self._file)
        self.addCleanup(setattr, rag_chat, "PROJECT", self._proj)

    def _write(self, data):
        import json
        with open(self.rag_chat.PROJECTS_FILE, "w") as f:
            json.dump(data, f)

    def test_exclusions_become_cli_flags(self):
        self._write({"so3": {"path": "/x", "kind": "generic",
                             "exclude": ["lvgl", "micropython"]}})
        self.rag_chat.PROJECT = "so3"
        self.assertEqual(["--exclude", "lvgl", "--exclude", "micropython"],
                         self.rag_chat.reindex_options())

    def test_project_without_options_passes_nothing(self):
        self._write({"lvgl": {"path": "/x", "kind": "generic"}})
        self.rag_chat.PROJECT = "lvgl"
        self.assertEqual([], self.rag_chat.reindex_options())

    def test_unregistered_adhoc_project_is_not_fatal(self):
        self._write({})
        self.rag_chat.PROJECT = "adhoc:quelque-chose"
        self.assertEqual([], self.rag_chat.reindex_options())

    def test_extra_keys_survive_a_load_save_round_trip(self):
        """save_projects() writes back what load_projects() returned. If load
        rebuilt specs from path/kind alone, the first `spear-corpus add` would
        erase every exclusion."""
        self._write({"so3": {"path": "/x", "kind": "generic",
                             "exclude": ["lvgl"], "include_build": True}})
        loaded = self.rag_chat.load_projects()
        self.rag_chat.save_projects(loaded)
        again = self.rag_chat.load_projects()
        self.assertEqual(["lvgl"], again["so3"]["exclude"])
        self.assertTrue(again["so3"]["include_build"])

    def test_include_build_flag(self):
        self._write({"p": {"path": "/x", "kind": "generic",
                           "include_build": True}})
        self.rag_chat.PROJECT = "p"
        self.assertIn("--include-build", self.rag_chat.reindex_options())


class ReindexTargetsTheCorpusNotTheCwdTest(unittest.TestCase):
    """/reindex rebuilds the CORPUS's index, wherever the session stands.

    A workspace split registers the components and leaves you in the directory
    above them. Pointed at the cwd, /reindex walked that umbrella instead of
    the chosen corpus: four sibling trees, the file cap, a refusal. Under the
    cap it would have been worse than the refusal — the destination collection
    is derived from the same path, so the chunks would have landed in
    adhoc_<md5(cwd)> while the session went on reading adhoc_<md5(corpus)>.
    """

    def setUp(self):
        import rag_chat
        self.rag_chat = rag_chat
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = os.path.realpath(self.tmp.name)

        for name in ("PROJECTS_FILE", "PROJECT", "PROJECT_SPEC", "PROJECT_KIND",
                     "PROJECT_ROOT", "CORPUS_ROOT", "COLLECTION_NAME"):
            self.addCleanup(setattr, rag_chat, name, getattr(rag_chat, name))

        rag_chat.PROJECTS_FILE = os.path.join(self.root, "projects.json")

        # the session the transcript showed: corpus `agency`, cwd one level up

        self._write({
            "agency": {"path": f"{self.root}/agency", "kind": "generic"},
            "buildroot": {"path": f"{self.root}/buildroot", "kind": "generic"},
            "opencn-qemu": {"path": f"{self.root}/qemu", "kind": "generic"},
        })
        rag_chat.PROJECT = "agency"
        rag_chat.PROJECT_SPEC = {"name": "agency", "path": f"{self.root}/agency",
                                 "kind": "generic"}
        rag_chat.PROJECT_KIND = "generic"
        rag_chat.PROJECT_ROOT = self.root                  # tools run here
        rag_chat.CORPUS_ROOT = f"{self.root}/agency"       # the corpus is here
        rag_chat.COLLECTION_NAME = "adhoc_deadbeef"

    def _write(self, data):
        import json
        with open(self.rag_chat.PROJECTS_FILE, "w") as f:
            json.dump(data, f)

    def test_it_indexes_the_corpus_tree(self):
        cmd = self.rag_chat.reindex_command()
        self.assertIn(f"{self.root}/agency", cmd)
        self.assertNotIn(self.root, cmd)

    def test_it_names_the_collection_the_session_reads(self):
        """Derived from the cwd, the name silently misses: the indexer would
        report `Done` into a collection nobody queries."""
        cmd = self.rag_chat.reindex_command()
        self.assertEqual("adhoc_deadbeef", cmd[cmd.index("--collection") + 1])

    def test_a_buildsystem_corpus_uses_the_curated_indexer(self):
        """Declared by the registry entry, not inferred from its type.

        The curated walk was selected by `kind == "edgem1"`, which meant a
        corpus could not ask for it without also taking a collection name, an
        autoindex policy and a domain prompt it might not want.
        """
        # Declared where a real corpus declares it. The registry is
        # authoritative over the spec a session was launched with, which is
        # the point: the entry describes the corpus.
        self._write({"agency": {"path": f"{self.root}/agency",
                                "kind": "buildsystem",
                                "indexer": "buildsystem"}})
        cmd = self.rag_chat.reindex_command()
        self.assertTrue(cmd[1].endswith("index_corpus.py"), cmd)
        self.assertIn(f"{self.root}/agency", cmd)
        # The curated walk derives its own collection; nothing to name.
        self.assertNotIn("--collection", cmd)

    def test_the_generic_indexer_is_the_default(self):
        cmd = self.rag_chat.reindex_command()
        self.assertTrue(cmd[1].endswith("index_dir.py"), cmd)
        self.assertIn("--collection", cmd)

    def test_corpora_below_are_excluded_from_an_umbrella_reindex(self):
        """The umbrella case, where the cap actually fired: the components own
        their indexes and the session federates them, so re-walking them into
        one collection buys nothing but the runaway."""
        self.rag_chat.PROJECT = "workspace:opencn"
        self.rag_chat.PROJECT_SPEC = {"name": "workspace:opencn",
                                      "path": self.root, "kind": "generic"}
        self.rag_chat.CORPUS_ROOT = self.root
        opts = self.rag_chat.reindex_options()
        self.assertEqual(["--exclude", "agency", "--exclude", "buildroot",
                          "--exclude", "qemu"], opts)

    def test_a_declared_exclusion_is_not_repeated(self):
        self._write({"agency": {"path": f"{self.root}/agency",
                                "kind": "generic", "exclude": ["vendor"]},
                     "vendor": {"path": f"{self.root}/agency/vendor",
                                "kind": "generic"}})
        self.assertEqual(["--exclude", "vendor"], self.rag_chat.reindex_options())


class CorpusCommandTest(unittest.TestCase):
    """/corpus and the `spear-corpus` CLI are one implementation.

    The shell script used to carry its own load/save, its own seed list and its
    own component walk against a hardcoded projects.json. Two copies of a
    registry is one too many: the script's skip list had already drifted from
    the indexer's, and it could not see SPEAR_CORPUS_ROOT at all.
    """

    def setUp(self):
        import rag_chat
        self.rag_chat = rag_chat
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = os.path.realpath(self.tmp.name)
        self.addCleanup(setattr, rag_chat, "PROJECTS_FILE",
                        rag_chat.PROJECTS_FILE)
        rag_chat.PROJECTS_FILE = os.path.join(self.root, "projects.json")
        self._write({})

        for name in ("agency", "linux"):
            os.makedirs(os.path.join(self.root, "tree", name), exist_ok=True)

    def _write(self, data):
        with open(self.rag_chat.PROJECTS_FILE, "w") as f:
            json.dump(data, f)

    def _registry(self):
        with open(self.rag_chat.PROJECTS_FILE) as f:
            return json.load(f)

    def _corpus(self, *args, current=None):
        return self.rag_chat.handle_corpus_command(args, current=current)

    def test_add_registers_the_tree(self):
        out = self._corpus("add", "agency", f"{self.root}/tree/agency")
        self.assertIn("registered 'agency'", out)
        self.assertEqual(f"{self.root}/tree/agency",
                         self._registry()["agency"]["path"])

    def test_add_defaults_to_the_current_directory(self):
        cwd = os.getcwd()
        self.addCleanup(os.chdir, cwd)
        os.chdir(f"{self.root}/tree/agency")
        self._corpus("add", "agency")
        self.assertEqual(f"{self.root}/tree/agency",
                         self._registry()["agency"]["path"])

    def test_add_refuses_a_name_that_is_really_a_path(self):
        """`/corpus add agency/linux`, typed from the tree above it, registered
        the NAME "agency/linux" pointing at the cwd — the path defaults to it,
        and nothing said the first argument had been read as a name. The
        session then announced "corpus: agency/linux" while indexing the
        parent."""
        cwd = os.getcwd()
        self.addCleanup(os.chdir, cwd)
        os.chdir(f"{self.root}/tree")
        out = self._corpus("add", "agency/linux")
        self.assertIn("not a corpus name", out)
        self.assertEqual({}, self._registry())

    def test_that_refusal_shows_the_command_that_was_meant(self):
        """Only when the path exists: a refusal that guesses is worse than one
        that states the rule."""
        cwd = os.getcwd()
        self.addCleanup(os.chdir, cwd)
        os.chdir(self.root)
        self.assertIn("add agency tree/agency", self._corpus("add", "tree/agency"))
        self.assertNotIn("You gave a path", self._corpus("add", "no/such/tree"))

    def test_a_name_with_a_space_is_refused_too(self):
        self.assertIn("not a corpus name", self._corpus("add", "my corpus"))

    def test_the_names_already_in_use_all_pass(self):
        """The rule is a guard on new names, not a migration: every corpus the
        registry holds today has to keep working."""
        for name in ("so3", "sye_sol", "llama.cpp-next", "avz.back",
                     "opencn-u-boot", "spear", "virt64"):
            self.assertTrue(self.rag_chat.CORPUS_NAME_RE.fullmatch(name), name)

    def test_add_refuses_a_path_that_is_not_a_directory(self):
        self.assertIn("not a directory",
                      self._corpus("add", "x", f"{self.root}/nope"))
        self.assertEqual({}, self._registry())

    def test_re_adding_the_same_tree_keeps_its_declared_options(self):
        """exclude/collection are hand-written. Rebuilding the spec from the
        name and the path alone would drop them without a word."""
        self._write({"agency": {"path": f"{self.root}/tree/agency",
                                "kind": "generic", "exclude": ["linux"]}})
        self._corpus("add", "agency", f"{self.root}/tree/agency")
        self.assertEqual(["linux"], self._registry()["agency"]["exclude"])

    def test_add_refuses_to_repoint_an_existing_name(self):
        self._write({"agency": {"path": f"{self.root}/tree/agency",
                                "kind": "generic"}})
        out = self._corpus("add", "agency", f"{self.root}/tree/linux")
        self.assertIn("already points at", out)
        self.assertEqual(f"{self.root}/tree/agency",
                         self._registry()["agency"]["path"])

    def test_rm_leaves_the_session_own_corpus_alone(self):
        """Its index, history and memories are bound to it at launch."""
        self._write({"agency": {"path": f"{self.root}/tree/agency",
                                "kind": "generic"}})
        out = self._corpus("rm", "agency", current="agency")
        self.assertIn("relaunch", out)
        self.assertIn("agency", self._registry())

    def test_rm_removes_and_says_the_index_stays(self):
        self._write({"agency": {"path": f"{self.root}/tree/agency",
                                "kind": "generic"}})
        out = self._corpus("rm", "agency")
        self.assertIn("removed", out)
        self.assertIn("index is left", out)
        self.assertNotIn("agency", self._registry())

    def test_scan_registers_the_components_and_names_the_owned_ones(self):
        for i in range(5):
            Path(f"{self.root}/tree/agency/f{i}.c").write_text("int x;\n")
            Path(f"{self.root}/tree/linux/f{i}.c").write_text("int y;\n")

        self._write({"linux": {"path": f"{self.root}/tree/linux",
                               "kind": "generic"}})
        out = self._corpus("scan", f"{self.root}/tree", "--min", "3")
        self.assertIn("already: linux", out)
        self.assertIn("-> corpus 'agency'", out)
        self.assertIn("agency", self._registry())

    def test_scan_takes_min_as_a_value_not_as_a_directory(self):
        """`--min 500` used to leave 500 in the positionals, where it was read
        as the directory to scan."""
        out = self._corpus("scan", "--min", "500")
        self.assertNotIn("not a directory: 500", out)

    def test_a_read_only_registry_does_not_kill_the_session(self):
        """save_projects raises SystemExit there — right for a CLI, fatal in a
        chat loop, which would lose the conversation to a typo."""
        os.chmod(self.rag_chat.PROJECTS_FILE, 0o444)
        os.chmod(self.root, 0o555)
        self.addCleanup(os.chmod, self.root, 0o755)
        out = self._corpus("add", "agency", f"{self.root}/tree/agency")
        self.assertIn("corpus registry", out)

    def test_list_marks_the_session_corpus(self):
        self._write({"agency": {"path": f"{self.root}/tree/agency",
                                "kind": "generic"}})
        out = self._corpus("list", current="agency")
        self.assertIn("* agency", out)
        self.assertIn("* = this session", out)

    def test_no_subcommand_lists(self):
        """Bare `/corpus` is the question people actually ask."""
        self._write({"agency": {"path": f"{self.root}/tree/agency",
                                "kind": "generic"}})
        self.assertIn("agency", self._corpus())

    def test_an_unknown_subcommand_answers_with_the_usage(self):
        self.assertEqual(self.rag_chat.CORPUS_USAGE, self._corpus("frobnicate"))


class IndexTargetTest(unittest.TestCase):
    """What gets indexed is decided at import, from the command line.

    There is no built-in root any more. The platform ships nobody's checkout,
    so the tree is the one named on the command line or, failing that, the
    one the caller is standing in. PROJECT_ROOT is computed at import, so
    each case goes through a fresh subprocess rather than a module reload."""

    APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _project_root(self, *argv, cwd=None):
        out = subprocess.run(
            [sys.executable, "-c",
             f"import sys; sys.path.insert(0, {self.APP!r});"
             f" sys.argv = ['x', *{list(argv)!r}];"
             " import index_corpus as m; print(m.PROJECT_ROOT)"],
            capture_output=True, text=True, cwd=cwd or self.APP)
        self.assertEqual(0, out.returncode, out.stderr)
        return out.stdout.strip()

    def test_the_argument_names_the_tree(self):
        self.assertEqual("/ailleurs/monrepo",
                         self._project_root("/ailleurs/monrepo"))

    def test_without_an_argument_it_is_the_working_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(os.path.realpath(tmp),
                             os.path.realpath(self._project_root(cwd=tmp)))

    def test_a_relative_argument_is_resolved_against_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.mkdir(os.path.join(tmp, "sub"))
            self.assertEqual(
                os.path.realpath(os.path.join(tmp, "sub")),
                os.path.realpath(self._project_root("sub", cwd=tmp)))


class GpuPinningTest(unittest.TestCase):
    """GPU pinning on a shared machine. The subprocess is mandatory: _pin_gpu()
    runs at import, and CUDA reads CUDA_VISIBLE_DEVICES at initialisation — a
    test re-importing the module in place would not reproduce the real
    sequence."""

    APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    UUID = "GPU-00000000-0000-0000-0000-000000000000"

    def _visible(self, env, conf=None):
        """The subprocess must start from a pristine state.

        Two traps. (1) `import embedding` already runs _pin_gpu() with the
        machine's REAL GPU_CONF: on a deployed host that has one, the variable
        would be set before we redirect anything. (2) _pin_gpu() returns
        immediately if CUDA_VISIBLE_DEVICES is already set. So we clear it
        after the import, except when the test case is precisely checking that
        a caller's choice survives."""
        e = {k: v for k, v in os.environ.items()
             if k not in ("CUDA_VISIBLE_DEVICES", "SPEAR_GPU_UUID")}
        e.update(env)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = os.path.join(tmp.name, "active-gpu.conf")
        if conf is not None:
            with open(path, "w") as f:
                f.write(conf)
        keep = "CUDA_VISIBLE_DEVICES" in env
        code = (
            f"import sys, os; sys.path.insert(0, {self.APP!r});"
            f" import embedding;"
            f" ({keep!r}) or os.environ.pop('CUDA_VISIBLE_DEVICES', None);"
            f" embedding.GPU_CONF = {path!r};"
            f" embedding._pin_gpu();"
            f" print(os.environ.get('CUDA_VISIBLE_DEVICES', '<absent>'))")
        out = subprocess.run([sys.executable, "-c", code],
                             capture_output=True, text=True, env=e, cwd=self.APP)
        self.assertEqual(0, out.returncode, out.stderr)
        return out.stdout.strip()

    def test_env_uuid_pins_the_card(self):
        self.assertEqual(self.UUID, self._visible({"SPEAR_GPU_UUID": self.UUID}))

    def test_explicit_caller_choice_wins(self):
        """An already-set CUDA_VISIBLE_DEVICES is a deliberate choice: we do
        not touch it, or we break a hand-launched run."""
        self.assertEqual("7", self._visible(
            {"CUDA_VISIBLE_DEVICES": "7", "SPEAR_GPU_UUID": self.UUID}))

    def test_nothing_configured_leaves_cuda_untouched(self):
        """On a single-GPU machine, no pinning must appear."""
        self.assertEqual("<absent>", self._visible({}))

    def test_conf_file_is_read_when_env_is_absent(self):
        self.assertEqual(self.UUID, self._visible({}, conf=self.UUID + "\n"))

    def test_empty_conf_file_pins_nothing(self):
        self.assertEqual("<absent>", self._visible({}, conf="\n"))


class EnclosingCorpusTest(unittest.TestCase):
    """Switching to a corpus that sits UNDER the current directory.

    detect_project answers "is the cwd inside a corpus"; this is the opposite
    case, which used to leave the session with no RAG at all and the model
    answering from parametric memory.
    """

    def setUp(self):
        import rag_chat
        self.rag_chat = rag_chat
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = os.path.realpath(self.tmp.name)

    def _mk(self, *rel):
        for r in rel:
            os.makedirs(os.path.join(self.root, r), exist_ok=True)
        return {os.path.basename(r): {"path": os.path.join(self.root, r),
                                      "kind": "generic"} for r in rel}

    def test_single_child_is_unambiguous(self):
        projects = self._mk("lvgl/lvgl")
        pick, _ = self.rag_chat.enclosing_corpus(
            projects, os.path.join(self.root, "lvgl"))
        self.assertEqual("lvgl", pick)

    def test_same_name_child_wins_among_siblings(self):
        """The umbrella-checkout shape: ~/soo/so3 holds so3, u-boot, avz, qemu,
        atf — six candidates, but only one carries the parent's name."""
        projects = self._mk("soo/so3/so3", "soo/so3/u-boot", "soo/so3/avz",
                            "soo/so3/qemu", "soo/so3/atf")
        pick, cand = self.rag_chat.enclosing_corpus(
            projects, os.path.join(self.root, "soo/so3"))
        self.assertEqual("so3", pick)
        self.assertEqual(5, len(cand))

    def test_ambiguous_siblings_switch_nothing(self):
        """Two equally plausible children must NOT be picked by accident —
        the caller names them instead."""
        projects = self._mk("umbrella/target-a", "umbrella/target-b")
        pick, cand = self.rag_chat.enclosing_corpus(
            projects, os.path.join(self.root, "umbrella"))
        self.assertIsNone(pick)
        self.assertEqual(["target-a", "target-b"], cand)

    def test_a_corpus_is_not_its_own_child(self):
        projects = self._mk("so3")
        pick, cand = self.rag_chat.enclosing_corpus(
            projects, os.path.join(self.root, "so3"))
        self.assertIsNone(pick)
        self.assertEqual([], cand)

    def test_grandchildren_do_not_trigger_a_switch(self):
        """Only a direct child is close enough to be meant. A corpus buried
        two levels down is a different tree, not this one."""
        projects = self._mk("top/a/deep")
        pick, cand = self.rag_chat.enclosing_corpus(
            projects, os.path.join(self.root, "top"))
        self.assertIsNone(pick)
        self.assertEqual(["deep"], cand)

    def test_unrelated_corpora_are_ignored(self):
        projects = self._mk("elsewhere/thing")
        os.makedirs(os.path.join(self.root, "here"), exist_ok=True)
        pick, cand = self.rag_chat.enclosing_corpus(
            projects, os.path.join(self.root, "here"))
        self.assertIsNone(pick)
        self.assertEqual([], cand)

    def test_an_umbrella_keeps_the_tools_and_federates_the_parts(self):
        """Launching above the corpora attaches them; it does not pick one.

        Picking one and chdir-ing into it is what this used to do, and from
        ~/soo/so3 it moved the session into ~/soo/so3/so3 -- losing avz/,
        build/ and doc/, which is half of what the questions are about, while
        the orientation map (written for the umbrella) pointed at paths that no
        longer existed from there. The chunk-path prefix in federated_corpora
        is what makes staying put safe: the model reads a path bash can open.
        """
        projects = self._mk("soo/so3/so3", "soo/so3/u-boot", "soo/so3/avz")
        parent = os.path.join(self.root, "soo/so3")
        prev = os.getcwd()
        self.addCleanup(os.chdir, prev)
        os.chdir(parent)
        argv, pf = sys.argv, self.rag_chat.PROJECTS_FILE
        self.rag_chat.PROJECTS_FILE = os.path.join(self.tmp.name, "p.json")
        with open(self.rag_chat.PROJECTS_FILE, "w") as f:
            json.dump(projects, f)
        sys.argv = ["spear-chat"]
        try:
            spec = self.rag_chat.resolve_project_at_startup()
        finally:
            sys.argv, self.rag_chat.PROJECTS_FILE = argv, pf
        self.assertEqual("workspace:so3", spec["name"])
        self.assertEqual(os.path.realpath(parent), os.path.realpath(os.getcwd()))
        self.assertEqual(["avz", "so3", "u-boot"], spec["corpora"])

    def test_a_snapshot_tree_is_never_federated(self):
        """avz.back and friends are registered, indexed, and never the point."""
        projects = self._mk("soo/so3/so3", "soo/so3/avz.back")
        parent = os.path.join(self.root, "soo/so3")
        found = self.rag_chat.corpora_below(projects, parent)
        self.assertEqual(["so3"], sorted(found))

    def test_here_pins_the_session_to_the_current_directory(self):
        projects = self._mk("soo/so3/so3")
        parent = os.path.join(self.root, "soo/so3")
        prev = os.getcwd()
        self.addCleanup(os.chdir, prev)
        os.chdir(parent)
        argv, pf = sys.argv, self.rag_chat.PROJECTS_FILE
        self.rag_chat.PROJECTS_FILE = os.path.join(self.tmp.name, "p2.json")
        with open(self.rag_chat.PROJECTS_FILE, "w") as f:
            json.dump(projects, f)
        sys.argv = ["spear-chat", "--here"]
        try:
            spec = self.rag_chat.resolve_project_at_startup()
        finally:
            sys.argv, self.rag_chat.PROJECTS_FILE = argv, pf
        self.assertTrue(spec["name"].startswith("adhoc:"))
        self.assertEqual(os.path.realpath(parent), os.path.realpath(os.getcwd()))


class WriteRequiresPriorReadTest(unittest.TestCase):
    """write_file must not overwrite a file the model has not opened.

    Reconstructing an existing file from whatever the retrieved context held
    works only while the whole file fits in that context; past it the model
    rewrites from a partial view and silently drops what it never saw. The
    pre-existing size check cannot catch this — a same-length hallucination
    passes it.
    """

    def setUp(self):
        import rag_chat
        from tool_runtime import Workspace
        self.rag_chat = rag_chat
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self._ws = rag_chat.WORKSPACE
        self.addCleanup(setattr, rag_chat, "WORKSPACE", self._ws)
        rag_chat.WORKSPACE = Workspace.from_path(self.tmp.name)
        self.target = os.path.join(self.tmp.name, "ping.c")
        with open(self.target, "w") as f:
            f.write("int main(void)\n{\n\treturn 0;\n}\n")

    def _write(self, cache, content="int main(void)\n{\n\treturn 1;\n}"):
        # The rule under test is read-before-write, which only has anything
        # to say in a session that may write at all: the process default is
        # safe mode, where the handler refuses on the mode before it ever
        # asks whether the file was read.

        from unittest.mock import patch
        from tool_runtime import ExecutionMode

        with patch.object(self.rag_chat, "EXECUTION_MODE", ExecutionMode.AUTO):
            return self.rag_chat.execute_tool(
                "write_file", {"path": "ping.c", "content": content}, cache)

    def test_overwrite_without_reading_is_refused(self):
        out = self._write({})
        self.assertIn("ERROR", out)
        self.assertIn("not read it this turn", out)
        self.assertIn("return 0", open(self.target).read())   # untouched

    def test_reading_it_first_unlocks_the_write(self):
        cache = {}
        self.rag_chat._note_files_read(cache, "cat ping.c")
        self.assertTrue(self.rag_chat._was_read_this_turn(cache, self.target))

    def test_ls_is_not_reading(self):
        """Knowing a file exists is not knowing what is in it."""
        cache = {}
        self.rag_chat._note_files_read(cache, "ls -la ping.c")
        self.assertFalse(self.rag_chat._was_read_this_turn(cache, self.target))

    def test_a_new_file_needs_no_prior_read(self):
        out = self.rag_chat.execute_tool(
            "write_file", {"path": "brand_new.c", "content": "int x;"}, {})
        self.assertNotIn("not read it this turn", out)

    def test_flags_are_not_mistaken_for_paths(self):
        cache = {}
        self.rag_chat._note_files_read(cache, "grep -n --color=never main ping.c")
        self.assertTrue(self.rag_chat._was_read_this_turn(cache, self.target))

    def test_a_malformed_command_is_not_fatal(self):
        cache = {}
        self.rag_chat._note_files_read(cache, "cat 'unclosed")
        self.assertEqual(set(), cache.get(self.rag_chat.READ_PATHS, set()))


class QuietEmbedderLoadTest(unittest.TestCase):
    """Loading the embedder must not narrate itself mid-turn.

    Two sources, two levers: the "Loading weights" bar belongs to transformers
    (HF_HUB_DISABLE_PROGRESS_BARS does nothing to it), and the
    unauthenticated-Hub warning comes from the native client on stderr and
    exists in no Python module — it can only be avoided by not contacting the
    Hub, hence the conditional offline switch.

    The decision is exercised through a fake environment. An earlier version
    of these tests flipped the real HF_HUB_OFFLINE and leaked it into every
    later test that loads an actual model, which then failed on
    local_files_only.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _env_with(self, *models):
        for m in models:
            os.makedirs(os.path.join(self.tmp.name,
                                     "models--" + m.replace("/", "--")))
        return {"HF_HUB_CACHE": self.tmp.name}

    def test_a_cached_model_goes_offline(self):
        self.assertTrue(embedding.should_go_offline(
            "BAAI/bge-m3", self._env_with("BAAI/bge-m3")))

    def test_an_uncached_model_stays_online(self):
        """Offline mode turns a first download into an OSError, so it must
        never be imposed on a model that is not there yet."""
        self.assertFalse(embedding.should_go_offline(
            "BAAI/bge-m3", self._env_with()))

    def test_an_explicit_choice_is_left_alone(self):
        env = self._env_with("BAAI/bge-m3")
        env["HF_HUB_OFFLINE"] = "0"
        self.assertFalse(embedding.should_go_offline("BAAI/bge-m3", env))

    def test_hf_home_is_honoured(self):
        os.makedirs(os.path.join(self.tmp.name, "hub", "models--a--b"))
        self.assertTrue(embedding._is_cached("a/b", {"HF_HOME": self.tmp.name}))

    def test_cache_path_matches_huggingface_hubs_own(self):
        """The path is recomputed to avoid importing the Hub client too early;
        it must not drift from what that client would use."""
        from huggingface_hub.constants import HF_HUB_CACHE
        marker = "models--zz--probe"
        os.makedirs(os.path.join(HF_HUB_CACHE, marker), exist_ok=True)
        self.addCleanup(os.rmdir, os.path.join(HF_HUB_CACHE, marker))
        self.assertTrue(embedding._is_cached("zz/probe", {}))


class EditFileTest(unittest.TestCase):
    """edit_file raised TypeError on every edit of an existing file.

    resolve_path returns a Path and re.search refuses one, so the
    generated-file guard blew up before any edit could happen. The tool was
    dead from the moment the workspace started resolving to Path objects,
    which is why the model always fell back to rewriting whole files.
    """

    def setUp(self):
        import rag_chat
        from tool_runtime import Workspace, ExecutionMode
        self.rag_chat = rag_chat
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        for attr, val in (("WORKSPACE", Workspace.from_path(self.tmp.name)),
                          ("EXECUTION_MODE", ExecutionMode.AUTO),
                          ("BYPASS_PERMISSIONS", True)):
            self.addCleanup(setattr, rag_chat, attr, getattr(rag_chat, attr))
            setattr(rag_chat, attr, val)

    def _write(self, name, text):
        p = os.path.join(self.tmp.name, name)
        with open(p, "w") as f:
            f.write(text)
        return p

    def test_an_ordinary_edit_applies(self):
        p = self._write("x.c", "int main(void)\n{\n\treturn 0;\n}\n")
        out = self.rag_chat.edit_file("x.c", "return 0", "return 1")
        self.assertIn("OK", out)
        self.assertIn("return 1", open(p).read())

    def test_a_generated_file_is_still_refused(self):
        """The guard the crash was hiding must keep working."""
        self._write("gen.c", "/* automatically generated */\nint x;\n")
        out = self.rag_chat.edit_file("gen.c", "int x", "int y")
        self.assertIn("GENERATED", out)

    def test_a_path_under_generated_is_refused(self):
        os.makedirs(os.path.join(self.tmp.name, "generated"))
        self._write("generated/tbl.c", "int t;\n")
        out = self.rag_chat.edit_file("generated/tbl.c", "int t", "int u")
        self.assertIn("GENERATED", out)


class ToolErrorHandlerTest(unittest.TestCase):
    """The handler that keeps a tool bug from killing the session must not
    itself crash: it subscripted a ModelToolCall as if it were a dict, so a
    tool failure took the whole process down on a traceback."""

    def test_a_tool_call_exposes_name_as_an_attribute(self):
        from model_backend import ModelToolCall
        c = ModelToolCall(id="1", name="edit_file", arguments={"path": "x"})
        self.assertEqual("edit_file", c.name)
        with self.assertRaises(TypeError):
            c["name"]


class RemoteEmbeddingTest(unittest.TestCase):
    """Offloading the indexing GPU work to another machine.

    Measured on identical real chunks: 28 chunks/s on the laptop, 129 through
    the offload. Only the compute moves — chunking and the Chroma writes stay
    local, so there is no tree to mirror and no index to copy back.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.conf = os.path.join(self.tmp.name, "active-embed-remote.conf")
        self._prev = (embedding.REMOTE_CONF,
                      os.environ.get("SPEAR_EMBED_REMOTE"),
                      os.environ.get("SPEAR_EMBED_REMOTE_SSH_OPTS"))
        self.addCleanup(self._restore)
        embedding.REMOTE_CONF = self.conf
        for k in ("SPEAR_EMBED_REMOTE", "SPEAR_EMBED_REMOTE_SSH_OPTS"):
            os.environ.pop(k, None)

    def _restore(self):
        embedding.REMOTE_CONF = self._prev[0]
        for k, v in zip(("SPEAR_EMBED_REMOTE", "SPEAR_EMBED_REMOTE_SSH_OPTS"),
                        self._prev[1:]):
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v

    def test_no_config_means_local(self):
        self.assertIsNone(embedding.remote_target())

    def test_the_config_names_the_target(self):
        with open(self.conf, "w") as f:
            f.write("user@host\n")
        self.assertEqual("user@host", embedding.remote_target())

    def test_an_empty_env_var_disables_the_offload(self):
        """`in os.environ`, not truthiness: the worker on the GPU host sets it
        to "" so it does not ship the batch on again. Falling through to the
        config file there would make it recurse."""
        with open(self.conf, "w") as f:
            f.write("user@host\n")
        os.environ["SPEAR_EMBED_REMOTE"] = ""
        self.assertIsNone(embedding.remote_target())

    def test_the_env_var_overrides_the_config(self):
        with open(self.conf, "w") as f:
            f.write("user@host\n")
        os.environ["SPEAR_EMBED_REMOTE"] = "other@elsewhere"
        self.assertEqual("other@elsewhere", embedding.remote_target())

    def test_ssh_options_travel_with_the_target(self):
        """A host with a low MaxAuthTries needs IdentitiesOnly, so indexing
        must work from cron or a bare shell, not only where the variable
        happens to be exported."""
        with open(self.conf, "w") as f:
            f.write("user@host\n-o IdentitiesOnly=yes -i $HOME/.ssh/k\n")
        opts = embedding.remote_ssh_opts()
        self.assertIn("-o", opts)
        self.assertIn("IdentitiesOnly=yes", opts)
        self.assertTrue(any(o.endswith("/.ssh/k") for o in opts), opts)
        self.assertNotIn("$HOME/.ssh/k", opts)

    def test_options_without_a_config_are_empty(self):
        self.assertEqual([], embedding.remote_ssh_opts())


class MultiLinePasteTest(unittest.TestCase):
    """Pasting several lines must arrive as one message.

    The drain loop used select() on the raw fd together with
    sys.stdin.readline(). The buffered reader pulls EVERYTHING pending into
    Python's own buffer and hands back one line; select then looks at the
    kernel fd, finds it empty, and the loop stops while the rest sits
    unreachable in that buffer. A three-line paste arrived as two, a
    thirty-line paste as two.
    """

    APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # The reader, lifted verbatim from rag_chat's read_user_input, exercised
    # under a real pty: the bug only exists when stdin is a terminal.
    CHILD = r'''
import sys, os, fcntl, select, readline
PROMPT = "> "
first = input(PROMPT)
PSTART, PEND = "\x1b[200~", "\x1b[201~"
if PSTART in first:
    buf = first.split(PSTART, 1)[1]; parts = []
    while PEND not in buf:
        parts.append(buf); nxt = sys.stdin.readline()
        if not nxt: break
        buf = nxt.rstrip("\n")
    parts.append(buf.split(PEND, 1)[0]); text = "\n".join(parts).strip()
else:
    fl = fcntl.fcntl(0, fcntl.F_GETFL)
    fcntl.fcntl(0, fcntl.F_SETFL, fl | os.O_NONBLOCK)
    try:
        rest, gap = "", 0.05
        while select.select([sys.stdin], [], [], gap)[0]:
            try: block = os.read(0, 65536)
            except BlockingIOError: break
            if not block: break
            rest += block.decode("utf-8", "replace"); gap = 0.4
    finally:
        fcntl.fcntl(0, fcntl.F_SETFL, fl)
    text = "\n".join([first] + rest.splitlines()).strip()
sys.stderr.write("LINES=%d\n" % (text.count("\n") + 1))
'''

    def _paste(self, payload):
        import pty, select as sel, time
        pid, fd = pty.fork()
        if pid == 0:
            os.execv(sys.executable, [sys.executable, "-c", self.CHILD])
        time.sleep(1.0)
        os.write(fd, payload.encode())
        out, deadline = b"", time.time() + 6
        while time.time() < deadline:
            if sel.select([fd], [], [], 0.3)[0]:
                try:
                    out += os.read(fd, 65536)
                except OSError:
                    break
        os.close(fd)
        for line in out.decode("utf-8", "replace").splitlines():
            if line.strip().startswith("LINES="):
                return int(line.strip().split("=")[1])
        return 0

    def test_a_typed_line_stays_one_line(self):
        self.assertEqual(1, self._paste("une seule ligne\n"))

    def test_a_raw_three_line_paste_arrives_whole(self):
        self.assertEqual(3, self._paste("a\nb\nc\n"))

    def test_a_long_raw_paste_arrives_whole(self):
        """Thirty lines used to arrive as two."""
        payload = "\n".join(f"ligne {i}" for i in range(1, 31)) + "\n"
        self.assertEqual(30, self._paste(payload))

    def test_bracketed_paste_still_works(self):
        self.assertEqual(3, self._paste("\x1b[200~a\nb\nc\x1b[201~\n"))


class FileCapRefusesRatherThanTruncatesTests(unittest.TestCase):
    """A partial index answers confidently from a fraction of the tree.

    It used to be a WARNING, printed into buffered output that scrolled past:
    an infrabase index came out 99.8% vendored QEMU and u-boot, with none of
    the build system it existed for, and nothing failed. The walk stops
    wherever it stops, so what it collects is not a sample -- it is whichever
    directories os.walk reached first.
    """

    APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "vendored").mkdir()
        (self.root / "build").mkdir()
        for i in range(30):
            (self.root / "vendored" / f"f{i}.c").write_text(f"int x{i};\n")
        for i in range(5):
            (self.root / "build" / f"r{i}.bb").write_text('SUMMARY = "x"\n')

    def tearDown(self):
        self.temp.cleanup()

    def run_indexer(self, *args, cap="10"):
        # Its own store. These cases index for real, and pointing them at the
        # default one left throwaway collections in the corpus the assistant
        # queries -- three of them, found only when an image was about to bake
        # them in.
        env = dict(os.environ, SPEAR_INDEX_MAX_FILES=cap,
                   SPEAR_DB_PATH=str(self.root / "_chromadb"))
        return subprocess.run(
            [sys.executable, os.path.join(self.APP, "index_dir.py"),
             str(self.root), *args],
            capture_output=True, text=True, env=env, cwd=self.APP)

    def test_hitting_the_cap_fails_and_says_what_to_exclude(self):
        out = self.run_indexer()
        self.assertEqual(2, out.returncode, out.stdout + out.stderr)
        self.assertIn("REFUSING to build a partial index", out.stderr)
        # The remedy, and the directory that actually caused it.
        self.assertIn("--exclude", out.stderr)
        self.assertIn("vendored", out.stderr)

    def test_the_previous_index_is_left_untouched(self):
        """The refusal happens BEFORE anything is embedded or committed."""
        out = self.run_indexer()
        self.assertNotIn("Done:", out.stdout)
        self.assertNotIn("indexed", out.stdout)

    def test_a_partial_index_stays_possible_on_purpose(self):
        out = self.run_indexer("--allow-partial")
        self.assertEqual(0, out.returncode, out.stdout + out.stderr)
        self.assertIn("Done:", out.stdout)

    def test_explicit_collection_name_is_used(self):
        out = self.run_indexer("--allow-partial", "--collection", "notes")
        self.assertEqual(0, out.returncode, out.stdout + out.stderr)
        self.assertIn("Done:", out.stdout)
        self.assertIn("chunks in notes", out.stdout)


class UmbrellaSessionTests(unittest.TestCase):
    """A session launched above the corpora keeps its map and its federation."""

    def setUp(self):
        import rag_chat
        self.rag_chat = rag_chat

    def test_the_federation_survives_a_spec_that_is_not_in_the_registry(self):
        """An umbrella spec is synthesised, so a lookup by name finds nothing.

        federated_corpora() looked the session up in projects.json by name; an
        umbrella is not in there, so the eight corpora it had just announced
        were silently never attached.
        """
        previous = self.rag_chat.PROJECT_SPEC
        self.addCleanup(setattr, self.rag_chat, "PROJECT_SPEC", previous)
        self.rag_chat.PROJECT_SPEC = {"name": "workspace:x", "path": "/tmp",
                                      "kind": "generic",
                                      "corpora": ["a", "b"]}
        spec = ({} or self.rag_chat.PROJECT_SPEC)
        self.assertEqual(["a", "b"],
                         self.rag_chat.attached_corpus_names(spec, {}))

    def test_the_umbrella_uses_the_map_named_after_its_directory(self):
        previous = (self.rag_chat.PROJECT, self.rag_chat.PROJECT_ROOT,
                    self.rag_chat.CORPUS_ROOT, self.rag_chat.SHIPPED_CORPUS_RULES)

        def restore():
            (self.rag_chat.PROJECT, self.rag_chat.PROJECT_ROOT,
             self.rag_chat.CORPUS_ROOT,
             self.rag_chat.SHIPPED_CORPUS_RULES) = previous
        self.addCleanup(restore)

        # A shipped map of this test's own. rules.d/corpora/ ships empty --
        # a map describes one organisation's tree -- so the fixture provides
        # the file the lookup is supposed to find.

        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        Path(temp.name, "probe.md").write_text(
            "The probe tree: sources in src/, tests in tests/.\n",
            encoding="utf-8")

        self.rag_chat.SHIPPED_CORPUS_RULES = temp.name
        self.rag_chat.PROJECT = "workspace:probe"
        self.rag_chat.PROJECT_ROOT = self.rag_chat.CORPUS_ROOT = "/nonexistent"
        rules = self.rag_chat.load_corpus_rules()
        self.assertIn("The probe tree", rules)
