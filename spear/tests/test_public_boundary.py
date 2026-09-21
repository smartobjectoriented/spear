"""Nothing private, and nothing of the retired platform, in the tracked tree.

This repository becomes a public one. Two classes of string must not be in it,
and the difference between them matters.

PRIVATE DEPLOYMENT VALUES are a disclosure: an institute address, an account
name, a key filename, somebody's home directory, a customer's name. They were
in here -- a complete deployment dossier sat in a tracked example file, host,
account, two real GPU identifiers and six paths -- and the first push would
have published all of it. There is no justified occurrence of any of them, so
this test allows none.

RETIRED PLATFORM BRANDING is not a disclosure, only a lie about what the
product is called. The bar is the same but the reason is weaker, so the test
records the two places that spell an old name deliberately: the checks that
assert those names are gone.

The needles are assembled from fragments. Spelled out, they would appear in
this file, and the scan -- which reads every tracked file, this one included
-- would find its own assertion list and fail on its own evidence.

DELIBERATELY NOT CHECKED HERE, and pending rather than accepted:

  edgem1      a corpus KIND, bound to live state: two registry entries and two
              Chroma collections holding 8 609 rows. Renaming the literal
              without migrating both would silently empty those indexes, which
              is a state migration and not a branding edit.
  edgemtech   the container registry and the fine-tuning corpus builders still
              name one organisation's product trees. That is customer-specific
              functionality to externalize, not text to rewrite: the client
              reads two files from that tree.

Adding either to the lists below without doing that work would turn this test
into a thing people disable.
"""

from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT.parent

#: Binary formats a substring scan cannot read usefully.
SKIP_SUFFIX = (".svg", ".png", ".drawio", ".jpg", ".jpeg", ".ico", ".pdf",
               ".woff", ".woff2", ".gz", ".zip")


def tracked():
    out = subprocess.run(["git", "ls-files"], cwd=REPO,
                         capture_output=True, text=True).stdout.split()

    return [REPO / name for name in out if not name.endswith(SKIP_SUFFIX)]


def scan(needles):
    """{file: [needle, ...]} for every tracked file containing one."""
    found = {}

    for path in tracked():
        try:
            body = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        for needle in needles:
            if needle in body:
                found.setdefault(str(path.relative_to(REPO)), []).append(needle)

    return found


class NoPrivateDeploymentValue(unittest.TestCase):
    """No exceptions. Each of these names a real machine, account or client."""

    NEEDLES = (
        "/home/" + "rossierd",          # one person's home directory
        "10.190." + "161.59",           # the institute's GPU host
        "/home/re" + "ds-ml",           # its account
        "re" + "ds-ml@",                # user@host, as typed into ssh
        "gitlab." + "edgemtech.ch",     # a private forge
        "id_pod" + "_gpu",              # a private key filename
        "arma" + "suisse",              # a customer
        "GPU-ac" + "51ffcd",            # a real card's UUID
    )

    def test_none_of_them_is_tracked(self):
        self.assertEqual(scan(self.NEEDLES), {})

    def test_the_scan_reads_the_whole_tree(self):
        """A scan that reads nothing passes for the wrong reason.

        The bound is deliberately loose. A public export drops the two corpora
        and the run logs -- 3592 of 3967 files -- so a threshold tuned to the
        private tree fails in the exported one, which is exactly where this
        check matters most. What it has to rule out is an empty list, not a
        small tree.
        """
        files = tracked()

        self.assertGreater(len(files), 100, len(files))
        self.assertIn("README.md", {p.name for p in files})

    def test_the_scan_would_notice(self):
        """Prove the detector detects, on a string built the same way."""
        probe = "ssh " + self.NEEDLES[3] + "host"

        self.assertTrue(any(n in probe for n in self.NEEDLES))


class NoRetiredPlatformName(unittest.TestCase):
    """The product is SPEAR. The old name is not an alias, it is nothing.

    Including the separators. The hyphenated spelling was searched for,
    replaced, and declared gone -- while the same word with a MIDDLE DOT sat
    in the first line of two READMEs through three passes, because every scan
    looked for a hyphen. The needles below cover the separators; they are
    assembled from fragments so this file does not contain what it forbids.
    """

    NEEDLES = (
        "EDGEM" + "-AI", "EDGEM" + "\u00b7AI", "EDGEM" + "_AI", "EDGEM" + " AI",
        "edgem" + "-ai", "edgem" + "1-rag",
        "edgem" + "-chat", "edgem" + "-corpus", "edgem" + "-index",
        "edgem" + "-reindex", "edgem" + "-server", "edgem" + "-model",
        "edgem" + "-docker", "edgem" + "-llm", "edgem" + "-stop",
        "gen_" + "edgem", "edgem" + ".drawio",
    )

    #: The only files allowed to spell an old name, and why. A file, not a
    #: token: a token allowlist says "this word is fine anywhere", which is
    #: how a word comes back somewhere it is not fine.
    JUSTIFIED = {
        "spear/tests/test_client_server_boundary.py":
            "asserts the retired launcher and the retired remote path are gone; "
            "it has to name them to check for their absence",
    }

    def test_only_the_absence_checks_spell_them(self):
        unexplained = {f: n for f, n in scan(self.NEEDLES).items()
                       if f not in self.JUSTIFIED}

        self.assertEqual(unexplained, {})

    def test_every_justification_still_has_a_file(self):
        """A reason left behind after its file went is dead weight."""
        found = set(scan(self.NEEDLES))

        self.assertEqual(sorted(set(self.JUSTIFIED) - found), [])


class TheOrganisationIsNamedOnlyWhereItMustBe(unittest.TestCase):
    """EDGEMTech, its product line and its build system.

    Unlike the platform name, these are not always wrong. A measurement made
    against a real tree names that tree, and a persistent collection
    identifier cannot be renamed without abandoning the index it names. What
    must not happen is a NEW occurrence appearing unnoticed.

    So: a file-and-reason map, checked both ways. A file not in it may not
    mention them at all; an entry whose file stopped mentioning them is
    removed, so the map cannot rot into a blanket permission.
    """

    NEEDLES = ("EDGEM" + "Tech", "edgem" + "tech", "Infra" + "base",
               "infra" + "base", "edgem" + "1")

    #: Substrings that are the persistent Chroma collection identifiers. They
    #: name indexes that exist; renaming the literal abandons 8 609 rows.
    PERSISTENT = ("edgem1_verdin", "edgem1_virt64")

    JUSTIFIED = {
        "spear/index_corpus.py":
            "derives the collection name that produced the two persistent "
            "indexes; the literal IS their identity",
        "spear/rag_chat.py":
            "one anecdote naming the trees a copyright-header rule was "
            "observed against",
        "spear/tool_runtime.py":
            "a measured incident: the entry points a path check refused",
        "spear/index_dir.py":
            "measured incidents -- the snapshot suffixes a real build system "
            "leaves, and an index that came out 99.8% vendored",
        "spear/benchmarks/runner.py":
            "names the repositories the benchmark was actually run against",
        "spear/tests/test_benchmarks.py":
            "the same repository set, as a fixture",
        "spear/tests/test_indexing.py":
            "quotes the measured incident it was written for",
        "spear/tests/test_tool_runtime.py":
            "quotes the measured incident it was written for",
        "spear/tests/test_corpus_behaviour_is_declared.py":
            "names the retired corpus kind to assert no behaviour hides behind "
            "it any more",
        "spear/tests/test_normative_binding_reaches_every_corpus.py":
            "names the retired corpus kind in the same absence assertion",
        "spear/tests/test_public_boundary.py":
            "this file: the needles and the reasons",
        "doc/source/retrieval.rst":
            "reports measurements made against a real build system, including "
            "the 37-question recall figure",
    }

    def scan_ignoring_persistent(self):
        """Occurrences that are not one of the persistent identifiers."""
        found = {}

        for path in tracked():
            try:
                body = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue

            hits = [line for line in body.splitlines()
                    if any(n in line for n in self.NEEDLES)
                    and not any(p in line for p in self.PERSISTENT)]

            if hits:
                found[str(path.relative_to(REPO))] = len(hits)

        return found

    def test_every_file_that_mentions_them_has_a_reason(self):
        found = self.scan_ignoring_persistent()
        unexplained = sorted(set(found) - set(self.JUSTIFIED))

        self.assertEqual(unexplained, [], "\n".join(
            f"{f}: {found[f]} occurrence(s) and no entry in JUSTIFIED"
            for f in unexplained))

    def test_no_reason_outlives_its_file(self):
        found = set(self.scan_ignoring_persistent())
        stale = sorted(set(self.JUSTIFIED) - found)

        self.assertEqual(stale, [], "these files no longer mention it")

    def test_the_persistent_identifiers_are_still_reachable(self):
        """They are exempt because they name live indexes. If the registry
        stopped pinning them, the exemption would be protecting nothing."""
        import json

        registry = ROOT / "projects.json"

        if not registry.exists():
            self.skipTest("no local registry on this machine")

        pinned = {spec.get("collection")
                  for spec in json.loads(registry.read_text()).values()}

        for name in self.PERSISTENT:
            with self.subTest(collection=name):
                self.assertIn(name, pinned)


class NoStaleProductPrefixEscapes(unittest.TestCase):
    """Any `edgem` token at all, not a list of the ones we thought of.

    Four passes hunted specific spellings and four passes missed something: a
    middle-dot variant in two README first lines, a model directory nobody had
    listed, and two retired commands still documented as the way to do things.
    A needle list can only find what someone remembered to put in it.

    So this one matches the PREFIX and requires every survivor to be named,
    per file, with what it is. Persistent identifiers -- a Chroma collection,
    a file users have written in their own trees, a systemd unit prefix that
    live scopes carry -- are not renameable without abandoning or breaking
    what they name, and they are listed as such rather than exempted by
    pattern.
    """

    #: Matched case-insensitively, after the exemptions below are removed from
    #: the line. `edgement` and friends are ordinary English and would match a
    #: bare prefix, so the boundary is a word-ish one.
    PATTERN = re.compile(r"edgem(?![a-z])|edgem[-_.]", re.IGNORECASE)

    #: Substrings whose presence on a line exempts that line. Each names
    #: something that exists and cannot be renamed by editing a string.
    PERSISTENT = (
        "edgem1_verdin",        # Chroma collection, 4260 rows
        "edgem1_virt64",        # Chroma collection, 4349 rows
        "edgem_skills",         # Chroma collection, the skill library
        "edgem_archive",        # Chroma collection, the conversation archive
        ".edgem-rules.md",      # written by hand in users' own trees
    )

    JUSTIFIED = {
        # --- persistent identifiers, named where they are defined ---
        "spear/index_corpus.py":
            "derives the collection name that produced the two build-system "
            "indexes; the literal IS their identity",
        ".gitignore":
            "guards a 47 GB directory this deployment still has on disk; "
            "renaming the rule without moving the directory exposes it",

        # --- historical evidence: transcripts and measured incidents ---
        "spear/rag_chat.py":
            "one anecdote naming the trees a copyright rule was observed against",
        "spear/TRAINING_DATA.md":
            "narrates a run against a path that was actually tried",
        "spear/doc/source/training_host_setup.rst":
            "the same narrative: the path Axolotl was not usable at",
        "doc/source/model_serving.rst":
            "a recorded backend probe and its literal reply",
        "doc/source/operations.rst":
            "a systemctl glob that must match the unit prefix the code emits",
        "pod-artifacts-qwen3coder/gguf-download.log":
            "a download log from one run; evidence, not source",

        # --- checks that must name what they forbid ---
        "spear/tests/test_public_boundary.py":
            "this file: the pattern, the exemptions and the reasons",
        "server/tests/test_server_public_surface.py":
            "the server tree's own scan, needles assembled from fragments",
        "spear/tests/test_remote_embedding_policy.py":
            "asserts no retired remote path is named in the runtime",
        "spear/tests/test_environment_namespace.py":
            "asserts the retired environment namespace is gone",
        "spear/tests/test_client_server_boundary.py":
            "asserts the retired launcher and remote path are gone",
        "spear/tests/test_corpus_behaviour_is_declared.py":
            "names the retired corpus kind to assert no behaviour hides behind it",
        "spear/tests/test_normative_binding_reaches_every_corpus.py":
            "names the retired corpus kind in an absence assertion",
        "spear/tests/test_normative_contract.py":
            "asserts the contract names no particular project",
        "spear/tests/test_indexing.py":
            "quotes the measured incident it was written for",
    }

    def offenders(self):
        found = {}

        for path in tracked():
            try:
                body = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue

            for line in body.splitlines():
                stripped = line
                for exempt in self.PERSISTENT:
                    stripped = stripped.replace(exempt, "")

                if self.PATTERN.search(stripped):
                    found.setdefault(str(path.relative_to(REPO)), []).append(
                        line.strip()[:90])

        return found

    def test_every_file_with_one_is_named_and_explained(self):
        found = self.offenders()
        unexplained = sorted(set(found) - set(self.JUSTIFIED))

        self.assertEqual(unexplained, [], "\n".join(
            f"{f}\n    " + "\n    ".join(found[f][:3]) for f in unexplained))

    def test_no_reason_outlives_its_file(self):
        """A reason whose file stopped matching is dead weight.

        A reason whose file is not in THIS tree is a different thing: a public
        export leaves some of them behind, and the entry still applies to the
        repository it came from. Only files that are present are judged.
        """
        present = {str(p.relative_to(REPO)) for p in tracked()}
        stale = sorted((set(self.JUSTIFIED) & present) - set(self.offenders()))

        self.assertEqual(stale, [], "these files no longer contain one")

    def test_the_pattern_catches_what_the_needle_lists_missed(self):
        """The four spellings that escaped four hand-written scans."""
        for missed in ("EDGEM\u00b7AI", "edgem-gguf/models",
                       "run edgem-fetch-coder32 to stage it",
                       "~/.local/bin/edgem-*"):
            with self.subTest(spelling=missed):
                self.assertTrue(self.PATTERN.search(missed), missed)

    def test_ordinary_words_are_not_swept_up(self):
        for innocent in ("acknowledgements", "edgements", "the edge of the map"):
            with self.subTest(word=innocent):
                self.assertIsNone(self.PATTERN.search(innocent), innocent)


class TheProductIdentifiesItself(unittest.TestCase):
    """What the tracked tree says the product is."""

    def test_the_banner_says_spear(self):
        """Checked on the RENDERED banner, not on the source.

        The expansion is written across two source lines, so scanning the file
        for it finds nothing -- which is how this test first passed for the
        wrong reason and then failed for the right one.
        """
        import io, sys
        sys.path.insert(0, str(ROOT))
        import rag_chat

        out = io.StringIO()
        stdout, sys.stdout = sys.stdout, out
        try:
            rag_chat.banner_art()
        finally:
            sys.stdout = stdout

        rendered = out.getvalue()
        self.assertIn("Specification-driven Platform for Embedded "
                      "Agentic Reasoning", rendered)
        self.assertIn("HEIG-VD/REDS", rendered)
        self.assertNotIn("EDGEM", rendered.upper().replace("SPEAR", ""))

    def test_the_help_names_the_current_commands(self):
        """The banner is not the only place a name is printed."""
        source = (ROOT / "rag_chat.py").read_text(encoding="utf-8")

        self.assertIn("spear-chat [options]", source)
        self.assertNotIn("edgem" + "-chat", source)


if __name__ == "__main__":
    unittest.main()
