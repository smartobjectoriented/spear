"""The generic set and the tested profile, held apart and held in step.

Two files, two jobs. `requirements.txt` is unpinned because SPEAR does not
know what card a deployment has, and a pin chosen centrally would make the
project quietly CUDA-13-specific for everyone. `constraints-reds-tested.txt`
records the one stack the equivalence gate was actually run against.

The failure these tests exist to prevent is drift: a version bumped in one
file and not the other, or a README table that stops describing the pins
beside it. That drift is invisible until someone reproduces the "tested" stack
and gets something else -- at which point the word "tested" was a lie for an
unknown length of time.

The other thing asserted here is honesty about the index. `torch==2.12.0+cu130`
is not on PyPI. A constraints file that pinned it without saying so would fail
for every reader who tried it, and would be claiming a provenance it does not
have.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

EMBED = Path(__file__).resolve().parent.parent / "embed"
GENERIC = EMBED / "requirements.txt"
TESTED = EMBED / "constraints-reds-tested.txt"
README = EMBED / "README.md"

PIN = re.compile(r"^([A-Za-z0-9._-]+)==([^\s#]+)", re.MULTILINE)


def prose(path):
    """The file's text as one flowed line, comment markers and wrapping gone.

    A sentence that happens to wrap across two comment lines is the same
    sentence. Asserting on the raw bytes would make these tests fail on a
    reflow and, worse, would push someone to reformat the file to please the
    test rather than to read well.
    """
    body = path.read_text(encoding="utf-8")

    return " ".join(body.replace("#", " ").split()).lower()

#: The PyTorch wheel index. Named here so a change to either file has to
#: change this test too, rather than slipping through.
TORCH_INDEX = "https://download.pytorch.org/whl/cu130"


def requirements(path):
    """Package names a pip file asks for, ignoring comments."""
    names = []

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()

        if line:
            names.append(re.split(r"[=<>!~\[]", line)[0].strip().lower())

    return names


def pins(path):
    body = "\n".join(line for line in path.read_text(encoding="utf-8").splitlines()
                     if not line.lstrip().startswith("#"))

    return {name.lower(): version for name, version in PIN.findall(body)}


class TheGenericSetStaysGeneric(unittest.TestCase):
    """A pin here would make every deployment inherit one host's CUDA."""

    def test_nothing_in_it_is_pinned(self):
        self.assertEqual(pins(GENERIC), {},
                         "requirements.txt must stay unpinned; the tested "
                         "versions belong in constraints-reds-tested.txt")

    def test_it_still_asks_for_the_four_things_the_worker_needs(self):
        self.assertEqual(set(requirements(GENERIC)),
                         {"sentence-transformers", "torch", "transformers",
                          "numpy"})

    def test_it_does_not_pull_the_client_dependency_set(self):
        """The machine that only encodes should not carry a vector store."""
        for unwanted in ("chromadb", "openai", "anthropic"):
            self.assertNotIn(unwanted, requirements(GENERIC))


class TheTestedProfileIsComplete(unittest.TestCase):
    def test_every_generic_dependency_is_pinned_in_it(self):
        missing = set(requirements(GENERIC)) - set(pins(TESTED))

        self.assertEqual(missing, set(),
                         f"the tested profile does not pin {missing}")

    def test_it_pins_the_versions_the_gate_was_run_against(self):
        expected = {"torch": "2.12.0+cu130",
                    "sentence-transformers": "5.5.1",
                    "transformers": "5.9.0",
                    "numpy": "2.4.6"}

        for name, version in expected.items():
            with self.subTest(package=name):
                self.assertEqual(pins(TESTED).get(name), version)

    def test_the_python_version_is_recorded(self):
        self.assertIn("3.12.3", TESTED.read_text(encoding="utf-8"))


class TheIndexIsNotMisrepresented(unittest.TestCase):
    """A local version segment means the wheel is not from PyPI."""

    def test_any_local_version_pin_names_the_index_it_comes_from(self):
        local = {name: v for name, v in pins(TESTED).items() if "+" in v}

        self.assertTrue(local, "expected at least one non-PyPI pin to check")

        body = TESTED.read_text(encoding="utf-8")

        for name in local:
            with self.subTest(package=name):
                self.assertIn(TORCH_INDEX, body,
                              f"{name} is pinned to a local version but the "
                              f"file never says which index provides it")

    def test_the_file_says_plainly_that_it_is_not_on_pypi(self):
        self.assertIn("does not exist on the normal pypi index", prose(TESTED))

    def test_the_install_is_documented_as_two_steps(self):
        body = TESTED.read_text(encoding="utf-8")

        self.assertIn("--index-url " + TORCH_INDEX, body)
        self.assertIn("-c constraints-reds-tested.txt", body)


class TheReadmeCannotDriftFromThePins(unittest.TestCase):
    """The table in the README is the same claim, written twice."""

    def setUp(self):
        self.readme = README.read_text(encoding="utf-8")

    def test_every_headline_version_appears_in_the_readme(self):
        for name in ("torch", "sentence-transformers", "transformers", "numpy"):
            with self.subTest(package=name):
                self.assertIn(pins(TESTED)[name], self.readme,
                              f"the README does not state the pinned {name} "
                              f"version")

    def test_the_readme_names_the_constraints_file(self):
        self.assertIn("constraints-reds-tested.txt", self.readme)

    def test_the_readme_repeats_the_index_caveat(self):
        self.assertIn(TORCH_INDEX, self.readme)
        self.assertIn("not on PyPI", self.readme)

    def test_the_equivalence_result_is_recorded_in_both(self):
        """"Tested" is only meaningful if it says what the test concluded."""
        for name, path in (("constraints", TESTED), ("readme", README)):
            with self.subTest(file=name):
                self.assertIn("bit-identical", prose(path))


class TheProfileIsNamedHonestly(unittest.TestCase):
    """What these files must NOT contain is checked once, not twice.

    `test_server_public_surface` already scans every file under server/ for
    host names, addresses, accounts and card UUIDs, and it covers these two --
    verified by planting a real address in the constraints file and watching
    it caught. Repeating that scan here would mean repeating its needle list,
    and a second copy of a list of forbidden strings is itself a file
    containing forbidden strings. So this class asserts only what the other
    scan cannot: that the file's NAME does not overstate what it is.
    """

    def test_the_name_says_tested_not_required(self):
        """`requirements-reds.txt` would read as a mandate. It is a record."""
        self.assertIn("tested", TESTED.name)
        self.assertTrue(TESTED.name.startswith("constraints-"))

    def test_the_file_says_it_is_not_the_required_stack(self):
        body = prose(TESTED)

        self.assertIn("this is not the required stack", body)
        self.assertIn("not a lockfile for the project", body)


if __name__ == "__main__":
    unittest.main()
