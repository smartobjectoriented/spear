"""The reds profile: an image for the institute, cut to the corpora it bakes.

Between the public image, which carries no corpus, and the private one, which
carries everything the building host has, sits the image handed to the people
of the institute: the trees they share, those trees' index, and the rules and
skills written for them -- nothing of the other corpora on the host, whose
rules name their build, their platforms and their customer.

What travels is decided from what the content declares, matched against what
the image bakes; so the rule is tested on synthetic content and a synthetic
registry, the way the standards profile is.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DOCKER = REPO / "scripts" / "docker"
BUILD_SH = (DOCKER / "build.sh").read_text()


def run(script, *args, env=None):
    return subprocess.run([sys.executable, str(DOCKER / script), *args],
                          capture_output=True, text=True, env=env)


class ContentIsCutToTheBakedCorpora(unittest.TestCase):
    """stage-content.py keeps a rule or skill only where the image could use it."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.rules = root / "rules.d"
        self.skills = root / "skills"
        self.benches = root / "benches"
        (self.rules / "corpora").mkdir(parents=True)
        self.skills.mkdir()
        self.benches.mkdir()

        def rule(name, header, body="the rule\n"):
            (self.rules / name).write_text(f"---\n{header}\n---\n{body}")

        rule("10-product.md", "scope: corpus verdin, virt64")
        rule("15-family.md", "scope: family buildsys")
        rule("20-global.md", "scope: global\ntasks: implementation, mixed")
        rule("25-shared.md", "scope: corpus so3, verdin")
        rule("40-host.md", "scope: path /home/someone/tree")
        (self.rules / "90-unscoped.md").write_text("no header at all\n")
        (self.rules / "README").write_text("about rules\n")
        (self.rules / "corpora" / "so3.md").write_text("so3 map\n")
        (self.rules / "corpora" / "verdin.md").write_text("verdin map\n")

        def skill(name, scope):
            (self.skills / f"{name}.md").write_text(
                f"---\nname: {name}\nscope: {scope}\n---\n## Procedure\n1. do\n")

        skill("generic", "[any]")
        skill("for-family", "[family:buildsys]")
        skill("for-product", "[verdin, virt64]")
        skill("for-so3", "[so3]")
        skill("customer", "[vita492]")
        (self.skills / "undeclared.md").write_text("## Procedure\n1. do\n")
        (self.skills / "README.md").write_text("# skills\n")

        (self.benches / "so3-ls.sh").write_text("#!/bin/sh\nexit 0\n")
        (self.benches / "product.sh").write_text("#!/bin/sh\nexit 0\n")
        (self.benches / "README.md").write_text("# benches\n")

        self.registry = root / "registry.json"
        self.registry.write_text(json.dumps({
            "so3": {"path": "soo/so3", "bench": "so3-ls.sh"},
            "buildsys": {"path": "soo/buildsys", "families": ["buildsys"]},
            "verdin": {"path": "product/verdin", "families": ["buildsys"],
                       "bench": "product.sh"},
        }))
        self.out = root / "out"

    def tearDown(self):
        self.temp.cleanup()

    def stage(self, bake):
        result = run("stage-content.py", "--bake", bake, "--registry", str(self.registry),
                     "--rules", str(self.rules), "--skills", str(self.skills),
                     "--benches", str(self.benches), str(self.out))
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def staged(self, sub):
        return sorted(p.name for p in (self.out / sub).iterdir() if p.is_file())

    def test_rules_follow_their_scope(self):
        self.stage("so3,buildsys")

        self.assertEqual(self.staged("rules.d"),
                         ["15-family.md", "20-global.md", "25-shared.md", "README"])

    def test_a_family_rule_needs_a_corpus_that_declares_the_family(self):
        self.stage("so3")

        self.assertNotIn("15-family.md", self.staged("rules.d"))

    def test_the_map_travels_with_its_corpus_only(self):
        self.stage("so3,buildsys")

        self.assertEqual(self.staged("rules.d/corpora"), ["so3.md"])

    def test_skills_follow_their_scope(self):
        self.stage("so3,buildsys")

        self.assertEqual(self.staged("skills"),
                         ["README.md", "for-family.md", "for-so3.md", "generic.md"])

    def test_a_bench_travels_with_the_corpus_that_names_it(self):
        self.stage("so3,buildsys")

        self.assertEqual(self.staged("benches"), ["README.md", "so3-ls.sh"])

    def test_the_build_says_what_it_left_out_and_why(self):
        out = self.stage("so3,buildsys")

        self.assertIn("10-product.md  LEFT OUT — corpus verdin, virt64", out)
        self.assertIn("40-host.md  LEFT OUT — scoped to a host path", out)
        self.assertIn("90-unscoped.md  LEFT OUT — declares no scope", out)
        self.assertIn("customer.md  LEFT OUT — vita492", out)
        self.assertIn("undeclared.md  LEFT OUT — declares no scope", out)
        self.assertIn("corpora/verdin.md  LEFT OUT", out)
        self.assertIn("product.sh  LEFT OUT", out)


class ATreeIsCopiedAsGitHasIt(unittest.TestCase):
    """stage-corpora.py: tracked files for a checkout, a pruned whole for a
    fetched component, the registry's exclusions for both."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        (self.repo / "src").mkdir(parents=True)
        (self.repo / "src" / "main.c").write_text("int main(void) { return 0; }\n")
        (self.repo / "src" / "main.o").write_bytes(b"\0obj")
        (self.repo / "vendored").mkdir()
        (self.repo / "vendored" / "lib.c").write_text("/* vendored */\n")
        (self.repo / "generated").mkdir()
        (self.repo / "generated" / "x.c").write_text("/* generated */\n")

        git = ["git", "-C", str(self.repo)]
        subprocess.run(git + ["init", "-q"], check=True)
        subprocess.run(git + ["add", "src/main.c", "vendored/lib.c"], check=True)
        subprocess.run(git + ["-c", "user.email=t@t", "-c", "user.name=t",
                              "commit", "-q", "-m", "init"], check=True)

        # The fetched component sits inside the checkout, untracked, with its
        # build output beside its sources.
        fetched = self.repo / "u-boot"
        (fetched / "arch").mkdir(parents=True)
        (fetched / "arch" / "start.S").write_text("nop\n")
        (fetched / "build").mkdir()
        (fetched / "build" / "u-boot.bin").write_bytes(b"\0bin")
        (fetched / "u-boot.elf").write_bytes(b"\0elf")
        (fetched / "generated").mkdir()
        (fetched / "generated" / "y.h").write_text("#define Y\n")
        (fetched / "lib.back").mkdir()
        (fetched / "lib.back" / "old.c").write_text("/* snapshot */\n")

        self.host = {
            "tree": {"path": str(self.repo), "kind": "generic", "exclude": ["generated"]},
            "u-boot": {"path": str(fetched), "kind": "generic", "exclude": ["./generated"]},
        }
        self.image = {
            "tree": {"path": "work/repo", "kind": "generic", "exclude": ["generated"],
                     "collection": "adhoc_11111111"},
            "u-boot": {"path": "work/repo/u-boot", "kind": "generic",
                       "exclude": ["./generated"], "collection": "adhoc_22222222"},
        }
        self.out = self.root / "baked"

    def tearDown(self):
        self.temp.cleanup()

    def stage(self, bake):
        """Run the stager against a repository layout of our own making."""
        fake = self.root / "fake-repo"
        (fake / "client").mkdir(parents=True)
        (fake / "docker").mkdir()
        (fake / "client" / "projects.json").write_text(json.dumps(self.host))
        (fake / "docker" / "projects.docker.json").write_text(json.dumps(self.image))
        script = (DOCKER / "stage-corpora.py").read_text().replace(
            "REPO = pathlib.Path(__file__).resolve().parents[2]",
            f"REPO = pathlib.Path({str(fake)!r})")
        copy = fake / "stage-corpora.py"
        copy.write_text(script)
        result = subprocess.run([sys.executable, str(copy), "--bake", bake, str(self.out)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def files(self):
        return sorted(str(p.relative_to(self.out)) for p in self.out.rglob("*")
                      if p.is_file())

    def test_a_checkout_travels_as_its_tracked_files(self):
        out = self.stage("tree")

        self.assertEqual(self.files(), ["work/repo/src/main.c", "work/repo/vendored/lib.c"])
        self.assertIn("tracked files", out)

    def test_a_fetched_component_travels_without_its_build_output(self):
        """Nor its `.back` snapshots: a previous attached tree is never a source."""
        out = self.stage("u-boot")

        self.assertEqual(self.files(), ["work/repo/u-boot/arch/start.S"])
        self.assertIn("fetched tree", out)

    def test_naming_both_bakes_both_each_by_its_own_rule(self):
        self.stage("u-boot,tree")

        self.assertEqual(self.files(), ["work/repo/src/main.c",
                                        "work/repo/u-boot/arch/start.S",
                                        "work/repo/vendored/lib.c"])


class TheOtherStagersKnowTheProfile(unittest.TestCase):

    def test_standards_take_only_what_says_public(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp) / "standards"

            for sid, origin in (("PUB", "PUBLIC"), ("LIC", "LICENSED_STANDARD")):
                d = store / sid / "R1"
                (d / "corpus").mkdir(parents=True)
                (d / "manifest.json").write_text(json.dumps(
                    {"standard_id": sid, "revision": "R1", "source_origin": origin}))

            out = Path(tmp) / "out"
            result = run("stage-standards.py", "--profile", "reds",
                         "--store", str(store), str(out))

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(sorted(p.name for p in out.iterdir()), ["PUB"])
            self.assertIn("LEFT OUT of a reds image", result.stdout)

    def test_the_registry_generator_withholds_nothing_for_reds(self):
        """The cut to the baked corpora is stage-corpora's, after the bake."""
        source = (DOCKER / "gen-registry.py").read_text()

        self.assertIn('if profile == "public":', source)
        self.assertNotIn('profile == "reds"', source)


class TheBuilderHonoursTheProfile(unittest.TestCase):

    def test_reds_is_a_profile(self):
        self.assertIn("public|reds|private) ;;", BUILD_SH)

    def test_reds_needs_a_bake_list(self):
        self.assertIn('[ "$PROFILE" = reds ] && [ -z "$BAKE" ]', BUILD_SH)

    def test_the_index_and_the_content_are_cut_for_reds(self):
        for stager in ("stage-index.py", "stage-content.py"):
            with self.subTest(stager=stager):
                self.assertIn(stager, BUILD_SH)

    def test_the_registry_exists_before_the_cut_needs_it(self):
        self.assertLess(BUILD_SH.index("gen-registry.py"), BUILD_SH.index("stage-index.py"))

    def test_the_notes_corpus_is_withheld(self):
        self.assertIn("a reds image carries no notes corpus", BUILD_SH)

    def test_only_a_public_image_is_redistributable(self):
        label = re.search(r'if \[ "\$PROFILE" = public \]; then\n'
                          r'\s+LABELS\+=\(--label "ch\.heig-vd\.reds\.spear\.redistributable=true"\)',
                          BUILD_SH)

        self.assertIsNotNone(label)


FAKE_DOCKER = """#!/bin/bash
echo "docker $*" >> "$DOCKER_LOG"
case "$1 $2" in
    "image inspect")
        case "$*" in
            *redistributable*) echo false ;;
            *profile*) echo reds ;;
        esac ;;
esac
exit 0
"""


class ARedsImageGoesToTheInstitutesRegistry(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        bin_dir = Path(self.temp.name) / "bin"
        bin_dir.mkdir()
        (bin_dir / "docker").write_text(FAKE_DOCKER)
        (bin_dir / "docker").chmod(0o755)
        self.log = Path(self.temp.name) / "docker.log"
        self.log.write_text("")
        self.env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}",
                        DOCKER_LOG=str(self.log))
        self.env.pop("SPEAR_REDS_REGISTRY", None)

    def tearDown(self):
        self.temp.cleanup()

    def push(self, *args):
        return subprocess.run([str(DOCKER / "push.sh"), *args], env=self.env,
                              capture_output=True, text=True, timeout=30)

    def test_its_own_registry_needs_no_word(self):
        for image in ("reds-gitlab.heig-vd.ch/reds/spear:1-reds",
                      "reds-gitlab.heig-vd.ch:5050/reds/spear:1-reds"):
            with self.subTest(image=image):
                result = self.push(image)

                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f"docker push {image}", self.log.read_text())

    def test_any_other_registry_is_refused(self):
        result = self.push("ghcr.io/someone/spear:1-reds")

        self.assertEqual(result.returncode, 1)
        self.assertIn("Refusing to push", result.stderr)
        self.assertIn("reds-gitlab.heig-vd.ch", result.stderr)
        self.assertNotIn("docker push", self.log.read_text())

    def test_unless_said_so(self):
        result = self.push("--allow-push", "ghcr.io/someone/spear:1-reds")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("docker push ghcr.io/someone/spear:1-reds", self.log.read_text())

    def test_the_registry_can_be_named(self):
        self.env["SPEAR_REDS_REGISTRY"] = "registry.example.org"
        result = self.push("registry.example.org/reds/spear:1-reds")

        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()


class TheStandaloneLauncherNeedsOnlyDocker(unittest.TestCase):
    """scripts/docker/spear-run.sh: for a machine that has the image and
    nothing else. It is also copied into the image, so a recipient can get it
    from there."""

    SCRIPT = DOCKER / "spear-run.sh"

    def run_it(self, *args, env=None):
        return subprocess.run([str(self.SCRIPT), *args], env=env,
                              capture_output=True, text=True, timeout=30)

    def test_help_names_every_option(self):
        out = self.run_it("--help").stdout

        for option in ("--image", "--pull", "--reds", "--key", "--endpoint",
                       "--corpus", "--mount", "--state", "--list"):
            with self.subTest(option=option):
                self.assertIn(option, out)

    def test_it_refuses_to_run_without_a_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = Path(tmp) / "bin"
            bin_dir.mkdir()
            (bin_dir / "docker").write_text("#!/bin/bash\nexit 0\n")
            (bin_dir / "docker").chmod(0o755)
            env = {k: v for k, v in os.environ.items()
                   if not k.startswith("SPEAR_")}
            env["PATH"] = f"{bin_dir}:{env['PATH']}"
            result = self.run_it("--corpus", "x", env=env)

        self.assertEqual(result.returncode, 1)
        self.assertIn("--reds USER@HOST, or --endpoint URL", result.stderr)

    def test_the_image_carries_it(self):
        dockerfile = (REPO / "docker" / "Dockerfile").read_text()

        self.assertIn("scripts/docker/spear-run.sh /opt/spear/spear-run.sh", dockerfile)
