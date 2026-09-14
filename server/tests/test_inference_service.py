"""The unit supervises the server. It must not also configure it.

serve.sh exists because three implementations of the same command line
disagreed and nothing reconciled them -- the running server held a 98304-token
context while the restart script would have brought it back at 65536. A
systemd unit is the fourth place that could hold such a number, and the
easiest one to put it in, because `ExecStart=` accepts flags happily.

So these tests say what the unit may NOT contain. They are static: a unit file
is configuration, and what matters about it is what it says.

The other thing checked here is honesty about persistence. `systemctl --user
enable` on a host without lingering produces a service that looks enabled and
does not come back after a reboot. Documentation that omitted that would be
worse than none.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

INFERENCE = Path(__file__).resolve().parent.parent / "inference"
UNIT = INFERENCE / "spear-inference.service"
README = INFERENCE / "README.md"


def directives(section=None):
    """{key: [values]} from the unit, optionally restricted to one section."""
    found, current = {}, None

    for line in UNIT.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            continue

        if not line or line.startswith("#") or "=" not in line:
            continue

        if section is None or current == section:
            key, _, value = line.partition("=")
            found.setdefault(key.strip(), []).append(value.strip())

    return found


def prose(path):
    return " ".join(path.read_text(encoding="utf-8").replace("#", " ").split()).lower()


class TheUnitDecidesNothingAboutServing(unittest.TestCase):
    """Every serving value belongs to configuration serve.sh reads."""

    #: The flags that decide what is served and how. Any of them in ExecStart
    #: would be a second source of truth for a number that already has one.
    SERVING_FLAGS = ("--model", "--ctx-size", "--parallel", "--n-gpu-layers",
                     "--threads", "--port", "--host", "--lora", "--n-cpu-moe",
                     "--cache-type-k", "--cache-type-v", "--flash-attn")

    def test_execstart_only_invokes_the_launcher(self):
        exec_start = directives("Service")["ExecStart"]

        self.assertEqual(len(exec_start), 1)
        self.assertTrue(exec_start[0].endswith("server/inference/serve.sh"),
                        exec_start[0])

    def test_execstart_passes_no_serving_flag(self):
        body = " ".join(directives("Service")["ExecStart"])

        for flag in self.SERVING_FLAGS:
            with self.subTest(flag=flag):
                self.assertNotIn(flag, body)

    def test_no_serving_flag_appears_anywhere_in_the_unit(self):
        """Not in Environment= either, which is the other way in."""
        body = "\n".join(line for line in UNIT.read_text().splitlines()
                         if not line.lstrip().startswith("#"))

        for flag in self.SERVING_FLAGS:
            with self.subTest(flag=flag):
                self.assertNotIn(flag, body)

    def test_it_points_at_the_configuration_root_and_nothing_deeper(self):
        env = directives("Service").get("Environment", [])

        self.assertIn("SPEAR_SERVER_ROOT=%h/spear-runtime", env)

    def test_the_environment_file_is_optional(self):
        """A missing optional file must not fail the unit."""
        for value in directives("Service").get("EnvironmentFile", []):
            with self.subTest(value=value):
                self.assertTrue(value.startswith("-"), value)


class TheUnitNamesNoMachine(unittest.TestCase):
    """It is a template. A deployment's values are the deployment's."""

    def test_no_absolute_home_directory(self):
        """%h, so the same file works for any account."""
        body = "\n".join(line for line in UNIT.read_text().splitlines()
                         if not line.lstrip().startswith("#"))

        self.assertNotIn("/home/", body)
        self.assertIn("%h", body)

    def test_no_user_or_group_is_hardcoded(self):
        service = directives("Service")

        self.assertNotIn("User", service)
        self.assertNotIn("Group", service)

    def test_no_card_uuid(self):
        self.assertIsNone(re.search(r"GPU-[0-9a-f]{8}-", UNIT.read_text()))

    def test_no_address(self):
        body = "\n".join(line for line in UNIT.read_text().splitlines()
                         if not line.lstrip().startswith("#"))

        self.assertIsNone(re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", body))


class TheRestartPolicyIsDeliberate(unittest.TestCase):
    def test_it_restarts_on_failure_and_not_always(self):
        """`always` would undo an operator's deliberate stop."""
        self.assertEqual(directives("Service")["Restart"], ["on-failure"])

    def test_a_crash_loop_is_rate_limited(self):
        unit = directives("Unit")

        self.assertIn("StartLimitBurst", unit)
        self.assertIn("StartLimitIntervalSec", unit)

    def test_the_rate_limit_is_in_the_section_systemd_reads(self):
        """Both moved to [Unit]; in [Service] they are silently ignored --
        a rate limit that does not exist rather than one that does."""
        self.assertNotIn("StartLimitBurst", directives("Service"))
        self.assertNotIn("StartLimitIntervalSec", directives("Service"))

    def test_the_start_timeout_outlasts_a_large_model_load(self):
        """A timeout shorter than the load turns a working server into a
        restart loop. A quantised 80B takes minutes."""
        value = directives("Service")["TimeoutStartSec"][0]

        self.assertGreaterEqual(int(value), 600)


class ThePersistenceCaveatIsStated(unittest.TestCase):
    """`enable` without lingering is a service that does not come back."""

    def test_both_files_explain_lingering(self):
        for name, path in (("unit", UNIT), ("readme", README)):
            with self.subTest(file=name):
                self.assertIn("linger", prose(path))

    def test_the_readme_gives_the_command_that_reveals_it(self):
        self.assertIn('loginctl show-user "$user" -p linger', prose(README))

    def test_the_readme_offers_the_system_unit_alternative(self):
        body = prose(README)

        self.assertIn("system unit", body)
        self.assertIn("/etc/systemd/system", body)

    def test_readiness_is_not_claimed(self):
        """active means started, not serving -- said, not assumed.

        Read from the parsed directive, not the raw text: the unit EXPLAINS
        why Type=notify is unavailable, and a scan of the bytes would find
        that explanation and call it a claim.
        """
        self.assertEqual(directives("Service")["Type"], ["exec"])
        self.assertIn("readiness is not startedness", prose(README))
        self.assertIn("`active` means started", prose(README))
        self.assertIn("/health", README.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
