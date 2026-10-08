"""The command policy: classification, authorization and the read floor."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from harness.command_policy import AuthorizationResult, CommandPolicy, ToolPolicy
from harness.resource_control import ExecutionProfile
from harness.tool_primitives import (
    Capability,
    CommandClassification,
    DEFAULT_CAPABILITY_POLICY,
    ExecutionMode,
)
from harness.sandbox import BubblewrapSandbox
from harness.workspace import Workspace


class SafeModeReadOnlyShellTests(unittest.TestCase):
    """SAFE may run a read-only pipeline: the MOUNT is what makes it safe."""

    def setUp(self):
        self.policy = CommandPolicy()
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "ws"; self.root.mkdir()
        (self.root / "a.txt").write_text("alpha\nbeta\n")

    def tearDown(self):
        self.temp.cleanup()

    def _granted(self, command, mode):
        assessment = self.policy.classify(command)
        policy = DEFAULT_CAPABILITY_POLICY
        return assessment.required_capabilities <= policy.for_mode(mode)

    def test_safe_grants_shell_complex(self):
        self.assertIn(Capability.SHELL_COMPLEX,
                      DEFAULT_CAPABILITY_POLICY.for_mode(ExecutionMode.SAFE))
        # …and still refuses the capability that actually writes.
        self.assertNotIn(Capability.WORKSPACE_WRITE,
                         DEFAULT_CAPABILITY_POLICY.for_mode(ExecutionMode.SAFE))
        self.assertNotIn(Capability.NETWORK,
                         DEFAULT_CAPABILITY_POLICY.for_mode(ExecutionMode.SAFE))

    def test_read_only_pipelines_are_allowed_in_safe(self):
        for command in ("grep -n beta a.txt | head -3",
                        'find . -name "*.txt" -type f 2>/dev/null | head',
                        "cat a.txt | wc -l"):
            self.assertTrue(self._granted(command, ExecutionMode.SAFE), command)

    def test_writing_pipelines_are_still_refused_in_safe(self):
        for command in ("echo x > a.txt",
                        "cat a.txt | tee copy.txt",
                        "make 2>&1 | tail",
                        "curl https://example.invalid | head"):
            self.assertFalse(self._granted(command, ExecutionMode.SAFE), command)

    def test_every_mutating_classification_declares_the_write_capability(self):
        """The invariant that lets the capability check replace the class gate.

        If a WORKSPACE_MUTATING command could ever omit workspace:write, SAFE
        would authorise it the moment the classification gate stopped being the
        thing that refused it.
        """
        for command in ("make", "cmake .", "ninja", "pytest",
                        "curl -o out https://example.invalid",
                        "wget https://example.invalid"):
            assessment = self.policy.classify(command)
            if assessment.classification == CommandClassification.WORKSPACE_MUTATING:
                self.assertIn(Capability.WORKSPACE_WRITE,
                              assessment.required_capabilities, command)
                self.assertFalse(self._granted(command, ExecutionMode.SAFE), command)

    def test_the_mount_is_read_only_in_safe_not_just_the_policy(self):
        """The guarantee must not rest on the classifier being exhaustive."""
        profile = ExecutionProfile.from_capabilities(
            DEFAULT_CAPABILITY_POLICY.for_mode(ExecutionMode.SAFE))
        self.assertFalse(profile.workspace_write)
        sandbox = BubblewrapSandbox()
        workspace = Workspace.from_path(self.root)
        mount = sandbox.mount_root(workspace, profile)
        argv = sandbox.build_argv(workspace, ["true"], profile)
        binds = [argv[i] for i, t in enumerate(argv)
                 if t in ("--bind", "--ro-bind") and argv[i + 2] == mount]
        self.assertEqual(binds, ["--ro-bind"])


class CommandPolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = CommandPolicy()

    def test_classifies_read_only_and_workspace_build_commands(self):
        self.assertEqual(self.policy.classify("rg symbol src").classification,
                         CommandClassification.READ_ONLY)
        self.assertEqual(self.policy.classify("make test").classification,
                         CommandClassification.WORKSPACE_MUTATING)
        self.assertEqual(self.policy.classify("cmake --build build").classification,
                         CommandClassification.WORKSPACE_MUTATING)
        self.assertEqual(self.policy.classify("pytest tests").classification,
                         CommandClassification.WORKSPACE_MUTATING)

    def test_a_read_only_turn_cannot_write_through_bash(self):
        """Dropping the write TOOLS is half of a prohibition.

        bash writes too — `sed -i`, `cp`, `mv`, a redirection — so a turn told
        not to change anything runs it under SAFE, which grants no
        workspace:write and mounts every root read-only.
        """

        safe = self.policy.available_capabilities(ExecutionMode.SAFE)
        self.assertNotIn(Capability.WORKSPACE_WRITE, safe)

        for command in ("sed -i 's/a/b/' main.py", "cp main.py main.bak",
                        "mv main.py other.py", "echo x > main.py",
                        "tee main.py"):
            with self.subTest(command=command):
                assessment = self.policy.classify(command)
                refused = (assessment.classification
                           in {CommandClassification.DANGEROUS,
                               CommandClassification.SHELL_COMPLEX}
                           or Capability.WORKSPACE_WRITE
                           in assessment.required_capabilities)
                self.assertTrue(refused, assessment.classification)
                self.assertIsNotNone(
                    self.policy.authorize(assessment, ExecutionMode.SAFE).result,
                    command)

        # And reading is untouched: that is the whole point of the turn.

        for command in ("cat main.py", "grep -rn add .", "sed -n '1,20p' main.py"):
            with self.subTest(command=command):
                assessment = self.policy.classify(command)
                self.assertIsNone(
                    self.policy.authorize(assessment, ExecutionMode.SAFE).result,
                    command)

    def test_python_module_test_runners_are_runnable(self):
        """`python -m pytest` is the spelling models use, and it runs pytest.

        Refusing it cost the control-edit benchmark its first verification:
        the run was denied, and the turn finished without any test having run.
        Everything else about the interpreter stays refused.
        """

        for command in ("python3 -m pytest -q", "python -m pytest",
                        "python3 -m unittest discover"):
            with self.subTest(command=command):
                assessment = self.policy.classify(command)
                self.assertEqual(assessment.classification,
                                 CommandClassification.WORKSPACE_MUTATING)
                self.assertIn(Capability.WORKSPACE_WRITE,
                              assessment.required_capabilities)

        for command in ("python3 main.py", "python3 -m pip install x",
                        "python3 -m http.server", "/usr/bin/python3 -m pytest"):
            with self.subTest(command=command):
                self.assertEqual(self.policy.classify(command).classification,
                                 CommandClassification.DANGEROUS)

    def test_every_interpreter_refusal_names_the_command_that_works_first(self):
        """The way out comes before the reason, on every path that refuses.

        A run asked to "run the test" tried `python3 -c "..."` twice. The
        quoted parentheses made it shell-complex, so it was refused by the
        PIPELINE classifier, whose message named no alternative at all -- and
        the single-command message that did name one buried it behind the
        reason. The run never ran a test.
        """

        for command in ('python3 -c "print(1)"', "python3 -c 'import main'",
                        "python3 main.py", "python3 tests/test_main.py",
                        'echo x | python3 -c "y"', "python -c 'x'"):
            with self.subTest(command=command):
                reason = self.policy.classify(command).reason
                self.assertEqual(self.policy.classify(command).classification,
                                 CommandClassification.DANGEROUS)
                self.assertTrue(
                    reason.startswith("run tests with `python3 -m unittest "
                                      "discover -s tests`"), reason)
                self.assertIn("`python3 -c` and running a script directly are "
                              "refused", reason)

    def test_the_generic_refusal_leads_with_what_can_run(self):
        reason = self.policy.classify("frobnicate --all").reason
        self.assertTrue(reason.startswith("build and test with"), reason)
        self.assertIn("python3 -m unittest discover -s tests", reason)
        self.assertIn("'frobnicate' is none of those", reason)

    def test_classifies_shell_and_dangerous_commands(self):
        self.assertEqual(self.policy.classify("rg symbol src | head").classification,
                         CommandClassification.SHELL_COMPLEX)
        self.assertEqual(self.policy.classify("rm -rf build").classification,
                         CommandClassification.DANGEROUS)
        self.assertEqual(self.policy.classify("git reset --hard").classification,
                         CommandClassification.DANGEROUS)

    def test_redirection_target_is_a_file_not_a_command(self):
        """`2>/dev/null` must not make a command dangerous.

        The stage splitter treated the token after a redirection operator as a
        new command stage, so `/dev/null` hit the "argv[0] contains /" rule and
        every command carrying the most common idiom in shell was refused as a
        "dangerous shell stage" — including `ls 2>/dev/null`.
        """
        for command in ('find . -name "ping.c" -type f 2>/dev/null | head -20',
                        "ls 2>/dev/null",
                        "echo hi 2>/dev/null",
                        'grep -r so3 . --include="*.md" 2>/dev/null | head',
                        "make 2>&1 | tail -20",
                        "cat notes.txt > copy.txt",
                        "sort < input.txt"):
            assessment = self.policy.classify(command)
            self.assertNotEqual(assessment.classification,
                                CommandClassification.DANGEROUS,
                                f"{command!r}: {assessment.reason}")

        # The guard the bug was accidentally providing must survive on its own:
        # a redirection escaping the workspace stays dangerous, and only the
        # harmless /dev sinks are exempt.
        for command in ("echo pwned > /etc/spear-marker",
                        "echo pwned >> /usr/bin/spear-marker",
                        "cat secrets > ../outside.txt",
                        "cat secrets > /home/other/loot"):
            assessment = self.policy.classify(command)
            self.assertEqual(assessment.classification,
                             CommandClassification.DANGEROUS,
                             f"{command!r} must stay dangerous")

        # A write redirection still declares that it mutates the workspace.
        self.assertIn(Capability.WORKSPACE_WRITE,
                      self.policy.classify("echo hi > note.txt").required_capabilities)
        # Reading a pipeline stage after a redirect is still analysed: a real
        # command behind the redirect keeps its own capabilities.
        self.assertIn(Capability.NETWORK,
                      self.policy.classify("curl https://example.invalid 2>/dev/null")
                      .required_capabilities)

    def test_sandbox_mount_paths_are_not_escapes(self):
        """Paths under the sandbox mounts are inside, by construction.

        The blanket "argv contains an absolute path" refusal predates multiple
        roots. It left the feature half-wired: write_file could reach a second
        root while bash could not.
        """
        for command in ("ls /workspaces/so3/usr/src",
                        "cat /workspace/notes.txt",
                        "grep -r main /workspaces/lvgl",
                        "echo ok > /workspaces/so3/usr/src/ping.c",
                        "grep main /workspace/a.c /workspaces/so3/b.c"):
            assessment = self.policy.classify(command)
            self.assertNotEqual(assessment.classification,
                                CommandClassification.DANGEROUS,
                                f"{command!r}: {assessment.reason}")

        # WRITING outside stays refused, and a mount name is not a licence to
        # traverse out of it. Reading outside is a separate grant — see
        # OutsideReadPolicyTests.
        for command in ("echo pwned > /workspace/../etc/x",
                        "echo pwned > /etc/x",
                        "cat ../outside/secret"):
            self.assertEqual(self.policy.classify(command).classification,
                             CommandClassification.DANGEROUS, command)

    def test_assessments_declare_required_capabilities(self):
        cases = {
            "pwd": {Capability.FILESYSTEM_READ},
            "git status": {Capability.FILESYSTEM_READ},
            "git diff": {Capability.FILESYSTEM_READ},
            "make": {Capability.FILESYSTEM_READ, Capability.WORKSPACE_WRITE},
            "pytest": {Capability.FILESYSTEM_READ, Capability.WORKSPACE_WRITE},
            "curl https://example.invalid": {Capability.NETWORK},
            "git clone https://example.invalid/repo": {
                Capability.FILESYSTEM_READ, Capability.WORKSPACE_WRITE, Capability.NETWORK,
            },
            "git fetch": {
                Capability.FILESYSTEM_READ, Capability.WORKSPACE_WRITE, Capability.NETWORK,
            },
            "git fetch origin": {
                Capability.FILESYSTEM_READ, Capability.WORKSPACE_WRITE, Capability.NETWORK,
            },
            "git push": {
                Capability.FILESYSTEM_READ, Capability.NETWORK, Capability.REMOTE_WRITE,
            },
            "wget https://example.invalid": {Capability.NETWORK, Capability.WORKSPACE_WRITE},
            "curl -o result https://example.invalid": {
                Capability.NETWORK, Capability.WORKSPACE_WRITE,
            },
            "curl --output result https://example.invalid": {
                Capability.NETWORK, Capability.WORKSPACE_WRITE,
            },
            "curl -O https://example.invalid": {Capability.NETWORK, Capability.WORKSPACE_WRITE},
            "curl --remote-name https://example.invalid": {
                Capability.NETWORK, Capability.WORKSPACE_WRITE,
            },
            "ssh host": {Capability.SSH, Capability.NETWORK, Capability.REMOTE_WRITE},
            "ssh host command": {Capability.SSH, Capability.NETWORK, Capability.REMOTE_WRITE},
            "scp file host:": {
                Capability.SSH, Capability.NETWORK, Capability.FILESYSTEM_READ,
                Capability.REMOTE_WRITE,
            },
            "scp host:path local-file": {
                Capability.SSH, Capability.NETWORK, Capability.WORKSPACE_WRITE,
            },
            "scp host:path .": {Capability.SSH, Capability.NETWORK, Capability.WORKSPACE_WRITE},
            "docker ps": {Capability.CONTAINER_RUNTIME},
            "podman ps": {Capability.CONTAINER_RUNTIME},
            "nvidia-smi": {Capability.GPU},
            "ls | head": {Capability.FILESYSTEM_READ, Capability.SHELL_COMPLEX},
            # filesystem:read is unconditional on a pipeline now: without it
            # the sandbox mounts an empty directory instead of the trees.
            "curl https://example.invalid | cat": {
                Capability.NETWORK, Capability.SHELL_COMPLEX,
                Capability.FILESYSTEM_READ},
            "make && pytest": {
                Capability.FILESYSTEM_READ, Capability.WORKSPACE_WRITE,
                Capability.SHELL_COMPLEX,
            },
            "curl -o result https://example.invalid | cat": {
                Capability.NETWORK, Capability.WORKSPACE_WRITE,
                Capability.SHELL_COMPLEX, Capability.FILESYSTEM_READ,
            },
            "git push && git status": {
                Capability.FILESYSTEM_READ, Capability.NETWORK,
                Capability.REMOTE_WRITE, Capability.SHELL_COMPLEX,
            },
            "curl -X POST https://example.invalid | cat": {
                Capability.NETWORK, Capability.REMOTE_WRITE,
                Capability.SHELL_COMPLEX, Capability.FILESYSTEM_READ,
            },
            "scp file host:path && echo done": {
                Capability.FILESYSTEM_READ, Capability.SSH, Capability.NETWORK,
                Capability.REMOTE_WRITE, Capability.SHELL_COMPLEX,
            },
            "wget https://example.invalid && cat file": {
                Capability.NETWORK, Capability.WORKSPACE_WRITE,
                Capability.FILESYSTEM_READ, Capability.SHELL_COMPLEX,
            },
            "git fetch && git status": {
                Capability.NETWORK, Capability.WORKSPACE_WRITE,
                Capability.FILESYSTEM_READ, Capability.SHELL_COMPLEX,
            },
            "printf 'ok\\n' > result.txt": {
                Capability.FILESYSTEM_READ, Capability.WORKSPACE_WRITE,
                Capability.SHELL_COMPLEX,
            },
        }
        for command, expected in cases.items():
            with self.subTest(command=command):
                assessment = self.policy.classify(command)
                self.assertEqual(assessment.required_capabilities, frozenset(expected))
        for command in ("git status", "git diff"):
            self.assertNotIn(Capability.NETWORK, self.policy.classify(command).required_capabilities)

    def test_remote_mutation_capability_for_curl(self):
        for command in (
            "curl -X POST https://example.invalid",
            "curl --request DELETE https://example.invalid",
            "curl -d x=1 https://example.invalid",
            "curl --data x=1 https://example.invalid",
            "curl -F file=@x https://example.invalid",
        ):
            with self.subTest(command=command):
                capabilities = self.policy.classify(command).required_capabilities
                self.assertIn(Capability.NETWORK, capabilities)
                self.assertIn(Capability.REMOTE_WRITE, capabilities)
        output = self.policy.classify("curl -o result -X POST https://example.invalid")
        self.assertEqual(output.required_capabilities, frozenset({
            Capability.NETWORK, Capability.WORKSPACE_WRITE, Capability.REMOTE_WRITE,
        }))
        for command in (
            "curl https://example.invalid",
            "curl -I https://example.invalid",
            "curl --head https://example.invalid",
            "git fetch",
            "scp host:file .",
        ):
            with self.subTest(command=command):
                self.assertNotIn(Capability.REMOTE_WRITE,
                                 self.policy.classify(command).required_capabilities)

    def test_sensitive_capabilities_are_classified_without_becoming_dangerous(self):
        for command in ("curl https://example.invalid", "ssh host", "docker ps", "nvidia-smi"):
            with self.subTest(command=command):
                self.assertNotEqual(self.policy.classify(command).classification,
                                    CommandClassification.DANGEROUS)

    def test_sensitive_capabilities_are_refused_in_every_mode(self):
        assessments = [
            self.policy.classify("git push"),
            self.policy.classify("ssh host"),
            self.policy.classify("docker ps"),
            self.policy.classify("nvidia-smi"),
        ]
        for mode in ExecutionMode:
            for assessment in assessments:
                with self.subTest(mode=mode, command=assessment.command):
                    result = self.policy.authorize(assessment, mode, lambda _: True)
                    self.assertIsNotNone(result.result)
                self.assertEqual(result.result.status, "denied")

    def test_ask_network_is_confirmable_but_remote_write_is_denied_first(self):
        curl = self.policy.classify("curl https://example.invalid")
        fetch = self.policy.classify("git fetch origin")
        post = self.policy.classify("curl -X POST https://example.invalid")
        push = self.policy.classify("git push")
        ssh = self.policy.classify("ssh example.invalid")
        status = self.policy.classify("git status")

        self.assertTrue(self.policy.authorize(curl, ExecutionMode.ASK, lambda _: True).allowed)
        self.assertEqual(self.policy.authorize(curl, ExecutionMode.ASK, lambda _: False).result.status,
                         "cancelled")
        self.assertTrue(self.policy.authorize(fetch, ExecutionMode.ASK, lambda _: True).allowed)
        for assessment in (post, push, ssh):
            with self.subTest(command=assessment.command):
                approve = MagicMock(return_value=True)
                result = self.policy.authorize(assessment, ExecutionMode.ASK, approve)
                self.assertEqual(result.result.status, "denied")
                approve.assert_not_called()
        self.assertTrue(self.policy.authorize(status, ExecutionMode.ASK, lambda _: False).allowed)
        approve = MagicMock(return_value=True)
        result = self.policy.authorize(curl, ExecutionMode.SAFE, approve)
        self.assertEqual(result.result.status, "denied")
        approve.assert_not_called()
        # AUTO grants it and asks nothing — that is what AUTO means. It used
        # to deny, which made `-y` the mode that still refused curl.
        approve = MagicMock(return_value=True)
        self.assertTrue(self.policy.authorize(curl, ExecutionMode.AUTO,
                                              approve).allowed)
        approve.assert_not_called()

    def test_modes_enforce_expected_command_policy(self):
        readonly = self.policy.classify("ls")
        mutate = self.policy.classify("ninja")
        complex_command = self.policy.classify("ls | head")
        self.assertTrue(self.policy.authorize(readonly, ExecutionMode.SAFE).allowed)
        self.assertEqual(self.policy.authorize(mutate, ExecutionMode.SAFE).result.status, "denied")
        # A read-only pipeline is now allowed in SAFE — the mount keeps it
        # read-only. A pipeline that writes still fails the capability check.
        self.assertTrue(self.policy.authorize(complex_command, ExecutionMode.SAFE).allowed)
        self.assertEqual(
            self.policy.authorize(self.policy.classify("echo x > f"),
                                  ExecutionMode.SAFE).result.status, "denied")
        self.assertTrue(self.policy.authorize(mutate, ExecutionMode.AUTO).allowed)
        # Authorization is distinct from execution: AUTO shell-complex still
        # requires Bubblewrap and is fail-closed by the integration route.
        self.assertTrue(self.policy.authorize(complex_command, ExecutionMode.AUTO).allowed)
        self.assertEqual(self.policy.authorize(mutate, ExecutionMode.ASK, lambda _: False).result.status,
                         "cancelled")
        self.assertTrue(self.policy.authorize(mutate, ExecutionMode.ASK, lambda _: True).allowed)

    def test_capability_policy_default_and_network_restriction(self):
        # SAFE now also grants shell:complex — read-only pipelines. What keeps
        # SAFE read-only is the mount, not the classifier; see
        # SafeModeReadOnlyShellTests.
        # host:read is granted in SAFE too: looking at a tree outside the
        # declared roots is reading, which is what SAFE is for, and the paths
        # are bind-mounted read-only in every mode.
        self.assertEqual(DEFAULT_CAPABILITY_POLICY.for_mode(ExecutionMode.SAFE),
                         frozenset({Capability.FILESYSTEM_READ,
                                    Capability.HOST_READ,
                                    Capability.SHELL_COMPLEX}))
        self.assertIn(Capability.NETWORK, DEFAULT_CAPABILITY_POLICY.for_mode(ExecutionMode.ASK))
        # AUTO is ASK without the prompt, so it cannot grant less than ASK.
        # It used to withhold the network, which made `-y` — reached for to
        # stop being blocked — the mode that still refused curl.
        self.assertIn(Capability.NETWORK,
                      DEFAULT_CAPABILITY_POLICY.for_mode(ExecutionMode.AUTO))
        restricted = CommandPolicy(DEFAULT_CAPABILITY_POLICY.without_network())
        approve = MagicMock(return_value=True)
        outcome = restricted.authorize(
            restricted.classify("curl https://example.invalid"), ExecutionMode.ASK, approve
        )
        self.assertFalse(outcome.allowed)
        self.assertEqual(outcome.result.status, "denied")
        approve.assert_not_called()

    def test_a_capability_refusal_names_the_mode_that_grants_it(self):
        """A refusal that only states the rule is a dead end: told `network`
        was not allowed and nothing else, the model ran six more web searches
        and told the user to open a browser."""
        policy = CommandPolicy(DEFAULT_CAPABILITY_POLICY)
        outcome = policy.authorize(
            policy.classify("curl https://example.invalid"), ExecutionMode.SAFE)
        message = str(outcome.result)
        self.assertIn("--ask", message)
        self.assertIn("--auto", message)
        # …and the way that needs no capability at all.
        self.assertIn("fetch_url", message)

    def test_a_refusal_no_mode_can_lift_says_so(self):
        offline = CommandPolicy(DEFAULT_CAPABILITY_POLICY.without_network())
        outcome = offline.authorize(
            offline.classify("curl https://example.invalid"), ExecutionMode.AUTO)
        message = str(outcome.result)
        self.assertIn("No mode grants", message)
        self.assertNotIn("--ask", message)
        # …and it does not send the model to fetch_url either: --no-network
        # takes the web tools out of the session, so that would be a dead end
        # of a different kind.
        self.assertNotIn("fetch_url", message)

    def test_authorization_result_is_the_granted_capability_source(self):
        assessment = self.policy.classify("git fetch origin")
        outcome = self.policy.authorize(assessment, ExecutionMode.ASK, lambda _: True)
        self.assertIsInstance(outcome, AuthorizationResult)
        self.assertTrue(outcome.allowed)
        self.assertEqual(outcome.granted_capabilities, assessment.required_capabilities)

    def test_heredoc_body_is_data_not_path_operands(self):
        """A file's CONTENT must not be read as arguments of the command.

        Writing a documentation chapter with `cat > doc/source/ls.rst <<EOF`
        was refused with "/ contains the sandbox's own mounts and cannot be
        exposed": the chapter contained the shell-prompt example line `/ % ls`,
        and shlex handed that bare `/` over as a path operand of `cat`. The
        refusal named a path the request never contained.
        """
        command = ("cat > /tmp/ls.rst << 'EOF'\n"
                   ".. _ls:\n\nExamples\n========\n\n::\n\n"
                   "   / % ls\n   / % ls -l /etc\n"
                   "EOF")
        assessment = self.policy.classify(command)
        self.assertNotEqual(assessment.classification,
                            CommandClassification.DANGEROUS)
        self.assertNotIn("sandbox's own mounts", assessment.reason)

    def test_a_heredoc_that_feeds_an_interpreter_is_still_scanned(self):
        """Stripping the body must not become a way to smuggle commands.

        `bash <<EOF` and `cat <<EOF | bash` both execute their body, so there
        the body is code and stays visible to the danger scan.
        """
        for command in ("bash << 'EOF'\nrm -rf /\nEOF",
                        "cat << 'EOF' | bash\nrm -rf /\nEOF"):
            with self.subTest(command=command.splitlines()[0]):
                self.assertEqual(self.policy.classify(command).classification,
                                 CommandClassification.DANGEROUS)

    def test_the_heredoc_target_path_is_still_checked(self):
        """Only the body goes; the redirection target is still an operand."""
        command = ("cat > /etc/shadow << 'EOF'\nharmless text\nEOF")
        assessment = self.policy.classify(command)
        self.assertIn("/etc/shadow", assessment.command)


class ToolPolicyTests(unittest.TestCase):
    def test_safe_ask_and_auto_mutation_behavior(self):
        self.assertEqual(ToolPolicy(ExecutionMode.SAFE).authorize_mutation("Write file").status,
                         "denied")
        self.assertEqual(ToolPolicy(ExecutionMode.ASK, lambda _: False)
                         .authorize_mutation("Write file").status, "cancelled")
        self.assertIsNone(ToolPolicy(ExecutionMode.ASK, lambda _: True)
                          .authorize_mutation("Write file"))
        self.assertIsNone(ToolPolicy(ExecutionMode.AUTO).authorize_mutation("Write file"))


class SedCommandPolicyTests(unittest.TestCase):
    """sed reads like cat but can also write, so it is allowlisted narrowly."""

    def setUp(self):
        self.policy = CommandPolicy()

    def test_reading_forms_are_read_only(self):
        for command in ("sed -n '85,130p' os_desktop.cmake",
                        "sed -n '/BEGIN/,/END/p' f.txt",
                        "sed -e /pattern/d f.txt",
                        "sed 's/a/b/' f.txt"):
            with self.subTest(command=command):
                self.assertEqual(self.policy.classify(command).classification,
                                 CommandClassification.READ_ONLY)

    def test_a_sed_script_is_not_treated_as_a_path(self):
        """`/BEGIN/,/END/p` starts with a slash and names no file.

        Checking it as a path refused ordinary sed with "path argument may
        escape the workspace" -- a reason the model cannot act on, because the
        premise is false.
        """
        assessment = self.policy.classify("sed -n '/BEGIN/,/END/p' f.txt")
        self.assertNotEqual(assessment.reason, "path argument may escape the workspace")

    def test_in_place_editing_is_refused_in_every_spelling(self):
        for command in ("sed -i s/a/b/ f.txt", "sed -i.bak s/a/b/ f.txt",
                        "sed --in-place s/a/b/ f.txt",
                        "sed --in-place=.bak s/a/b/ f.txt",
                        "sed -ni p f.txt", "sed -n -i p f.txt"):
            with self.subTest(command=command):
                assessment = self.policy.classify(command)
                self.assertEqual(assessment.classification,
                                 CommandClassification.DANGEROUS)
                self.assertIn("edit_file", assessment.reason)

    def test_file_operands_are_still_checked_for_escapes(self):
        # Only the script is exempt from the path check, not the operands --
        # and -f takes a script FILE, which is a real path. An absolute operand
        # outside the roots is now a read grant; a RELATIVE `..` still is not,
        # because the classifier cannot know which cwd it resolves against.
        assessment = self.policy.classify("sed -n p ../outside/secret.txt")
        self.assertEqual(assessment.classification, CommandClassification.DANGEROUS)
        self.assertIn("relative path leaves the workspace", assessment.reason)

        for command in ("sed -n p /etc/passwd", "sed -f /etc/evil.sed f.txt"):
            with self.subTest(command=command):
                assessment = self.policy.classify(command)
                self.assertEqual(assessment.classification,
                                 CommandClassification.READ_ONLY)

    def test_other_binaries_keep_checking_every_argument(self):
        assessment = self.policy.classify("cat /etc/passwd")
        self.assertEqual(assessment.classification, CommandClassification.READ_ONLY)
        self.assertIn(Capability.HOST_READ, assessment.required_capabilities)


class FeedbackLoopPolicyTests(unittest.TestCase):
    """A model that cannot run its code cannot check it."""

    def setUp(self):
        self.policy = CommandPolicy()

    def test_a_compiler_is_workspace_mutating_like_make(self):
        for command in ("gcc -o /tmp/t so3/usr/src/ls.c", "cc -o /tmp/t x.c",
                        "g++ -o /tmp/t x.cc"):
            with self.subTest(command=command):
                self.assertEqual(self.policy.classify(command).classification,
                                 CommandClassification.WORKSPACE_MUTATING)

    def test_a_program_built_in_the_sandbox_may_be_run(self):
        """Refusing it denied the only way to TEST a change.

        `make` is already allowed and a Makefile runs anything, so the binary
        the model just compiled is no wider an exposure -- same sandbox, same
        confinement, no network.
        """
        for command in ("/tmp/t 'ts*'", "./t 'ts*'"):
            with self.subTest(command=command):
                self.assertEqual(self.policy.classify(command).classification,
                                 CommandClassification.WORKSPACE_MUTATING)

    def test_a_project_entry_point_may_live_in_a_subdirectory(self):
        """`./scripts/build.sh` is the command the model is meant to run.

        Relative paths with a subdirectory used to be refused outright, which
        refused every Infrabase front-end script. `..` is still caught, and the
        confinement is the sandbox: a program reachable by a relative path
        inside it is inside a mounted tree by construction.
        """
        assessment = self.policy.classify("./scripts/build.sh usr-so3")
        self.assertEqual(assessment.classification,
                         CommandClassification.WORKSPACE_MUTATING)
        self.assertIn(Capability.WORKSPACE_WRITE, assessment.required_capabilities)

    def test_programs_outside_the_sandbox_are_still_refused(self):
        for command in ("/bin/sh -c evil", "../outside/prog", "/etc/passwd",
                        "./../a/prog", "/usr/bin/env sh"):
            with self.subTest(command=command):
                assessment = self.policy.classify(command)
                self.assertEqual(assessment.classification,
                                 CommandClassification.DANGEROUS, command)

    def test_running_a_binary_still_needs_workspace_write(self):
        # SAFE mode grants no workspace:write, so the loop stays unavailable
        # there rather than being classified read-only by accident.
        assessment = self.policy.classify("/tmp/t 'ts*'")
        self.assertIn(Capability.WORKSPACE_WRITE, assessment.required_capabilities)


class OutsideReadPolicyTests(unittest.TestCase):
    """Reading a tree no declared root contains.

    The refusal this replaces ended a real session: asked to look at the notes
    in /opt/llm/claude, the assistant was told "path argument may escape the
    workspace", re-ran the same command, then handed the question back to the
    user. Looking at a file is not changing it, and the two are now separate
    grants.
    """

    def setUp(self):
        # NOT under /tmp: the sandbox's own tmpfs lives there, so a path in it
        # already counts as inside and would not exercise an outside read.
        self.temp = tempfile.TemporaryDirectory(dir=Path.home())
        self.root = Path(self.temp.name) / "workspace"
        self.root.mkdir()
        self.outside = Path(self.temp.name) / "notes"
        self.outside.mkdir()
        (self.outside / "memory.md").write_text("remembered\n")
        self.workspace = Workspace.from_path(self.root)
        self.policy = CommandPolicy()
        self.policy.bind_workspace(self.workspace)

    def tearDown(self):
        self.temp.cleanup()

    def test_a_read_only_command_may_name_an_outside_tree(self):
        assessment = self.policy.classify(f"ls -la {self.outside}")
        self.assertEqual(assessment.classification, CommandClassification.READ_ONLY)
        self.assertIn(Capability.HOST_READ, assessment.required_capabilities)
        self.assertEqual(assessment.host_read_paths, (str(self.outside),))

    def test_the_grant_survives_a_pipeline(self):
        # The shell branch used to check nothing at all: the command ran with
        # the tree unmounted and reported "No such file or directory" about a
        # file that plainly exists -- worse than either allowing or refusing.
        assessment = self.policy.classify(
            f"cat {self.outside}/memory.md | head -3")
        self.assertEqual(assessment.classification, CommandClassification.SHELL_COMPLEX)
        self.assertIn(Capability.HOST_READ, assessment.required_capabilities)
        self.assertIn(str(self.outside / "memory.md"), assessment.host_read_paths)

    def test_an_input_redirection_reads_and_an_output_redirection_does_not(self):
        readable = self.policy.classify(f"cat < {self.outside}/memory.md")
        self.assertNotEqual(readable.classification, CommandClassification.DANGEROUS)
        self.assertIn(str(self.outside / "memory.md"), readable.host_read_paths)

        writable = self.policy.classify(f"echo x > {self.outside}/memory.md")
        self.assertEqual(writable.classification, CommandClassification.DANGEROUS)

    def test_a_missing_outside_path_binds_nothing_and_is_not_refused(self):
        # ENOENT from the command is the truthful answer and the one the model
        # can act on; a policy error about a file that was never there is not.
        assessment = self.policy.classify(f"cat {self.outside}/absent.md")
        self.assertEqual(assessment.classification, CommandClassification.READ_ONLY)
        self.assertEqual(assessment.host_read_paths, ())

    def test_credential_stores_stay_refused(self):
        for path in ("/home/someone/.ssh/id_ed25519", "/home/someone/.aws/credentials",
                     "/etc/shadow", "/home/someone/.netrc"):
            with self.subTest(path=path):
                assessment = self.policy.classify(f"cat {path}")
                self.assertEqual(assessment.classification,
                                 CommandClassification.DANGEROUS)
                self.assertIn("credentials", assessment.reason)

    def test_a_sandbox_mount_point_cannot_be_bound_over(self):
        # Binding /home would land on top of the sandbox's own $HOME.
        for path in ("/", "/home", "/usr", "/tmp"):
            with self.subTest(path=path):
                assessment = self.policy.classify(f"ls {path}")
                self.assertEqual(assessment.classification,
                                 CommandClassification.DANGEROUS)
                self.assertIn("subdirectory", assessment.reason)

    def test_a_glob_mounts_the_directory_the_shell_will_expand_in(self):
        # The pattern names nothing on the host; the directory holding the
        # matches is what has to be there when the sandbox's shell expands it.
        assessment = self.policy.classify(f"grep -l x {self.outside}/*.md")
        self.assertEqual(assessment.host_read_paths, (str(self.outside),))

    def test_a_symlinked_tree_is_bound_where_the_command_spells_it(self):
        # /opt/llm/claude is a symlink; binding only its target answers ENOENT
        # for the exact path the user named.
        link = Path(self.temp.name) / "link-to-notes"
        link.symlink_to(self.outside, target_is_directory=True)
        assessment = self.policy.classify(f"ls -la {link}/")
        self.assertEqual(assessment.host_read_paths, (str(link),))

    def test_a_symlink_is_not_a_way_round_the_checks(self):
        link = Path(self.temp.name) / "innocent"
        link.symlink_to("/etc/shadow")
        assessment = self.policy.classify(f"cat {link}")
        self.assertEqual(assessment.classification, CommandClassification.DANGEROUS)
        self.assertIn("credentials", assessment.reason)

    def test_a_relative_traversal_is_refused_and_says_what_to_do(self):
        assessment = self.policy.classify("cat ../notes/memory.md")
        self.assertEqual(assessment.classification, CommandClassification.DANGEROUS)
        self.assertIn("absolute path", assessment.reason)
        self.assertIn(str(self.root), assessment.reason)

    def test_every_mode_grants_the_read_and_ask_confirms_it(self):
        assessment = self.policy.classify(f"ls {self.outside}")
        for mode in (ExecutionMode.SAFE, ExecutionMode.AUTO):
            with self.subTest(mode=mode):
                self.assertTrue(self.policy.authorize(assessment, mode).allowed)
        refused = self.policy.authorize(assessment, ExecutionMode.ASK, lambda _: False)
        self.assertFalse(refused.allowed)
        self.assertTrue(
            self.policy.authorize(assessment, ExecutionMode.ASK, lambda _: True).allowed)

    def test_a_refusal_names_the_trees_that_are_reachable(self):
        reason = self.policy.classify("cat /etc/shadow").reason
        self.assertIn(str(self.root), reason)
        self.assertIn("READ-ONLY", reason)


if __name__ == "__main__":
    unittest.main()
