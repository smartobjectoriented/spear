"""serve.sh takes its settings from configuration, or refuses to start.

Every value that used to be written into a script is now something a
deployment says. The tests below run the real script with a stub binary in
place of llama-server, so what is checked is the command line it actually
builds -- not a re-implementation of its logic in Python, which would agree
with itself and with nothing else.

No GPU, no weights, no network.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

SERVER = Path(__file__).resolve().parent.parent
SERVE = SERVER / "inference" / "serve.sh"

#: A stand-in for llama-server: it records its argv and exits. serve.sh execs
#: it, so this is the only way to see the command line it decided on.
STUB = """#!/bin/bash
printf '%s\\n' "$@" > "$SPEAR_TEST_ARGV"
printf '%s\\n' "gpu=${CUDA_VISIBLE_DEVICES:-<unpinned>}" >> "$SPEAR_TEST_ARGV"
printf '%s\\n' "ldpath=${LD_LIBRARY_PATH:-}" >> "$SPEAR_TEST_ARGV"
exit 0
"""


class Serving(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

        self.bin = self.root / "llama.cpp" / "build" / "bin" / "llama-server"
        self.bin.parent.mkdir(parents=True)
        self.bin.write_text(STUB)
        self.bin.chmod(0o755)

        self.model = self.root / "models" / "Stub-Q8_0-00001-of-00002.gguf"
        self.model.parent.mkdir(parents=True)
        self.model.write_bytes(b"not really a gguf")

        (self.root / "config").mkdir()
        self.argv = self.root / "argv.txt"

    def conf(self, text):
        (self.root / "config" / "server.conf").write_text(text)

    def gpu_conf(self, text):
        (self.root / "config" / "gpu.conf").write_text(text)

    def serve(self, *args, env=None, expect=0):
        """Run serve.sh with nothing inherited but what a shell needs.

        Not called `run`: TestCase.run is how unittest executes a test,
        and overriding it makes the runner call this instead.
        """
        e = {"PATH": os.environ["PATH"], "HOME": str(self.root),
             "SPEAR_SERVER_ROOT": str(self.root),
             "SPEAR_TEST_ARGV": str(self.argv)}
        e.update(env or {})
        proc = subprocess.run(["bash", str(SERVE), *args], env=e,
                              capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, expect,
                         f"stdout={proc.stdout}\nstderr={proc.stderr}")
        return proc

    def served(self):
        """The argv llama-server was actually given, as a flag -> value map."""
        lines = self.argv.read_text().splitlines()
        flags, i = {}, 0
        while i < len(lines):
            if lines[i].startswith("--"):
                nxt = lines[i + 1] if i + 1 < len(lines) else ""
                if nxt.startswith("--") or "=" in nxt and not nxt.startswith("/"):
                    flags[lines[i]] = True
                    i += 1
                else:
                    flags[lines[i]] = nxt
                    i += 2
            else:
                key, _, value = lines[i].partition("=")
                flags[key] = value
                i += 1
        return flags

    def complete(self, **extra):
        return dict({"SPEAR_SERVER_LLAMA_BIN": str(self.bin),
                     "SPEAR_SERVER_MODEL": str(self.model),
                     "SPEAR_SERVER_CTX": "4096"}, **extra)

    # ── A. it reads configuration ────────────────────────────────────

    def test_the_config_file_supplies_everything(self):
        self.conf(f'SPEAR_SERVER_LLAMA_BIN={self.bin}\n'
                  f'SPEAR_SERVER_MODEL={self.model}\n'
                  f'SPEAR_SERVER_CTX=4096\n'
                  f'SPEAR_SERVER_PORT=9999\n')
        self.serve()
        served = self.served()
        self.assertEqual(served["--model"], str(self.model))
        self.assertEqual(served["--ctx-size"], "4096")
        self.assertEqual(served["--port"], "9999")

    def test_the_environment_beats_the_file(self):
        """A one-off run must not need the file edited -- and sourcing the
        file must not quietly undo what the caller asked for."""
        self.conf(f'SPEAR_SERVER_LLAMA_BIN={self.bin}\n'
                  f'SPEAR_SERVER_MODEL={self.model}\n'
                  f'SPEAR_SERVER_CTX=4096\n')
        self.serve(env={"SPEAR_SERVER_CTX": "131072"})
        self.assertEqual(self.served()["--ctx-size"], "131072")

    def test_extra_arguments_reach_the_server(self):
        self.serve("--verbose", env=self.complete())
        self.assertIn("--verbose", self.served())

    # ── B. it refuses to guess ───────────────────────────────────────

    def test_nothing_configured_names_every_missing_value(self):
        proc = self.serve(expect=78)
        for name in ("SPEAR_SERVER_LLAMA_BIN", "SPEAR_SERVER_MODEL",
                     "SPEAR_SERVER_CTX"):
            self.assertIn(name, proc.stderr)
        self.assertFalse(self.argv.exists(), "the server was started anyway")

    def test_a_missing_context_size_is_refused_not_defaulted(self):
        """The one value with no default. Two numbers were in circulation for
        it, and a script that picks one silently is how that happened."""
        env = self.complete()
        del env["SPEAR_SERVER_CTX"]
        proc = self.serve(expect=78, env=env)
        self.assertIn("SPEAR_SERVER_CTX", proc.stderr)
        self.assertFalse(self.argv.exists())

    def test_a_non_numeric_context_size_is_refused(self):
        proc = self.serve(expect=78, env=self.complete(SPEAR_SERVER_CTX="lots"))
        self.assertIn("not a number", proc.stderr)

    def test_a_missing_binary_says_how_to_build_it(self):
        proc = self.serve(expect=78,
                        env=self.complete(SPEAR_SERVER_LLAMA_BIN="/nope/llama-server"))
        self.assertIn("install-llamacpp.sh", proc.stderr)

    def test_a_missing_model_says_how_to_fetch_it(self):
        proc = self.serve(expect=78,
                        env=self.complete(SPEAR_SERVER_MODEL="/nope/model.gguf"))
        self.assertIn("fetch-model.sh", proc.stderr)

    # ── C. nothing is resolved against the checkout ──────────────────

    def test_no_path_is_derived_from_the_repository(self):
        """Every path the server touches came from configuration. If any of
        them pointed into the checkout, this temporary tree could not supply
        all of them and the run would fail."""
        self.serve(env=self.complete())
        served = self.served()
        for value in (served["--model"], served.get("ldpath", "")):
            self.assertTrue(value.startswith(str(self.root)), value)
            self.assertNotIn(str(SERVER.parent), value)

    def test_the_library_path_follows_the_configured_binary(self):
        self.serve(env=self.complete())
        self.assertEqual(self.served()["ldpath"].split(":")[0],
                         str(self.bin.parent))

    # ── D. the card ──────────────────────────────────────────────────

    def test_the_gpu_file_pins_the_card(self):
        self.gpu_conf("SPEAR_SERVER_GPU_UUID=GPU-0123\n")
        self.serve(env=self.complete())
        self.assertEqual(self.served()["gpu"], "GPU-0123")

    def test_no_gpu_configured_means_unpinned(self):
        self.serve(env=self.complete())
        self.assertEqual(self.served()["gpu"], "<unpinned>")

    def test_an_explicit_choice_by_the_caller_wins(self):
        self.gpu_conf("SPEAR_SERVER_GPU_UUID=GPU-0123\n")
        self.serve(env=self.complete(CUDA_VISIBLE_DEVICES="GPU-9999"))
        self.assertEqual(self.served()["gpu"], "GPU-9999")

    # ── E. offload and adapter ───────────────────────────────────────

    def test_a_dense_model_gets_no_expert_offload(self):
        dense = self.root / "models" / "Dense-32B-Q8_0.gguf"
        dense.write_bytes(b"x")
        self.serve(env=self.complete(SPEAR_SERVER_MODEL=str(dense)))
        self.assertNotIn("--n-cpu-moe", self.served())

    def test_a_mixture_of_experts_model_offloads_its_experts(self):
        moe = self.root / "models" / "Model-A3B-Q8_0.gguf"
        moe.write_bytes(b"x")
        self.serve(env=self.complete(SPEAR_SERVER_MODEL=str(moe)))
        self.assertEqual(self.served()["--n-cpu-moe"], "99")

    def test_the_guess_can_be_overridden(self):
        moe = self.root / "models" / "Model-A3B-Q8_0.gguf"
        moe.write_bytes(b"x")
        self.serve(env=self.complete(SPEAR_SERVER_MODEL=str(moe),
                                   SPEAR_SERVER_NCPUMOE="0"))
        self.assertNotIn("--n-cpu-moe", self.served())

    def test_threads_unset_lets_llama_cpp_choose(self):
        """A thread count chosen on another machine is worse than none."""
        self.serve(env=self.complete())
        self.assertNotIn("--threads", self.served())

    def test_threads_configured_are_passed_on(self):
        self.serve(env=self.complete(SPEAR_SERVER_THREADS="8"))
        self.assertEqual(self.served()["--threads"], "8")

    def test_none_means_no_adapter(self):
        self.serve(env=self.complete(SPEAR_SERVER_LORA="none"))
        self.assertNotIn("--lora", self.served())

    def test_a_stale_adapter_path_does_not_stop_the_server(self):
        proc = self.serve(env=self.complete(SPEAR_SERVER_LORA="/nope/a.gguf"))
        self.assertNotIn("--lora", self.served())
        self.assertIn("serving the base model", proc.stderr)

    def test_a_readable_adapter_is_served(self):
        lora = self.root / "adapter.gguf"
        lora.write_bytes(b"x")
        self.serve(env=self.complete(SPEAR_SERVER_LORA=str(lora)))
        self.assertEqual(self.served()["--lora"], str(lora))


if __name__ == "__main__":
    unittest.main()
