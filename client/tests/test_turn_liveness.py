import sys
import threading
import time
import unittest
from unittest.mock import patch

sys.argv = ["rag_chat", "--safe"]
from cli import terminal_ui
from runtime.cancellation import CancellationSource


class TheHeartbeat(unittest.TestCase):
    """The wall-clock budget is cooperative; a turn suspended in a read that
    never returns charges nothing and so never consults the clock. One ran
    for eleven hours inside an eight-minute budget."""

    def test_a_notice_counts_as_life(self):
        terminal_ui.LIVENESS.at = time.monotonic() - 60
        terminal_ui.CliRuntimeObserver().notice("verification_nudge", {})

        self.assertLess(terminal_ui.LIVENESS.silent_for(), 1)

    def test_a_token_counts_as_life(self):
        with terminal_ui.CliRuntimeObserver().model_activity("Thinking…") as tick:
            terminal_ui.LIVENESS.at = time.monotonic() - 60
            tick()

        self.assertLess(terminal_ui.LIVENESS.silent_for(), 1)

    def test_the_spinner_alone_is_not_life(self):
        """It animates whether or not the provider answers — which is
        precisely the case being watched for."""
        with terminal_ui.Spinner("Thinking…"):
            terminal_ui.LIVENESS.at = time.monotonic() - 60
            time.sleep(0.5)

        self.assertGreater(terminal_ui.LIVENESS.silent_for(), 30)


class ABoundedCommand(unittest.TestCase):
    """A build is silent while it compiles: under the 300 s heartbeat alone,
    the watchdog interrupted a 600 s `build.sh linux` the sandbox was still
    running within its own timeout."""

    def test_a_running_command_is_alive_until_its_own_timeout(self):
        with terminal_ui.LIVENESS.bounded(600):
            terminal_ui.LIVENESS.at = time.monotonic() - 400

            self.assertEqual(terminal_ui.LIVENESS.silent_for(), 0.0)

    def test_past_its_timeout_the_silence_counts_again(self):
        with terminal_ui.LIVENESS.bounded(600):
            terminal_ui.LIVENESS.at = time.monotonic() - 400
            terminal_ui.LIVENESS.until = time.monotonic() - 310

            self.assertGreater(terminal_ui.LIVENESS.silent_for(), 300)

    def test_the_end_of_the_command_is_a_sign_of_life(self):
        with terminal_ui.LIVENESS.bounded(600):
            terminal_ui.LIVENESS.at = time.monotonic() - 400

        self.assertLess(terminal_ui.LIVENESS.silent_for(), 1)


class TheWatchdog(unittest.TestCase):
    def run_turn(self, *, silence, threshold, work=0.6):
        source = CancellationSource()

        with patch.object(terminal_ui, "TURN_LIVENESS_SECONDS", threshold):
            with terminal_ui.interruptible(source):
                terminal_ui.LIVENESS.at = time.monotonic() - silence
                time.sleep(work)

        return source.token

    def test_a_quiet_turn_is_cancelled(self):
        token = self.run_turn(silence=30, threshold=1, work=6.5)

        self.assertTrue(token.is_cancelled)
        self.assertIn("no sign of life", token.reason or "")

    def test_a_working_turn_is_left_alone(self):
        token = self.run_turn(silence=0, threshold=600)

        self.assertFalse(token.is_cancelled)

    def test_the_watchdog_stops_with_the_turn(self):
        before = threading.active_count()
        self.run_turn(silence=0, threshold=600)
        time.sleep(6)

        self.assertLessEqual(threading.active_count(), before + 1,
                             "no watchdog outlives its turn")


if __name__ == "__main__":
    unittest.main()
