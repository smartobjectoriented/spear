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
