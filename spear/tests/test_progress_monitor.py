import unittest

from progress_monitor import ProgressMonitor, action_fingerprint, read_evidence


class ProgressMonitorTests(unittest.TestCase):
    def test_identical_and_whitespace_normalized_commands_repeat(self):
        monitor = ProgressMonitor(3)
        monitor.observe_action("bash", {"command": "make   test"})
        observed = monitor.observe_action("bash", {"command": "make test"})
        self.assertTrue(observed.repeated)

    def test_meaningfully_changed_command_is_not_repeat(self):
        monitor = ProgressMonitor(3)
        monitor.observe_action("bash", {"command": "make test"})
        self.assertFalse(monitor.observe_action(
            "bash", {"command": "make integration"}).repeated)

    def test_repeated_file_read_stalls_only_at_threshold(self):
        monitor = ProgressMonitor(3)
        monitor.observe_action("read_file", {"path": "a.py", "start": 1})
        self.assertFalse(monitor.observe_action(
            "read_file", {"path": "a.py", "start": 1}).stalled)
        self.assertTrue(monitor.observe_action(
            "read_file", {"path": "a.py", "start": 1}).stalled)

    def test_new_evidence_and_mutation_reset_progress(self):
        monitor = ProgressMonitor(3)
        monitor.observe_action("read_file", {"path": "a.py"})
        monitor.observe_action("read_file", {"path": "a.py"})
        observed = monitor.observe_action(
            "read_file", {"path": "a.py"}, evidence=("a.py:new-range",))
        self.assertFalse(observed.stalled)
        observed = monitor.observe_action("write_file", {"path": "a.py"}, mutation=True)
        self.assertEqual(observed.consecutive_without_progress, 0)

    def test_verification_counts_as_progress(self):
        monitor = ProgressMonitor(3)
        monitor.observe_action("bash", {"command": "make"})
        observed = monitor.observe_action("bash", {"command": "make"}, verification=True)
        self.assertFalse(observed.repeated)

    def test_state_round_trip_preserves_stall_memory(self):
        monitor = ProgressMonitor(4)
        monitor.observe_action("bash", {"command": "make"})
        monitor.observe_action("bash", {"command": "make"})
        restored = ProgressMonitor.from_dict(monitor.to_dict())
        self.assertEqual(restored.consecutive_without_progress, 1)
        self.assertTrue(restored.observe_action(
            "bash", {"command": "make"}).repeated)

    def test_fingerprint_is_deterministic(self):
        self.assertEqual(action_fingerprint("x", {"b": 2, "a": 1}),
                         action_fingerprint("x", {"a": 1, "b": 2}))



class WindowedReads(unittest.TestCase):
    """Paging through a long file is one action that keeps showing new lines.

    A model read an 897-line file in four 200-line sed windows and was
    stopped as stalled on the fourth, one round before the edit the task had
    asked for. The file repeats; the content does not. The fingerprint stays
    one -- nine looks at the same three lines are still one look -- and the
    unseen blocks are the evidence that says this is not a loop.
    """

    FILE = "src/command/command_wire.c"

    def observe(self, monitor, command):
        return monitor.observe_action("bash", {"command": command},
                                      evidence=(self.FILE,))

    def test_four_windows_of_one_file_do_not_stall(self):
        monitor = ProgressMonitor(stall_threshold=4)
        seen = [self.observe(monitor, f"sed -n '{a},{b}p' {self.FILE}")
                for a, b in ((1, 200), (200, 400), (400, 600), (600, 897))]

        self.assertFalse(any(item.stalled for item in seen))
        self.assertEqual(len({item.fingerprint for item in seen}), 1)

    def test_the_same_window_four_times_still_does(self):
        monitor = ProgressMonitor(stall_threshold=4)
        seen = [self.observe(monitor, f"sed -n '1,200p' {self.FILE}")
                for _ in range(4)]

        self.assertTrue(seen[-1].stalled)

    def test_the_same_three_lines_five_ways_still_do(self):
        monitor = ProgressMonitor(stall_threshold=4)
        seen = [self.observe(monitor, command) for command in (
            f"sed -n '59,61p' {self.FILE}",
            f"sed -n '58,63p' {self.FILE} | cat -A",
            f"sed -n '59,61p' {self.FILE} | od -c",
            f"sed -n '59,61p' {self.FILE} | hexdump -C")]

        self.assertTrue(seen[-1].stalled)

    def test_looks_with_no_window_still_collapse(self):
        monitor = ProgressMonitor(stall_threshold=4)
        seen = [self.observe(monitor, command) for command in (
            f"cat -A {self.FILE}", f"od -c {self.FILE}",
            f"hexdump -C {self.FILE}", f"strings {self.FILE}")]

        self.assertTrue(seen[-1].stalled)

    def test_blocks(self):
        self.assertEqual(read_evidence(f"sed -n '200,400p' {self.FILE}"),
                         (f"{self.FILE}#L100", f"{self.FILE}#L200",
                          f"{self.FILE}#L300"))
        self.assertEqual(read_evidence(f"head -n 40 {self.FILE}"),
                         (f"{self.FILE}#L0",))
        self.assertEqual(read_evidence(f"tail -n 40 {self.FILE}"),
                         (f"{self.FILE}#tail40",))
        self.assertEqual(read_evidence(f"cat {self.FILE}"), ())
        self.assertEqual(read_evidence(f"grep -n x {self.FILE}"), ())
