import io
import re
import sys
import time
import unittest
from contextlib import redirect_stdout

sys.argv = ["rag_chat", "--safe"]
import rag_chat

CLEAR = "\r\033[K"
ANSI = re.compile(r"\033\[[0-9;]*m")


def visible(raw):
    """What the line holds after every carriage return and erase is applied."""
    line = ""

    for chunk in raw.split(CLEAR):
        line = chunk

    return ANSI.sub("", line)


class TheLineIsNeverBlanked(unittest.TestCase):
    """A one-second call followed by three seconds of silence used to show
    the label, wipe it, and leave the operator staring at nothing."""

    def test_a_finished_activity_still_holds_the_line(self):
        buffer = io.StringIO()

        with redirect_stdout(buffer):
            with rag_chat.Spinner("Analyzing results…"):
                time.sleep(1.0)

        self.assertIn("Analyzing results…", visible(buffer.getvalue()))

    def test_even_a_very_short_one(self):
        buffer = io.StringIO()

        with redirect_stdout(buffer):
            with rag_chat.Spinner("Compacting context…"):
                time.sleep(0.2)

        self.assertIn("Compacting context…", visible(buffer.getvalue()))

    def test_what_it_leaves_is_static(self):
        buffer = io.StringIO()

        with redirect_stdout(buffer):
            with rag_chat.Spinner("Thinking…"):
                time.sleep(0.3)

        last = visible(buffer.getvalue())

        self.assertNotIn("ctrl+c", last)

        for frame in rag_chat.Spinner.FRAMES:
            self.assertNotIn(frame, last)


class TheLineIsNeverStacked(unittest.TestCase):
    """The first fix left every finished activity on its own line, and a
    turn trailed a column of "Thinking… 4s" above the work."""

    def test_no_activity_ever_writes_a_newline(self):
        buffer = io.StringIO()

        with redirect_stdout(buffer):
            for label in ("Thinking…", "Analyzing results…", "Compacting…"):
                with rag_chat.Spinner(label):
                    time.sleep(0.2)

        self.assertNotIn("\n", buffer.getvalue())

    def test_the_next_activity_overwrites_the_last(self):
        buffer = io.StringIO()

        with redirect_stdout(buffer):
            with rag_chat.Spinner("Thinking…"):
                time.sleep(0.2)
            with rag_chat.Spinner("Compacting context…"):
                time.sleep(0.2)

        line = visible(buffer.getvalue())

        self.assertIn("Compacting context…", line)
        self.assertNotIn("Thinking…", line)


class RealOutputTakesTheLineBack(unittest.TestCase):
    """Every print would otherwise have to remember, and one that forgot
    would append to a spinner mid-frame."""

    class Stream:
        def __init__(self):
            self.written = []

        def write(self, text):
            self.written.append(text)

            return len(text)

        def flush(self):
            pass

    def written(self, *, pending, writing=False, text="⏺ Bash(ls)"):
        """What the terminal receives, through the real chain.

        The status line writes to sys.stdout, so the proxy has to BE
        sys.stdout for the test to exercise anything -- wrapping a stream
        beside it tests a proxy nobody uses.
        """
        stream = self.Stream()
        previous, previous_pending = sys.stdout, rag_chat.STATUS.pending
        sys.stdout = rag_chat._StatusAwareStdout(stream)
        rag_chat.STATUS.pending = pending
        rag_chat.STATUS.writing = writing

        try:
            sys.stdout.write(text)
        finally:
            sys.stdout = previous
            rag_chat.STATUS.pending = previous_pending
            rag_chat.STATUS.writing = False

        return stream.written

    def test_a_print_clears_a_pending_status_first(self):
        written = self.written(pending=True)

        self.assertEqual(written[0], CLEAR)
        self.assertEqual(written[-1], "⏺ Bash(ls)")

    def test_with_no_status_pending_nothing_is_erased(self):
        self.assertEqual(self.written(pending=False, text="plain output"),
                         ["plain output"])

    def test_the_status_line_does_not_erase_itself_forever(self):
        """Its own writes are exempt, or clearing would recurse."""

        self.assertEqual(
            self.written(pending=True, writing=True, text="spinner frame"),
            ["spinner frame"])


class OneClockForTheTurn(unittest.TestCase):
    """The label does not change between rounds, so a counter jumping back
    to zero was the only visible signal — and it read as a restart when the
    turn was simply still going."""

    def seconds(self, raw):
        return [int(n) for n in re.findall(r"\((\d+)s", ANSI.sub("", raw))]

    def test_the_count_never_goes_backwards_across_activities(self):
        buffer = io.StringIO()
        rag_chat.STATUS.begin_turn()

        try:
            with redirect_stdout(buffer):
                for label in ("Thinking…", "Analyzing results…", "Compacting…"):
                    with rag_chat.Spinner(label):
                        time.sleep(1.1)
        finally:
            rag_chat.STATUS.end_turn()

        counts = self.seconds(buffer.getvalue())

        self.assertEqual(counts, sorted(counts), "the clock only goes up")
        self.assertGreaterEqual(counts[-1], 2, "it spans the whole turn")

    def test_outside_a_turn_each_activity_times_itself(self):
        """Startup work — loading the embedder — is not part of any turn."""
        rag_chat.STATUS.end_turn()
        buffer = io.StringIO()

        with redirect_stdout(buffer):
            with rag_chat.Spinner("Loading…"):
                time.sleep(1.1)
            with rag_chat.Spinner("Loading…"):
                time.sleep(0.2)

        self.assertEqual(min(self.seconds(buffer.getvalue())), 0)

    def test_minutes_appear_once_seconds_stop_being_readable(self):
        self.assertEqual(rag_chat.Spinner._fmt_elapsed(59), "59s")
        self.assertEqual(rag_chat.Spinner._fmt_elapsed(60), "1m00s")
        self.assertEqual(rag_chat.Spinner._fmt_elapsed(754), "12m34s")


class AToolLooksLikeWorkNotLikeAFreeze(unittest.TestCase):
    """Only the model call had an indicator.

    So when a tool ran, the line kept the finished call's last frame -- its
    label, its token count -- and stopped moving. Every standard.* tool
    prints nothing while it works, and one turn spent 85 of its 145 seconds
    in tool calls: 25 of 30 of them over two seconds of apparent freeze.
    """

    @staticmethod
    def shape(raw, label):
        """The rendered line with the label taken out: the form, not the text."""

        return [ANSI.sub("", chunk).replace(label, "LABEL")
                for chunk in raw.split(CLEAR) if chunk]

    @staticmethod
    def render(label, *, tokens=0):
        buffer = io.StringIO()

        with redirect_stdout(buffer):
            with rag_chat.Spinner(label) as spinner:
                spinner.tokens = tokens
                time.sleep(0.4)

        return buffer.getvalue()

    def test_a_tool_is_drawn_exactly_like_a_model_call(self):
        """Same frames, same colours, same layout -- only the label changes."""

        model = self.render("Analyzing results…")
        tool = self.render("standard.fetch…")

        model_shape = self.shape(model, "Analyzing results…")
        tool_shape = self.shape(tool, "standard.fetch…")

        self.assertEqual(model_shape[:2], tool_shape[:2])

        # The same animation, drawn from the same set, in the same order.

        frames = [line[0] for line in tool_shape[:-1]]
        self.assertEqual(frames, [line[0] for line in model_shape[:len(frames)]])
        self.assertTrue(set(frames) <= set(rag_chat.Spinner.FRAMES))
        self.assertIn("ctrl+c to interrupt", ANSI.sub("", tool))

    def test_the_clock_carries_on_from_the_model_call_into_the_tool(self):
        """One turn, one clock: the handover is not a restart."""

        buffer = io.StringIO()
        rag_chat.STATUS.begin_turn()

        try:
            with redirect_stdout(buffer):
                with rag_chat.Spinner("Analyzing results…"):
                    time.sleep(1.1)

                with rag_chat.Spinner("standard.fetch…"):
                    time.sleep(1.1)
        finally:
            rag_chat.STATUS.end_turn()

        counts = [int(n) for n in re.findall(r"\((\d+)s",
                                            ANSI.sub("", buffer.getvalue()))]

        self.assertEqual(counts, sorted(counts), "the clock only goes up")
        self.assertGreaterEqual(counts[-1], 2, "it spans both activities")

    def test_the_tool_line_carries_no_leftover_token_count(self):
        """The count belonged to a call that has finished."""

        tool = ANSI.sub("", self.render("standard.fetch…"))

        self.assertNotIn("tokens", tool)


if __name__ == "__main__":
    unittest.main()
