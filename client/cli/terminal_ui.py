"""The operator's terminal: status line, spinners, interrupts and printing."""

import os
import re
import sys
import signal
import time
import difflib
import threading
from contextlib import contextmanager


@contextmanager
def interruptible(source):
    """Make Ctrl+C cancel the running turn instead of doing nothing.

    The runtime already polls a cancellation token at six points -- between
    rounds, before a tool call, after one -- and ends the turn cleanly when
    it is set. Nothing ever set it: AgentContext defaults to NEVER_CANCELLED
    and the CLI never passed anything else, so the cooperative path existed
    and was never armed. Measured, not assumed: SIGINT to the interpreter
    mid-turn left the turn running and the prompt gone, thirty seconds on.

    The FIRST interrupt asks the turn to stop, which lets the harness write
    its session, keep what the turn already did, and answer. A SECOND one
    restores Python's own handler, so a turn wedged somewhere that never
    polls can still be killed the ordinary way rather than trapping the
    operator in their own shell.
    """
    def on_interrupt(signum, frame):
        if source.cancel("interrupted by the operator"):
            print(f"\n{C_DIM}⏺ stopping this turn — ctrl+c again to force"
                  f"{C_RST}", flush=True)

            return

        signal.signal(signal.SIGINT, previous)
        raise KeyboardInterrupt

    stop = threading.Event()

    def watchdog():
        """Cancel a turn that has gone quiet, since it cannot cancel itself."""
        while not stop.wait(5):
            if LIVENESS.silent_for() < TURN_LIVENESS_SECONDS:
                continue

            if source.cancel(f"no sign of life for "
                             f"{int(LIVENESS.silent_for())}s"):
                print(f"\n{C_DIM}⏺ nothing has happened for "
                      f"{TURN_LIVENESS_SECONDS}s — stopping this turn{C_RST}",
                      flush=True)

            return

    LIVENESS.touch()
    watcher = threading.Thread(target=watchdog, daemon=True)
    watcher.start()

    try:
        previous = signal.signal(signal.SIGINT, on_interrupt)
    except ValueError:
        # Not the main thread; leave the default handler alone.
        try:
            yield
        finally:
            stop.set()

        return

    try:
        yield
    finally:
        stop.set()
        signal.signal(signal.SIGINT, previous)

C_TOOL = "\033[36m"
C_WARN = "\033[33m"
C_OK = "\033[32m"
C_ERR = "\033[31m"
C_RST = "\033[0m"
C_DIM = "\033[2m"
C_BOLD = "\033[1m"
C_ACCENT = "\033[38;5;77m"    # the platform green
C_CODE = "\033[38;5;114m"     # soft green for inline code


# ── Claude Code-style UI ─────────────────────────────────────────────

ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def vlen(s):
    """Visible length (ANSI codes stripped)."""
    return len(ANSI_RE.sub("", s))


# The spinner runs in its own thread and repaints with "\r\033[K" -- carriage
# return, then erase to end of line. Anything another thread is midway through
# printing on that line is erased with it, and the line simply never appears.
# That is how an `edit_file` that HAD applied showed a diff and then no result
# at all: the model, told nothing, abandoned the tool that had just worked and
# went off rewriting the file with shell splices. Every terminal write that
# must survive takes this lock; the spinner holds it while it repaints.

TERMINAL_LOCK = threading.RLock()


class StatusLine:
    """The one line the harness uses to say what it is doing right now.

    Two rules, and both come from watching it get them wrong.

    IT IS NEVER BLANKED. The old spinner erased its line when an activity
    ended, so a one-second call followed by three seconds of silence showed
    the label, wiped it, and left the operator staring at nothing during the
    part they most wanted explained. The text stays until something replaces
    it.

    IT IS NEVER STACKED. The first attempt at a fix left each finished
    activity behind on its own line, and a turn then trailed a column of
    "Thinking… 4s" above the work. There is one status line. A new activity
    overwrites it; real output erases it and takes the line for itself.

    Erasing on real output is why this exists as an object rather than a
    convention: every print in the client would otherwise have to remember,
    and one that forgot would append to a spinner mid-frame.
    """

    def __init__(self):
        self.pending = False
        self.writing = False

        # One clock for the whole turn. Each round used to start a fresh
        # spinner, and since the label does not change between rounds
        # ("Analyzing results…" over and over) the only visible signal was a
        # counter jumping back to zero -- which reads as a restart when the
        # turn is simply still going. It runs from the prompt to the answer.
        self.turn_t0 = None

    def begin_turn(self):
        self.turn_t0 = time.time()

    def end_turn(self):
        self.turn_t0 = None

    def elapsed(self, fallback):
        """Seconds to show: the turn's, when a turn is running."""
        return time.time() - (self.turn_t0 if self.turn_t0 is not None
                              else fallback)

    def show(self, text):
        with TERMINAL_LOCK:
            self.writing = True

            try:
                sys.stdout.write("\r\033[K" + text)
                sys.stdout.flush()
                self.pending = True
            finally:
                self.writing = False

    def clear(self):
        """Take the line back, for something that needs to keep it."""
        with TERMINAL_LOCK:
            if not self.pending:
                return

            self.writing = True

            try:
                sys.stdout.write("\r\033[K")
                sys.stdout.flush()
                self.pending = False
            finally:
                self.writing = False


STATUS = StatusLine()


class _StatusAwareStdout:
    """Anything printed takes the status line back before it prints.

    A single point of control, so no caller has to remember. The status
    line's own writes are exempt, or clearing would recurse forever.
    """

    def __init__(self, stream):
        self._stream = stream

    def write(self, text):
        if text and not STATUS.writing and STATUS.pending:
            STATUS.clear()

        return self._stream.write(text)

    def flush(self):
        return self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


sys.stdout = _StatusAwareStdout(sys.stdout)


@contextmanager
def terminal_output():
    with TERMINAL_LOCK:
        yield


class Spinner:
    """Animated spinner during LLM calls, Claude Code style:
    ✻ Thinking… (3s · ctrl+c to interrupt)"""
    FRAMES = "·✢✳✶✻✶✳✢"

    def __init__(self, label="Thinking…"):
        self.label = label
        self._stop = threading.Event()
        self._t = None
        self._t0 = 0.0
        self.tokens = 0   # live counter updated by the streaming consumer

    def __enter__(self):
        self._t0 = time.time()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

        return self

    @staticmethod
    def _fmt_tokens(n):
        # Claude-style: 999 -> "999", 1200 -> "1.2k", 12345 -> "12k"

        if n < 1000:
            return str(n)

        if n < 10000:
            return f"{n / 1000:.1f}".rstrip("0").rstrip(".") + "k"

        return f"{round(n / 1000)}k"

    @staticmethod
    def _fmt_elapsed(seconds):
        """Seconds while that stays readable, then minutes and seconds."""
        seconds = int(seconds)

        return (f"{seconds}s" if seconds < 60
                else f"{seconds // 60}m{seconds % 60:02d}s")

    def _run(self):
        i = 0

        while not self._stop.is_set():
            frame = self.FRAMES[i % len(self.FRAMES)]
            el = self._fmt_elapsed(STATUS.elapsed(self._t0))
            tok = (f" · {self._fmt_tokens(self.tokens)} tokens"
                   if self.tokens else "")

            STATUS.show(
                f"{C_ACCENT}{frame}{C_RST} {self.label} "
                f"{C_DIM}({el}{tok} · ctrl+c to interrupt){C_RST}")

            time.sleep(0.12)
            i += 1

    def __exit__(self, *exc):
        self._stop.set()

        if self._t:
            self._t.join(timeout=1)

        elapsed = self._fmt_elapsed(STATUS.elapsed(self._t0))
        tokens = (f" · {self._fmt_tokens(self.tokens)} tokens"
                  if self.tokens else "")

        # Left standing, static: no glyph, no interrupt hint, no newline.
        # It is what the harness last did, and it holds the line until the
        # next activity overwrites it or real output takes it away.
        STATUS.show(f"{C_DIM}  {self.label} {elapsed}{tokens}{C_RST}")


class SpinnerProgress:
    """The phases of a long operator command, on the spinner.

    The phase being worked on is the spinner's label, with its percentage and
    an estimate where it walks a known count; a phase that ends is left behind
    as a line of its own with how long it took. A progress line printed beside
    the spinner would have fought it for the same terminal line.
    """

    def __init__(self, spinner):
        self.spinner = spinner
        self.label = None
        self.t0 = 0.0
        self.phase = 0

    def __call__(self, label, done=None, total=None):
        now = time.time()

        if label != self.label:
            self._close(now)
            self.label, self.t0 = label, now
            self.phase += 1

        text = f"[{self.phase}] {label}"
        elapsed = now - self.t0

        if done is not None and total:
            text += f" {done * 100 // total}% ({done}/{total})"

            if 0 < done < total and elapsed >= 1:
                text += (" ~" + Spinner._fmt_elapsed(elapsed * (total - done) / done)
                         + " left")

        self.spinner.label = text

    def finish(self):
        self._close(time.time())

    def _close(self, now):
        if self.label is not None:
            with terminal_output():
                print(f"  {C_DIM}⎿ [{self.phase}] {self.label} "
                      f"{Spinner._fmt_elapsed(now - self.t0)}{C_RST}")

        self.label = None


class Liveness:
    """When the turn last did something observable.

    The wall-clock budget is COOPERATIVE: check_wall_time runs when
    something is charged -- a tool call, a model call. A turn suspended in a
    read that never returns charges nothing, so it never consults the clock.
    One ran for eleven hours inside an eight-minute budget, overnight, and
    the only reason it stopped was that I killed it.
    So the clock needs a heartbeat that does not depend on the turn's
    cooperation: tokens arriving, a result printed, a notice. The spinner is
    deliberately NOT a heartbeat -- it animates whether or not the provider
    is answering, which is precisely the case being watched for.
    """

    def __init__(self):
        self.at = time.monotonic()

    def touch(self):
        self.at = time.monotonic()

    def silent_for(self):
        return time.monotonic() - self.at


LIVENESS = Liveness()

# How long a turn may show no sign of life before it is cancelled. Generous:
# a long build or a slow first token is not a hang, and the cancellation is
# cooperative anyway -- it asks the runtime to stop at its next poll point.
TURN_LIVENESS_SECONDS = int(os.environ.get("SPEAR_TURN_LIVENESS", "300"))


class CliRuntimeObserver:
    """Keep terminal rendering outside the provider-neutral runtime."""

    @contextmanager
    def model_activity(self, label):
        LIVENESS.touch()

        with Spinner(label) as spinner:
            def tick():
                spinner.tokens += 1
                LIVENESS.touch()

            yield tick

    def intermediate_text(self, text):
        LIVENESS.touch()
        print_assistant(text)
        print()

    def notice(self, kind, metadata):
        LIVENESS.touch()

        if kind == "verification_nudge":
            print(f"  {C_DIM}↪ nothing ran after the change — "
                  f"asking for verification{C_RST}")
        elif kind == "investigation_nudge":
            print(f"  {C_DIM}↪ enough investigation — asking the model "
                  f"to conclude{C_RST}")
        elif kind == "intent_judged":
            if metadata.get("write"):
                print(f"  {C_DIM}↪ read as a request to change the code "
                      f"(the verb list did not recognise it){C_RST}")
        elif kind == "identical_result":
            print(f"  {C_DIM}↪ {metadata.get('tool')} returned a result "
                  f"already seen this turn — saying so{C_RST}")
        elif kind == "stalled_without_redirect":
            spent = metadata.get("redirects_spent", {})
            print(f"  {C_DIM}↪ looping and nothing sent it back "
                  f"(wrote={metadata.get('wrote')}, "
                  f"write_request={metadata.get('write_request')}, "
                  f"redirects={spent}){C_RST}")
        elif kind == "wall_time_wrap_up":
            print(f"  {C_DIM}↪ four fifths of the time budget spent — asking "
                  f"for a landing{C_RST}")
        elif kind == "compaction_failed":
            print(f"  {C_DIM}↪ the context could not be compacted "
                  f"({metadata.get('reason', '')[:90]}) — later rounds will "
                  f"carry the full history{C_RST}")
        elif kind == "empty_final_synthesis":
            print(f"  {C_DIM}↪ no conclusion from the model — reporting "
                  f"what was found{C_RST}")
        elif kind == "clauses_unaddressed":
            missing = metadata.get("missing") or []
            print(f"  {C_DIM}↪ {len(missing)} of "
                  f"{metadata.get('carried', 0)} established clauses not "
                  f"addressed — asking for them{C_RST}")
        elif kind == "next_work_item":
            print(f"  {C_DIM}↪ that change is finished — pointing at the next "
                  f"open requirement{C_RST}")
        elif kind == "validation_owed":
            print(f"  {C_DIM}↪ source changed and its test not written yet — "
                  f"asking for the validation{C_RST}")
        elif kind == "build_broken_work_item":
            print(f"  {C_DIM}↪ build down, nothing proved — sending it back "
                  f"to the work item that broke it{C_RST}")
        elif kind == "contract_closed":
            print(f"  {C_DIM}↪ every carried requirement closed and covered — "
                  f"asking it to land{C_RST}")
        elif kind == "plan_owed":
            print(f"  {C_DIM}↪ both sides read and nothing planned — asking "
                  f"for the plan{C_RST}")
        elif kind == "exploration_without_evidence":
            print(f"  {C_DIM}↪ the last few calls established nothing new — "
                  f"asking for a synthesis ({metadata.get('phase', '')})"
                  f"{C_RST}")
        elif kind == "write_request_unanswered":
            print(f"  {C_DIM}↪ nothing was changed — asking for the edit"
                  f"{C_RST}")
        elif kind == "project_build_failed":
            print(f"  {C_DIM}↪ the change does not survive "
                  f"`{metadata.get('command', '')[:52]}` — asking for a fix"
                  f"{C_RST}")
        elif kind == "work_order_sections_skipped":
            print(f"  {C_DIM}↪ work order sections not done "
                  f"({', '.join(metadata.get('sections') or [])}) — "
                  f"asking for them{C_RST}")
        elif kind == "standard_evidence_injected":
            # The opening retrieval and the boundary bootstrap are the same
            # call at two moments, and they must not be announced the same
            # way: the opening runs BEFORE the model has said anything, so
            # "the answer cited the standard without reading it" describes a
            # failure that has not happened -- and on a good turn it was the
            # only thing the line ever said.

            if metadata.get("origin") == "OPENING_RETRIEVAL":
                print(f"  {C_DIM}↪ reading the bound standard before "
                      f"answering — {metadata.get('tool_name')}{C_RST}")
            else:
                print(f"  {C_DIM}↪ the answer cited the standard without "
                      f"reading it — {metadata.get('tool_name')} run for it"
                      f"{C_RST}")
        elif kind == "tool_exception":
            tool_result(str(metadata.get("error_summary") or "tool failed"))
            print()


def render_md(text):
    """Light terminal markdown rendering: bold, colored inline code,
    bold headers, code blocks with a side bar."""
    out = []
    in_code = False

    for line in text.split("\n"):
        stripped = line.strip()

        if stripped.startswith("```"):
            in_code = not in_code
            continue

        if in_code:
            out.append(f"  {C_DIM}│{C_RST} {C_CODE}{line}{C_RST}")
            continue

        l = re.sub(r"^#{1,4}\s+(.*)", f"{C_BOLD}\\1{C_RST}", line)
        l = re.sub(r"\*\*(.+?)\*\*", f"{C_BOLD}\\1{C_RST}", l)
        l = re.sub(r"`([^`]+)`", f"{C_CODE}\\1{C_RST}", l)
        out.append(l)

    return "\n".join(out)



def runtime_tool_log(task_result):
    """How much the turn managed to do before it failed — one number, safely."""
    result = getattr(task_result, "agent_result", None)

    return getattr(result, "tool_log", ()) or ()


def print_assistant(text):
    """Assistant response: ⏺ bullet on the first line, indented continuation."""
    text = render_md(text.strip())

    if not text:
        print(f"{C_DIM}⏺ (no response){C_RST}")
        return

    lines = text.split("\n")
    print(f"{C_BOLD}⏺{C_RST} {lines[0]}")

    for l in lines[1:]:
        print(f"  {l}")


def tool_use(name, arg, color=C_OK):
    """Tool-use header: ⏺ Bash(command)"""
    arg = arg if vlen(arg) <= 90 else arg[:87] + "…"

    with terminal_output():
        print(f"{color}⏺{C_RST} {C_BOLD}{name}{C_RST}({arg})", flush=True)


def tool_result(text, max_lines=8):
    """Tool result indented under ⎿ , truncated Claude Code style."""

    # An empty result is a result: "" rstripped and split is [""], which is
    # truthy, so the (empty) fallback never fired and a bare ⎿ was printed.

    lines = [line for line in (text or "").rstrip().split("\n")] or ["(empty)"]

    if lines == [""]:
        lines = ["(empty)"]

    with terminal_output():
        for i, l in enumerate(lines[:max_lines]):
            pfx = "⎿  " if i == 0 else "   "
            l = l if len(l) <= 160 else l[:157] + "…"
            print(f"  {C_DIM}{pfx}{l}{C_RST}")

        if len(lines) > max_lines:
            print(f"  {C_DIM}   … +{len(lines) - max_lines} lines{C_RST}")

        sys.stdout.flush()


def show_web_sources(result, max_sources=5):
    """Claude Code-style web source listing: title + URL per result."""
    shown = 0
    first = True

    for line in result.split("\n"):
        s = line.strip()

        if s.startswith("[") and shown < max_sources:
            pfx = "⎿  " if first else "   "
            first = False
            print(f"  {C_DIM}{pfx}{s[s.index(']')+1:].strip()}{C_RST}")
        elif s.startswith("http") and shown < max_sources:
            print(f"     {C_TOOL}{s}{C_RST}")
            shown += 1

    if shown == 0:
        tool_result(result, max_lines=4)
    else:
        total = result.count("\nhttp") + result.count("    http")

        if total > max_sources:
            print(f"  {C_DIM}   … +{total - max_sources} sources{C_RST}")

def show_diff(old_text, new_text, max_lines=14):
    """Show the lines that actually CHANGED.

    Printing the head of old_text and then the head of new_text is not a diff.
    When the edit lands past the cut -- an #include inserted below six
    unchanged ones -- both sides render identically and the user reads
    "OK: updated" under a diff showing no change at all. That is unreviewable
    in auto mode, where the diff is the only thing standing between the model
    and the file.

    An edit whose sides are identical is reported as such rather than drawn as
    a change: a no-op edit is worth seeing, not hiding.
    """
    old, new = old_text.split("\n"), new_text.split("\n")
    rows = []

    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
            None, old, new, autojunk=False).get_opcodes():
        if tag == "equal":
            continue

        rows += [("-", C_ERR, l) for l in old[i1:i2]]
        rows += [("+", C_OK, l) for l in new[j1:j2]]

    if not rows:
        print(f"  {C_DIM}⎿  (no change: old and new text are identical){C_RST}")
        return

    for i, (sign, color, l) in enumerate(rows[:max_lines]):
        pfx = "⎿  " if i == 0 else "   "
        print(f"  {C_DIM}{pfx}{C_RST}{color}{sign} {l[:150]}{C_RST}")

    if len(rows) > max_lines:
        print(f"  {C_DIM}   … +{len(rows) - max_lines} more changed lines{C_RST}")


DIFF_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@", re.M)


def looks_like_diff(text):
    """True if text is a unified diff (hunk header, or ---/+++ file headers)."""

    if not text:
        return False

    if DIFF_HUNK_RE.search(text):
        return True

    return text.lstrip().startswith("--- ") and "\n+++ " in text


def render_diff(diff_text, max_lines=60):
    """Print a unified diff Claude Code style: + green, - red, @@ cyan,
    file headers and context dim."""
    lines = diff_text.split("\n")

    for n, ln in enumerate(lines):
        if n >= max_lines:
            print(f"  {C_DIM}   … (+{len(lines) - n} more diff lines){C_RST}")
            break

        pfx = "⎿  " if n == 0 else "   "
        body = ln[:200]

        if ln.startswith(("+++", "---")):
            print(f"  {C_DIM}{pfx}{body}{C_RST}")
        elif ln.startswith("@@"):
            print(f"  {C_DIM}{pfx}{C_RST}{C_TOOL}{body}{C_RST}")
        elif ln.startswith("+"):
            print(f"  {C_DIM}{pfx}{C_RST}{C_OK}{body}{C_RST}")
        elif ln.startswith("-"):
            print(f"  {C_DIM}{pfx}{C_RST}{C_ERR}{body}{C_RST}")
        else:
            print(f"  {C_DIM}{pfx}{body}{C_RST}")
