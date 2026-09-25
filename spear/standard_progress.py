"""Progress of the long standard operations, for an operator to watch.

Ingesting a document of a few hundred pages took seconds and needed none. A
17 000-page reference manual yields some 330 000 units, and every phase after
extraction -- writing them, indexing them, embedding them -- runs over all of
them, silently, for as long as it takes.

A progress callback is `progress(label, done=None, total=None)`. Producers
call it through `report`, which rate-limits a per-unit loop to about two
hundred updates, so passing one costs nothing measurable and passing None
costs nothing at all.
"""

from __future__ import annotations

import sys
import time

#: Updates per phase, at most. Enough to move by half a percent at a time.
STEPS = 200


def report(progress, label: str, done: int | None = None,
           total: int | None = None) -> None:
    """Forward one step to `progress`, throttled; a no-op when it is None."""
    if progress is None:
        return

    if done is None or not total:
        progress(label)
        return

    step = max(1, total // STEPS)

    if done == total or done % step == 0:
        progress(label, done, total)


def _duration(seconds: float) -> str:
    seconds = int(seconds)

    if seconds < 60:
        return f"{seconds}s"

    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"

    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


class TerminalProgress:
    """One line per phase, rewritten in place: percentage, elapsed, remaining.

    The estimate is per phase, extrapolated from the phase's own rate. One
    percentage for the whole ingestion would be a guess dressed as a
    measurement: the phases differ in cost by orders of magnitude, and which
    one dominates depends on the document and on whether embedding runs.
    """

    def __init__(self, stream=None) -> None:
        self.stream = stream if stream is not None else sys.stderr
        self.label = None
        self.started = 0.0
        self.phase = 0

    def __call__(self, label: str, done: int | None = None,
                 total: int | None = None) -> None:
        now = time.monotonic()

        if label != self.label:
            if self.label is not None:
                self._write("\n")

            self.label, self.started = label, now
            self.phase += 1

        line = f"  ⎿ [{self.phase}] {label}"
        elapsed = now - self.started

        if done is not None and total:
            line += f"  {done * 100 // total:3d}%  ({done}/{total})"

            if 0 < done < total and elapsed >= 1:
                line += f"  ~{_duration(elapsed * (total - done) / done)} left"
        elif elapsed >= 1:
            line += f"  {_duration(elapsed)}"

        self._write("\r\x1b[K" + line)

    def finish(self) -> None:
        if self.label is not None:
            self._write("\n")

        self.label = None
        self.phase = 0

    def _write(self, text: str) -> None:
        try:
            self.stream.write(text)
            self.stream.flush()
        except (OSError, ValueError):
            pass
