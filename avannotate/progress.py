"""One line that moves, instead of a thousand that scroll past.

A stage over a thousand videos used to write two lines per video -- where it
had got to, and that it had finished -- which is two thousand lines per stage
and a log nobody reads.  What a person watching actually wants is one number:
how many videos are done, out of how many.

**Where the bar goes is the whole design.**  The driver pipes the batch through
``tee`` so the log survives a disconnect, which means stdout is a pipe even
when somebody is watching it.  A bar redrawn with carriage returns into that
pipe is a log file full of carriage returns -- the same pile of intermediate
output, in a worse format.  So the bar is drawn on ``/dev/tty``, the terminal
*under* the redirection: the person sees it move, and the log keeps only the
lines worth keeping.

When there is no terminal at all -- nohup, cron, CI -- there is nothing to
redraw, and the same numbers are written as a line every ``interval`` seconds.
A bar is a thing you look at; a line is a thing you read later, and the two
cases want different formats.

Failures are not progress and never go through here.  The bar says how many;
only a line can say which video and why, so the caller writes those and this
module only gets out of the way: :meth:`Progress.clear`.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import TextIO

#: How often a line is written when there is no terminal to draw a bar on.
DEFAULT_INTERVAL = 30.0

#: Cells between the brackets.  Fixed, rather than counted from the terminal's
#: width, so the bar does not change size when somebody resizes their window
#: mid-run -- the numbers beside it are what carry the meaning either way.
BAR_WIDTH = 24

#: What to assume when the terminal will not say how wide it is.
_FALLBACK_WIDTH = 80


def format_duration(seconds: float) -> str:
    """``1h 04m 09s``, or ``4m 09s``, or ``9s`` -- whichever fits."""

    total = int(round(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def bar(fraction: float, width: int = BAR_WIDTH) -> str:
    """``======>                 `` -- the moving part, on its own.

    ASCII on purpose.  Block-drawing characters read better and cost an
    encoding failure the first time one of these runs under ``LANG=C`` on a
    cluster: a bar that raises ``UnicodeEncodeError`` partway through a stage
    is worse than a bar made of equals signs.
    """

    if width < 1:
        raise ValueError(f"a bar needs at least one cell, not {width}")

    filled = min(max(int(fraction * width), 0), width)
    if filled >= width:
        return "=" * width
    return "=" * filled + ">" + " " * (width - filled - 1)


def open_terminal() -> TextIO | None:
    """The user's terminal, if this process has one, else ``None``.

    Not ``sys.stdout``: see the module docstring.  ``/dev/tty`` is the
    controlling terminal, which survives the redirection into ``tee`` and is
    exactly the thing a person is looking at.

    Everything that can go wrong here means "no terminal" -- a cluster job with
    no controlling tty, a platform without ``/dev/tty`` -- so the failure is
    caught rather than raised.  The caller falls back to lines.
    """

    try:
        # Pinned to UTF-8: the terminal's locale is the user's business, and a
        # video id is a filename, which is not necessarily ASCII.
        return open("/dev/tty", "w", encoding="utf-8", errors="replace")
    except OSError:
        return None


def _width_of(stream: TextIO) -> int:
    """How many cells fit, so the bar does not wrap onto a second line.

    The zero case is not hypothetical.  A pty whose window size was never set
    reports no columns at all -- ``pty.fork`` gives you one -- and truncating
    the bar to zero cells draws an empty line, which is the one thing a
    progress display must never do: it is indistinguishable from a run that has
    stopped.  Better a bar that is the wrong width than no bar.
    """

    try:
        columns = os.get_terminal_size(stream.fileno()).columns
    except (OSError, ValueError, AttributeError):
        return _FALLBACK_WIDTH
    return columns if columns > 0 else _FALLBACK_WIDTH


class Progress:
    """Progress through one stage of a corpus run.

    ``bar_stream`` decides the format: a terminal draws a bar in place, and
    ``None`` writes a line every ``interval`` seconds.  ``lines`` is where
    those lines go, and is unused when there is a terminal.
    """

    def __init__(
        self,
        *,
        title: str,
        total: int,
        lines: TextIO,
        bar_stream: TextIO | None = None,
        interval: float = DEFAULT_INTERVAL,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.title = title
        self.total = total
        self.lines = lines
        self.bar_stream = bar_stream
        self.interval = interval
        self.clock = clock

        self.done = 0
        self.succeeded = 0
        self.failed = 0

        self._started = clock()
        self._last_line_at: float | None = None
        self._drawn_width = 0

    # -- reporting ---------------------------------------------------------- #

    def advance(self, *, ok: bool) -> None:
        """One more video is finished, one way or the other."""

        self.done += 1
        if ok:
            self.succeeded += 1
        else:
            self.failed += 1

        if self.bar_stream is not None:
            self._draw(self.bar_stream)
            return

        now = self.clock()
        # The first completion always writes a line, whatever the interval: a
        # corpus small enough to finish inside one interval would otherwise
        # record no progress at all, which reads exactly like a run that did
        # nothing.
        if self._last_line_at is None or now - self._last_line_at >= self.interval:
            self._last_line_at = now
            self.lines.write(self.render() + "\n")
            self.lines.flush()

    def render(self) -> str:
        """The line, as it would appear on a terminal that is wide enough."""

        elapsed = self.clock() - self._started
        share = self.done / self.total if self.total else 1.0

        parts = [
            self.title,
            f"[{bar(share)}]",
            f"{self.done}/{self.total}",
            # Truncated, not rounded.  Rounding shows 100% at 997 of 1000, and
            # "finished" is the one thing a progress figure must not say early.
            f"{int(share * 100):3d}%",
            format_duration(elapsed),
        ]
        if 0 < self.done < self.total and elapsed > 0:
            remaining = elapsed / self.done * (self.total - self.done)
            parts.append(f"eta {format_duration(remaining)}")
        if self.failed:
            # Beside the count, not instead of it: "3 failed" beside 997 done
            # is the same as "3 failed" alone, but the pair is what tells you
            # whether the run is healthy.
            parts.append(f"failed {self.failed}")

        return "  ".join(part for part in parts if part)

    # -- drawing ------------------------------------------------------------ #

    def clear(self) -> None:
        """Erase the bar so a line can be written where it was.

        The caller owes this before anything that is not the bar -- a failure,
        a warning out of the pool -- or the two are written to the same cells
        and neither is readable.
        """

        if self.bar_stream is None or not self._drawn_width:
            return
        self.bar_stream.write("\r" + " " * self._drawn_width + "\r")
        self.bar_stream.flush()
        self._drawn_width = 0

    def close(self) -> None:
        """Leave the cursor where the next line of output belongs."""

        self.clear()

    def _draw(self, stream: TextIO) -> None:
        room = _width_of(stream)
        text = self.render()[:room]

        # Padding out to the previous length is what erases it.  The bar gets
        # shorter as the percentage loses a digit and as the eta appears, and
        # without this the tail of the longer version stays on screen.
        padded = text.ljust(min(self._drawn_width, room))
        self._drawn_width = len(text)

        stream.write("\r" + padded)
        stream.flush()
