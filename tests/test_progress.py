"""Tests for the progress bar.

No terminal is involved: `Progress` takes the stream it draws on, so a
`StringIO` is a terminal that can be inspected.  What is worth pinning is which
of the two streams output goes to and when -- a bar that leaks a carriage
return into the log, or a line-based fallback that writes a line per video, is
the same complaint this module exists to answer, arriving from the other side.
"""

from __future__ import annotations

import io
import os

import pytest

from avannotate.progress import (
    BAR_WIDTH,
    DEFAULT_INTERVAL,
    Progress,
    bar,
    format_duration,
    open_terminal,
)


class _Clock:
    """A clock that only moves when a test says so."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def tick(self, seconds: float) -> None:
        self.now += seconds


class _Terminal(io.StringIO):
    """A StringIO that claims a file descriptor, so the width lookup runs.

    A plain StringIO raises on ``fileno``, which is the *other* branch of
    ``_width_of`` -- a test using one would pass whatever the width did.
    """

    def fileno(self) -> int:
        # Any descriptor at all: the size lookup is patched in the test that
        # cares, and a real one would make this depend on where it is run.
        return 1


def _lines(**kwargs: object) -> tuple[Progress, io.StringIO, _Clock]:
    """A line-mode progress and the stream it writes to."""

    clock = _Clock()
    stream = io.StringIO()
    progress = Progress(title="s3-cluster", total=4, lines=stream, clock=clock, **kwargs)  # type: ignore[arg-type]
    return progress, stream, clock


def _bar(**kwargs: object) -> tuple[Progress, io.StringIO, io.StringIO, _Clock]:
    """A bar-mode progress, with the terminal and the log kept apart."""

    clock = _Clock()
    terminal = io.StringIO()
    lines = io.StringIO()
    progress = Progress(
        title="s3-cluster",
        total=4,
        lines=lines,
        bar_stream=terminal,
        clock=clock,
        **kwargs,  # type: ignore[arg-type]
    )
    return progress, terminal, lines, clock


# --------------------------------------------------------------------------- #
# the bar itself
# --------------------------------------------------------------------------- #


def test_the_bar_fills_left_to_right() -> None:
    assert bar(1.0, 10) == "=" * 10
    assert bar(0.0, 10) == ">" + " " * 9
    assert bar(0.5, 10) == "=" * 5 + ">" + " " * 4


def test_the_bar_is_always_exactly_its_width() -> None:
    """A bar one cell too long wraps and the in-place redraw comes apart."""

    for fraction in (-1.0, 0.0, 0.001, 0.25, 0.999, 1.0, 2.0):
        assert len(bar(fraction, 10)) == 10


def test_the_bar_is_ascii_and_defaults_to_its_advertised_width() -> None:
    """Block characters would raise UnicodeEncodeError under LANG=C."""

    default = bar(0.5)

    default.encode("ascii")
    assert len(default) == BAR_WIDTH


def test_a_bar_needs_room() -> None:
    with pytest.raises(ValueError):
        bar(0.5, 0)


def test_durations_read_as_durations() -> None:
    assert format_duration(9) == "9s"
    assert format_duration(69) == "1m 09s"
    assert format_duration(3661) == "1h 01m 01s"


# --------------------------------------------------------------------------- #
# with a terminal
# --------------------------------------------------------------------------- #


def test_a_terminal_gets_the_bar_and_the_log_gets_nothing() -> None:
    """The point of drawing on /dev/tty: the log stays free of redraws."""

    progress, terminal, lines, _ = _bar()

    progress.advance(ok=True)

    assert lines.getvalue() == ""
    assert terminal.getvalue() == "\r" + progress.render()


def test_the_bar_is_redrawn_in_place() -> None:
    progress, terminal, _, _ = _bar()

    progress.advance(ok=True)
    progress.advance(ok=True)

    assert terminal.getvalue().count("\r") == 2
    assert "\n" not in terminal.getvalue(), "a newline would scroll the bar away"


def test_a_shorter_bar_is_padded_over_the_longer_one() -> None:
    """Otherwise the tail of the previous draw stays on screen."""

    progress, terminal, _, _ = _bar()

    progress.advance(ok=True)
    longest = max(len(part) for part in terminal.getvalue().split("\r"))

    terminal.seek(0)
    terminal.truncate()
    progress.advance(ok=True)

    assert len(terminal.getvalue()) - 1 >= longest


def test_clear_erases_exactly_what_was_drawn() -> None:
    progress, terminal, _, _ = _bar()
    progress.advance(ok=True)

    drawn = len(terminal.getvalue()) - 1  # minus the leading carriage return
    terminal.seek(0)
    terminal.truncate()

    progress.clear()

    assert terminal.getvalue() == "\r" + " " * drawn + "\r"


def test_a_terminal_that_reports_no_width_still_gets_a_bar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pty whose window size was never set reports zero columns.

    Truncating the bar to zero cells draws an empty line, which is the one
    thing a progress display must not do: it cannot be told apart from a run
    that has stopped.  Found by running one under a real pty.
    """

    terminal = _Terminal()
    progress = Progress(
        title="s3-cluster",
        total=4,
        lines=io.StringIO(),
        bar_stream=terminal,
        clock=_Clock(),
    )
    monkeypatch.setattr(os, "get_terminal_size", lambda fd: os.terminal_size((0, 0)))

    progress.advance(ok=True)

    drawn = terminal.getvalue()[1:]  # minus the carriage return
    assert drawn.strip(), "the bar drew nothing at all"
    assert "1/4" in drawn


def test_clear_before_anything_was_drawn_writes_nothing() -> None:
    progress, terminal, _, _ = _bar()

    progress.clear()

    assert terminal.getvalue() == ""


def test_close_clears_the_bar() -> None:
    """So the next line printed starts at the margin, not over the bar."""

    progress, terminal, _, _ = _bar()
    progress.advance(ok=True)
    terminal.seek(0)
    terminal.truncate()

    progress.close()

    assert terminal.getvalue().startswith("\r")
    assert terminal.getvalue().strip() == ""


# --------------------------------------------------------------------------- #
# without one
# --------------------------------------------------------------------------- #


def test_without_a_terminal_the_first_video_always_writes_a_line() -> None:
    """A corpus small enough to finish inside one interval must not be silent."""

    progress, lines, _ = _lines()

    progress.advance(ok=True)
    progress.advance(ok=True)

    assert lines.getvalue().count("\n") == 1


def test_without_a_terminal_a_line_arrives_every_interval() -> None:
    progress, lines, clock = _lines()
    progress.advance(ok=True)

    clock.tick(DEFAULT_INTERVAL)
    progress.advance(ok=True)

    assert lines.getvalue().count("\n") == 2


def test_without_a_terminal_nothing_is_written_between_intervals() -> None:
    progress, lines, clock = _lines()
    progress.advance(ok=True)

    # Four videos inside 0.6 of an interval: several completions, one line.
    for _ in range(3):
        clock.tick(DEFAULT_INTERVAL / 5)
        progress.advance(ok=True)

    assert lines.getvalue().count("\n") == 1


def test_without_a_terminal_there_is_no_carriage_return() -> None:
    progress, lines, clock = _lines()
    progress.advance(ok=True)
    clock.tick(DEFAULT_INTERVAL)
    progress.advance(ok=True)

    assert "\r" not in lines.getvalue()


# --------------------------------------------------------------------------- #
# what the line says
# --------------------------------------------------------------------------- #


def test_the_line_carries_the_count_and_the_stage() -> None:
    progress, _, _ = _lines()

    progress.advance(ok=True)
    progress.advance(ok=True)

    assert progress.render().startswith("s3-cluster")
    assert "2/4" in progress.render()
    assert "50%" in progress.render()


def test_the_line_says_how_many_failed_beside_how_many_did_not() -> None:
    """`failed 1` alone does not say whether the other 999 worked."""

    progress, _, _ = _lines()
    progress.advance(ok=True)
    progress.advance(ok=False)

    text = progress.render()

    assert "failed 1" in text
    assert "2/4" in text
    assert progress.succeeded == 1
    assert progress.failed == 1


def test_a_run_with_no_failures_does_not_mention_failures() -> None:
    progress, _, _ = _lines()
    progress.advance(ok=True)

    assert "failed" not in progress.render()


def test_the_percentage_does_not_say_100_before_it_is() -> None:
    """997 of 1000 rounds to 100, and "finished" is the one thing a progress
    figure must not say early."""

    progress = Progress(
        title="s3-cluster", total=1000, lines=io.StringIO(), clock=_Clock()
    )
    for _ in range(997):
        progress.advance(ok=True)

    text = progress.render()

    assert "997/1000" in text
    assert " 99%" in text
    assert "100%" not in text


def test_the_percentage_says_100_at_the_end() -> None:
    progress = Progress(
        title="s3-cluster", total=1000, lines=io.StringIO(), clock=_Clock()
    )
    for _ in range(1000):
        progress.advance(ok=True)

    assert " 100%" in progress.render()


def test_the_eta_appears_only_between_the_ends() -> None:
    """At the start there is nothing to extrapolate from, and at the end no rest."""

    progress, lines, clock = _lines()

    assert "eta" not in progress.render()

    clock.tick(60)
    progress.advance(ok=True)
    assert "eta" in progress.render()

    for _ in range(3):
        progress.advance(ok=True)
    assert "eta" not in progress.render()


# --------------------------------------------------------------------------- #
# finding the terminal
# --------------------------------------------------------------------------- #


def test_looking_for_a_terminal_never_raises() -> None:
    """Whether there is one is the machine's business; failing is not.

    This runs both ways depending on where the suite is started from -- with a
    controlling terminal under a shell, without one under CI -- and the branch
    that matters is the one that catches the OSError rather than propagating it
    into the middle of a stage.
    """

    stream = open_terminal()

    if stream is not None:
        stream.write("\r")
        stream.close()
