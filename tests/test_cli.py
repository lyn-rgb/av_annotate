"""Tests for the CLI's input handling.

The piece worth testing is which files a directory scan decides are videos,
which used to be a suffix check.  A suffix is not enough: macOS writes an
AppleDouble sidecar beside every file it copies onto a filesystem that cannot
hold extended attributes, names it ``._`` plus the original, and gives it the
original's suffix and none of its bytes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from avannotate import cli
from avannotate.cli import _read_inputs, is_media_file

#: What a real sidecar starts with.  Not valid video of any kind.
_APPLEDOUBLE = b"\x00\x05\x16\x07\x00\x02\x00\x00Mac OS X        "


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("clip.mp4", True),
        ("clip.mov", True),
        ("clip.mkv", True),
        ("clip.MP4", True),
        ("notes.txt", False),
        ("clip", False),
        # The one that cost four tests: the sidecar for a video called clip.mp4.
        ("._clip.mp4", False),
        ("._b0e05a7c907ea178e0254148611e67d6.mp4", False),
        (".clip.mp4", False),
        (".DS_Store", False),
    ],
)
def test_only_real_videos_are_videos(name: str, expected: bool) -> None:
    assert is_media_file(Path(name)) is expected


def test_a_scan_takes_the_video_and_leaves_the_sidecar(tmp_path: Path) -> None:
    """The production shape of the bug: a good video beside its own sidecar.

    Trusting the suffix, this returned both, and the batch then failed a
    perfectly good video with ffprobe's ``moov atom not found`` -- a message
    that reads as the video's fault and is not.
    """

    real = tmp_path / "clip.mp4"
    real.write_bytes(b"the file we mean, whatever it decodes to")
    (tmp_path / "._clip.mp4").write_bytes(_APPLEDOUBLE)
    (tmp_path / ".DS_Store").write_bytes(b"\x00\x00\x00\x01Bud1")

    assert _read_inputs(str(tmp_path), base=tmp_path) == [real]


def test_a_scan_of_only_sidecars_finds_no_videos(tmp_path: Path) -> None:
    """It has to say so, rather than hand back things ffprobe will reject."""

    (tmp_path / "._clip.mp4").write_bytes(_APPLEDOUBLE)

    with pytest.raises(SystemExit, match="no video files"):
        _read_inputs(str(tmp_path), base=tmp_path)


def test_naming_a_sidecar_explicitly_still_opens_it(tmp_path: Path) -> None:
    """Guards the guard, in the other direction.

    The filter governs what a scan picks up on its own.  Someone who names a
    path meant that path, and handing it to ffprobe gives a clearer error than
    being silently dropped from a list.
    """

    sidecar = tmp_path / "._clip.mp4"
    sidecar.write_bytes(_APPLEDOUBLE)

    assert _read_inputs(str(sidecar), base=tmp_path) == [sidecar.resolve()]


# --------------------------------------------------------------------------- #
# whose arguments they are
# --------------------------------------------------------------------------- #


def test_the_arguments_are_gone_before_the_handler_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A library calling ``parse_args()`` with no list reads ``sys.argv[1:]``.

    That is argparse's documented default, not an accident, and it is how S7
    died: ClearerVoice's AV model is built on TalkNet, whose inference code
    parses its own arguments that way, so it read our ``batch --input ...`` and
    exited 2.  ``SystemExit`` is a ``BaseException``, so ``multiprocessing``'s
    worker loop does not catch it -- the worker died and the batch hung on a
    result that was never coming.
    """

    seen: list[list[str]] = []

    def spy(value: str, *, base: Path) -> list[Path]:
        seen.append(list(sys.argv))
        raise SystemExit(0)

    monkeypatch.setattr(cli, "_read_inputs", spy)
    monkeypatch.setattr(
        sys,
        "argv",
        ["cli.py", "run", "--stage", "s0-preprocess", "--input", "x", "--output", "y"],
    )

    with pytest.raises(SystemExit):
        cli.main()

    # Truncated, not emptied: argv[0] is the program's own name.
    assert seen == [["cli.py"]]


def test_a_caller_passing_a_list_does_not_touch_the_runners_argv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guards the guard, in the direction that would break the test suite.

    Every test calls ``main([...])`` with an explicit list, and pytest's own
    ``sys.argv`` is the runner's -- truncating it here would edit the arguments
    of the process running the tests.
    """

    monkeypatch.setattr(sys, "argv", ["pytest", "-q", "tests/test_cli.py"])

    cli.main(["stages"])

    assert sys.argv == ["pytest", "-q", "tests/test_cli.py"]


# --------------------------------------------------------------------------- #
# what a batch prints
# --------------------------------------------------------------------------- #


def _stub_corpus(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the pool, and the GPU probe, with something that returns at once.

    What is under test is which callbacks ``_cmd_batch`` wires up, so this
    drives them the way ``run_corpus`` would and asserts on what came out.
    """

    from avannotate import batch as batch_module

    def run_corpus(jobs: object, **kwargs: object) -> list[object]:
        on_stage = kwargs.get("on_stage")
        on_done = kwargs.get("on_done")
        for job in jobs:  # type: ignore[union-attr]
            if callable(on_stage):
                on_stage(job.video_id, "s0-preprocess", "start")
                on_stage(job.video_id, "s0-preprocess", "run")
            if callable(on_done):
                on_done(True)
        return []

    monkeypatch.setattr(batch_module, "run_corpus", run_corpus)
    monkeypatch.setattr(batch_module, "detect_gpus", lambda: ())


def _batch_args(tmp_path: Path, *extra: str) -> object:
    (tmp_path / "clip.mp4").write_bytes(b"\x00")
    listing = tmp_path / "list.txt"
    listing.write_text("clip.mp4\n", encoding="utf-8")
    return cli.build_parser().parse_args(
        ["batch", "--input", str(listing), "--output", str(tmp_path / "out"), *extra]
    )


def test_a_batch_does_not_print_a_line_per_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two thousand lines per stage was the complaint; the bar replaces them."""

    _stub_corpus(monkeypatch)

    assert cli._cmd_batch(_batch_args(tmp_path)) == 0  # type: ignore[arg-type]

    assert "start" not in capsys.readouterr().out


def test_verbose_brings_the_lines_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The escape hatch.  A stalled bar says *that* it stopped, not *where*."""

    _stub_corpus(monkeypatch)

    assert cli._cmd_batch(_batch_args(tmp_path, "--verbose")) == 0  # type: ignore[arg-type]

    out = capsys.readouterr().out
    assert "start" in out
    assert "s0-preprocess" in out
