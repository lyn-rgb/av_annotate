"""Tests for the CLI's input handling.

The piece worth testing is which files a directory scan decides are videos,
which used to be a suffix check.  A suffix is not enough: macOS writes an
AppleDouble sidecar beside every file it copies onto a filesystem that cannot
hold extended attributes, names it ``._`` plus the original, and gives it the
original's suffix and none of its bytes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

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
