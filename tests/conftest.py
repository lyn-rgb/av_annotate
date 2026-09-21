"""Shared fixtures.

Video fixtures are generated with ffmpeg rather than committed, so the suite
runs anywhere ffmpeg exists instead of only where the sample corpus was copied.
The generated clips are deliberately tiny -- a few hundred kilobytes and a few
seconds -- because their job is to exercise the plumbing, not the models.

The encoder is asked of ffmpeg rather than named here.  These fixtures used to
say ``libx264`` outright, which made the whole suite unrunnable on a build
without it -- and that is not a rare build: x264 is GPL, so cluster images
routinely leave it out.  The suite is meant to run wherever ffmpeg is, and
naming a codec it may not have contradicted that.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from avannotate.ffmpeg import FFmpegError, video_encoder

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = REPO_ROOT / "data" / "examples"


def _ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if executable is None:
        pytest.skip("ffmpeg is not installed")
    return executable


def _codec() -> str:
    """An encoder this ffmpeg has, skipping the test if it has none of them."""

    _ffmpeg()
    try:
        return video_encoder()
    except FFmpegError as error:
        pytest.skip(str(error))


def _generate(target: Path, *args: str) -> Path:
    result = subprocess.run(
        [_ffmpeg(), "-y", "-v", "error", *args, str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(f"could not generate fixture video: {result.stderr.strip()}")
    return target


@pytest.fixture(scope="session")
def single_shot_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Three seconds, one continuous shot, with a tone on the audio track."""

    target = tmp_path_factory.mktemp("fixtures") / "single.mp4"
    return _generate(
        target,
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=320x240:rate=25:duration=3",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=16000:duration=3",
        "-c:v",
        _codec(),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
    )


@pytest.fixture(scope="session")
def two_shot_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Two seconds of test pattern, then two seconds of colour bars.

    A hard cut that any content detector should find, which is what makes it
    useful: it distinguishes "no cuts in this video" from "the detector is
    broken".
    """

    target = tmp_path_factory.mktemp("fixtures") / "two_shot.mp4"
    return _generate(
        target,
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=320x240:rate=25:duration=2",
        "-f",
        "lavfi",
        "-i",
        "smptebars=size=320x240:rate=25:duration=2",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=16000:duration=4",
        "-filter_complex",
        "[0:v][1:v]concat=n=2:v=1:a=0[v]",
        "-map",
        "[v]",
        "-map",
        "2:a",
        "-c:v",
        _codec(),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
    )


@pytest.fixture(scope="session")
def video_without_audio(tmp_path_factory: pytest.TempPathFactory) -> Path:
    target = tmp_path_factory.mktemp("fixtures") / "silent.mp4"
    return _generate(
        target,
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=320x240:rate=25:duration=2",
        "-c:v",
        _codec(),
        "-pix_fmt",
        "yuv420p",
    )


@pytest.fixture(scope="session")
def sample_videos() -> tuple[Path, ...]:
    """The real corpus, when it has been copied next to the repo."""

    if not SAMPLE_DIR.is_dir():
        pytest.skip(f"sample corpus not present at {SAMPLE_DIR}")
    found = tuple(sorted(SAMPLE_DIR.glob("*.mp4")))
    if not found:
        pytest.skip(f"no videos in {SAMPLE_DIR}")
    return found
