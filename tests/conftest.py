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
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from avannotate.ffmpeg import FFmpegError, video_encoder

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = REPO_ROOT / "data" / "examples"


class _Absent:
    """A meta-path finder that refuses one module and its submodules."""

    def __init__(self, name: str) -> None:
        self.name = name

    def find_spec(self, fullname: str, path: object = None, target: object = None) -> None:
        if fullname == self.name or fullname.startswith(self.name + "."):
            raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
        return None


@pytest.fixture
def absent_module() -> Callable[[str], contextmanager[None]]:
    """Make a module unimportable, whatever this machine actually has.

    Two tests assert the message a stage gives when its optional dependency is
    missing.  Both were written to "run for real" by relying on the machine not
    having it -- which holds on a development box and fails on a server, where
    the dependency is installed because the pipeline needs it.  The test that
    was meant to check the error path ends up checking nothing and failing.

    Both halves are needed: dropping it from ``sys.modules`` is not enough,
    because the next import would find it on disk again, and a meta-path finder
    alone is not enough either, because an already-imported module never
    consults one.
    """

    @contextmanager
    def _absent(name: str) -> Iterator[None]:
        saved = {
            key: value
            for key, value in sys.modules.items()
            if key == name or key.startswith(name + ".")
        }
        for key in saved:
            del sys.modules[key]
        finder = _Absent(name)
        sys.meta_path.insert(0, finder)
        try:
            yield
        finally:
            sys.meta_path.remove(finder)
            sys.modules.update(saved)

    return _absent


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
