"""Decode frames through ffmpeg, one raw frame at a time.

ffmpeg rather than OpenCV's ``VideoCapture`` because frame *identity* matters
here: a face track is a list of (frame index, time, box), and OpenCV's seek is
approximate on many codecs, so a "frame 300" that is really frame 298 would
misalign every timestamp downstream by 80 ms.  Piping raw frames in order makes
the index arithmetic exact and lets the caller check it.

Streamed rather than buffered: an hour of 1080p raw frames is about 150 GB, so
:func:`iter_frames` yields as it decodes and holds one frame at a time.
"""

from __future__ import annotations

import subprocess
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import numpy as np

from avannotate.faces.types import Frame
from avannotate.ffmpeg import FFmpegError, find_ffmpeg

#: Channels in the ``rgb24`` pixel format every reader here requests.
_CHANNELS = 3


@dataclass(frozen=True)
class FrameSampling:
    """How densely to look.

    Detecting every Nth frame and letting the tracker bridge the gap is the
    standard trade: faces do not move meaningfully in 80 ms, and detection is
    the expensive half of the stage.  Stride 1 is for debugging, not for a batch.
    """

    stride: int = 3

    def __post_init__(self) -> None:
        if self.stride < 1:
            raise ValueError(f"stride must be at least 1, got {self.stride}")

    def indices(self, frame_count: int) -> tuple[int, ...]:
        """The frame indices this sampling selects from a video of that length."""

        if frame_count < 0:
            raise ValueError(f"frame_count cannot be negative, got {frame_count}")
        return tuple(range(0, frame_count, self.stride))


def _decode_command(source: Path, *, stride: int) -> list[str]:
    return [
        find_ffmpeg(),
        "-v",
        "error",
        "-nostdin",
        "-i",
        str(source),
        "-an",
        "-sn",
        "-vf",
        f"select='not(mod(n\\,{stride}))'",
        # Without passthrough ffmpeg re-times the selected frames to a constant
        # rate and duplicates them, and the count no longer matches the indices.
        "-fps_mode",
        "passthrough",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]


def _drain(stream: IO[bytes], sink: list[bytes]) -> None:
    """Consume a pipe in the background so a full buffer cannot deadlock the decode."""

    for line in stream:
        sink.append(line)


def iter_frames(
    source: Path,
    *,
    width: int,
    height: int,
    sampling: FrameSampling,
    frame_count: int,
) -> Iterator[tuple[int, Frame]]:
    """Yield ``(frame_index, rgb_array)`` for the sampled frames, in order.

    Raises if ffmpeg yields a different number of frames than the sampling
    predicts.  That check is the point of doing this in one place: a short read
    would otherwise shift every later index by one and misattribute a face to
    the wrong instant, silently.
    """

    expected = sampling.indices(frame_count)
    frame_bytes = width * height * _CHANNELS
    if frame_bytes <= 0:
        raise ValueError(f"invalid frame size {width}x{height}")

    process = subprocess.Popen(
        _decode_command(source, stride=sampling.stride),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
    )
    assert process.stdout is not None and process.stderr is not None

    captured: list[bytes] = []
    drainer = threading.Thread(target=_drain, args=(process.stderr, captured), daemon=True)
    drainer.start()

    produced = 0
    try:
        while produced < len(expected):
            chunk = process.stdout.read(frame_bytes)
            if len(chunk) < frame_bytes:
                break
            # Copy: the array is handed to a caller that keeps it, and the next
            # read must not mutate what it holds.
            frame = np.frombuffer(chunk, dtype=np.uint8).reshape(height, width, _CHANNELS).copy()
            yield expected[produced], frame
            produced += 1
    finally:
        process.stdout.close()
        process.wait()
        drainer.join(timeout=5.0)

    if produced != len(expected):
        detail = b"".join(captured).decode("utf-8", "replace").strip()
        raise FFmpegError(
            f"decoded {produced} frames from {source.name}, expected {len(expected)} "
            f"(stride {sampling.stride}, {frame_count} frames); frame indices would be wrong"
            + (f": {detail}" if detail else "")
        )


def iter_gray_window(
    source: Path, *, width: int, height: int, start_time: float, count: int
) -> Iterator[Frame]:
    """Decode ``count`` consecutive grey frames from ``start_time``."""

    return iter_window_frames(
        source, width=width, height=height, start_time=start_time, count=count, gray=True
    )


def iter_window_frames(
    source: Path,
    *,
    width: int,
    height: int,
    start_time: float,
    count: int,
    gray: bool = False,
) -> Iterator[Frame]:
    """Decode ``count`` consecutive frames from ``start_time``.

    One seek for the whole run rather than one per frame: both consumers need
    every frame in a range, and :func:`read_frame`'s per-call seek would
    dominate their runtime.

    ``gray`` asks ffmpeg for the single-channel format directly, which moves a
    third of the bytes over the pipe and skips a conversion nothing needs.  The
    ASD stages want greyscale; the extractor's visual encoder was trained on
    colour and wants ``rgb24``.
    """

    if count < 0:
        raise ValueError(f"count cannot be negative, got {count}")
    if count == 0:
        return

    channels = 1 if gray else _CHANNELS
    frame_bytes = width * height * channels
    if frame_bytes <= 0:
        raise ValueError(f"invalid frame size {width}x{height}")

    process = subprocess.Popen(
        [
            find_ffmpeg(),
            "-v",
            "error",
            "-nostdin",
            "-ss",
            f"{max(0.0, start_time):.4f}",
            "-i",
            str(source),
            "-frames:v",
            str(count),
            "-an",
            "-sn",
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray" if gray else "rgb24",
            "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
    )
    assert process.stdout is not None and process.stderr is not None

    captured: list[bytes] = []
    drainer = threading.Thread(target=_drain, args=(process.stderr, captured), daemon=True)
    drainer.start()

    produced = 0
    try:
        while produced < count:
            chunk = process.stdout.read(frame_bytes)
            if len(chunk) < frame_bytes:
                break
            shape = (height, width) if gray else (height, width, channels)
            yield np.frombuffer(chunk, dtype=np.uint8).reshape(shape).copy()
            produced += 1
    finally:
        process.stdout.close()
        process.wait()
        drainer.join(timeout=5.0)

    if produced != count:
        detail = b"".join(captured).decode("utf-8", "replace").strip()
        raise FFmpegError(
            f"decoded {produced} frames from {source.name} starting at {start_time:.3f}s, "
            f"expected {count}" + (f": {detail}" if detail else "")
        )


def iter_sampled_frames(
    source: Path,
    *,
    width: int,
    height: int,
    start_time: float,
    duration: float,
    count: int,
    gray: bool = False,
) -> Iterator[Frame]:
    """Decode ``count`` frames spread evenly across a span.

    For consumers that want the span *represented* rather than every frame of
    it -- describing a shot needs a handful of stills, not all three hundred.
    The sampling is ffmpeg's ``fps`` filter rather than a stride over decoded
    output, so the frames that are not wanted are never decoded: a twelve-second
    shot costs four frames of work here and three hundred in
    :func:`iter_window_frames`.

    The returned frames land at ``start_time + (k + 0.5) * duration / count``
    give or take the encoder's own frame grid, which is why this is for scene
    description and not for anything that has to line up with a face track.
    """

    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")
    if duration <= 0.0:
        raise ValueError(f"duration must be positive, got {duration}")

    channels = 1 if gray else _CHANNELS
    frame_bytes = width * height * channels
    if frame_bytes <= 0:
        raise ValueError(f"invalid frame size {width}x{height}")

    process = subprocess.Popen(
        [
            find_ffmpeg(),
            "-v",
            "error",
            "-nostdin",
            "-ss",
            f"{max(0.0, start_time):.4f}",
            "-t",
            f"{duration:.4f}",
            "-i",
            str(source),
            # One output frame per input interval of this length, so the count
            # asked for is the count that comes out of a span of this duration.
            "-vf",
            f"fps={count / duration:.6f}",
            "-frames:v",
            str(count),
            "-an",
            "-sn",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray" if gray else "rgb24",
            "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
    )
    assert process.stdout is not None and process.stderr is not None

    captured: list[bytes] = []
    drainer = threading.Thread(target=_drain, args=(process.stderr, captured), daemon=True)
    drainer.start()

    produced = 0
    try:
        while produced < count:
            chunk = process.stdout.read(frame_bytes)
            if len(chunk) < frame_bytes:
                break
            shape = (height, width) if gray else (height, width, channels)
            yield np.frombuffer(chunk, dtype=np.uint8).reshape(shape).copy()
            produced += 1
    finally:
        process.stdout.close()
        process.wait()
        drainer.join(timeout=5.0)

    if produced != count:
        detail = b"".join(captured).decode("utf-8", "replace").strip()
        raise FFmpegError(
            f"decoded {produced} frames from {source.name} across "
            f"{start_time:.3f}s+{duration:.3f}s, expected {count}"
            + (f": {detail}" if detail else "")
        )


def read_frame(source: Path, *, width: int, height: int, time: float) -> Frame | None:
    """Decode the single frame at ``time``.

    Used by later stages to crop a face for a reference still or a lip video.
    Returns ``None`` when the seek lands past the end rather than raising, since
    a track that runs to the last frame is ordinary and not an error.
    """

    result = subprocess.run(
        [
            find_ffmpeg(),
            "-v",
            "error",
            "-nostdin",
            "-ss",
            f"{max(0.0, time):.4f}",
            "-i",
            str(source),
            "-frames:v",
            "1",
            "-an",
            "-sn",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        capture_output=True,
        check=False,
    )
    frame_bytes = width * height * _CHANNELS
    if result.returncode != 0 or len(result.stdout) < frame_bytes:
        return None
    return (
        np.frombuffer(result.stdout[:frame_bytes], dtype=np.uint8)
        .reshape(height, width, _CHANNELS)
        .copy()
    )
