"""Writing the face-track video the extractor is conditioned on.

The extractor's own pipeline detects faces and picks a speaker by lip motion.
That is the wrong choice for this pipeline -- we already know which face belongs
to which person, from three stages and an embedding match -- and its public API
exposes no way to say so.  Feeding it a video containing exactly one face is the
way to make the decision ours: with only one candidate it cannot pick wrong.

So the crop follows a tracked face: decode the frames, cut the box the tracker
reported (expanded, as the model's training preprocessing would), resize to what
the visual encoder takes, and hand the result to ffmpeg to encode.  Frames go
through a pipe rather than a temporary image directory -- a ten-second segment
is 250 frames, and a thousand of those is a hundred thousand files.

One frame at a time, end to end.  Ten seconds of 1080p is 1.5 GB of raw frames,
and materialising a segment before encoding it would make the stage's memory
scale with its resolution instead of with its pipe buffer.
"""

from __future__ import annotations

import subprocess
import threading
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import IO

import numpy as np
from numpy.typing import NDArray

from avannotate.asd.crop import crop_box
from avannotate.faces.frames import iter_window_frames
from avannotate.faces.types import Frame
from avannotate.ffmpeg import FFmpegError, find_ffmpeg
from avannotate.matching import Box

#: What the extractor's visual encoder takes.  Larger than ASD's 112: this one
#: reads lips for separation, not for a talking/not-talking decision.
DEFAULT_CROP_SIZE = 224

#: The crop is expanded the way the model's training preprocessing expands it.
#: TalkNet's ``cropScale``, inherited down this family of models.
DEFAULT_MARGIN = 0.40


def crop_tile(
    frame: Frame,
    box: Box | None,
    *,
    size: int = DEFAULT_CROP_SIZE,
    margin: float = DEFAULT_MARGIN,
) -> NDArray[np.uint8]:
    """One face, cut out and resized, as ``[size, size, 3]`` RGB.

    A frame with no box yields a black tile.  The extractor's input is a
    fixed-length sequence and dropping a frame would shift the rest against the
    audio -- the same reason the ASD batch pads rather than skips.
    """

    tile = np.zeros((size, size, 3), dtype=np.uint8)
    if box is None:
        return tile

    height, width = frame.shape[:2]
    cut = crop_box(box, frame_width=width, frame_height=height, margin=margin)
    if cut.area == 0:
        return tile
    patch = frame[cut.y : cut.y + cut.height, cut.x : cut.x + cut.width]
    if patch.size == 0:
        return tile

    import cv2

    return np.asarray(
        cv2.resize(patch, (size, size), interpolation=cv2.INTER_LINEAR), dtype=np.uint8
    )


def iter_tiles(
    frames: Iterable[Frame],
    boxes: Sequence[Box | None],
    *,
    size: int = DEFAULT_CROP_SIZE,
    margin: float = DEFAULT_MARGIN,
) -> Iterator[NDArray[np.uint8]]:
    """Crop each frame as it arrives, consuming the box sequence in step.

    Raises if the two run out at different times: a short read would silently
    condition the extractor's last frames on the wrong boxes, or on none.
    """

    index = 0
    for frame in frames:
        if index >= len(boxes):
            raise ValueError(
                f"more frames than boxes: frame {index} has no box to crop"
            )
        yield crop_tile(frame, boxes[index], size=size, margin=margin)
        index += 1
    if index != len(boxes):
        raise ValueError(f"more boxes than frames: {len(boxes)} boxes, {index} frames")


def _drain(stream: IO[bytes], sink: list[bytes]) -> None:
    """Consume a pipe in the background so a full buffer cannot deadlock ffmpeg."""

    for line in stream:
        sink.append(line)


def write_crop_video(
    source: Path,
    target: Path,
    *,
    width: int,
    height: int,
    start_time: float,
    frame_count: int,
    boxes: Sequence[Box | None],
    fps: float,
    size: int = DEFAULT_CROP_SIZE,
    margin: float = DEFAULT_MARGIN,
) -> Path:
    """Decode a range, crop the tracked face out of each frame, and encode."""

    if frame_count <= 0:
        raise ValueError(f"frame_count must be positive, got {frame_count}")
    if fps <= 0.0:
        raise ValueError(f"fps must be positive, got {fps}")

    target.parent.mkdir(parents=True, exist_ok=True)
    encoder = subprocess.Popen(
        [
            find_ffmpeg(),
            "-y",
            "-v",
            "error",
            "-nostdin",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{size}x{size}",
            "-r",
            f"{fps:.6f}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(target),
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert encoder.stdin is not None and encoder.stderr is not None

    # Drained in the background: a full stderr buffer would block ffmpeg, which
    # would block the write below, which would stop us reading the buffer.
    captured: list[bytes] = []
    drainer = threading.Thread(target=_drain, args=(encoder.stderr, captured), daemon=True)
    drainer.start()

    frames = iter_window_frames(
        source, width=width, height=height, start_time=start_time, count=frame_count
    )
    try:
        for tile in iter_tiles(frames, boxes, size=size, margin=margin):
            encoder.stdin.write(tile.tobytes())
    except BrokenPipeError as error:
        encoder.wait()
        detail = b"".join(captured).decode("utf-8", "replace").strip()
        raise FFmpegError(
            f"ffmpeg closed the pipe while encoding {target.name}: {detail}"
        ) from error
    finally:
        encoder.stdin.close()

    encoder.wait()
    drainer.join(timeout=5.0)
    if encoder.returncode != 0:
        detail = b"".join(captured).decode("utf-8", "replace").strip()
        raise FFmpegError(
            f"encoding {target.name} failed (exit {encoder.returncode}): {detail}"
        )
    return target
