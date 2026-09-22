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

**The sound comes along.**  The model is audio-visual: the crop says which face,
and the audio is the thing being separated, so the segment's own sound is muxed
into the file alongside the frames.  A crop without it is one the extractor can
do nothing with -- and it reports that by going looking for a per-track wav that
was never written, which names a missing file rather than a missing audio track.
This wrote ``-an`` for a while, so every crop was silent.

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
from avannotate.ffmpeg import FFmpegError, find_ffmpeg, video_encoder
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


#: Quality flags per codec, for the ones :func:`video_encoder` can return.
#:
#: CRF where the encoder has a quality scale this module is sure of -- x264 and
#: x265, where 18 and 20 are the visually-lossless-ish settings.  Bitrate
#: everywhere else, because ``-b:v`` is generic ffmpeg and every encoder takes
#: it, while the per-encoder quality knobs differ in both name and meaning:
#: nvenc's ``-cq`` and its ``-preset`` values changed names across versions, and
#: guessing them wrong is not an error ffmpeg is obliged to raise.
#:
#: 4 Mbps is deliberately generous.  These are 224x224 clips of one face, so it
#: is far more than the resolution needs -- the point is to be sure the setting
#: is not what limits the lip detail the extractor reads.
_QUALITY_ARGS = {
    "libx264": ("-preset", "veryfast", "-crf", "18"),
    "libx265": ("-preset", "veryfast", "-crf", "20"),
    "libopenh264": ("-b:v", "4M"),
    "h264_nvenc": ("-b:v", "4M"),
    "mpeg4": ("-qscale:v", "3"),
}


def _quality_args(codec: str) -> tuple[str, ...]:
    """The quality flags ``codec`` understands, empty if it is not a known one.

    Empty rather than guessed: an unknown encoder is one whose options this
    module has not been told about, and ffmpeg's defaults are a better answer
    than another codec's flags.
    """

    return _QUALITY_ARGS.get(codec, ())


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
    codec = video_encoder()
    encoder = subprocess.Popen(
        [
            find_ffmpeg(),
            "-y",
            "-v",
            "error",
            "-nostdin",
            # Input 0: the cropped face, one raw frame at a time down the pipe.
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
            # Input 1: the original, for its audio and nothing else.
            #
            # The extractor is **audio-visual**, and that is the whole design of
            # this stage: the crop says *which* face, and the sound is the thing
            # being separated.  A crop with no soundtrack is one it can do
            # nothing with -- and it says so a long way from here, by going
            # looking for the per-track wav it means to write from this audio:
            #
            #     FileNotFoundError: '.../F001_0000/py_faceTracks/00000.wav'
            #
            # Nothing in that names a missing audio track.  This used to pass
            # ``-an``, so every crop was silent and S7 could never have worked.
            "-ss",
            f"{max(0.0, start_time):.4f}",
            "-t",
            f"{frame_count / fps:.4f}",
            "-i",
            str(source),
            "-map",
            "0:v",
            # ``?`` so that a source with no audio is still a source: the crop
            # comes out silent, and the extractor's own complaint about that is
            # clearer than anything raised here would be.
            "-map",
            "1:a?",
            "-c:v",
            codec,
            # Quality settings are per-codec, not per-encoder-role: -crf and
            # -preset are libx264's, and mpeg4 rejects both in favour of
            # -qscale:v.  Passing x264's flags to mpeg4 is not an error ffmpeg
            # is obliged to raise, so getting this wrong could quietly encode
            # at the default quality instead of the one asked for.
            *_quality_args(codec),
            "-c:a",
            "aac",
            "-b:a",
            "128k",
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
