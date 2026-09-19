"""Reading and writing spans of PCM16 WAV.

One definition of "read a window of audio", because three stages need it and
they need to agree: the demuxer writes the file, ASD slices it per window, and
extraction slices its own output back to the segment.  A second implementation
would be a second rounding convention for the sample offsets, and a half-sample
disagreement between a segment and its transcript is invisible until someone
listens to both.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

#: Full scale for signed 16-bit samples.  Dividing by 32768 rather than 32767
#: keeps the mapping symmetric: -32768 and +32767 both land inside [-1, 1].
_PCM16_SCALE = 32768.0


class WavError(ValueError):
    """The file is not the PCM16 WAV this module reads and writes."""


def read_info(path: str | Path) -> tuple[int, int]:
    """``(sample_rate, frame_count)`` without decoding anything."""

    with wave.open(str(path)) as handle:
        rate = int(handle.getframerate())
        if int(handle.getsampwidth()) != 2:
            raise WavError(f"{path} is not PCM16")
        return rate, int(handle.getnframes())


def read_window(
    path: str | Path, *, start_seconds: float, duration_seconds: float
) -> NDArray[np.float32]:
    """A span as float32 in ``[-1, 1]``, clamped to the file.

    Clamping rather than raising: a caller asking for a window at the end of a
    file wants the part that exists, and the alternative is every caller
    checking the length first.
    """

    with wave.open(str(path)) as handle:
        rate = int(handle.getframerate())
        if int(handle.getsampwidth()) != 2:
            raise WavError(f"{path} is not PCM16")
        if int(handle.getnchannels()) != 1:
            raise WavError(f"{path} is not mono")

        first = int(round(max(0.0, start_seconds) * rate))
        count = max(0, int(round(duration_seconds * rate)))
        handle.setpos(min(first, handle.getnframes()))
        raw = handle.readframes(count)

    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / _PCM16_SCALE


def write_pcm16(
    path: str | Path, samples: NDArray[np.float32], *, sample_rate: int
) -> Path:
    """Write float32 in ``[-1, 1]`` as PCM16 mono.

    Clipped rather than wrapped: a sample outside the range is a processing
    artefact, and letting it wrap turns a loud passage into a burst of noise.
    """

    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm = np.round(clipped * (_PCM16_SCALE - 1.0)).astype("<i2")

    with wave.open(str(target), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return target


def slice_samples(
    samples: NDArray[np.float32], *, sample_rate: int, start: float, end: float
) -> NDArray[np.float32]:
    """The part of an in-memory buffer between two times."""

    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    first = max(0, int(round(start * sample_rate)))
    last = min(len(samples), int(round(end * sample_rate)))
    if last <= first:
        return np.zeros(0, dtype=np.float32)
    return np.asarray(samples[first:last], dtype=np.float32)
