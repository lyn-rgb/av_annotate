"""Reading the span of audio a segment's transcript should come from.

Thin, because :mod:`avannotate.audio.wav` already defines what "a window of
audio" means, refuses a file at the wrong sample rate, and S7 already writes one
file per segment.  What is left is turning a segment's relative path and its two
origins into a window.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from avannotate.asr.types import SegmentSource
from avannotate.audio.wav import WavError, read_mono


class AsrAudioError(WavError):
    """The audio a segment points at is not the audio a recogniser can take.

    A :class:`WavError` so that a caller has one exception to catch for anything
    wrong with an audio file -- a missing one, a malformed one, or one at a rate
    that would silently shift every timestamp.
    """


def read_source(
    source: SegmentSource, *, root: Path, sample_rate: int
) -> NDArray[np.float32]:
    """One segment's audio as float32 mono at ``sample_rate``.

    Clamped to the file rather than padded: a window that runs past the end is
    the normal case at the end of a video, and silence appended to reach a
    length would be transcribed as a pause that is not in the recording.
    """

    # ``Path`` every time: an absolute path is already a ``str``, and returning
    # it as one makes the check below fail on a file that is right there.
    relative = Path(source.path)
    path = relative if relative.is_absolute() else root / relative
    if not path.is_file():
        raise AsrAudioError(
            f"no audio at {path} for segment {source.segment.name} "
            f"(source: {source.source})"
        )

    samples = read_mono(
        path,
        start_seconds=source.seek,
        duration_seconds=source.duration,
        sample_rate=sample_rate,
    )
    if len(samples) == 0:
        raise AsrAudioError(
            f"reading {source.seek:.3f}s+{source.duration:.3f}s of {path} "
            f"yielded no samples for segment {source.segment.name}"
        )
    return samples


def silence(*, seconds: float, sample_rate: int) -> NDArray[np.float32]:
    """Padding for a concatenated detection sample, which is not a transcript."""

    return np.zeros(max(0, int(round(seconds * sample_rate))), dtype=np.float32)


def concat(samples: list[NDArray[np.float32]], *, gap_seconds: float, sample_rate: int
) -> NDArray[np.float32]:
    """Join samples with a short silence between each.

    The gap is not cosmetic: butting two utterances together without one gives
    the recogniser a word boundary that is not there, and the language it then
    reports is a language nobody spoke.
    """

    if not samples:
        return np.zeros(0, dtype=np.float32)
    pieces: list[NDArray[np.float32]] = []
    for index, chunk in enumerate(samples):
        if index:
            pieces.append(silence(seconds=gap_seconds, sample_rate=sample_rate))
        pieces.append(chunk)
    return np.concatenate(pieces)
