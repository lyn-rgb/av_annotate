"""Reading the span of audio a segment's transcript should come from.

Thin, because :mod:`avannotate.audio.wav` already defines what "a window of
audio" means and S7 already writes one file per segment.  What is left is the
one thing that cannot be got wrong quietly: the sample rate.

A recogniser assumes 16 kHz.  Handed 8 kHz audio without being told, it does not
fail -- it transcribes the wrong frequencies against the wrong time base and
returns timestamps at half scale, so every word lands in the wrong place and
reads plausibly while doing it.  So the rate is checked here and a mismatch is
an error, not something to resample past: both sources are known to be 16 kHz
by construction, and a file that is not means something upstream has changed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from avannotate.asr.types import SegmentSource
from avannotate.audio.wav import read_info, read_window


class AsrAudioError(ValueError):
    """The audio a segment points at is not the audio a recogniser can take."""


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

    rate, _ = read_info(path)
    if rate != sample_rate:
        raise AsrAudioError(
            f"{path} is {rate} Hz but the recogniser takes {sample_rate} Hz; "
            "transcribing it anyway would put every timestamp at the wrong scale"
        )

    samples = read_window(
        path, start_seconds=source.seek, duration_seconds=source.duration
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
