"""Combining the per-window predictions into one trace per tracklet.

Every frame near a window boundary is scored with half its context, so its
prediction is worse than one made from the middle of a window.  Overlapping the
windows and averaging where they cover the same frame is what recovers that: a
frame at the edge of one window sits in the interior of its neighbour, and the
average is dominated by the better-informed of the two.

Averaging rather than picking one: both predictions are evidence, and a rule
that prefers the interior would have to define "interior" by another arbitrary
threshold.  The cost is that a genuine onset -- a face starting to talk exactly
at a window edge -- is smoothed across two frames, which the segmentation stage
downstream does not care about.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from avannotate.asd.types import SpeakingSample, Window


def stitch_predictions(
    predictions: Sequence[tuple[Window, NDArray[np.float32]]], *, fps: float
) -> tuple[SpeakingSample, ...]:
    """Average overlapping window predictions into one series.

    Each entry is a window and that window's per-frame probabilities for one
    track, in frame order.  The result is in time order, one sample per frame
    that any window covered.
    """

    if fps <= 0.0:
        raise ValueError(f"fps must be positive, got {fps}")

    totals: dict[int, float] = {}
    counts: dict[int, int] = {}

    for window, probabilities in predictions:
        values = np.asarray(probabilities, dtype=np.float64).reshape(-1)
        if len(values) != window.frame_count:
            # A silent mismatch would shift every later sample by the
            # difference, attributing speech to the wrong instant.
            raise ValueError(
                f"window {window.index} covers {window.frame_count} frames but "
                f"{len(values)} predictions were given; the two would not align"
            )
        for offset, value in enumerate(values):
            frame = window.start_frame + offset
            totals[frame] = totals.get(frame, 0.0) + float(value)
            counts[frame] = counts.get(frame, 0) + 1

    return tuple(
        SpeakingSample(time=frame / fps, probability=totals[frame] / counts[frame])
        for frame in sorted(totals)
    )


def mean_probability(samples: Sequence[SpeakingSample]) -> float | None:
    """Mean over a series, or ``None`` when it is empty."""

    if not samples:
        return None
    return sum(sample.probability for sample in samples) / len(samples)
