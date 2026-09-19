"""Half-open time intervals and the set operations both S8 and S10 need.

Kept separate from :mod:`avannotate.associate` because segmentation needs the
same primitives without wanting the matching logic, and separate from
:mod:`avannotate.schema` because none of this appears in the deliverable.

Every interval is ``[start, end)`` in seconds.  Half-open matters here: a
speaker turn ending at 3.0 and another starting at 3.0 are adjacent, not
overlapping, and treating them as overlapping would inflate the overlap metrics
that gate a video.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class Interval:
    start: float
    end: float

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"interval ends before it starts: {self.start} > {self.end}")

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def overlap(self, other: Interval) -> float:
        """Length of the shared span; zero when they only touch."""

        return max(0.0, min(self.end, other.end) - max(self.start, other.start))

    def contains(self, time: float) -> bool:
        return self.start <= time < self.end


def merge(intervals: Iterable[Interval], *, max_gap: float = 0.0) -> tuple[Interval, ...]:
    """Sort and coalesce, joining intervals separated by less than ``max_gap``."""

    ordered = sorted(intervals)
    if not ordered:
        return ()

    merged: list[Interval] = [ordered[0]]
    for interval in ordered[1:]:
        last = merged[-1]
        if interval.start - last.end <= max_gap:
            merged[-1] = Interval(last.start, max(last.end, interval.end))
        else:
            merged.append(interval)
    return tuple(merged)


def total_duration(intervals: Sequence[Interval]) -> float:
    """Time covered, counting overlaps once."""

    return sum(interval.duration for interval in merge(intervals))


def overlap_duration(left: Sequence[Interval], right: Sequence[Interval]) -> float:
    """Total shared time between two interval sets, counting overlaps once per set.

    Both sides are merged first: a doubled interval on either side (a diarization
    segment listed twice, two VAD regions that touch) must not inflate the answer.
    """

    merged_left = merge(left)
    merged_right = merge(right)
    return sum(a.overlap(b) for a in merged_left for b in merged_right)


def intersect(left: Sequence[Interval], right: Sequence[Interval]) -> tuple[Interval, ...]:
    """Every maximal span covered by both sets."""

    result: list[Interval] = []
    for a in merge(left):
        for b in merge(right):
            start = max(a.start, b.start)
            end = min(a.end, b.end)
            if end > start:
                result.append(Interval(start, end))
    return merge(result)


def subtract(left: Sequence[Interval], right: Sequence[Interval]) -> tuple[Interval, ...]:
    """The parts of ``left`` not covered by ``right``."""

    result: list[Interval] = []
    holes = merge(right)
    for interval in merge(left):
        cursor = interval.start
        for hole in holes:
            if hole.end <= cursor:
                continue
            if hole.start >= interval.end:
                break
            if hole.start > cursor:
                result.append(Interval(cursor, hole.start))
            cursor = max(cursor, hole.end)
        if cursor < interval.end:
            result.append(Interval(cursor, interval.end))
    return tuple(result)
