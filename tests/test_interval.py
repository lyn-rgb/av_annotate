"""Tests for the interval primitives.

Half-open semantics are the thing to pin: a turn ending at 3.0 and one starting
at 3.0 are adjacent, and getting that wrong inflates the overlap metrics that
gate a video.
"""

from __future__ import annotations

import pytest

from avannotate.interval import (
    Interval,
    intersect,
    merge,
    overlap_duration,
    subtract,
    total_duration,
)


def test_interval_rejects_inverted_range() -> None:
    with pytest.raises(ValueError):
        Interval(5.0, 1.0)


def test_touching_intervals_do_not_overlap() -> None:
    assert Interval(0.0, 3.0).overlap(Interval(3.0, 5.0)) == 0.0
    assert Interval(0.0, 3.0).contains(2.999)
    assert not Interval(0.0, 3.0).contains(3.0)


def test_merge_coalesces_and_respects_max_gap() -> None:
    intervals = [Interval(0.0, 1.0), Interval(1.1, 2.0), Interval(5.0, 6.0)]
    assert merge(intervals) == (Interval(0.0, 1.0), Interval(1.1, 2.0), Interval(5.0, 6.0))
    assert merge(intervals, max_gap=0.2) == (Interval(0.0, 2.0), Interval(5.0, 6.0))


def test_merge_handles_nesting_and_unsorted_input() -> None:
    intervals = [Interval(5.0, 6.0), Interval(0.0, 10.0), Interval(2.0, 3.0)]
    assert merge(intervals) == (Interval(0.0, 10.0),)


def test_total_duration_counts_overlaps_once() -> None:
    intervals = [Interval(0.0, 3.0), Interval(1.0, 5.0)]
    assert total_duration(intervals) == pytest.approx(5.0)


def test_overlap_duration_merges_both_sides_first() -> None:
    """A doubled interval on either side must not inflate the answer."""

    left = [Interval(0.0, 3.0), Interval(0.0, 3.0)]
    right = [Interval(1.0, 2.0), Interval(1.5, 4.0)]
    # Union left = [0,3]; union right = [1,4]; overlap = [1,3] = 2.0
    assert overlap_duration(left, right) == pytest.approx(2.0)


def test_intersect_finds_shared_spans() -> None:
    left = [Interval(0.0, 5.0)]
    right = [Interval(1.0, 2.0), Interval(4.0, 9.0)]
    assert intersect(left, right) == (Interval(1.0, 2.0), Interval(4.0, 5.0))


def test_intersect_of_disjoint_sets_is_empty() -> None:
    assert intersect([Interval(0.0, 1.0)], [Interval(2.0, 3.0)]) == ()


def test_subtract_removes_holes() -> None:
    assert subtract([Interval(0.0, 10.0)], [Interval(2.0, 4.0), Interval(7.0, 20.0)]) == (
        Interval(0.0, 2.0),
        Interval(4.0, 7.0),
    )


def test_subtract_with_no_holes_is_identity() -> None:
    assert subtract([Interval(0.0, 10.0)], []) == (Interval(0.0, 10.0),)


def test_subtract_handles_hole_before_and_after() -> None:
    assert subtract([Interval(5.0, 6.0)], [Interval(0.0, 1.0), Interval(9.0, 10.0)]) == (
        Interval(5.0, 6.0),
    )


def test_empty_inputs() -> None:
    assert merge([]) == ()
    assert total_duration([]) == 0.0
    assert intersect([], [Interval(0.0, 1.0)]) == ()
    assert subtract([], [Interval(0.0, 1.0)]) == ()
