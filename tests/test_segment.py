"""Tests for utterance segmentation."""

from __future__ import annotations

import pytest

from avannotate.interval import Interval
from avannotate.segment import (
    Segment,
    SegmentationConfig,
    merge_adjacent,
    segment_speech,
    tag_eligible,
    total_speech,
)


def test_single_run_is_padded_on_both_ends() -> None:
    segments = segment_speech([Interval(1.0, 2.0)])
    assert len(segments) == 1
    assert segments[0].start == pytest.approx(0.95)
    assert segments[0].end == pytest.approx(2.05)


def test_short_gap_is_joined_long_gap_is_not() -> None:
    close = segment_speech([Interval(0.0, 1.0), Interval(1.1, 2.0)])
    assert len(close) == 1

    far = segment_speech([Interval(0.0, 1.0), Interval(3.0, 4.0)])
    assert len(far) == 2


def test_runs_below_min_duration_are_dropped() -> None:
    # 0.2s of speech plus 0.1s of padding is still under the 0.30s floor.
    assert segment_speech([Interval(0.0, 0.2)]) == ()
    assert len(segment_speech([Interval(0.0, 0.4)])) == 1


def test_just_over_the_floor_is_kept_and_flagged_short() -> None:
    segments = segment_speech([Interval(0.0, 0.4)])
    assert segments[0].flags == ("short",)
    assert segments[0].is_low_confidence


def test_long_run_is_not_flagged() -> None:
    segments = segment_speech([Interval(0.0, 3.0)])
    assert segments[0].flags == ()


def test_vad_gates_out_speech_the_face_did_not_produce() -> None:
    """ASD can light up where there is no audio; the VAD is what removes it."""

    speaking = [Interval(0.0, 10.0)]
    vad = [Interval(2.0, 4.0)]
    segments = segment_speech(speaking, vad)
    assert len(segments) == 1
    assert segments[0].start == pytest.approx(1.95)
    assert segments[0].end == pytest.approx(4.05)


def test_vad_that_misses_everything_yields_nothing() -> None:
    assert segment_speech([Interval(0.0, 5.0)], [Interval(20.0, 25.0)]) == ()


def test_padding_never_goes_negative_or_past_the_limit() -> None:
    at_start = segment_speech([Interval(0.0, 2.0)])
    assert at_start[0].start == 0.0

    at_end = segment_speech([Interval(9.0, 10.0)], limit=10.0)
    assert at_end[0].end == 10.0


def test_padding_does_not_undo_a_split_the_segmentation_already_made() -> None:
    """max_gap is applied to the real speech; padding must not reopen the question."""

    # The 0.25s gap exceeds max_gap=0.20, so these are two utterances even
    # though padding narrows the visible gap to 0.15s.
    segments = segment_speech([Interval(0.0, 1.0), Interval(1.25, 2.0)])
    assert len(segments) == 2
    assert segments[0].end == pytest.approx(1.05)
    assert segments[1].start == pytest.approx(1.20)


def test_padding_induced_overlap_is_coalesced() -> None:
    """A large pad can make two spans cross; the result must not overlap itself."""

    config = SegmentationConfig(pad=0.3)
    segments = segment_speech([Interval(0.0, 1.0), Interval(1.25, 2.0)], config=config)
    assert len(segments) == 1
    assert segments[0].start == pytest.approx(0.0)
    assert segments[0].end == pytest.approx(2.3)


def test_negative_pad_is_rejected() -> None:
    with pytest.raises(ValueError):
        segment_speech([Interval(0.0, 1.0)], config=SegmentationConfig(pad=-0.1))


def test_config_is_honoured() -> None:
    generous = SegmentationConfig(min_duration=0.01, low_confidence_duration=0.02)
    segments = segment_speech([Interval(0.0, 0.05)], config=generous)
    assert len(segments) == 1


def test_tag_eligibility_follows_duration() -> None:
    long = Segment(start=0.0, end=1.0)
    borderline = Segment(start=0.0, end=0.5)
    short = Segment(start=0.0, end=0.3)

    assert tag_eligible(long)
    assert tag_eligible(borderline)  # the floor is inclusive
    assert not tag_eligible(short)


def test_merge_adjacent_joins_and_unions_flags() -> None:
    merged = merge_adjacent(
        [
            Segment(start=0.0, end=1.0, flags=("short",)),
            Segment(start=1.1, end=2.0),
            Segment(start=5.0, end=6.0),
        ]
    )
    assert len(merged) == 2
    assert merged[0].start == 0.0
    assert merged[0].end == pytest.approx(2.0)
    assert merged[0].flags == ("short",)


def test_merge_adjacent_handles_slight_overlap() -> None:
    """Per-segment extraction can return a length one frame off what it was given."""

    merged = merge_adjacent([Segment(start=0.0, end=1.0), Segment(start=0.98, end=2.0)])
    assert len(merged) == 1
    assert merged[0].end == pytest.approx(2.0)


def test_merge_adjacent_on_empty_input() -> None:
    assert merge_adjacent([]) == ()


def test_total_speech_sums_durations() -> None:
    assert total_speech([Segment(0.0, 1.0), Segment(2.0, 2.5)]) == pytest.approx(1.5)
