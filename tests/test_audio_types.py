"""Tests for diarization turn geometry.

Overlap is the number this pipeline turns on -- it is what target-speaker
extraction exists to undo -- so the sweep that measures it gets the most
attention here.
"""

from __future__ import annotations

import pytest

from avannotate.audio.diarize import normalize_label
from avannotate.audio.types import (
    DiarizationResult,
    SpeakerTurn,
    overlap_intervals,
    speaker_overlap_seconds,
)
from avannotate.interval import Interval


def _turn(speaker: str, start: float, end: float) -> SpeakerTurn:
    return SpeakerTurn(speaker=speaker, start=start, end=end)


# --------------------------------------------------------------------------- #
# turns
# --------------------------------------------------------------------------- #


def test_a_turn_rejects_an_inverted_span() -> None:
    with pytest.raises(ValueError, match="ends before it starts"):
        _turn("spk_00", 5.0, 1.0)


def test_a_zero_length_turn_is_allowed() -> None:
    """A diarizer can emit one, and the stage's duration filter removes it."""

    assert _turn("spk_00", 2.0, 2.0).duration == 0.0


def test_turn_round_trips_through_json() -> None:
    turn = _turn("spk_00", 1.25, 3.5)
    assert SpeakerTurn.from_dict(turn.to_dict()) == turn


# --------------------------------------------------------------------------- #
# overlap
# --------------------------------------------------------------------------- #


def test_no_turns_means_no_overlap() -> None:
    assert overlap_intervals(()) == ()


def test_a_single_speaker_never_overlaps_themselves() -> None:
    """Two adjacent turns from one person are not simultaneous speech."""

    turns = [_turn("spk_00", 0.0, 1.0), _turn("spk_00", 1.0, 2.0)]
    assert overlap_intervals(turns) == ()


def test_two_disjoint_speakers_do_not_overlap() -> None:
    turns = [_turn("spk_00", 0.0, 1.0), _turn("spk_01", 2.0, 3.0)]
    assert overlap_intervals(turns) == ()


def test_touching_speakers_do_not_overlap() -> None:
    """Half-open: one ending where the next begins is a turn change, not overlap."""

    turns = [_turn("spk_00", 0.0, 1.0), _turn("spk_01", 1.0, 2.0)]
    assert overlap_intervals(turns) == ()


def test_two_speakers_at_once() -> None:
    turns = [_turn("spk_00", 0.0, 4.0), _turn("spk_01", 2.0, 6.0)]
    assert overlap_intervals(turns) == (Interval(2.0, 4.0),)


def test_nested_turns() -> None:
    turns = [_turn("spk_00", 0.0, 10.0), _turn("spk_01", 3.0, 5.0)]
    assert overlap_intervals(turns) == (Interval(3.0, 5.0),)


def test_two_separate_overlap_spans() -> None:
    turns = [
        _turn("spk_00", 0.0, 3.0),
        _turn("spk_01", 1.0, 2.0),
        _turn("spk_00", 5.0, 8.0),
        _turn("spk_01", 6.0, 7.0),
    ]
    assert overlap_intervals(turns) == (Interval(1.0, 2.0), Interval(6.0, 7.0))


def test_three_speakers_at_once_counts_once() -> None:
    turns = [
        _turn("spk_00", 0.0, 5.0),
        _turn("spk_01", 1.0, 4.0),
        _turn("spk_02", 2.0, 3.0),
    ]
    # 1-2 is two speakers, 2-3 three, 3-4 two: one contiguous contested span.
    assert overlap_intervals(turns) == (Interval(1.0, 4.0),)


def test_a_gap_inside_an_overlap_splits_it() -> None:
    turns = [
        _turn("spk_00", 0.0, 10.0),
        _turn("spk_01", 1.0, 2.0),
        _turn("spk_01", 5.0, 6.0),
    ]
    assert overlap_intervals(turns) == (Interval(1.0, 2.0), Interval(5.0, 6.0))


def test_overlap_repeated_speaker_labels_still_count_once() -> None:
    """One speaker with two overlapping turns is still one speaker."""

    turns = [_turn("spk_00", 0.0, 5.0), _turn("spk_00", 2.0, 7.0)]
    assert overlap_intervals(turns) == ()


# --------------------------------------------------------------------------- #
# the result
# --------------------------------------------------------------------------- #


def _result(*turns: SpeakerTurn) -> DiarizationResult:
    return DiarizationResult(turns=turns)


def test_speakers_are_sorted_and_distinct() -> None:
    result = _result(
        _turn("spk_02", 0.0, 1.0), _turn("spk_00", 1.0, 2.0), _turn("spk_02", 2.0, 3.0)
    )
    assert result.speakers == ("spk_00", "spk_02")


def test_speech_seconds_counts_simultaneous_speakers_once() -> None:
    result = _result(_turn("spk_00", 0.0, 4.0), _turn("spk_01", 2.0, 6.0))
    assert result.speech_seconds == pytest.approx(6.0)
    assert result.speaker_seconds == pytest.approx(8.0)


def test_overlap_ratio_is_against_speech_not_duration() -> None:
    """A quiet clip with two words over each other is not 40% overlapped."""

    result = _result(_turn("spk_00", 0.0, 1.0), _turn("spk_01", 0.5, 1.5))
    assert result.overlap_seconds == pytest.approx(0.5)
    assert result.overlap_ratio == pytest.approx(0.5 / 1.5)


def test_overlap_ratio_of_silence_is_zero() -> None:
    assert _result().overlap_ratio == 0.0


def test_clamped_trims_to_the_video_and_drops_what_falls_outside() -> None:
    """S0 established the audio runs longer than the video."""

    result = _result(
        _turn("spk_00", 0.0, 2.0),
        _turn("spk_01", 2.5, 5.1),  # ends past a 5.0s video
        _turn("spk_01", 5.5, 6.0),  # wholly past the end
    )
    clamped = result.clamped(5.0)
    assert [(t.start, t.end) for t in clamped.turns] == [(0.0, 2.0), (2.5, 5.0)]


def test_clamped_keeps_metadata() -> None:
    result = DiarizationResult(turns=(_turn("spk_00", 0.0, 9.0),), metadata={"model": "m"})
    assert result.clamped(5.0).metadata == {"model": "m"}


def test_merged_joins_one_speakers_nearby_turns() -> None:
    result = _result(
        _turn("spk_00", 0.0, 1.0),
        _turn("spk_00", 1.1, 2.0),
        _turn("spk_00", 5.0, 6.0),
        _turn("spk_01", 3.0, 4.0),
    )
    merged = result.merged(max_gap=0.2)

    spans = [(t.speaker, t.start, t.end) for t in merged.turns]
    assert spans == [
        ("spk_00", 0.0, 2.0),   # the two nearby ones joined
        ("spk_01", 3.0, 4.0),
        ("spk_00", 5.0, 6.0),   # far from the first pair, left alone
    ]
    assert [t.start for t in merged.turns] == sorted(t.start for t in merged.turns)


def test_merged_does_not_bridge_a_wide_gap() -> None:
    result = _result(_turn("spk_00", 0.0, 1.0), _turn("spk_00", 3.0, 4.0))
    assert len(result.merged(max_gap=0.2).turns) == 2


def test_result_round_trips_through_json() -> None:
    result = DiarizationResult(
        turns=(_turn("spk_00", 0.0, 3.0), _turn("spk_01", 1.0, 2.0)),
        metadata={"model": "x"},
    )
    assert DiarizationResult.from_dict(result.to_dict()) == result


def test_round_trip_is_stable_for_unsorted_turns() -> None:
    """``to_dict`` sorts, so the round trip normalises order rather than failing.

    The sort is deliberate -- it makes the file deterministic -- so the property
    worth asserting is that a second trip changes nothing.
    """

    unsorted = DiarizationResult(
        turns=(_turn("spk_01", 1.0, 2.0), _turn("spk_00", 0.0, 3.0))
    )
    once = DiarizationResult.from_dict(unsorted.to_dict())
    twice = DiarizationResult.from_dict(once.to_dict())
    assert once == twice
    assert [t.speaker for t in once.turns] == ["spk_00", "spk_01"]


def test_result_to_dict_sorts_turns() -> None:
    result = _result(_turn("spk_01", 5.0, 6.0), _turn("spk_00", 0.0, 1.0))
    turns = result.to_dict()["turns"]
    assert isinstance(turns, list)
    assert [item["speaker"] for item in turns] == ["spk_00", "spk_01"]


def test_summary_reports_the_headline_numbers() -> None:
    result = _result(_turn("spk_00", 0.0, 4.0), _turn("spk_01", 2.0, 6.0))
    summary = result.summary()
    assert summary["turns"] == 2
    assert summary["speakers"] == 2
    assert summary["speech_seconds"] == pytest.approx(6.0)
    assert summary["overlap_seconds"] == pytest.approx(2.0)


def test_speaker_pair_overlap() -> None:
    turns = [_turn("spk_00", 0.0, 4.0), _turn("spk_01", 2.0, 6.0), _turn("spk_02", 3.0, 4.0)]
    assert speaker_overlap_seconds(turns, "spk_00", "spk_01") == pytest.approx(2.0)
    assert speaker_overlap_seconds(turns, "spk_00", "spk_02") == pytest.approx(1.0)
    assert speaker_overlap_seconds(turns, "spk_00", "spk_09") == 0.0


# --------------------------------------------------------------------------- #
# labels
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (0, "spk_00"),
        ("0", "spk_00"),
        (7, "spk_07"),
        (12, "spk_12"),
        ("SPEAKER_03", "spk_SPEAKER_03"),
    ],
)
def test_labels_normalize_to_sortable_ids(raw: object, expected: str) -> None:
    assert normalize_label(raw) == expected


def test_zero_padded_labels_sort_in_numeric_order() -> None:
    assert sorted([normalize_label(10), normalize_label(2)]) == ["spk_02", "spk_10"]
