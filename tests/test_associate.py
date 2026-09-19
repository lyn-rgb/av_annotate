"""Tests for the face/speaker assignment.

The solver itself is checked in test_matching.py; these are about what the
assignment means.
"""

from __future__ import annotations

import pytest

from avannotate.associate import (
    AssociationConfig,
    FaceObservation,
    SpeakerTurn,
    SpeechSample,
    assign_speakers,
)
from avannotate.interval import Interval
from avannotate.schema import OFFSCREEN_FACE_ID


def _trace(face_id: str, span: tuple[float, float], speaking: list[tuple[float, float]],
           **kwargs: object) -> FaceObservation:
    """A face visible over ``span`` with ASD probability 1 inside ``speaking``."""

    start, end = span
    samples = []
    step = 0.1
    t = start
    while t < end:
        hit = any(lo <= t < hi for lo, hi in speaking)
        samples.append(SpeechSample(time=round(t, 4), probability=1.0 if hit else 0.0))
        t += step
    intervals = tuple(kwargs.pop("visible", ())) or (Interval(start, end),)
    return FaceObservation(face_id=face_id, visible=intervals, speech=tuple(samples))


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #


def test_clear_two_speaker_assignment() -> None:
    faces = [
        _trace("F001", (0.0, 10.0), [(0.0, 3.0)]),
        _trace("F002", (0.0, 10.0), [(5.0, 8.0)]),
    ]
    turns = [
        SpeakerTurn("spkA", 0.0, 3.0),
        SpeakerTurn("spkB", 5.0, 8.0),
    ]
    result = {item.speaker: item for item in assign_speakers(faces, turns)}

    assert result["spkA"].face_id == "F001"
    assert result["spkB"].face_id == "F002"
    assert not result["spkA"].ambiguous
    assert not result["spkB"].ambiguous
    assert result["spkA"].coverage == pytest.approx(1.0)


def test_constant_talker_loses_to_the_speaker_specific_face() -> None:
    """The contrast term is what stops a always-talking face from winning everything."""

    # F001 talks the whole time; F002 talks only during spkA's turn.
    faces = [
        _trace("F001", (0.0, 10.0), [(0.0, 10.0)]),
        _trace("F002", (0.0, 10.0), [(0.0, 3.0)]),
    ]
    turns = [SpeakerTurn("spkA", 0.0, 3.0)]
    result = assign_speakers(faces, turns)[0]

    assert result.face_id == "F002"
    assert result.score == pytest.approx(1.0)  # F002: 1.0 inside, 0.0 outside
    # F001 would have scored 1.0 - 0.5 * 1.0 = 0.5.
    assert result.margin == pytest.approx(0.5)


def test_offscreen_speaker_is_not_pinned_to_a_face() -> None:
    """A speaker nobody on screen matches becomes F000, not the least-bad face."""

    faces = [_trace("F001", (0.0, 10.0), [(0.0, 1.0)])]
    turns = [SpeakerTurn("spkVoiceOver", 5.0, 8.0)]
    result = assign_speakers(faces, turns)[0]

    assert result.face_id == OFFSCREEN_FACE_ID
    assert result.is_offscreen
    assert "floor" in result.reason
    # F001 is visible during the turn but its ASD is flat zero there.
    assert result.candidate == "F001"


def test_low_coverage_face_is_ineligible() -> None:
    """A face glimpsed briefly cannot claim a long speech."""

    faces = [
        FaceObservation(
            face_id="F001",
            visible=(Interval(0.0, 0.2),),
            speech=tuple(SpeechSample(time=t / 10, probability=1.0) for t in range(2)),
        ),
        _trace("F002", (0.0, 10.0), [(0.0, 6.0)]),
    ]
    turns = [SpeakerTurn("spkA", 0.0, 6.0)]
    result = assign_speakers(faces, turns)[0]
    assert result.face_id == "F002"


def test_look_alikes_are_flagged_ambiguous() -> None:
    """Two faces that explain a speaker equally well must not be silently resolved."""

    faces = [
        _trace("F001", (0.0, 10.0), [(0.0, 3.0)]),
        _trace("F002", (0.0, 10.0), [(0.0, 3.0)]),
    ]
    turns = [SpeakerTurn("spkA", 0.0, 3.0)]
    result = assign_speakers(faces, turns)[0]

    assert result.ambiguous
    assert result.margin == pytest.approx(0.0)
    assert "margin below" in result.reason


def test_margin_can_be_relaxed_by_config() -> None:
    faces = [
        _trace("F001", (0.0, 10.0), [(0.0, 3.0)]),
        _trace("F002", (0.0, 10.0), [(2.0, 3.0)]),
    ]
    turns = [SpeakerTurn("spkA", 0.0, 3.0)]
    strict = assign_speakers(faces, turns, AssociationConfig(min_margin=0.9))[0]
    relaxed = assign_speakers(faces, turns, AssociationConfig(min_margin=0.01))[0]
    assert strict.ambiguous
    assert not relaxed.ambiguous


def test_over_segmented_diarization_shares_a_face_instead_of_going_offscreen() -> None:
    """One person split into two clusters must not have half their speech relabelled
    as narration -- which is what a strict one-to-one assignment would do, and what
    nothing downstream could detect."""

    faces = [_trace("F001", (0.0, 10.0), [(0.0, 3.0), (6.0, 9.0)])]
    turns = [SpeakerTurn("spkA", 0.0, 3.0), SpeakerTurn("spkB", 6.0, 9.0)]
    result = {item.speaker: item for item in assign_speakers(faces, turns)}

    assert result["spkA"].face_id == "F001"
    assert result["spkB"].face_id == "F001"
    assert not result["spkA"].merged
    assert result["spkB"].merged
    assert "split of the same person" in result["spkB"].reason


def test_a_second_cluster_below_the_floor_still_goes_offscreen() -> None:
    """Sharing a face is only for a speaker that genuinely matches it."""

    faces = [_trace("F001", (0.0, 10.0), [(0.0, 3.0)])]
    turns = [SpeakerTurn("spkA", 0.0, 3.0), SpeakerTurn("spkB", 6.0, 9.0)]
    result = {item.speaker: item for item in assign_speakers(faces, turns)}

    assert result["spkA"].face_id == "F001"
    assert result["spkB"].face_id == OFFSCREEN_FACE_ID
    assert not result["spkB"].merged


def test_more_speakers_than_faces_leaves_the_extras_offscreen() -> None:
    faces = [_trace("F001", (0.0, 10.0), [(0.0, 3.0)])]
    turns = [SpeakerTurn("spkA", 0.0, 3.0), SpeakerTurn("spkB", 4.0, 7.0)]
    result = {item.speaker: item for item in assign_speakers(faces, turns)}

    assert result["spkA"].face_id == "F001"
    assert result["spkB"].face_id == OFFSCREEN_FACE_ID


def test_more_faces_than_speakers_leaves_the_extras_unassigned() -> None:
    faces = [
        _trace("F001", (0.0, 10.0), [(0.0, 3.0)]),
        _trace("F002", (0.0, 10.0), []),
    ]
    turns = [SpeakerTurn("spkA", 0.0, 3.0)]
    result = assign_speakers(faces, turns)
    assert len(result) == 1
    assert result[0].face_id == "F001"


def test_no_faces_reports_every_speaker_as_offscreen() -> None:
    turns = [SpeakerTurn("spkA", 0.0, 3.0), SpeakerTurn("spkB", 4.0, 5.0)]
    result = assign_speakers([], turns)
    assert [item.speaker for item in result] == ["spkA", "spkB"]
    assert all(item.is_offscreen for item in result)
    assert "no faces" in result[0].reason


def test_no_turns_yields_nothing() -> None:
    faces = [_trace("F001", (0.0, 10.0), [(0.0, 3.0)])]
    assert assign_speakers(faces, []) == ()


def test_assignment_is_deterministic_under_input_reordering() -> None:
    """Resume and cross-run diffing depend on this."""

    faces = [
        _trace("F002", (0.0, 10.0), [(5.0, 8.0)]),
        _trace("F001", (0.0, 10.0), [(0.0, 3.0)]),
    ]
    turns = [
        SpeakerTurn("spkB", 5.0, 8.0),
        SpeakerTurn("spkA", 0.0, 3.0),
    ]
    first = assign_speakers(faces, turns)
    second = assign_speakers(list(reversed(faces)), list(reversed(turns)))
    assert [(a.speaker, a.face_id) for a in first] == [(a.speaker, a.face_id) for a in second]


@pytest.mark.parametrize("probability", [-0.1, 1.1])
def test_speech_sample_rejects_out_of_range(probability: float) -> None:
    with pytest.raises(ValueError):
        SpeechSample(time=0.0, probability=probability)
