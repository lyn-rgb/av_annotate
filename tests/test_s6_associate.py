"""Tests for stage S6.

Unlike S4 and S5 this stage is fully testable here: the matching algorithm is
pure and has its own tests, and S3's, S4's and S5's artifacts are plain files
that this stage only reads.  So the tests write those files directly, which
lets each case vary exactly one thing -- who is off-screen, who was
over-segmented -- without running three models to get there.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from avannotate.asd.types import SpeakingSample
from avannotate.audio.types import SpeakerTurn
from avannotate.faces.types import Detection
from avannotate.schema import OFFSCREEN_FACE_ID
from avannotate.stages import (
    s0_preprocess,
    s1_faces,
    s2_tracks,
    s3_cluster,
    s4_diarize,
    s5_asd,
    s6_associate,
)
from avannotate.stages.base import StageContext


class _TwoFaceDetector:
    """Two faces, both on screen for the whole clip."""

    name = "stub"
    provides_embeddings = True

    def __init__(self) -> None:
        self.calls = 0

    def detect(self, frame: np.ndarray) -> tuple[Detection, ...]:
        self.calls += 1
        first = np.zeros(16, dtype=np.float32)
        first[0] = 1.0
        second = np.zeros(16, dtype=np.float32)
        second[1] = 1.0
        drift = 0.2 * self.calls
        return (
            Detection(x=100.0 + drift, y=100.0, width=40.0, height=50.0, score=0.9,
                      embedding=tuple(float(v) for v in first)),
            Detection(x=400.0 + drift, y=100.0, width=40.0, height=50.0, score=0.9,
                      embedding=tuple(float(v) for v in second)),
        )


def _context(source: Path, root: Path, **config: object) -> StageContext:
    return StageContext(
        video_id=source.stem, source=source, work_dir=root / "work" / source.stem,
        config=config,
    )


def _write_identities(context: StageContext, mapping: dict[str, list[int]]) -> None:
    path = context.output(s3_cluster.STAGE, s3_cluster.IDENTITIES_NAME)
    path.write_text(
        json.dumps(
            {
                "schema_version": "avannotate-identities-v1",
                "config": {},
                "identities": [
                    {"face_id": face_id, "track_ids": track_ids}
                    for face_id, track_ids in mapping.items()
                ],
                "dropped": [],
            }
        ),
        encoding="utf-8",
    )


def _write_turns(context: StageContext, turns: list[tuple[str, float, float]]) -> None:
    path = context.output(s4_diarize.STAGE, s4_diarize.TURNS_NAME)
    with path.open("w", encoding="utf-8") as handle:
        for speaker, start, end in turns:
            handle.write(
                json.dumps({"speaker": speaker, "start": start, "end": end}) + "\n"
            )


def _write_speaking(
    context: StageContext, traces: dict[int, list[tuple[float, float]]]
) -> None:
    path = context.output(s5_asd.STAGE, s5_asd.SPEAKING_NAME)
    with path.open("w", encoding="utf-8") as handle:
        for track_id, samples in traces.items():
            handle.write(
                json.dumps(
                    {
                        "track_id": track_id,
                        "samples": [
                            {"time": time, "probability": probability}
                            for time, probability in samples
                        ],
                    }
                )
                + "\n"
            )


def _ramp(start: float, end: float, *, step: float, value: float) -> list[tuple[float, float]]:
    times = np.arange(start, end, step)
    return [(float(t), value) for t in times]


@pytest.fixture
def staged(single_shot_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """S0-S2 run for real; S3, S4 and S5's artifacts written by the caller."""

    context = _context(single_shot_video, tmp_path, stride=1)
    s0_preprocess.run(context)
    monkeypatch.setattr(s1_faces, "build_detector", lambda _: _TwoFaceDetector())
    s1_faces.run(context)
    s2_tracks.run(context)

    timeline = s0_preprocess.load_timeline(context)
    return context, timeline


# --------------------------------------------------------------------------- #
# adaptation
# --------------------------------------------------------------------------- #


def test_algorithm_turns_maps_every_field() -> None:
    """The two turn types are kept separate on purpose, so the mapping between
    them is asserted rather than assumed."""

    diarizer_turns = (SpeakerTurn("spk_00", 1.25, 3.5), SpeakerTurn("spk_01", 4.0, 5.0))
    adapted = s6_associate.algorithm_turns(diarizer_turns)

    assert [(t.speaker, t.start, t.end) for t in adapted] == [
        ("spk_00", 1.25, 3.5),
        ("spk_01", 4.0, 5.0),
    ]
    assert adapted[0].interval.start == 1.25


def test_face_observations_merges_tracklets_into_one_person(
    staged: Any,
) -> None:
    context, _ = staged
    tracklets = s2_tracks.load_tracklets(context)
    # Pretend the two tracklets are the same person.
    identity_tracks = {"F001": [t.track_id for t in tracklets]}
    speaking = s5_asd.AsdResult(tracks=())

    observations = s6_associate.face_observations(
        identity_tracks, tracklets, speaking, s6_associate.S6Config()
    )
    assert len(observations) == 1
    assert observations[0].face_id == "F001"
    assert observations[0].visible, "the merged presence should not be empty"


def test_face_observations_carries_the_speaking_trace(staged: Any) -> None:
    context, _ = staged
    tracklets = s2_tracks.load_tracklets(context)
    speaking = s5_asd.AsdResult(
        tracks=(
            s5_asd.TrackSpeaking(
                track_id=1, samples=(SpeakingSample(0.5, 0.9),)
            ),
        )
    )
    observations = s6_associate.face_observations(
        {"F001": [1]}, tracklets, speaking, s6_associate.S6Config()
    )
    assert [s.time for s in observations[0].speech] == [0.5]
    assert observations[0].speech[0].probability == pytest.approx(0.9)


def test_face_observations_ignores_an_unknown_tracklet(staged: Any) -> None:
    context, _ = staged
    tracklets = s2_tracks.load_tracklets(context)
    observations = s6_associate.face_observations(
        {"F001": [999]}, tracklets, s5_asd.AsdResult(), s6_associate.S6Config()
    )
    assert observations[0].visible == ()
    assert observations[0].speech == ()


def test_speaking_intervals_unions_an_identitys_speakers() -> None:
    from avannotate.associate import SpeakerAssignment

    turns = [
        SpeakerTurn("spk_00", 0.0, 1.0),
        SpeakerTurn("spk_01", 1.1, 2.0),
        SpeakerTurn("spk_02", 5.0, 6.0),
    ]
    assignments = [
        SpeakerAssignment("spk_00", "F001", 0.9, 0.5, 1.0, "F001", False, "ok"),
        SpeakerAssignment("spk_01", "F001", 0.8, 0.4, 1.0, "F001", False, "ok", merged=True),
        SpeakerAssignment("spk_02", "F002", 0.9, 0.5, 1.0, "F002", False, "ok"),
    ]
    intervals = s6_associate.speaking_intervals(
        "F001", assignments, turns, gap=0.2
    )
    # The two speakers' turns are adjacent and merge into one span.
    assert [(round(i.start, 3), round(i.end, 3)) for i in intervals] == [(0.0, 2.0)]


def test_speaking_intervals_of_an_unassigned_face_is_empty() -> None:
    assert s6_associate.speaking_intervals("F009", [], [], gap=0.2) == ()


# --------------------------------------------------------------------------- #
# the stage
# --------------------------------------------------------------------------- #


def _two_people(staged: Any, *, turns: list[tuple[str, float, float]],
                traces: dict[int, list[tuple[float, float]]],
                identities: dict[str, list[int]] | None = None) -> StageContext:
    context, timeline = staged
    _write_identities(context, identities or {"F001": [1], "F002": [2]})
    _write_turns(context, turns)
    _write_speaking(context, traces)
    return context


def test_two_faces_and_two_speakers_match(staged: Any) -> None:
    """Track 1 is the left face at x=100; track 2 the right at x=400.  Speaker
    A talks first, speaker B second, and the ASD traces say which face was
    talking when."""

    context = _two_people(
        staged,
        turns=[("spk_00", 0.0, 1.5), ("spk_01", 1.5, 3.0)],
        traces={
            1: _ramp(0.0, 1.5, step=0.1, value=0.95) + _ramp(1.5, 3.0, step=0.1, value=0.02),
            2: _ramp(0.0, 1.5, step=0.1, value=0.02) + _ramp(1.5, 3.0, step=0.1, value=0.95),
        },
    )
    result = s6_associate.run(context)

    assert not result.skipped
    assert result.summary["assigned"] == 2
    assert result.summary["offscreen"] == 0

    mapping = {
        str(item["speaker"]): str(item["face_id"])
        for item in s6_associate.load_assignments(context)
    }
    assert mapping == {"spk_00": "F001", "spk_01": "F002"}


def test_a_speaker_no_face_matches_becomes_offscreen(staged: Any) -> None:
    """Narration, a phone call, someone out of frame."""

    context = _two_people(
        staged,
        turns=[("spk_00", 0.0, 1.5), ("spk_99", 1.6, 3.0)],
        traces={
            # Neither face is scored as talking during spk_99's turn.
            1: _ramp(0.0, 1.5, step=0.1, value=0.95) + _ramp(1.5, 3.0, step=0.1, value=0.0),
            2: _ramp(0.0, 3.0, step=0.1, value=0.0),
        },
    )
    s6_associate.run(context)

    mapping = {
        str(item["speaker"]): str(item["face_id"])
        for item in s6_associate.load_assignments(context)
    }
    assert mapping["spk_00"] == "F001"
    assert mapping["spk_99"] == OFFSCREEN_FACE_ID

    payload = json.loads(s6_associate.assignments_path(context).read_text())
    assert payload["offscreen"]["speakers"] == ["spk_99"]
    assert payload["offscreen"]["seconds"] == pytest.approx(1.4)


def test_an_over_segmented_speaker_shares_its_face(staged: Any) -> None:
    """One person split into two diarization clusters must not have half their
    speech relabelled as narration.

    The shape needs more speakers than faces: with a face each, the assignment
    seats everybody and the merge path is never reached.
    """

    context = _two_people(
        staged,
        identities={"F001": [1]},
        turns=[("spk_00", 0.0, 1.5), ("spk_01", 1.5, 3.0)],
        # Both clusters belong to the one face on screen.
        traces={1: _ramp(0.0, 3.0, step=0.1, value=0.95)},
    )
    result = s6_associate.run(context)

    faces = {
        str(item["speaker"]): str(item["face_id"])
        for item in s6_associate.load_assignments(context)
    }
    assert faces == {"spk_00": "F001", "spk_01": "F001"}
    assert result.summary["merged"] == 1
    assert result.summary["offscreen"] == 0


def test_both_faces_talking_at_once_is_not_an_error(staged: Any) -> None:
    """Simultaneous speech is the case the design exists for; both faces
    scoring high is correct, not a conflict."""

    context = _two_people(
        staged,
        turns=[("spk_00", 0.0, 3.0), ("spk_01", 0.0, 3.0)],
        traces={
            1: _ramp(0.0, 3.0, step=0.1, value=0.95),
            2: _ramp(0.0, 3.0, step=0.1, value=0.95),
        },
    )
    result = s6_associate.run(context)
    assert result.summary["assigned"] == 2
    assert result.summary["merged"] == 0


def test_a_face_nobody_claims_is_counted(staged: Any) -> None:
    """On screen and silent throughout -- a listener, or set dressing."""

    context = _two_people(
        staged,
        turns=[("spk_00", 0.0, 1.5)],
        traces={
            1: _ramp(0.0, 1.5, step=0.1, value=0.95),
            2: _ramp(0.0, 3.0, step=0.1, value=0.0),
        },
    )
    s6_associate.run(context)

    summary = json.loads((context.work_dir / "s6-associate" / "summary.json").read_text())
    assert summary["counts"]["unclaimed_faces"] == 1


def test_identity_speech_intervals_are_written(staged: Any) -> None:
    """What S7 extracts audio for."""

    context = _two_people(
        staged,
        turns=[("spk_00", 0.0, 1.5)],
        traces={1: _ramp(0.0, 1.5, step=0.1, value=0.95), 2: _ramp(0.0, 3.0, step=0.1, value=0.0)},
    )
    s6_associate.run(context)

    speech = s6_associate.load_identity_speech(context)
    assert set(speech) == {"F001", "F002"}
    assert len(speech["F001"]) == 1
    assert speech["F001"][0].start == pytest.approx(0.0, abs=0.1)
    assert speech["F001"][0].end == pytest.approx(1.5, abs=0.1)
    # The face nobody claimed has no speaking time.
    assert speech["F002"] == ()


def test_no_speakers_still_writes_a_result(staged: Any) -> None:
    context = _two_people(staged, turns=[], traces={})
    result = s6_associate.run(context)
    assert result.summary["speakers"] == 0
    assert s6_associate.load_assignments(context) == ()


def test_second_run_skips(staged: Any) -> None:
    context = _two_people(staged, turns=[("spk_00", 0.0, 1.5)],
                          traces={1: _ramp(0.0, 1.5, step=0.1, value=0.95)})
    assert not s6_associate.run(context).skipped
    assert s6_associate.run(context).skipped


def test_force_reruns(staged: Any) -> None:
    context = _two_people(staged, turns=[("spk_00", 0.0, 1.5)],
                          traces={1: _ramp(0.0, 1.5, step=0.1, value=0.95)})
    s6_associate.run(context)
    assert not s6_associate.run(context, force=True).skipped


def test_a_config_change_invalidates_the_cache(staged: Any) -> None:
    context = _two_people(staged, turns=[("spk_00", 0.0, 1.5)],
                          traces={1: _ramp(0.0, 1.5, step=0.1, value=0.95)})
    s6_associate.run(context)

    rerun = s6_associate.run(
        _context(context.source, context.work_dir.parents[1], stride=1, min_margin=0.9)
    )
    assert not rerun.skipped
    assert "inputs changed" in rerun.reason


def test_editing_the_turns_invalidates_the_cache(staged: Any) -> None:
    context = _two_people(staged, turns=[("spk_00", 0.0, 1.5)],
                          traces={1: _ramp(0.0, 1.5, step=0.1, value=0.95)})
    s6_associate.run(context)

    path = s4_diarize.turns_path(context)
    path.write_text(path.read_text().replace("0.0", "0.1"), encoding="utf-8")
    assert not s6_associate.run(context).skipped


def test_running_before_s5_is_a_clear_error(staged: Any) -> None:
    context, _ = staged
    _write_identities(context, {"F001": [1]})
    _write_turns(context, [("spk_00", 0.0, 1.0)])
    with pytest.raises(FileNotFoundError, match="run s5-asd first"):
        s6_associate.run(context)


def test_loaders_fail_loudly_when_the_stage_has_not_run(tmp_path: Path) -> None:
    context = StageContext(
        video_id="x", source=tmp_path / "x.mp4", work_dir=tmp_path / "work" / "x"
    )
    with pytest.raises(FileNotFoundError, match="run s6-associate first"):
        s6_associate.load_assignments(context)
    with pytest.raises(FileNotFoundError, match="run s6-associate first"):
        s6_associate.assignments_path(context)
