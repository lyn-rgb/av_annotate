"""Tests for stage S2.

A stub detector drives S1 so the tracks are known in advance: what this stage
owns is the wiring, the seconds-to-steps conversion and the summary, not the
tracking algorithm, which test_track.py covers directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from avannotate.faces.types import Detection
from avannotate.stages import s0_preprocess, s1_faces, s2_tracks
from avannotate.stages.base import StageContext


class _MovingDetector:
    """One face drifting right, plus a static second face."""

    name = "stub"

    def __init__(self, *, faces: int = 2, score: float = 0.9) -> None:
        self.faces = faces
        self.score = score
        self.calls = 0

    def detect(self, frame: np.ndarray) -> tuple[Detection, ...]:
        self.calls += 1
        detections = [
            Detection(
                x=100.0 + 0.5 * self.calls,
                y=100.0,
                width=40.0,
                height=50.0,
                score=self.score,
            )
        ]
        if self.faces > 1:
            detections.append(
                Detection(x=400.0, y=100.0, width=40.0, height=50.0, score=self.score)
            )
        return tuple(detections)


def _context(source: Path, root: Path, **config: object) -> StageContext:
    return StageContext(
        video_id=source.stem,
        source=source,
        work_dir=root / "work" / source.stem,
        config=config,
    )


@pytest.fixture
def staged(
    single_shot_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[StageContext, _MovingDetector]:
    context = _context(single_shot_video, tmp_path, stride=1)
    s0_preprocess.run(context)
    detector = _MovingDetector()
    monkeypatch.setattr(s1_faces, "build_detector", lambda _: detector)
    s1_faces.run(context)
    return context, detector


# --------------------------------------------------------------------------- #
# the stage
# --------------------------------------------------------------------------- #


def test_run_writes_tracks_and_summary(staged: Any) -> None:
    context, _ = staged
    result = s2_tracks.run(context)

    assert not result.skipped
    assert result.summary["tracks"] == 2

    tracks = s2_tracks.load_tracks(context)
    assert len(tracks) == 2
    assert all(track["hits"] > 1 for track in tracks)
    assert all("quality" in track for track in tracks)


def test_track_rows_carry_the_detection_series(staged: Any) -> None:
    context, detector = staged
    s2_tracks.run(context)

    tracks = s2_tracks.load_tracks(context)
    total = sum(len(track["detections"]) for track in tracks)  # type: ignore[arg-type]
    assert total == detector.calls * 2
    first = tracks[0]["detections"][0]  # type: ignore[index]
    assert set(first) >= {"frame", "time", "x", "y", "w", "h", "score"}


def test_the_static_face_reports_no_motion(staged: Any) -> None:
    """The wall-art discriminator, end to end through the stage."""

    context, _ = staged
    s2_tracks.run(context)
    tracks = s2_tracks.load_tracks(context)

    motions = {
        round(float(track["detections"][0]["x"])): float(track["quality"]["motion"])  # type: ignore[index]
        for track in tracks
    }
    assert motions[400.0] == pytest.approx(0.0, abs=1e-6)
    assert motions[100.0] > 0.0


def test_summary_reports_fragmentation(staged: Any) -> None:
    context, _ = staged
    s2_tracks.run(context)

    summary = json.loads((context.work_dir / "s2-tracks" / "summary.json").read_text())
    fragmentation = summary["tracks"]["fragmentation"]
    assert fragmentation["single_hit"] == 0
    assert fragmentation["share_single_hit"] == 0.0
    assert summary["tracks"]["hits_per_track"]["median"] > 1


def test_summary_records_the_resolved_step_count(staged: Any) -> None:
    """The seconds-to-steps conversion is the one thing that can silently drift."""

    context, _ = staged
    s2_tracks.run(context)
    summary = json.loads((context.work_dir / "s2-tracks" / "summary.json").read_text())

    timeline = s0_preprocess.load_timeline(context)
    expected = round(1.0 * timeline.fps / 1)
    assert summary["resolved"]["max_time_lost_steps"] == expected
    assert summary["resolved"]["stride"] == 1


def test_an_empty_detection_set_yields_no_tracks(
    single_shot_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(single_shot_video, tmp_path, stride=1)
    s0_preprocess.run(context)

    class _Blind:
        name = "blind"

        def detect(self, frame: np.ndarray) -> tuple[Detection, ...]:
            return ()

    monkeypatch.setattr(s1_faces, "build_detector", lambda _: _Blind())
    s1_faces.run(context)
    result = s2_tracks.run(context)

    assert result.summary["tracks"] == 0
    summary = json.loads((context.work_dir / "s2-tracks" / "summary.json").read_text())
    assert summary["tracks"]["count"] == 0
    assert "note" in summary["tracks"]


def test_second_run_skips(staged: Any) -> None:
    context, _ = staged
    assert not s2_tracks.run(context).skipped
    assert s2_tracks.run(context).skipped


def test_force_reruns(staged: Any) -> None:
    context, _ = staged
    s2_tracks.run(context)
    assert not s2_tracks.run(context, force=True).skipped


def test_a_config_change_invalidates_the_cache(staged: Any) -> None:
    context, _ = staged
    s2_tracks.run(context)
    rerun = s2_tracks.run(
        _context(context.source, context.work_dir.parents[1], stride=1, det_thresh=0.95)
    )
    assert not rerun.skipped
    assert "inputs changed" in rerun.reason


def test_editing_the_detections_invalidates_the_cache(staged: Any) -> None:
    """Same-size content change is what a size check would miss."""

    context, _ = staged
    s2_tracks.run(context)

    path = s1_faces.detections_path(context)
    text = path.read_text()
    # Swap one digit for another: same length, different content.
    path.write_text(text.replace('"score": 0.9', '"score": 0.8', 1))

    assert not s2_tracks.run(context).skipped


def test_running_before_s1_is_a_clear_error(single_shot_video: Path, tmp_path: Path) -> None:
    context = _context(single_shot_video, tmp_path)
    s0_preprocess.run(context)
    with pytest.raises(FileNotFoundError, match="run s1-faces first"):
        s2_tracks.run(context)


def test_loaders_fail_loudly_when_the_stage_has_not_run(tmp_path: Path) -> None:
    context = StageContext(
        video_id="x", source=tmp_path / "x.mp4", work_dir=tmp_path / "work" / "x"
    )
    with pytest.raises(FileNotFoundError, match="run s2-tracks first"):
        s2_tracks.load_tracks(context)
    with pytest.raises(FileNotFoundError, match="run s2-tracks first"):
        s2_tracks.tracks_path(context)


def test_config_defaults_are_the_documented_ones() -> None:
    config = s2_tracks.S2Config.from_mapping({})
    assert config.track_thresh == 0.5
    assert config.det_thresh == 0.6
    assert config.max_time_lost_seconds == 1.0


def test_tracker_config_converts_seconds_through_the_stride() -> None:
    config = s2_tracks.S2Config.from_mapping({"max_time_lost_seconds": 2.0})
    assert config.tracker_config(fps=30.0, stride=3).max_time_lost == 20
    assert config.tracker_config(fps=30.0, stride=1).max_time_lost == 60
