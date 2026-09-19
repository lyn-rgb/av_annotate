"""Tests for stage S1.

Most of these inject a stub detector, because what the stage owns is not
detection accuracy -- it is the frame loop, the tally, and the resume record.
Those are exactly the parts a wrong answer hides in, and they are testable
without a model file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from avannotate.faces.types import Detection
from avannotate.stages import s0_preprocess, s1_faces
from avannotate.stages.base import StageContext


class _StubDetector:
    """Reports a fixed number of faces per frame, at a fixed place."""

    name = "stub"
    provides_embeddings = False

    def __init__(self, faces_per_frame: int = 2, score: float = 0.9) -> None:
        self.faces_per_frame = faces_per_frame
        self.score = score
        self.calls = 0

    def detect(self, frame: np.ndarray) -> tuple[Detection, ...]:
        self.calls += 1
        height, width = frame.shape[:2]
        return tuple(
            Detection(
                x=float(10 * index),
                y=10.0,
                width=float(width // 8),
                height=float(height // 8),
                score=self.score,
                landmarks=((1.0, 2.0), (3.0, 2.0), (2.0, 3.0), (1.0, 4.0), (3.0, 4.0)),
            )
            for index in range(self.faces_per_frame)
        )


def _context(source: Path, root: Path, **config: object) -> StageContext:
    return StageContext(
        video_id=source.stem,
        source=source,
        work_dir=root / "work" / source.stem,
        config=config,
    )


@pytest.fixture
def staged(single_shot_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A context with S0 already run and a stub detector wired in."""

    context = _context(single_shot_video, tmp_path)
    s0_preprocess.run(context)
    stub = _StubDetector()
    monkeypatch.setattr(s1_faces, "build_detector", lambda _: stub)
    return context, stub


def test_run_writes_detections_and_summary(staged: Any) -> None:
    context, stub = staged
    result = s1_faces.run(context)

    assert not result.skipped
    assert result.summary["detector"] == "stub"
    assert result.summary["detections"] == stub.calls * 2

    rows = s1_faces.load_detections(context)
    assert len(rows) == stub.calls
    assert all(len(row["faces"]) == 2 for row in rows)  # type: ignore[arg-type]

    summary = json.loads((context.work_dir / "s1-faces" / "summary.json").read_text())
    assert summary["detection"]["sampled_frames"] == stub.calls
    assert summary["detection"]["faces_per_frame"] == {"2": stub.calls}
    assert summary["detection"]["score"]["median"] == pytest.approx(0.9)


def test_detection_rows_carry_frame_time_and_boxes(staged: Any) -> None:
    context, _ = staged
    s1_faces.run(context)
    rows = s1_faces.load_detections(context)

    first = rows[0]
    assert first["frame"] == 0
    assert first["time"] == 0.0
    face = first["faces"][0]  # type: ignore[index]
    assert set(face) >= {"x", "y", "w", "h", "score", "landmarks"}
    assert len(face["landmarks"]) == 5


def test_second_run_skips(staged: Any) -> None:
    context, _ = staged
    assert not s1_faces.run(context).skipped
    assert s1_faces.run(context).skipped


def test_force_reruns(staged: Any) -> None:
    context, _ = staged
    s1_faces.run(context)
    assert not s1_faces.run(context, force=True).skipped


def test_changing_the_stride_invalidates_the_cache(staged: Any) -> None:
    context, _ = staged
    s1_faces.run(context)
    rerun = s1_faces.run(_context(context.source, context.work_dir.parents[1], stride=7))
    assert not rerun.skipped


def test_low_scores_are_recorded_not_dropped(single_shot_video: Path, tmp_path: Path,
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    """A background face scores low; S1 keeps it so a later stage can judge."""

    context = _context(single_shot_video, tmp_path)
    s0_preprocess.run(context)
    monkeypatch.setattr(s1_faces, "build_detector", lambda _: _StubDetector(score=0.62))
    s1_faces.run(context)

    summary = json.loads((context.work_dir / "s1-faces" / "summary.json").read_text())
    assert summary["detection"]["score"]["below_0.7"] > 0
    assert s1_faces.load_detections(context)[0]["faces"]  # type: ignore[index]


def test_max_frames_limits_the_decode(single_shot_video: Path, tmp_path: Path,
                                      monkeypatch: pytest.MonkeyPatch) -> None:
    context = _context(single_shot_video, tmp_path, stride=1, max_frames=5)
    s0_preprocess.run(context)
    stub = _StubDetector()
    monkeypatch.setattr(s1_faces, "build_detector", lambda _: stub)
    result = s1_faces.run(context)

    assert result.summary["sampled"] == 5
    assert stub.calls == 5


def test_running_before_s0_is_a_clear_error(single_shot_video: Path, tmp_path: Path) -> None:
    context = _context(single_shot_video, tmp_path)
    with pytest.raises(FileNotFoundError, match="run s0-preprocess before s1-faces"):
        s1_faces.run(context)


def test_a_detector_that_cannot_be_built_marks_the_stage_failed(
    single_shot_video: Path, tmp_path: Path
) -> None:
    """So the batch driver retries it instead of skipping it as done."""

    context = _context(single_shot_video, tmp_path, backend="yunet",
                       model_path="models/does-not-exist.onnx")
    s0_preprocess.run(context)

    from avannotate.faces.detect import DetectorError

    with pytest.raises(DetectorError):
        s1_faces.run(context)

    state = json.loads((context.work_dir / "stage_state.json").read_text())
    record = next(item for item in state["stages"] if item["stage"] == "s1-faces")
    assert record["status"] == "failed"
    assert "not found" in record["error"]


def test_config_paths_resolve_against_the_config_file(
    single_shot_video: Path, tmp_path: Path
) -> None:
    """A relative model path must not depend on the working directory."""

    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    context = _context(
        single_shot_video,
        tmp_path,
        backend="yunet",
        model_path="../models/yunet.onnx",
        config_root=str(config_dir),
    )
    s0_preprocess.run(context)

    config = s1_faces.S1Config.from_mapping(context.config)
    resolved = config.detector_config()
    assert resolved["model_path"] == str(tmp_path / "models" / "yunet.onnx")


def test_load_detections_fails_loudly_when_the_stage_has_not_run(tmp_path: Path) -> None:
    context = StageContext(
        video_id="x", source=tmp_path / "x.mp4", work_dir=tmp_path / "work" / "x"
    )
    with pytest.raises(FileNotFoundError, match="run s1-faces first"):
        s1_faces.load_detections(context)
    with pytest.raises(FileNotFoundError, match="run s1-faces first"):
        s1_faces.detections_path(context)


def test_sample_corpus_detects_two_people_without_runaway_false_positives(
    sample_videos: tuple[Path, ...], tmp_path: Path
) -> None:
    """The corpus is two-person footage, so two is the expected cast.

    A third box is expected on some clips -- a framed gold record on a wall
    scores 0.64 there, persistently, and is the reason S1 records scores instead
    of thresholding them away.  What is *not* expected is a detector producing
    boxes faster than there are faces, so the ceiling is what this pins.
    """

    model = Path(__file__).resolve().parents[1] / "models" / "yunet.onnx"
    if not model.is_file():
        pytest.skip(f"YuNet weights not present at {model}")

    for source in sample_videos:
        context = _context(
            source, tmp_path, backend="yunet", model_path=str(model),
            stride=3, score_threshold=0.6,
        )
        s0_preprocess.run(context)
        s1_faces.run(context)

        rows = s1_faces.load_detections(context)
        assert rows, f"{source.name} produced no detection rows at all"
        counts = [len(row["faces"]) for row in rows]  # type: ignore[arg-type]

        assert max(counts) >= 2, f"{source.name}: never found both people"
        assert max(counts) <= 3, (
            f"{source.name}: up to {max(counts)} faces per frame for two-person footage"
        )


def test_a_background_false_positive_is_kept_and_is_distinguishable(
    sample_videos: tuple[Path, ...], tmp_path: Path
) -> None:
    """The gold-record case: a spurious box that a later stage must be able to reject.

    Asserting on the score gap rather than on a count, because the point is that
    the *evidence* survives S1 -- drop low-scoring detections here and the
    filtering stage downstream has nothing to filter on.
    """

    model = Path(__file__).resolve().parents[1] / "models" / "yunet.onnx"
    if not model.is_file():
        pytest.skip(f"YuNet weights not present at {model}")

    best_gap = 0.0
    for source in sample_videos:
        context = _context(
            source, tmp_path, backend="yunet", model_path=str(model),
            stride=3, score_threshold=0.6,
        )
        s0_preprocess.run(context)
        s1_faces.run(context)

        for row in s1_faces.load_detections(context):
            faces = row["faces"]  # type: ignore[index]
            if len(faces) < 2:
                continue
            scores = sorted((face["score"] for face in faces), reverse=True)
            best_gap = max(best_gap, scores[0] - scores[1])

    assert best_gap > 0.1, (
        "no clip showed a score gap between its real faces and a background box; "
        "the recorded scores no longer separate them"
    )
