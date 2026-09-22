"""Tests for stage S3.

The clustering algorithm has its own tests in test_cluster.py; what this stage
owns is deciding which tracklets are credible people, joining them, and
numbering the result -- and the numbering is part of the deliverable, so it gets
asserted directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from avannotate.faces.track import TrackDetection, TrackQuality
from avannotate.faces.types import Detection
from avannotate.stages import s0_preprocess, s1_faces, s2_tracks, s3_cluster
from avannotate.stages.base import StageContext
from avannotate.stages.s2_tracks import Tracklet

DIM = 16


def _unit(channel: int) -> np.ndarray:
    vector = np.zeros(DIM, dtype=np.float32)
    vector[channel] = 1.0
    return vector


def _blend(base: np.ndarray, channel: int, weight: float) -> tuple[float, ...]:
    other = _unit(channel)
    blended = base * (1.0 - weight) + other * weight
    blended = blended / np.linalg.norm(blended)
    return tuple(float(value) for value in blended)


def _tracklet(
    track_id: int,
    *,
    frames: int = 10,
    span: float = 1.0,
    score: float = 0.9,
    height: float = 100.0,
    motion: float = 0.05,
    start: float = 0.0,
) -> Tracklet:
    detections = tuple(
        TrackDetection(
            frame_index=index,
            time=start + index * (span / max(1, frames - 1)),
            box=(100.0, 100.0, 60.0, height),
            score=score,
        )
        for index in range(frames)
    )
    return Tracklet(
        track_id=track_id,
        start_frame=0,
        end_frame=frames - 1,
        hits=frames,
        quality=TrackQuality(
            frames=frames,
            span_seconds=span,
            mean_score=score,
            mean_width=60.0,
            mean_height=height,
            motion=motion,
        ),
        detections=detections,
    )


# --------------------------------------------------------------------------- #
# crediting tracklets
# --------------------------------------------------------------------------- #


def test_rejection_reason_accepts_a_credible_tracklet() -> None:
    config = s3_cluster.S3Config()
    assert s3_cluster.rejection_reason(_tracklet(1), config) is None


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"frames": 1}, "detections"),
        ({"span": 0.1}, "visible for"),
        ({"score": 0.2}, "detection score"),
        ({"height": 5.0}, "px tall"),
    ],
)
def test_rejection_reason_names_the_failing_filter(
    kwargs: dict[str, object], expected: str
) -> None:
    config = s3_cluster.S3Config()
    reason = s3_cluster.rejection_reason(_tracklet(1, **kwargs), config)  # type: ignore[arg-type]
    assert reason is not None and expected in reason


def test_collect_vectors_averages_and_normalises() -> None:
    embeddings = np.asarray([_unit(0), _unit(0), _unit(1)], dtype=np.float32)
    tracklet = Tracklet(
        track_id=1,
        start_frame=0,
        end_frame=2,
        hits=3,
        quality=TrackQuality(3, 1.0, 0.9, 60.0, 100.0, 0.05),
        detections=tuple(
            TrackDetection(index, float(index), (100.0, 100.0, 60.0, 100.0), 0.9, index)
            for index in range(3)
        ),
    )
    collected = s3_cluster.collect_vectors(tracklet, embeddings)
    assert collected is not None
    assert collected.samples == 3
    assert float(np.linalg.norm(collected.centroid)) == pytest.approx(1.0, abs=1e-5)
    # Two of three agree, so the centroid leans their way and cohesion is low.
    assert collected.cohesion < 0.8


def test_collect_vectors_returns_none_without_links() -> None:
    embeddings = np.asarray([_unit(0)], dtype=np.float32)
    tracklet = Tracklet(
        track_id=1,
        start_frame=0,
        end_frame=0,
        hits=1,
        quality=TrackQuality(1, 0.1, 0.9, 60.0, 100.0, 0.0),
        detections=(TrackDetection(0, 0.0, (100.0, 100.0, 60.0, 100.0), 0.9, None),),
    )
    assert s3_cluster.collect_vectors(tracklet, embeddings) is None


# --------------------------------------------------------------------------- #
# numbering
# --------------------------------------------------------------------------- #


def _vectors_for(tracklets: list[Tracklet], vectors: list[np.ndarray]):
    collected = {
        tracklet.track_id: s3_cluster.TrackVectors(
            track_id=tracklet.track_id,
            centroid=np.asarray(vector, dtype=np.float32),
            cohesion=1.0,
            samples=5,
        )
        for tracklet, vector in zip(tracklets, vectors, strict=True)
    }
    return collected


def test_numbering_follows_screen_time_descending() -> None:
    """The ids are the deliverable, so the ordering rule is pinned."""

    short = _tracklet(1, span=1.0)
    long = _tracklet(2, span=9.0)
    tracklets = [short, long]
    vectors = _vectors_for(tracklets, [_unit(0), _unit(1)])

    identities = s3_cluster.cluster_tracklets(tracklets, vectors, s3_cluster.S3Config())
    assert [item["face_id"] for item in identities] == ["F001", "F002"]
    assert identities[0]["track_ids"] == [2]  # the longer one is F001
    assert identities[1]["track_ids"] == [1]


def test_a_split_person_becomes_one_identity() -> None:
    """The case the stage exists for."""

    first = _tracklet(1, span=3.0, start=0.0)
    second = _tracklet(2, span=3.0, start=5.0)
    other = _tracklet(3, span=1.0)
    tracklets = [first, second, other]

    same_person = _unit(0)
    vectors = _vectors_for(
        tracklets,
        [same_person, np.asarray(_blend(_unit(0), 1, 0.15), dtype=np.float32), _unit(8)],
    )

    identities = s3_cluster.cluster_tracklets(tracklets, vectors, s3_cluster.S3Config())
    assert len(identities) == 2
    assert identities[0]["track_ids"] == [1, 2]
    assert identities[1]["track_ids"] == [3]


def test_numbering_ties_go_to_the_earlier_appearance() -> None:
    later = _tracklet(1, span=2.0, start=5.0)
    earlier = _tracklet(2, span=2.0, start=0.0)
    tracklets = [later, earlier]
    vectors = _vectors_for(tracklets, [_unit(0), _unit(1)])

    identities = s3_cluster.cluster_tracklets(tracklets, vectors, s3_cluster.S3Config())
    assert identities[0]["track_ids"] == [2]


def test_identity_payload_reports_its_evidence() -> None:
    tracklets = [_tracklet(1, span=2.0, height=120.0, motion=0.07)]
    vectors = _vectors_for(tracklets, [_unit(0)])
    payload = s3_cluster.build_identity("F001", tracklets, list(vectors.values()),
                                        s3_cluster.S3Config())

    assert payload["face_id"] == "F001"
    quality = payload["quality"]
    assert isinstance(quality, dict)
    assert quality["frames"] == 10
    assert quality["mean_height"] == pytest.approx(120.0)
    assert payload["suspect_static"] is False


def test_a_motionless_tracklet_is_flagged_not_dropped() -> None:
    """A wall object moves during a pan too, so the flag cannot act alone."""

    tracklets = [_tracklet(1, motion=0.0001)]
    vectors = _vectors_for(tracklets, [_unit(0)])
    payload = s3_cluster.build_identity("F001", tracklets, list(vectors.values()),
                                        s3_cluster.S3Config())
    assert payload["suspect_static"] is True


def test_no_tracklets_yields_no_identities() -> None:
    assert s3_cluster.cluster_tracklets([], {}, s3_cluster.S3Config()) == []


# --------------------------------------------------------------------------- #
# the stage
# --------------------------------------------------------------------------- #


class _EmbeddingDetector:
    """Two people, each moving; both get distinct identity vectors."""

    name = "stub"
    provides_embeddings = True

    def detect(self, frame: np.ndarray) -> tuple[Detection, ...]:
        self.calls = getattr(self, "calls", 0) + 1
        return (
            Detection(x=100.0, y=100.0, width=40.0, height=50.0, score=0.9,
                      embedding=tuple(float(v) for v in _unit(0))),
            Detection(x=400.0, y=100.0, width=40.0, height=50.0, score=0.9,
                      embedding=tuple(float(v) for v in _unit(1))),
        )


def _context(source: Path, root: Path, **config: object) -> StageContext:
    return StageContext(
        video_id=source.stem, source=source, work_dir=root / "work" / source.stem,
        config=config,
    )


@pytest.fixture
def staged(single_shot_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    context = _context(single_shot_video, tmp_path, stride=1)
    s0_preprocess.run(context)
    monkeypatch.setattr(s1_faces, "build_detector", lambda _: _EmbeddingDetector())
    s1_faces.run(context)
    s2_tracks.run(context)
    return context


def test_stage_joins_two_people_end_to_end(staged: Any) -> None:
    result = s3_cluster.run(staged)
    assert not result.skipped
    assert result.summary["identities"] == 2

    identities = s3_cluster.load_identities(staged)
    assert [item["face_id"] for item in identities] == ["F001", "F002"]


def test_stage_writes_the_dropped_list(staged: Any) -> None:
    s3_cluster.run(staged)
    payload = json.loads(s3_cluster.identities_path(staged).read_text())
    assert payload["dropped"] == []
    assert payload["schema_version"] == "avannotate-identities-v1"


def test_second_run_skips(staged: Any) -> None:
    assert not s3_cluster.run(staged).skipped
    assert s3_cluster.run(staged).skipped


def test_force_reruns(staged: Any) -> None:
    s3_cluster.run(staged)
    assert not s3_cluster.run(staged, force=True).skipped


def test_a_config_change_invalidates_the_cache(staged: Any) -> None:
    s3_cluster.run(staged)
    rerun = s3_cluster.run(
        _context(staged.source, staged.work_dir.parents[1], stride=1, max_distance=0.01)
    )
    assert not rerun.skipped
    assert "inputs changed" in rerun.reason


def test_a_corpus_without_vectors_fails_with_an_actionable_message(
    single_shot_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Silently treating each tracklet as a person would fragment every identity
    and look like success."""

    context = _context(single_shot_video, tmp_path, stride=1)
    s0_preprocess.run(context)

    class _NoVectors(_EmbeddingDetector):
        provides_embeddings = False

        def detect(self, frame: np.ndarray) -> tuple[Detection, ...]:
            return tuple(
                Detection(
                    x=item.x, y=item.y, width=item.width, height=item.height, score=item.score
                )
                for item in super().detect(frame)
            )

    monkeypatch.setattr(s1_faces, "build_detector", lambda _: _NoVectors())
    s1_faces.run(context)
    s2_tracks.run(context)

    with pytest.raises(ValueError, match="no identity vectors"):
        s3_cluster.run(context)


def test_running_before_s2_is_a_clear_error(
    single_shot_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(single_shot_video, tmp_path, stride=1)
    s0_preprocess.run(context)
    monkeypatch.setattr(s1_faces, "build_detector", lambda _: _EmbeddingDetector())
    s1_faces.run(context)
    with pytest.raises(FileNotFoundError, match="run s2-tracks first"):
        s3_cluster.run(context)


def test_loaders_fail_loudly_when_the_stage_has_not_run(tmp_path: Path) -> None:
    context = StageContext(
        video_id="x", source=tmp_path / "x.mp4", work_dir=tmp_path / "work" / "x"
    )
    with pytest.raises(FileNotFoundError, match="run s3-cluster first"):
        s3_cluster.load_identities(context)
    with pytest.raises(FileNotFoundError, match="run s3-cluster first"):
        s3_cluster.identities_path(context)


# --------------------------------------------------------------------------- #
# the reference stills
# --------------------------------------------------------------------------- #
#
# The one thing this stage writes that can be looked at.  Worth its own tests
# for the reason it exists: a writer can produce nothing, report success, and
# leave a directory that every other assertion here walks straight past.


def test_an_identity_in_frame_gets_a_reference_still(staged: Any) -> None:
    """"Who is F001?" is not a question numbers answer.

    Everything else S3 produces is boxes and vectors.  Somebody checking that a
    transcript was attributed to the right face has to be able to see the face.
    """

    s3_cluster.run(staged)
    still = staged.stage_dir(s3_cluster.STAGE) / s3_cluster.REFERENCE_DIR / "F001.jpg"

    assert still.is_file()
    assert still.stat().st_size > 0


def test_the_reference_still_is_an_image_and_not_an_empty_file(staged: Any) -> None:
    """Guards the guard: a zero-byte file satisfies "the file is there"."""

    import cv2

    s3_cluster.run(staged)
    still = (
        staged.stage_dir(s3_cluster.STAGE) / s3_cluster.REFERENCE_DIR / "F001.jpg"
    )

    image = cv2.imread(str(still))

    assert image is not None, "the still does not decode as an image"
    # A face crop, not a one-pixel placeholder and not the whole frame.
    assert min(image.shape[:2]) > 8
    assert max(image.shape[:2]) <= s3_cluster.REFERENCE_MAX_EDGE


def test_an_identity_that_is_never_in_frame_gets_no_still(staged: Any) -> None:
    """And that is the correct answer, not a silent failure.

    This fixture's second person is reported at x=400 in a 320-wide frame, so
    there is no picture of them to cut out.  A blank tile would be worse than
    nothing -- it would look like a face that failed to render rather than a
    face the video never showed -- and the count in the summary is what makes
    the absence visible rather than silent.
    """

    s3_cluster.run(staged)

    assert not (
        staged.stage_dir(s3_cluster.STAGE) / s3_cluster.REFERENCE_DIR / "F002.jpg"
    ).is_file()

    summary = json.loads(
        (staged.stage_dir(s3_cluster.STAGE) / "summary.json").read_text()
    )
    assert summary["identities"] == 2
    assert summary["reference_frames"] == 1


def test_the_stills_are_recorded_as_artifacts(staged: Any) -> None:
    """Otherwise a deleted or edited one is never noticed, and the resume record
    goes on saying the stage is done."""

    s3_cluster.run(staged)
    state = json.loads((staged.work_dir / "stage_state.json").read_text())
    record = next(item for item in state["stages"] if item["stage"] == s3_cluster.STAGE)

    assert "s3-cluster/faces/F001.jpg" in {item["path"] for item in record["artifacts"]}
