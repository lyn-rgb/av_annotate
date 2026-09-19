"""Tests for the Kalman filter and the ByteTrack stage of the pipeline.

Most of these are about identity: a track that fragments, or that survives when
it should not, still produces plausible bounding boxes and is only detectable by
counting tracks and hits.
"""

from __future__ import annotations

import numpy as np
import pytest

from avannotate.faces.kalman import KalmanFilter
from avannotate.faces.track import (
    ByteTracker,
    TrackerConfig,
    TrackQuality,
    track_detections,
)
from avannotate.faces.types import Detection, FrameDetections


def _detection(x: float, y: float = 100.0, *, width: float = 40.0, height: float = 50.0,
               score: float = 0.9) -> Detection:
    return Detection(x=x, y=y, width=width, height=height, score=score)


def _frames(sequence: list[list[Detection]], *, stride: int = 3, fps: float = 30.0
            ) -> list[FrameDetections]:
    return [
        FrameDetections(frame_index=index * stride, time=index * stride / fps,
                        detections=tuple(detections))
        for index, detections in enumerate(sequence)
    ]


def _run(sequence: list[list[Detection]], config: TrackerConfig | None = None):
    return track_detections(_frames(sequence), config)


# --------------------------------------------------------------------------- #
# the filter
# --------------------------------------------------------------------------- #


def test_initiate_is_centred_on_the_measurement() -> None:
    kalman = KalmanFilter()
    mean, covariance = kalman.initiate(np.array([10.0, 20.0, 0.8, 50.0]))
    assert mean[0] == pytest.approx(10.0)
    assert mean[1] == pytest.approx(20.0)
    assert mean[2] == pytest.approx(0.8)
    assert mean[3] == pytest.approx(50.0)
    assert np.all(np.diag(covariance) > 0)
    assert np.allclose(mean[4:], 0.0)  # zero initial velocity


def test_predict_advances_along_the_velocity() -> None:
    kalman = KalmanFilter()
    mean, covariance = kalman.initiate(np.array([0.0, 0.0, 1.0, 50.0]))
    mean[4] = 10.0  # 10 px per step in x
    predicted, _ = kalman.predict(mean, covariance)
    assert predicted[0] == pytest.approx(10.0)


def test_update_pulls_the_state_toward_the_measurement() -> None:
    kalman = KalmanFilter()
    mean, covariance = kalman.initiate(np.array([0.0, 0.0, 1.0, 50.0]))
    corrected, _ = kalman.update(mean, covariance, np.array([20.0, 0.0, 1.0, 50.0]))
    assert 0.0 < corrected[0] < 20.0
    assert np.all(np.isfinite(corrected))


def test_repeated_updates_converge_on_a_constant_measurement() -> None:
    kalman = KalmanFilter()
    measurement = np.array([30.0, 40.0, 0.8, 60.0])
    mean, covariance = kalman.initiate(measurement)
    for _ in range(20):
        mean, covariance = kalman.predict(mean, covariance)
        mean, covariance = kalman.update(mean, covariance, measurement)
    assert mean[0] == pytest.approx(30.0, abs=1.0)
    assert mean[1] == pytest.approx(40.0, abs=1.0)
    assert mean[3] == pytest.approx(60.0, abs=2.0)


def test_following_a_moving_target_tracks_its_velocity() -> None:
    kalman = KalmanFilter()
    mean, covariance = kalman.initiate(np.array([0.0, 0.0, 1.0, 50.0]))
    for step in range(1, 31):
        mean, covariance = kalman.predict(mean, covariance)
        mean, covariance = kalman.update(mean, covariance, np.array([5.0 * step, 0.0, 1.0, 50.0]))
    assert mean[4] == pytest.approx(5.0, abs=1.0)


# --------------------------------------------------------------------------- #
# tracking
# --------------------------------------------------------------------------- #


def test_one_moving_face_yields_one_track() -> None:
    sequence = [[_detection(100.0 + 2 * step)] for step in range(10)]
    tracks = _run(sequence)

    assert len(tracks) == 1
    assert tracks[0].hits == 10
    assert len(tracks[0].detections) == 10


def test_two_faces_stay_separate() -> None:
    sequence = [
        [_detection(100.0 + step), _detection(400.0 + step)] for step in range(10)
    ]
    tracks = _run(sequence)

    assert len(tracks) == 2
    assert all(track.hits == 10 for track in tracks)
    starts = sorted(track.detections[0].box[0] for track in tracks)
    assert starts[0] == pytest.approx(100.0)
    assert starts[1] == pytest.approx(400.0)


def test_a_brief_occlusion_keeps_the_same_track() -> None:
    """The case the lost-track pool exists for."""

    sequence = [[_detection(100.0)] for _ in range(5)]
    sequence += [[] for _ in range(4)]                      # occluded
    sequence += [[_detection(100.0)] for _ in range(5)]     # back

    tracks = _run(sequence)
    assert len(tracks) == 1
    assert tracks[0].hits == 10
    assert tracks[0].time_since_update == 0  # re-associated on the last step


def test_a_long_absence_ends_the_track_and_starts_a_new_one() -> None:
    config = TrackerConfig(max_time_lost=2)
    sequence = [[_detection(100.0)] for _ in range(5)]
    sequence += [[] for _ in range(6)]                      # gone for longer than that
    sequence += [[_detection(100.0)] for _ in range(5)]

    tracks = _run(sequence, config)
    assert len(tracks) == 2
    assert [track.hits for track in tracks] == [5, 5]


def test_a_low_score_detection_rescues_an_existing_track() -> None:
    """ByteTrack's second association: a face turning away scores lower and must
    not lose its identity."""

    sequence = [[_detection(100.0, score=0.9)] for _ in range(4)]
    sequence += [[_detection(100.0, score=0.35)] for _ in range(4)]
    sequence += [[_detection(100.0, score=0.9)] for _ in range(4)]

    tracks = _run(sequence)
    assert len(tracks) == 1
    assert tracks[0].hits == 12


def test_a_low_score_detection_does_not_start_a_track() -> None:
    """The wall-art case: a persistent 0.4 box must not become a person."""

    sequence = [[_detection(100.0, score=0.4)] for _ in range(10)]
    assert _run(sequence) == ()


def test_a_low_score_detection_does_not_revive_a_lost_track() -> None:
    """Only confident evidence brings a track back, or a weak repeated detection
    could keep a dead track alive indefinitely."""

    config = TrackerConfig(max_time_lost=2)
    sequence = [[_detection(100.0, score=0.9)] for _ in range(3)]
    sequence += [[] for _ in range(5)]                       # lost
    sequence += [[_detection(100.0, score=0.35)] for _ in range(5)]

    tracks = _run(sequence, config)
    assert len(tracks) == 1
    assert tracks[0].hits == 3


def test_a_detection_below_the_floor_is_ignored_entirely() -> None:
    sequence = [[_detection(100.0, score=0.05)] for _ in range(10)]
    assert _run(sequence) == ()


def test_track_ids_are_unique_and_stable() -> None:
    sequence = [[_detection(100.0 + step), _detection(400.0 + step)] for step in range(20)]
    tracks = _run(sequence)
    ids = [track.track_id for track in tracks]
    assert len(set(ids)) == len(ids)
    assert sorted(ids) == [1, 2]
    assert len(tracks[0].detections) == 20


def test_finalize_closes_open_tracks() -> None:
    """Otherwise the last face of a clip never leaves ``tracked``."""

    tracker = ByteTracker()
    frames = _frames([[_detection(100.0)] for _ in range(3)])
    for frame in frames:
        tracker.update(frame.detections, frame.frame_index, frame.time)
    assert tracker.tracked
    tracker.finalize()
    assert tracker.tracked == []
    assert len(tracker.all_tracks) == 1


def test_more_detections_than_tracks_does_not_crash() -> None:
    """The transposed assignment path."""

    sequence = [
        [_detection(100.0 + index * 200) for index in range(5)] for _ in range(6)
    ]
    assert len(_run(sequence)) == 5


def test_an_empty_video_yields_no_tracks() -> None:
    assert _run([[] for _ in range(10)]) == ()


def test_max_time_lost_seconds_converts_through_fps_and_stride() -> None:
    config = TrackerConfig().with_max_time_lost_seconds(1.0, fps=30.0, stride=3)
    assert config.max_time_lost == 10


# --------------------------------------------------------------------------- #
# quality
# --------------------------------------------------------------------------- #


def test_a_static_track_has_near_zero_motion() -> None:
    """The discriminator for wall art: a picture does not breathe."""

    tracks = _run([[_detection(300.0, 200.0)] for _ in range(30)])
    quality = TrackQuality.from_track(tracks[0], fps=30.0)
    assert quality.motion == pytest.approx(0.0, abs=1e-6)
    assert quality.frames == 30


def test_a_moving_track_has_visible_motion() -> None:
    tracks = _run([[_detection(300.0 + 3 * step, 200.0)] for step in range(30)])
    quality = TrackQuality.from_track(tracks[0], fps=30.0)
    # 3 px per step against a 50 px face.
    assert quality.motion == pytest.approx(3.0 / 50.0, rel=0.3)


def test_motion_is_scale_invariant() -> None:
    """A distant face and a close-up moving the same fraction of their own size
    must score alike, or the threshold cannot be set once for a whole corpus."""

    # 1 px per step against a 25 px face, and 4 px against 100 px: both 4%.
    small = _run([[_detection(100.0 + step, width=20.0, height=25.0)] for step in range(20)])
    large = _run([[_detection(100.0 + 4 * step, width=80.0, height=100.0)] for step in range(20)])
    small_quality = TrackQuality.from_track(small[0], fps=30.0)
    large_quality = TrackQuality.from_track(large[0], fps=30.0)
    assert small_quality.motion == pytest.approx(0.04, rel=0.05)
    assert large_quality.motion == pytest.approx(0.04, rel=0.05)


def test_quality_reports_the_measured_extent() -> None:
    tracks = _run([[_detection(100.0, width=40.0, height=50.0, score=0.8)] for _ in range(6)])
    quality = TrackQuality.from_track(tracks[0], fps=30.0)
    assert quality.mean_width == pytest.approx(40.0)
    assert quality.mean_height == pytest.approx(50.0)
    assert quality.mean_score == pytest.approx(0.8)
    assert quality.frames == 6


def test_quality_to_dict_is_json_shaped() -> None:
    tracks = _run([[_detection(100.0)] for _ in range(4)])
    payload = TrackQuality.from_track(tracks[0], fps=30.0).to_dict()
    assert set(payload) == {
        "frames", "span_seconds", "mean_score", "mean_width", "mean_height", "motion"
    }
    assert isinstance(payload["frames"], int)
