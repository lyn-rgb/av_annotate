"""Tests for stage S5.

LoCoNet is not installed here, so a stub model stands in.  What that covers is
the loop: which windows are asked about, what shapes the model receives, how the
per-window answers become one trace per tracklet, and the failure modes.  The
arithmetic those depend on -- windowing, crops, groups, stitching -- has its own
tests in test_asd.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from avannotate.asd.model import _MEL_BINS as MEL_BINS
from avannotate.asd.model import AsdError
from avannotate.asd.types import SpeakingSample, TrackSpeaking
from avannotate.faces.track import TrackDetection, Tracklet, TrackQuality
from avannotate.faces.types import Detection
from avannotate.stages import s0_preprocess, s1_faces, s2_tracks, s5_asd
from avannotate.stages.base import StageContext


class _StubModel:
    """Returns a fixed probability, and remembers what it was asked."""

    name = "stub"
    max_window_frames = 200
    max_speakers = 3

    def __init__(self, probability: float = 0.75) -> None:
        self.probability = probability
        self.calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def score(self, crops: np.ndarray, audio: np.ndarray) -> np.ndarray:
        speakers, frames = crops.shape[0], crops.shape[1]
        self.calls.append((crops.shape, audio.shape))
        return np.full((speakers, frames), self.probability, dtype=np.float32)


class _EmbeddingDetector:
    """Two faces, each moving, so the tracker produces two tracklets."""

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
        drift = 0.3 * self.calls
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


@pytest.fixture
def staged(single_shot_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """S0-S2 run with a stub detector, and a stub ASD model wired in."""

    context = _context(single_shot_video, tmp_path, stride=1)
    s0_preprocess.run(context)
    monkeypatch.setattr(s1_faces, "build_detector", lambda _: _EmbeddingDetector())
    s1_faces.run(context)
    s2_tracks.run(context)

    # The audio frontend needs torch, which is not installed here.  Stubbing it
    # alongside the model keeps the test on what the stage owns -- the loop and
    # the shapes -- rather than on the frontend's arithmetic.
    feature_calls: list[int] = []

    def stub_features(samples: Any, *, video_frames: int) -> np.ndarray:
        feature_calls.append(len(samples))
        return np.zeros((video_frames * 4, MEL_BINS), dtype=np.float32)

    monkeypatch.setattr(s5_asd, "features_for_window", stub_features)

    def install(probability: float = 0.75):
        model = _StubModel(probability)
        monkeypatch.setattr(s5_asd, "build_asd_model", lambda _: model)
        return model

    return context, install, feature_calls


# --------------------------------------------------------------------------- #
# the stage
# --------------------------------------------------------------------------- #


def test_run_writes_a_trace_per_tracklet(staged: Any) -> None:
    context, install, _ = staged
    model = install()

    result = s5_asd.run(context)
    assert not result.skipped
    assert result.summary["tracks"] == 2
    assert model.calls, "the model was never asked anything"

    loaded = s5_asd.load_result(context)
    assert len(loaded.tracks) == 2
    assert all(track.samples for track in loaded.tracks)


def test_every_probability_survives_to_the_output(staged: Any) -> None:
    context, install, _ = staged
    install(probability=0.75)
    s5_asd.run(context)

    for track in s5_asd.load_result(context).tracks:
        assert all(sample.probability == pytest.approx(0.75) for sample in track.samples)


def test_the_model_receives_the_shapes_it_documents(staged: Any) -> None:
    """``[S, T, H, W]`` crops and ``[4T, 64]`` features, or the pass is wasted.

    The 64 is VGGish's mel-band count and is what the audio frontend's first
    convolution is wide.  Reading 128 off a later layer of the model is how this
    was wrong first; 128 bands into a 64-wide convolution does not fail, it
    convolves over nonsense.
    """

    context, install, _ = staged
    model = install()
    s5_asd.run(context)

    for crops_shape, audio_shape in model.calls:
        assert len(crops_shape) == 4
        assert crops_shape[2] == crops_shape[3] == 112
        assert audio_shape == (crops_shape[1] * 4, MEL_BINS)


def test_every_group_puts_the_target_first(staged: Any) -> None:
    """Only row 0 is kept, so a group that does not lead with its target would
    attribute one person's speech to another."""

    context, install, _ = staged
    model = install()
    s5_asd.run(context)

    loaded = s5_asd.load_result(context)
    # Both tracklets were scored, and each was the target of its own pass.
    assert {track.track_id for track in loaded.tracks} == {1, 2}
    assert len(model.calls) == 2


def test_the_frontend_is_given_samples_spanning_the_window(staged: Any) -> None:
    """The audio must cover the same span as the frames.

    A short read would score lip motion against the wrong sound, and the model
    would still return a confident answer -- about the wrong instant.
    """

    context, install, feature_calls = staged
    install()
    s5_asd.run(context)

    timeline = s0_preprocess.load_timeline(context)
    assert feature_calls, "the frontend was never called"
    for sample_count in feature_calls:
        assert sample_count > 0
        # A window is at most the whole clip, so no call can ask for more.
        assert sample_count <= int(timeline.audio_duration * timeline.sample_rate) + 1


def test_the_summary_records_the_window_plan(staged: Any) -> None:
    context, install, _ = staged
    install()
    s5_asd.run(context)

    summary = json.loads((context.work_dir / "s5-asd" / "summary.json").read_text())
    assert summary["window_plan"]["window_frames"] > 0
    assert summary["window_plan"]["dropped_tail_frames"] == 0
    assert summary["windows"], "the plan should be recorded"


def test_no_tracklets_still_writes_a_result(single_shot_video: Path, tmp_path: Path,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    """S6 must be able to tell "ran, found nothing" from "has not run"."""

    context = _context(single_shot_video, tmp_path, stride=1)
    s0_preprocess.run(context)

    class _Blind:
        name = "blind"
        provides_embeddings = True

        def detect(self, frame: np.ndarray) -> tuple[Detection, ...]:
            return ()

    monkeypatch.setattr(s1_faces, "build_detector", lambda _: _Blind())
    s1_faces.run(context)
    s2_tracks.run(context)

    result = s5_asd.run(context)
    assert result.summary["tracks"] == 0
    assert s5_asd.load_result(context).tracks == ()
    assert s5_asd.speaking_path(context).is_file()


def test_second_run_skips(staged: Any) -> None:
    context, install, _ = staged
    install()
    assert not s5_asd.run(context).skipped
    assert s5_asd.run(context).skipped


def test_force_reruns(staged: Any) -> None:
    context, install, _ = staged
    install()
    s5_asd.run(context)
    assert not s5_asd.run(context, force=True).skipped


def test_a_config_change_invalidates_the_cache(staged: Any) -> None:
    context, install, _ = staged
    install()
    s5_asd.run(context)

    rerun = s5_asd.run(
        _context(context.source, context.work_dir.parents[1], stride=1, window_seconds=1.0)
    )
    assert not rerun.skipped
    assert "inputs changed" in rerun.reason


def test_a_model_that_cannot_be_built_marks_the_stage_failed(staged: Any) -> None:
    context, _, _ = staged

    with pytest.raises(AsdError):
        s5_asd.run(context)

    state = json.loads((context.work_dir / "stage_state.json").read_text())
    record = next(item for item in state["stages"] if item["stage"] == "s5-asd")
    assert record["status"] == "failed"
    assert "checkpoint" in record["error"]


def test_running_before_s2_is_a_clear_error(
    single_shot_video: Path, tmp_path: Path
) -> None:
    context = _context(single_shot_video, tmp_path)
    s0_preprocess.run(context)
    with pytest.raises(FileNotFoundError, match="run s2-tracks first"):
        s5_asd.run(context)


def test_loaders_fail_loudly_when_the_stage_has_not_run(tmp_path: Path) -> None:
    context = StageContext(
        video_id="x", source=tmp_path / "x.mp4", work_dir=tmp_path / "work" / "x"
    )
    with pytest.raises(FileNotFoundError, match="run s5-asd first"):
        s5_asd.load_result(context)
    with pytest.raises(FileNotFoundError, match="run s5-asd first"):
        s5_asd.speaking_path(context)


# --------------------------------------------------------------------------- #
# building a pass
# --------------------------------------------------------------------------- #


def test_boxes_for_window_holds_the_last_sighting_forward() -> None:
    """A missed detection is a missed sighting, not an absent face."""

    from avannotate.asd.batch import boxes_for_window
    from avannotate.asd.types import Window

    tracklet = Tracklet(
        track_id=1,
        start_frame=0,
        end_frame=4,
        hits=2,
        quality=TrackQuality(2, 0.2, 0.9, 40.0, 50.0, 0.05),
        detections=(
            TrackDetection(0, 0.0, (10.0, 10.0, 40.0, 50.0), 0.9),
            TrackDetection(3, 0.3, (20.0, 10.0, 40.0, 50.0), 0.9),
        ),
    )
    window = Window(index=0, start_frame=0, end_frame=5, start=0.0, end=0.5)

    filled = boxes_for_window(tracklet, window)
    assert [box[0] for box in filled if box is not None] == [10.0, 10.0, 10.0, 20.0, 20.0]

    raw = boxes_for_window(tracklet, window, fill=False)
    assert raw[1] is None and raw[2] is None


def test_a_tracklet_absent_from_the_window_gets_no_boxes() -> None:
    from avannotate.asd.batch import boxes_for_window
    from avannotate.asd.types import Window

    tracklet = Tracklet(
        track_id=1, start_frame=0, end_frame=0, hits=1,
        quality=TrackQuality(1, 0.0, 0.9, 40.0, 50.0, 0.0),
        detections=(TrackDetection(0, 0.0, (10.0, 10.0, 40.0, 50.0), 0.9),),
    )
    window = Window(index=0, start_frame=100, end_frame=105, start=4.0, end=4.2)
    assert boxes_for_window(tracklet, window) == (None,) * 5


def test_stacking_speakers_needs_matching_lengths() -> None:
    from avannotate.asd.batch import stack_speakers

    with pytest.raises(ValueError, match="differing frame counts"):
        stack_speakers(
            [
                np.zeros((4, 112, 112), dtype=np.uint8),
                np.zeros((5, 112, 112), dtype=np.uint8),
            ]
        )


def test_stacking_leaves_the_pixels_in_the_range_the_model_expects() -> None:
    """0..255, because LoCoNet normalises inside its own visual frontend.

    Scaling to [0, 1] here as well would apply the shift twice and hand the
    network a batch centred near -2.5 instead of near 0.5.  Nothing raises; the
    first layer is simply wrong, which reads as a poor score rather than a bug.
    """

    from avannotate.asd.batch import stack_speakers

    stacked = stack_speakers([np.full((2, 112, 112), 255, dtype=np.uint8)])
    assert stacked.shape == (1, 2, 112, 112)
    assert stacked.dtype == np.float32
    assert float(stacked.max()) == pytest.approx(255.0)
    assert float(stacked.min()) == pytest.approx(255.0)


def test_track_speaking_round_trips() -> None:
    track = TrackSpeaking(track_id=3, samples=(SpeakingSample(0.0, 0.5),))
    assert TrackSpeaking.from_dict(track.to_dict()) == track
