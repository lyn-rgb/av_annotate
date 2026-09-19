"""Tests for the active speaker stage's arithmetic.

Windowing, crop geometry, group selection and stitching are all plain
arithmetic, and all of it goes wrong silently: a window plan with a gap drops
speech, a crop that misses the chin removes the signal, and a stitch that
misaligns attributes one person's speech to another.  None of that needs the
network to test.
"""

from __future__ import annotations

import numpy as np
import pytest

from avannotate.asd.crop import CropBox, clamp_box, crop_box
from avannotate.asd.stitch import stitch_predictions
from avannotate.asd.types import AsdResult, SpeakingSample, TrackSpeaking, Window
from avannotate.asd.window import (
    activity,
    context_speakers,
    groups_per_window,
    plan_windows,
    tracks_in_window,
)
from avannotate.faces.track import TrackDetection, Tracklet, TrackQuality

# --------------------------------------------------------------------------- #
# windowing
# --------------------------------------------------------------------------- #


def test_windows_cover_the_whole_video() -> None:
    plan = plan_windows(30.0, fps=25.0, window_seconds=8.0, overlap_seconds=1.0)
    assert plan.covers_everything
    assert plan.windows[0].start_frame == 0
    assert plan.windows[-1].end_frame == 750  # 30s at 25fps


def test_windows_never_exceed_the_frame_cap() -> None:
    """LoCoNet runs out of memory past 200 frames; the planner is the guard."""

    plan = plan_windows(120.0, fps=25.0, window_seconds=20.0, max_frames=200)
    assert plan.window_frames == 200
    assert all(window.frame_count <= 200 for window in plan.windows)


def test_windows_overlap_by_the_requested_amount() -> None:
    plan = plan_windows(30.0, fps=25.0, window_seconds=8.0, overlap_seconds=1.0)
    assert plan.window_frames == 200
    assert plan.step_frames == 175  # 8s window less 1s overlap
    first, second = plan.windows[0], plan.windows[1]
    assert second.start_frame - first.start_frame == 175


def test_a_video_shorter_than_a_window_gets_one_window() -> None:
    plan = plan_windows(3.0, fps=25.0, window_seconds=8.0)
    assert len(plan.windows) == 1
    assert plan.windows[0].frame_count == 75
    assert plan.covers_everything


def test_an_overlap_as_long_as_the_window_does_not_hang() -> None:
    """It would never advance; the step is clamped so the plan stays finite."""

    plan = plan_windows(10.0, fps=25.0, window_seconds=2.0, overlap_seconds=5.0)
    assert plan.step_frames == 1
    assert len(plan.windows) == 250 - 50 + 1


def test_the_last_window_is_short_rather_than_overshooting() -> None:
    plan = plan_windows(10.0, fps=25.0, window_seconds=3.0, overlap_seconds=0.5)
    assert plan.windows[-1].end_frame == 250
    assert plan.windows[-1].frame_count <= plan.window_frames


def test_windows_are_numbered_in_order() -> None:
    plan = plan_windows(30.0, fps=25.0)
    assert [window.index for window in plan.windows] == list(range(len(plan.windows)))


@pytest.mark.parametrize(
    ("duration", "fps", "max_frames"),
    [(0.0, 25.0, 200), (-1.0, 25.0, 200), (10.0, 0.0, 200), (10.0, 25.0, 0)],
)
def test_impossible_parameters_are_rejected(duration: float, fps: float, max_frames: int) -> None:
    with pytest.raises(ValueError):
        plan_windows(duration, fps=fps, max_frames=max_frames)


# --------------------------------------------------------------------------- #
# who is on screen
# --------------------------------------------------------------------------- #


def _tracklet(
    track_id: int, times: list[float], *, height: float = 100.0
) -> Tracklet:
    detections = tuple(
        TrackDetection(
            frame_index=int(round(time * 25.0)),
            time=time,
            box=(100.0, 100.0, 60.0, height),
            score=0.9,
        )
        for time in times
    )
    return Tracklet(
        track_id=track_id,
        start_frame=detections[0].frame_index if detections else 0,
        end_frame=detections[-1].frame_index if detections else 0,
        hits=len(detections),
        quality=TrackQuality(len(detections), 1.0, 0.9, 60.0, height, 0.05),
        detections=detections,
    )


def _window(index: int = 0, start: float = 0.0, end: float = 10.0) -> Window:
    return Window(
        index=index, start_frame=int(start * 25), end_frame=int(end * 25),
        start=start, end=end,
    )


def test_activity_spans_between_consecutive_detections() -> None:
    """A detection is an instant; the face is there until the next one.

    Treating each detection as a zero-length interval makes every pair of
    tracklets disjoint, and the context selection then falls back to track id.
    """

    # Detections one video frame apart, as a dense tracklet has them.
    spans = activity(_tracklet(1, [1.0, 1.04, 1.08]), _window())
    assert len(spans) == 1
    assert spans[0].start == pytest.approx(1.0)
    # The last detection is extended by the median spacing, 0.04s.
    assert spans[0].end == pytest.approx(1.12)


def test_the_tail_uses_the_same_rule_at_any_detection_count() -> None:
    """Deriving the tail from the span starts gave a different extension for two
    detections than for three at the same spacing."""

    two = activity(_tracklet(1, [1.0, 2.0]), _window())[0]
    three = activity(_tracklet(1, [1.0, 2.0, 3.0]), _window())[0]
    # n detections one second apart cover n seconds, whatever n is.
    assert two.duration == pytest.approx(2.0)
    assert three.duration == pytest.approx(3.0)


def test_activity_of_a_single_detection_is_not_empty() -> None:
    spans = activity(_tracklet(1, [2.0]), _window())
    assert len(spans) == 1
    assert spans[0].duration > 0.0


def test_activity_is_confined_to_the_window() -> None:
    spans = activity(_tracklet(1, [1.0, 2.0, 30.0]), _window(end=10.0))
    assert all(span.end <= 10.0 for span in spans)


def test_activity_of_a_tracklet_absent_from_the_window_is_empty() -> None:
    assert activity(_tracklet(1, [50.0, 51.0]), _window(end=10.0)) == ()


def test_tracks_in_window_reports_only_the_present_ones() -> None:
    tracklets = [_tracklet(1, [1.0, 2.0]), _tracklet(2, [50.0, 51.0])]
    assert tracks_in_window(tracklets, _window(end=10.0)) == (1,)


def test_context_speakers_ranks_by_overlap_not_by_id() -> None:
    """The bug this test exists for: ranking by id passes whenever the ids
    happen to line up with the overlaps, and fails on a corpus where they do
    not."""

    target = _tracklet(1, [1.0, 2.0, 3.0, 4.0, 5.0])
    # Id 2 barely overlaps; id 3 is present throughout.
    brief = _tracklet(2, [4.9, 5.0])
    constant = _tracklet(3, [1.0, 2.0, 3.0, 4.0, 5.0])

    chosen = context_speakers(1, [target, brief, constant], _window(), max_speakers=2)
    assert chosen == (1, 3)


def test_context_speakers_always_puts_the_target_first() -> None:
    target = _tracklet(5, [1.0, 2.0])
    other = _tracklet(1, [1.0, 2.0])
    assert context_speakers(5, [target, other], _window(), max_speakers=2) == (5, 1)


def test_context_speakers_caps_at_the_model_maximum() -> None:
    """AVA is 99% three-or-fewer; a sixth face has no learned behaviour."""

    target = _tracklet(1, [1.0, 2.0])
    others = [_tracklet(index, [1.0, 2.0]) for index in range(2, 8)]
    chosen = context_speakers(1, [target, *others], _window(), max_speakers=3)
    assert len(chosen) == 3
    assert chosen[0] == 1


def test_context_speakers_ties_break_by_id() -> None:
    target = _tracklet(9, [1.0, 2.0])
    first, second = _tracklet(3, [1.0, 2.0]), _tracklet(4, [1.0, 2.0])
    assert context_speakers(9, [target, second, first], _window(), max_speakers=2) == (9, 3)


def test_context_speakers_of_an_unknown_target_is_empty() -> None:
    assert context_speakers(99, [_tracklet(1, [1.0])], _window()) == ()


def test_context_speakers_rejects_a_zero_maximum() -> None:
    with pytest.raises(ValueError):
        context_speakers(1, [_tracklet(1, [1.0])], _window(), max_speakers=0)


def test_groups_per_window_gives_every_target_its_own_pass() -> None:
    """The same window is a different company depending on who is scored."""

    tracklets = [_tracklet(1, [1.0, 2.0]), _tracklet(2, [1.0, 2.0])]
    groups = groups_per_window(tracklets, _window(), max_speakers=2)
    assert groups == ((1, 2), (2, 1))


# --------------------------------------------------------------------------- #
# crops
# --------------------------------------------------------------------------- #


def test_crop_expands_the_box_by_the_margin() -> None:
    box = crop_box((100.0, 200.0, 50.0, 40.0), frame_width=1000, frame_height=1000,
                   margin=0.4)
    # 20px of margin on each side horizontally, 16px vertically.
    assert (box.x, box.y) == (80, 184)
    assert (box.width, box.height) == (90, 72)


def test_crop_is_clamped_to_the_frame() -> None:
    box = crop_box((0.0, 0.0, 50.0, 40.0), frame_width=100, frame_height=100, margin=0.4)
    assert box.x == 0 and box.y == 0
    assert box.x + box.width <= 100 and box.y + box.height <= 100


def test_crop_of_a_box_fully_outside_the_frame_is_empty() -> None:
    box = crop_box((500.0, 500.0, 10.0, 10.0), frame_width=100, frame_height=100)
    assert box.area == 0


def test_crop_of_a_degenerate_box_is_empty() -> None:
    assert crop_box((0.0, 0.0, 0.0, 40.0), frame_width=100, frame_height=100).area == 0


def test_crop_rejects_a_negative_margin() -> None:
    with pytest.raises(ValueError, match="margin cannot be negative"):
        crop_box((0.0, 0.0, 10.0, 10.0), frame_width=100, frame_height=100, margin=-0.1)


def test_clamp_rejects_an_empty_frame() -> None:
    with pytest.raises(ValueError, match="invalid frame size"):
        clamp_box(CropBox(0, 0, 10, 10), frame_width=0, frame_height=100)


def test_crop_box_round_trips() -> None:
    box = CropBox(1, 2, 3, 4)
    assert CropBox.from_dict(box.to_dict()) == box


# --------------------------------------------------------------------------- #
# stitching
# --------------------------------------------------------------------------- #


def test_stitch_averages_overlapping_windows() -> None:
    """Both predictions are evidence; a frame at one window's edge sits in the
    next window's interior."""

    first = Window(index=0, start_frame=0, end_frame=4, start=0.0, end=0.4)
    second = Window(index=1, start_frame=2, end_frame=6, start=0.2, end=0.6)
    samples = stitch_predictions(
        [
            (first, np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32)),
            (second, np.asarray([1.0, 1.0, 0.0, 0.0], dtype=np.float32)),
        ],
        fps=10.0,
    )
    values = [sample.probability for sample in samples]
    assert values == pytest.approx([0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    assert [sample.time for sample in samples] == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])


def test_stitch_of_one_window_is_that_window() -> None:
    window = Window(index=0, start_frame=0, end_frame=3, start=0.0, end=0.3)
    samples = stitch_predictions([(window, np.asarray([0.2, 0.5, 0.8], dtype=np.float32))],
                                 fps=10.0)
    assert [sample.probability for sample in samples] == pytest.approx([0.2, 0.5, 0.8])


def test_stitch_offsets_by_the_window_start() -> None:
    window = Window(index=1, start_frame=20, end_frame=22, start=2.0, end=2.2)
    samples = stitch_predictions([(window, np.asarray([0.5, 0.5], dtype=np.float32))], fps=10.0)
    assert [sample.time for sample in samples] == pytest.approx([2.0, 2.1])


def test_stitch_rejects_a_prediction_count_that_does_not_match_the_window() -> None:
    """A mismatch shifts every later sample, attributing speech to the wrong
    instant, and nothing downstream could detect it."""

    window = Window(index=0, start_frame=0, end_frame=5, start=0.0, end=0.5)
    with pytest.raises(ValueError, match="would not align"):
        stitch_predictions([(window, np.asarray([0.1, 0.2, 0.3], dtype=np.float32))], fps=10.0)


def test_stitch_of_nothing_is_nothing() -> None:
    assert stitch_predictions([], fps=10.0) == ()


def test_stitch_rejects_a_bad_frame_rate() -> None:
    with pytest.raises(ValueError, match="fps must be positive"):
        stitch_predictions([], fps=0.0)


# --------------------------------------------------------------------------- #
# the result's queries
# --------------------------------------------------------------------------- #


def _speaking() -> TrackSpeaking:
    return TrackSpeaking(
        track_id=7,
        samples=(
            SpeakingSample(0.0, 0.1),
            SpeakingSample(0.1, 0.9),
            SpeakingSample(0.2, 0.2),
        ),
    )


def test_probability_at_finds_the_nearest_sample() -> None:
    speaking = _speaking()
    assert speaking.probability_at(0.12) == pytest.approx(0.9)


def test_probability_at_returns_none_when_nothing_is_close() -> None:
    assert _speaking().probability_at(5.0) is None


def test_mean_probability_over_intervals() -> None:
    from avannotate.interval import Interval

    speaking = _speaking()
    assert speaking.mean_probability((Interval(0.0, 0.15),)) == pytest.approx(0.5)
    assert speaking.mean_probability((Interval(9.0, 10.0),)) is None


def test_speaking_span() -> None:
    from avannotate.interval import Interval

    assert _speaking().span == Interval(0.0, 0.2)


def test_speaking_sample_rejects_an_impossible_probability() -> None:
    with pytest.raises(ValueError, match="probability out of range"):
        SpeakingSample(0.0, 1.5)


def test_asd_result_round_trips() -> None:
    result = AsdResult(tracks=(_speaking(),), metadata={"model": "x"})
    assert AsdResult.from_dict(result.to_dict()) == result
    assert result.sample_count == 3
    assert result.for_track(7) is not None
    assert result.for_track(8) is None
