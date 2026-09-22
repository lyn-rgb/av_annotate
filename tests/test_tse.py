"""Tests for target speaker extraction's planning and crop writing.

The extractor itself is not installed here, but everything that decides what it
is handed is testable -- and the crop video really is written and read back,
because a crop that silently follows the wrong face would condition every
extraction on the wrong person.
"""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from avannotate.audio.wav import read_info, read_window, slice_samples, write_pcm16
from avannotate.faces.track import TrackDetection, Tracklet, TrackQuality
from avannotate.interval import Interval
from avannotate.segment import SegmentationConfig
from avannotate.stages import s0_preprocess, s2_tracks, s3_cluster, s6_associate, s7_tse
from avannotate.stages.base import StageContext
from avannotate.tse.crop_video import crop_tile, iter_tiles, write_crop_video
from avannotate.tse.model import TseError, build_extractor
from avannotate.tse.plan import (
    ExtractionSegment,
    context_window,
    identity_boxes,
    plan_extractions,
    segment_name,
)

# --------------------------------------------------------------------------- #
# planning
# --------------------------------------------------------------------------- #


def test_segment_names_sort_in_time_order() -> None:
    names = [segment_name("F001", index) for index in (0, 9, 10, 100)]
    assert names == ["F001_0000", "F001_0009", "F001_0010", "F001_0100"]
    assert names == sorted(names)


def test_segment_name_rejects_a_negative_index() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        segment_name("F001", -1)


def test_plan_extractions_uses_the_segmentation_rules() -> None:
    """The same rules S8 will use, so the file and its transcript agree."""

    config = SegmentationConfig(max_gap=0.2, min_duration=0.3, pad=0.05)
    speech = {
        "F001": [Interval(1.0, 2.0), Interval(2.1, 3.0)],  # one utterance after merging
        "F002": [Interval(4.0, 4.2)],  # too short to keep
    }
    plan = plan_extractions(speech, duration=10.0, config=config)

    assert len(plan["F001"]) == 1
    assert plan["F001"][0].start == pytest.approx(0.95)
    assert plan["F001"][0].end == pytest.approx(3.05)
    assert plan["F002"] == ()


def test_plan_extractions_indexes_each_persons_segments_from_zero() -> None:
    config = SegmentationConfig(max_gap=0.1, min_duration=0.1, pad=0.0)
    speech = {"F001": [Interval(0.0, 1.0), Interval(3.0, 4.0)]}
    plan = plan_extractions(speech, duration=5.0, config=config)

    assert [segment.index for segment in plan["F001"]] == [0, 1]
    assert [segment.name for segment in plan["F001"]] == ["F001_0000", "F001_0001"]


def test_plan_extractions_clamps_to_the_video() -> None:
    config = SegmentationConfig(min_duration=0.1, pad=0.5)
    plan = plan_extractions(
        {"F001": [Interval(0.0, 4.9)]}, duration=5.0, config=config
    )
    assert plan["F001"][0].start == 0.0
    assert plan["F001"][0].end == pytest.approx(5.0)


def test_context_window_is_wider_than_the_segment() -> None:
    segment = ExtractionSegment("F001", 0, start=2.0, end=3.0)
    start, end = context_window(segment, fps=25.0, frame_count=250, context_seconds=0.5)
    assert start == int(round(1.5 * 25))
    assert end == int(round(3.5 * 25))


def test_context_window_is_clamped_to_the_video() -> None:
    first = ExtractionSegment("F001", 0, start=0.1, end=0.5)
    assert context_window(first, fps=25.0, frame_count=250, context_seconds=0.5)[0] == 0

    last = ExtractionSegment("F001", 1, start=9.8, end=10.0)
    assert context_window(last, fps=25.0, frame_count=250, context_seconds=0.5)[1] == 250


def test_context_window_clamps_to_the_frame_count_not_the_duration() -> None:
    """The pair that disagree, and the reason this takes a count at all.

    Eight and a bit seconds at 25 fps rounds to more frames than a file which
    ends on frame 214 holds.  Clamping to the duration asked the decoder for a
    frame past the end, and the decoder refused the whole window rather than its
    last frame -- S7 lost a video whose only fault was ending between two.
    """

    # duration * fps would give 215 for this one.
    last = ExtractionSegment("F001", 0, start=8.0, end=8.6)
    assert context_window(last, fps=25.0, frame_count=214, context_seconds=0.5)[1] == 214


def test_context_window_rejects_a_negative_frame_count() -> None:
    segment = ExtractionSegment("F001", 0, start=0.0, end=1.0)
    with pytest.raises(ValueError, match="cannot be negative"):
        context_window(segment, fps=25.0, frame_count=-1, context_seconds=0.5)


def test_context_window_rejects_a_bad_frame_rate() -> None:
    segment = ExtractionSegment("F001", 0, start=0.0, end=1.0)
    with pytest.raises(ValueError, match="fps must be positive"):
        context_window(segment, fps=0.0, frame_count=250, context_seconds=0.5)


# --------------------------------------------------------------------------- #
# crop boxes
# --------------------------------------------------------------------------- #


def _tracklet(track_id: int, frames: list[int], *, x: float = 100.0) -> Tracklet:
    detections = tuple(
        TrackDetection(
            frame_index=frame,
            time=frame / 25.0,
            box=(x, 50.0, 40.0, 50.0),
            score=0.9,
        )
        for frame in frames
    )
    return Tracklet(
        track_id=track_id,
        start_frame=frames[0],
        end_frame=frames[-1],
        hits=len(frames),
        quality=TrackQuality(len(frames), 0.4, 0.9, 40.0, 50.0, 0.05),
        detections=detections,
    )


def test_identity_boxes_follows_one_tracklet() -> None:
    boxes = identity_boxes([_tracklet(1, [0, 1, 2])], start_frame=0, end_frame=5)
    assert len(boxes) == 5
    assert all(box is not None for box in boxes)


def test_identity_boxes_spans_a_split_between_tracklets() -> None:
    """An identity is several tracklets after a camera pan; the crop has to
    follow the person across the split."""

    early = _tracklet(1, [0, 1, 2], x=100.0)
    late = _tracklet(2, [5, 6, 7], x=300.0)
    boxes = identity_boxes([early, late], start_frame=0, end_frame=8)

    assert len(boxes) == 8
    assert boxes[0][0] == pytest.approx(100.0)  # type: ignore[index]
    # Frames 3 and 4 have no sighting: the last known box is held.
    assert boxes[4][0] == pytest.approx(100.0)  # type: ignore[index]
    assert boxes[5][0] == pytest.approx(300.0)  # type: ignore[index]


def test_identity_boxes_backfills_a_leading_gap() -> None:
    boxes = identity_boxes([_tracklet(1, [3, 4])], start_frame=0, end_frame=6)
    assert boxes[0] is not None
    assert boxes[0][0] == pytest.approx(100.0)  # type: ignore[index]


def test_identity_boxes_without_fill_keeps_the_gaps() -> None:
    boxes = identity_boxes([_tracklet(1, [3])], start_frame=0, end_frame=5, fill=False)
    assert boxes[0] is None and boxes[3] is not None


def test_identity_boxes_of_an_absent_tracklet_is_all_none() -> None:
    boxes = identity_boxes([_tracklet(1, [100])], start_frame=0, end_frame=3)
    # Leading gap filled from the first sighting later in the range -- there is
    # none inside it, so every frame is empty.
    assert all(box is None for box in boxes)


def test_identity_boxes_rejects_an_inverted_range() -> None:
    with pytest.raises(ValueError, match="precedes start_frame"):
        identity_boxes([], start_frame=10, end_frame=5)


# --------------------------------------------------------------------------- #
# crops, for real
# --------------------------------------------------------------------------- #


def test_crop_tile_of_no_box_is_black() -> None:
    tile = crop_tile(np.full((100, 100, 3), 255, dtype=np.uint8), None, size=32)
    assert tile.shape == (32, 32, 3)
    assert tile.max() == 0


def test_crop_tile_resizes_and_keeps_the_colour() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[40:70, 40:70] = (255, 0, 0)
    tile = crop_tile(frame, (40.0, 40.0, 30.0, 30.0), size=32, margin=0.0)
    assert tile.shape == (32, 32, 3)
    assert int(tile[:, :, 0].mean()) > 200


def test_iter_tiles_refuses_mismatched_lengths() -> None:
    frames = [np.zeros((10, 10, 3), dtype=np.uint8)] * 3
    with pytest.raises(ValueError, match="more frames than boxes"):
        list(iter_tiles(frames, [None, None]))
    with pytest.raises(ValueError, match="more boxes than frames"):
        list(iter_tiles(frames[:1], [None, None]))


def test_write_crop_video_produces_a_playable_video(
    single_shot_video: Path, tmp_path: Path
) -> None:
    """The extractor is handed this file; a crop that is the wrong size or
    length conditions every extraction on the wrong thing."""

    from avannotate.ffmpeg import probe_media

    info = probe_media(single_shot_video)
    target = tmp_path / "crop.mp4"
    boxes = [(20.0, 20.0, 40.0, 40.0)] * 10
    write_crop_video(
        single_shot_video,
        target,
        width=info.width,
        height=info.height,
        start_time=0.0,
        frame_count=10,
        boxes=boxes,
        fps=info.fps,
        size=64,
    )

    assert target.is_file()
    cropped = probe_media(target)
    assert (cropped.width, cropped.height) == (64, 64)
    assert cropped.frame_count == 10


def test_write_crop_video_rejects_an_empty_range(
    single_shot_video: Path, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="frame_count must be positive"):
        write_crop_video(
            single_shot_video, tmp_path / "x.mp4", width=320, height=240,
            start_time=0.0, frame_count=0, boxes=[], fps=25.0,
        )


# --------------------------------------------------------------------------- #
# wav helpers
# --------------------------------------------------------------------------- #


def test_write_and_read_a_wav_round_trip(tmp_path: Path) -> None:
    samples = (np.sin(np.linspace(0, 20, 8000)) * 0.5).astype(np.float32)
    path = write_pcm16(tmp_path / "tone.wav", samples, sample_rate=16000)

    rate, frames = read_info(path)
    assert rate == 16000
    assert frames == 8000

    back = read_window(path, start_seconds=0.0, duration_seconds=1.0)
    assert len(back) == 8000
    assert np.allclose(back, samples, atol=1e-4)


def test_read_window_clamps_past_the_end(tmp_path: Path) -> None:
    path = write_pcm16(tmp_path / "t.wav", np.zeros(1600, dtype=np.float32), sample_rate=16000)
    assert len(read_window(path, start_seconds=0.05, duration_seconds=5.0)) == 800


def test_read_window_of_a_non_pcm16_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(1)
        handle.setframerate(16000)
        handle.writeframes(b"\x00" * 100)
    with pytest.raises(ValueError, match="not PCM16"):
        read_window(path, start_seconds=0.0, duration_seconds=1.0)


def test_write_clips_rather_than_wraps(tmp_path: Path) -> None:
    """A sample outside the range is a processing artefact; wrapping it turns a
    loud passage into a burst of noise."""

    loud = np.array([2.0, -2.0, 0.0], dtype=np.float32)
    path = write_pcm16(tmp_path / "loud.wav", loud, sample_rate=16000)
    back = read_window(path, start_seconds=0.0, duration_seconds=1.0)
    assert back[0] == pytest.approx(1.0, abs=1e-4)
    assert back[1] == pytest.approx(-1.0, abs=1e-4)


def test_slice_samples_takes_a_window() -> None:
    samples = np.arange(1000, dtype=np.float32)
    assert list(slice_samples(samples, sample_rate=1000, start=0.1, end=0.2)) == [
        float(v) for v in range(100, 200)
    ]


def test_slice_samples_beyond_the_buffer_is_empty_not_an_error() -> None:
    silence = np.zeros(10, dtype=np.float32)
    assert len(slice_samples(silence, sample_rate=1000, start=5.0, end=6.0)) == 0


# --------------------------------------------------------------------------- #
# the adapter's contract, without the package
# --------------------------------------------------------------------------- #


def test_clearvoice_absent_gives_install_instructions(absent_module) -> None:
    """The error path, exercised whether or not this machine has the package.

    It used to rely on ClearerVoice being absent, which is true on a
    development box and false on any machine that runs S7.
    """

    with absent_module("clearvoice"), pytest.raises(TseError, match="pip install clearvoice"):
        build_extractor({"backend": "clearvoice"})


def test_an_unknown_backend_is_rejected() -> None:
    with pytest.raises(TseError, match="unknown extraction backend"):
        build_extractor({"backend": "spex"})


# --------------------------------------------------------------------------- #
# the stage
# --------------------------------------------------------------------------- #


class _ToneExtractor:
    """Writes a tone covering the crop window, as the real one would."""

    name = "stub"

    def __init__(self, *, silent: bool = False) -> None:
        self.silent = silent
        self.calls: list[Path] = []

    def extract(self, video: Path, output_dir: Path) -> Path:
        from avannotate.ffmpeg import probe_media

        self.calls.append(video)
        info = probe_media(video)
        count = max(1, int(round(info.duration * 16000)))
        samples = (
            np.zeros(count, dtype=np.float32)
            if self.silent
            else (np.sin(np.linspace(0, 100, count)) * 0.4).astype(np.float32)
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        return write_pcm16(output_dir / f"{video.stem}.wav", samples, sample_rate=16000)


def _context(source: Path, root: Path, **config: object) -> StageContext:
    return StageContext(
        video_id=source.stem, source=source, work_dir=root / "work" / source.stem,
        config=config,
    )




# --------------------------------------------------------------------------- #
# the stage
# --------------------------------------------------------------------------- #


class _ToneExtractor:
    """Writes a tone covering the crop window, as the real one would."""

    name = "stub"

    def __init__(self, *, silent: bool = False) -> None:
        self.silent = silent
        self.calls: list[Path] = []

    def extract(self, video: Path, output_dir: Path) -> Path:
        from avannotate.ffmpeg import probe_media

        self.calls.append(video)
        info = probe_media(video)
        count = max(1, int(round(info.duration * 16000)))
        samples = (
            np.zeros(count, dtype=np.float32)
            if self.silent
            else (np.sin(np.linspace(0, 100, count)) * 0.4).astype(np.float32)
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        return write_pcm16(output_dir / f"{video.stem}.wav", samples, sample_rate=16000)


def _write_upstream(
    context: StageContext,
    *,
    frames: tuple[int, ...] = tuple(range(10, 40, 2)),
    speaking: list[list[float]] | None = None,
) -> None:
    """S2's, S3's and S6's artifacts, written by hand.

    This stage only reads them, and a test pattern gives the tracker nothing to
    find -- so writing them directly is what lets each case vary one thing.
    An empty ``frames`` writes an empty track, which is how the "nothing to
    extract" case is set up.
    """

    detections = [
        {
            "frame": frame,
            "time": frame / 25.0,
            "x": 100.0,
            "y": 50.0,
            "w": 40.0,
            "h": 50.0,
            "score": 0.9,
        }
        for frame in frames
    ]
    span = (frames[-1] - frames[0]) / 25.0 if detections else 0.0
    context.output(s2_tracks.STAGE, s2_tracks.TRACKS_NAME).write_text(
        json.dumps(
            {
                "track_id": 1,
                "start_frame": frames[0] if detections else 0,
                "end_frame": frames[-1] if detections else 0,
                "hits": len(detections),
                "quality": {
                    "frames": len(detections),
                    "span_seconds": span,
                    "mean_score": 0.9,
                    "mean_width": 40.0,
                    "mean_height": 50.0,
                    "motion": 0.05,
                },
                "detections": detections,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    identities = [{"face_id": "F001", "track_ids": [1]}] if detections else []
    context.output(s3_cluster.STAGE, s3_cluster.IDENTITIES_NAME).write_text(
        json.dumps({"schema_version": "x", "identities": identities}), encoding="utf-8"
    )
    context.output(s6_associate.STAGE, s6_associate.ASSIGNMENTS_NAME).write_text(
        json.dumps(
            {
                "schema_version": "x",
                "assignments": [],
                "identities": [
                    {
                        "face_id": "F001",
                        "track_ids": [1],
                        "speaking_intervals": (
                            speaking if speaking is not None else [[0.5, 1.2]]
                        ),
                    }
                ]
                if detections
                else [],
            }
        ),
        encoding="utf-8",
    )


def _staged(source: Path, root: Path, **config: object) -> StageContext:
    context = _context(source, root, context_seconds=0.0, **config)
    s0_preprocess.run(context)
    _write_upstream(context, **config.pop("upstream", {}))  # type: ignore[arg-type]
    return context


def test_stage_writes_one_file_per_segment(
    single_shot_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(single_shot_video, tmp_path, context_seconds=0.0)
    s0_preprocess.run(context)
    _write_upstream(context)

    extractor = _ToneExtractor()
    monkeypatch.setattr(s7_tse, "build_extractor", lambda _: extractor)
    result = s7_tse.run(context)

    assert not result.skipped
    assert result.summary["written"] == 1
    assert len(extractor.calls) == 1

    written = s7_tse.load_segments(context)
    assert len(written) == 1
    assert written[0]["name"] == "F001_0000"
    assert Path(context.work_dir / str(written[0]["audio"])).is_file()


def test_the_written_file_covers_the_segment_not_the_crop_window(
    single_shot_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The context added for the extractor's benefit has to come back off, or
    every file would start half a second early."""

    context = _context(single_shot_video, tmp_path, context_seconds=0.4)
    s0_preprocess.run(context)
    _write_upstream(context, speaking=[[1.0, 2.0]])

    monkeypatch.setattr(s7_tse, "build_extractor", lambda _: _ToneExtractor())
    s7_tse.run(context)

    written = s7_tse.load_segments(context)[0]
    # The segment is 1.15s after padding; the crop window is 1.95s.
    assert written["duration"] == pytest.approx(1.15, abs=0.05)
    assert written["samples"] / written["sample_rate"] == pytest.approx(1.15, abs=0.05)


def test_a_silent_extraction_is_flagged(
    single_shot_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one extraction failure visible without a reference signal."""

    context = _context(single_shot_video, tmp_path, context_seconds=0.0)
    s0_preprocess.run(context)
    _write_upstream(context)

    monkeypatch.setattr(s7_tse, "build_extractor", lambda _: _ToneExtractor(silent=True))
    result = s7_tse.run(context)

    assert result.summary["silent"] == 1
    assert s7_tse.load_segments(context)[0]["silent"] is True


def test_the_crop_actually_contains_the_tracked_face(
    single_shot_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The claim the whole workaround rests on: hand over a video holding one
    face, and the extractor has no choice to get wrong."""

    context = _context(single_shot_video, tmp_path, context_seconds=0.0, keep_crops=True)
    s0_preprocess.run(context)
    _write_upstream(context, frames=tuple(range(5, 45, 2)), speaking=[[0.4, 1.4]])

    monkeypatch.setattr(s7_tse, "build_extractor", lambda _: _ToneExtractor())
    s7_tse.run(context)

    crops = sorted((context.stage_dir(s7_tse.STAGE) / "crops").glob("*.mp4"))
    assert len(crops) == 1

    from avannotate.ffmpeg import probe_media

    # 224 is what the extractor's visual encoder takes.
    assert probe_media(crops[0]).width == s7_tse.DEFAULT_CROP_SIZE
    assert probe_media(crops[0]).height == s7_tse.DEFAULT_CROP_SIZE


def test_crops_are_removed_unless_asked_for(
    single_shot_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A thousand segments is a hundred thousand files if they are kept."""

    context = _context(single_shot_video, tmp_path, context_seconds=0.0)
    s0_preprocess.run(context)
    _write_upstream(context)
    monkeypatch.setattr(s7_tse, "build_extractor", lambda _: _ToneExtractor())
    s7_tse.run(context)

    crops = list((context.stage_dir(s7_tse.STAGE) / "crops").glob("*.mp4"))
    assert crops == []


def test_an_identity_with_no_tracklets_is_reported_not_dropped(
    single_shot_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S6 can assign a speaker to a person whose tracklets S3 dropped."""

    context = _context(single_shot_video, tmp_path, context_seconds=0.0)
    s0_preprocess.run(context)
    _write_upstream(context, frames=())
    context.output(s6_associate.STAGE, s6_associate.ASSIGNMENTS_NAME).write_text(
        json.dumps(
            {
                "schema_version": "x",
                "assignments": [],
                "identities": [
                    {"face_id": "F001", "track_ids": [1], "speaking_intervals": [[0.5, 1.2]]}
                ],
            }
        ),
        encoding="utf-8",
    )

    extractor = _ToneExtractor()
    monkeypatch.setattr(s7_tse, "build_extractor", lambda _: extractor)
    result = s7_tse.run(context)

    assert result.summary["written"] == 0
    assert result.summary["skipped"] == 1
    assert extractor.calls == []

    payload = json.loads(s7_tse.segments_path(context).read_text())
    assert payload["skipped"][0]["reason"] == "F001 has no usable tracklets"


def test_second_run_skips(
    single_shot_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(single_shot_video, tmp_path, context_seconds=0.0)
    s0_preprocess.run(context)
    _write_upstream(context, frames=())
    monkeypatch.setattr(s7_tse, "build_extractor", lambda _: _ToneExtractor())

    assert not s7_tse.run(context).skipped
    assert s7_tse.run(context).skipped


def test_force_reruns(
    single_shot_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(single_shot_video, tmp_path, context_seconds=0.0)
    s0_preprocess.run(context)
    _write_upstream(context, frames=())
    monkeypatch.setattr(s7_tse, "build_extractor", lambda _: _ToneExtractor())

    s7_tse.run(context)
    assert not s7_tse.run(context, force=True).skipped


def test_loaders_fail_loudly_when_the_stage_has_not_run(tmp_path: Path) -> None:
    context = StageContext(
        video_id="x", source=tmp_path / "x.mp4", work_dir=tmp_path / "work" / "x"
    )
    with pytest.raises(FileNotFoundError, match="run s7-tse first"):
        s7_tse.load_segments(context)
    with pytest.raises(FileNotFoundError, match="run s7-tse first"):
        s7_tse.segments_path(context)
