"""Tests for caption planning, prompting, and the face-reference check.

The vision model is not installed here, so the adapter is not exercised.  What
is exercised is everything that decides what it is shown and what survives its
answer -- and the part that matters most is the check, because a caption naming
the wrong person is not caught by anything downstream.  Nothing else in the
pipeline reads a caption's names.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from avannotate.caption.plan import (
    frames_for,
    global_sample,
    identity_presence,
    plan_shot_samples,
    sample_times,
)
from avannotate.caption.prompt import global_prompt, shot_prompt
from avannotate.caption.types import ShotSample
from avannotate.caption.verify import check_references, flags_for, referenced
from avannotate.faces.track import TrackDetection, Tracklet, TrackQuality
from avannotate.interval import Interval
from avannotate.stages import s0_preprocess, s2_tracks, s3_cluster, s10_caption
from avannotate.stages.base import StageContext

RATE = 16000


def _tracklet(track_id: int, times: list[float]) -> Tracklet:
    detections = tuple(
        TrackDetection(
            frame_index=int(round(time * 25)),
            time=time,
            box=(100.0, 50.0, 40.0, 50.0),
            score=0.9,
        )
        for time in times
    )
    return Tracklet(
        track_id=track_id,
        start_frame=detections[0].frame_index,
        end_frame=detections[-1].frame_index,
        hits=len(detections),
        quality=TrackQuality(
            frames=len(detections),
            span_seconds=times[-1] - times[0] if len(times) > 1 else 0.0,
            mean_score=0.9,
            mean_width=40.0,
            mean_height=50.0,
            motion=0.05,
        ),
        detections=detections,
    )


def _tracklet_row(tracklet: Tracklet) -> dict[str, object]:
    """S2's on-disk row, which is not the dataclass -- the dataclass is what
    ``load_tracklets`` rebuilds from it."""

    quality = tracklet.quality
    return {
        "track_id": tracklet.track_id,
        "start_frame": tracklet.start_frame,
        "end_frame": tracklet.end_frame,
        "hits": tracklet.hits,
        "quality": {
            "frames": quality.frames,
            "span_seconds": quality.span_seconds,
            "mean_score": quality.mean_score,
            "mean_width": quality.mean_width,
            "mean_height": quality.mean_height,
            "motion": quality.motion,
        },
        "detections": [item.to_dict() for item in tracklet.detections],
    }


def _shot(index: int, start: float, end: float) -> ShotSample:
    return ShotSample(
        index=index, start=start, end=end, times=sample_times(start, end, count=2)
    )


# --------------------------------------------------------------------------- #
# which frames
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("duration", "expected"),
    [(0.5, 1), (4.0, 1), (12.0, 3), (32.0, 8), (600.0, 8)],
)
def test_frame_count_follows_the_shot_up_to_the_cap(duration: float, expected: int) -> None:
    """The floor because a scene cannot be described from nothing, the ceiling
    because request cost grows with the number of images."""

    assert (
        frames_for(duration, seconds_per_frame=4.0, low=1, high=8) == expected
    )


def test_sampling_avoids_the_edges_of_a_shot() -> None:
    """The first frame of a shot is the one most likely to be a transition."""

    times = sample_times(0.0, 8.0, count=4)

    assert times == (1.0, 3.0, 5.0, 7.0)
    assert 0.0 not in times


def test_a_single_frame_for_a_span_lands_at_its_start() -> None:
    """The global caption takes one frame per shot, and it should be from the
    beginning of the shot rather than from its middle."""

    assert sample_times(10.0, 20.0, count=1) == (10.0,)


def test_a_shot_past_the_end_of_the_video_is_clamped() -> None:
    """A shot detector reports a last shot that can run past the end, and
    sampling there asks ffmpeg for frames that do not exist."""

    samples = plan_shot_samples(
        [(1, 0.0, 30.0)],
        {},
        (),
        duration=10.0,
        seconds_per_frame=4.0,
        min_frames=1,
        max_frames=8,
        presence_gap=0.5,
        min_presence=0.2,
    )

    assert samples[0].end == 10.0
    assert max(samples[0].times) < 10.0


# --------------------------------------------------------------------------- #
# who is in them
# --------------------------------------------------------------------------- #


def test_presence_is_the_union_across_an_identitys_tracklets() -> None:
    """S3 rejoins fragments the tracker split, and missing that would drop a
    person from the roster whenever their time in a shot happened to be split
    -- and a missing name is a name the model cannot use."""

    found = identity_presence(
        {"F001": (1, 2)},
        [_tracklet(1, [1.0, 2.0]), _tracklet(2, [5.0, 6.0, 7.0])],
        window=Interval(0.0, 8.0),
        max_gap=0.5,
    )

    assert found == ("F001",)


def test_someone_briefly_visible_is_not_named() -> None:
    """A face detected once behind a shoulder should not put a name in the
    roster that the model is then asked to use."""

    found = identity_presence(
        {"F001": (1,)},
        [_tracklet(1, [1.0, 1.05])],
        window=Interval(0.0, 8.0),
        max_gap=0.0,
        min_presence=0.2,
    )

    assert found == ()


def test_someone_absent_from_a_shot_is_not_in_its_roster() -> None:
    found = identity_presence(
        {"F001": (1,), "F002": (2,)},
        [_tracklet(1, [1.0, 2.0]), _tracklet(2, [5.0, 6.0])],
        window=Interval(0.5, 2.5),
        max_gap=0.5,
    )

    assert found == ("F001",)


def test_a_track_id_that_no_longer_exists_is_skipped_not_fatal() -> None:
    """S3's mapping and S2's tracklets are separate files; a stale one should
    leave a person out, not stop the stage."""

    found = identity_presence(
        {"F001": (1, 99)}, [_tracklet(1, [1.0, 2.0])], window=Interval(0.0, 8.0), max_gap=0.5
    )

    assert found == ("F001",)


def test_the_global_caption_gets_one_frame_from_the_start_of_each_shot() -> None:
    shots = [
        ShotSample(
            index=index,
            start=float(index) * 10,
            end=float(index) * 10 + 5,
            # Early in the shot rather than at its first frame, which is the
            # one most likely to be a transition.
            times=(float(index) * 10 + 1.25,),
            identities=("F001",) if index == 1 else ("F002",),
        )
        for index in range(3)
    ]

    times, everyone = global_sample(shots, max_frames=12)

    assert times == (1.25, 11.25, 21.25)
    assert everyone == ("F001", "F002")


def test_more_shots_than_the_budget_are_thinned_evenly() -> None:
    """Thinning must span the video, not stop at its first half."""

    shots = [
        ShotSample(
            index=index,
            start=float(index) * 10,
            end=float(index) * 10 + 5,
            times=(float(index) * 10,),
            identities=(),
        )
        for index in range(100)
    ]

    times, _ = global_sample(shots, max_frames=4)

    assert len(times) == 4
    assert times[0] < 200.0 < times[-1]


# --------------------------------------------------------------------------- #
# what the model is told
# --------------------------------------------------------------------------- #


def test_the_prompt_names_the_roster_and_forbids_other_names() -> None:
    prompt = shot_prompt(_shot(3, 10.0, 20.0), total=8)

    assert "F001" not in prompt  # nobody in this shot's empty roster
    assert "No tracked people" in prompt


def test_the_prompt_uses_the_script_number_for_the_shot() -> None:
    """Shot indices are 1-based because the rendered script writes them that
    way, and adding one here would name every shot wrongly."""

    shot = ShotSample(index=3, start=10.0, end=20.0, times=(12.0, 18.0), identities=("F001",))

    assert "shot 3 of 8" in shot_prompt(shot, total=8)


def test_the_prompt_asks_for_observation_rather_than_inference() -> None:
    """Inference in a caption is indistinguishable from observation once it is
    rendered, so the instruction has to be explicit."""

    prompt = shot_prompt(_shot(1, 0.0, 5.0), total=1)

    assert "Do not infer feelings" in prompt
    assert "Do not quote or mention any dialogue" in prompt


def test_the_global_prompt_counts_its_own_frames() -> None:
    prompt = global_prompt(identities=("F001", "F002"), frames=6, shots=9)

    assert "6 frames" in prompt
    assert "9 shots" in prompt
    assert "F001, F002" in prompt


# --------------------------------------------------------------------------- #
# the names that come back
# --------------------------------------------------------------------------- #


def test_names_are_found_in_prose() -> None:
    assert referenced("F001 and F002 sit on a sofa") == ("F001", "F002")
    assert referenced("F001, then F001 again") == ("F001",)


def test_something_that_only_looks_like_a_name_is_not_one() -> None:
    """The pattern has to be exact, or a caption about a car will invent a
    person."""

    assert referenced("an F0012 model and XF001 and F0 and F01") == ()


def test_a_caption_that_names_nobody_is_left_alone() -> None:
    checked = check_references(
        "A bright living room with a sofa and a coffee table.", allowed=("F001",)
    )

    assert checked.dropped == ()
    assert checked.text == "A bright living room with a sofa and a coffee table."
    assert not checked.empty


def test_a_name_that_was_not_on_the_roster_is_removed() -> None:
    """A name the tracker never saw is a claim this pipeline cannot support,
    and the alternative to removing it is shipping it."""

    checked = check_references("F003 stands by the window.", allowed=("F001", "F002"))

    assert checked.dropped == ("F003",)
    assert checked.text == "stands by the window."
    assert checked.referenced == ()


def test_a_kept_name_stays_and_a_removed_one_goes() -> None:
    checked = check_references(
        "F001 and F007 are sitting on a sofa.", allowed=("F001", "F002")
    )

    assert checked.text == "F001 and are sitting on a sofa."
    assert checked.referenced == ("F001",)
    assert checked.dropped == ("F007",)


def test_removal_does_not_leave_a_double_space_or_a_stray_comma() -> None:
    """What is left has to read as prose, since nothing marks where a name was
    taken out."""

    checked = check_references("F007 , F008 then left.", allowed=())

    assert checked.text == "then left."
    assert "  " not in checked.text
    assert not checked.text.startswith(",")


def test_a_caption_that_was_only_names_is_reported_empty() -> None:
    """A blank line that looks like a description is worse than an obvious
    absence, so the caller has to be able to tell."""

    checked = check_references("F007", allowed=())

    assert checked.text == ""
    assert checked.empty


def test_the_two_ways_a_name_can_be_wrong_are_told_apart() -> None:
    """A name that exists nowhere is a hallucination; a real person named in a
    shot they are not in is a grounding failure.  A run with many of one and
    none of the other is telling you something."""

    invented = check_references("F009 arrives.", allowed=("F001",))
    misattributed = check_references("F002 arrives.", allowed=("F001",))

    assert flags_for(invented, known={"F001"}) == ("invented_name",)
    assert flags_for(misattributed, known={"F001", "F002"}) == ("misattributed_name",)


def test_a_clean_caption_has_no_flags() -> None:
    checked = check_references("A sofa and a window.", allowed=("F001",))
    assert flags_for(checked, known={"F001"}) == ()


# --------------------------------------------------------------------------- #
# the stage
# --------------------------------------------------------------------------- #


class _StubCaptioner:
    """Answers whatever the test queued, and records what it was shown."""

    name = "stub"

    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)
        self.calls: list[tuple[int, str]] = []

    def caption(self, frames: list[np.ndarray], *, prompt: str) -> str:
        self.calls.append((len(frames), prompt))
        if not self.answers:
            return "A room with a sofa."
        # The global request is the one with more than one shot's worth of
        # frames in it; the queue is consumed in the order the stage asks.
        return self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]


def _context(source: Path, root: Path, **config: object) -> StageContext:
    return StageContext(
        video_id=source.stem, source=source, work_dir=root / "work" / source.stem, config=config
    )


def _write_upstream(
    context: StageContext,
    *,
    shots: list[tuple[int, float, float]],
    tracks: list[Tracklet],
    identities: dict[str, list[int]],
) -> None:
    """S0's, S2's and S3's artifacts, written by hand: S10 only reads them."""

    context.output(s0_preprocess.STAGE, s0_preprocess.SHOTS_NAME).write_text(
        json.dumps(
            {
                "schema_version": "x",
                "shots": [
                    {"index": index, "start": start, "end": end}
                    for index, start, end in shots
                ],
            }
        ),
        encoding="utf-8",
    )
    context.output(s2_tracks.STAGE, s2_tracks.TRACKS_NAME).write_text(
        "".join(json.dumps(_tracklet_row(tracklet)) + "\n" for tracklet in tracks),
        encoding="utf-8",
    )
    context.output(s3_cluster.STAGE, s3_cluster.IDENTITIES_NAME).write_text(
        json.dumps(
            {
                "schema_version": "x",
                "identities": [
                    {"face_id": face_id, "track_ids": track_ids}
                    for face_id, track_ids in sorted(identities.items())
                ],
            }
        ),
        encoding="utf-8",
    )


def _staged(
    source: Path,
    root: Path,
    *,
    shots: list[tuple[int, float, float]],
    tracks: list[Tracklet],
    identities: dict[str, list[int]],
    **config: object,
) -> StageContext:
    context = _context(source, root, **config)
    s0_preprocess.run(context)
    _write_upstream(context, shots=shots, tracks=tracks, identities=identities)
    return context


UPSTREAM = {
    "shots": [(1, 0.0, 3.0)],
    "tracks": [_tracklet(1, [0.5, 1.0, 1.5, 2.0, 2.5])],
    "identities": {"F001": [1]},
}


def test_the_stage_captions_every_shot_and_the_video(
    single_shot_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _staged(single_shot_video, tmp_path, **UPSTREAM)
    monkeypatch.setattr(
        s10_caption, "build_captioner", lambda _: _StubCaptioner("A sofa.", "A living room.")
    )

    result = s10_caption.run(context)

    assert not result.skipped
    assert result.summary["shots"] == 1
    assert s10_caption.load_captions(context)[0].caption == "A sofa."
    assert s10_caption.load_global_caption(context).caption == "A living room."


def test_the_model_is_told_who_is_in_the_shot(
    single_shot_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _staged(single_shot_video, tmp_path, **UPSTREAM)
    stub = _StubCaptioner("A sofa.", "A living room.")
    monkeypatch.setattr(s10_caption, "build_captioner", lambda _: stub)

    s10_caption.run(context)

    shot_call = stub.calls[0]
    assert "F001" in shot_call[1]
    assert shot_call[0] >= 1


def test_a_name_the_model_invented_never_reaches_the_caption(
    single_shot_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _staged(single_shot_video, tmp_path, **UPSTREAM)
    monkeypatch.setattr(
        s10_caption,
        "build_captioner",
        lambda _: _StubCaptioner("F001 and F009 sit on a sofa.", "A living room."),
    )

    s10_caption.run(context)
    written = s10_caption.load_captions(context)[0]

    assert written.caption == "F001 and sit on a sofa."
    assert written.referenced == ("F001",)
    assert written.dropped == ("F009",)
    assert "invented_name" in written.flags


def test_the_global_caption_is_checked_against_every_identity(
    single_shot_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It is shown one frame per shot, so anyone in the video may appear in it."""

    context = _staged(single_shot_video, tmp_path, **UPSTREAM)
    monkeypatch.setattr(
        s10_caption,
        "build_captioner",
        lambda _: _StubCaptioner("A sofa.", "F001 and F002 in a living room."),
    )

    s10_caption.run(context)

    assert s10_caption.load_global_caption(context).caption == (
        "F001 and in a living room."
    )


def test_a_shot_with_no_roster_is_told_so_rather_than_left_silent(
    single_shot_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Silence leaves the model free to name somebody from the previous request."""

    context = _staged(
        single_shot_video,
        tmp_path,
        shots=[(1, 0.0, 3.0)],
        tracks=[],
        identities={},
    )
    stub = _StubCaptioner("An empty room.")
    monkeypatch.setattr(s10_caption, "build_captioner", lambda _: stub)

    s10_caption.run(context)

    assert "No tracked people" in stub.calls[0][1]
    assert s10_caption.load_captions(context)[0].caption == "An empty room."


def test_the_summary_counts_the_names_that_were_removed(
    single_shot_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _staged(single_shot_video, tmp_path, **UPSTREAM)
    monkeypatch.setattr(
        s10_caption,
        "build_captioner",
        lambda _: _StubCaptioner("F007 stands.", "F008 sits."),
    )

    s10_caption.run(context)
    summary = json.loads(
        (context.work_dir / s10_caption.STAGE / s10_caption.SUMMARY_NAME).read_text()
    )

    assert summary["dropped"] == {"F007": 1}
    assert summary["captioned"] == 1


def test_second_run_skips(
    single_shot_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _staged(single_shot_video, tmp_path, **UPSTREAM)
    monkeypatch.setattr(s10_caption, "build_captioner", lambda _: _StubCaptioner("A sofa."))

    assert not s10_caption.run(context).skipped
    assert s10_caption.run(context).skipped


def test_loaders_fail_loudly_when_the_stage_has_not_run(tmp_path: Path) -> None:
    context = StageContext(
        video_id="x", source=tmp_path / "x.mp4", work_dir=tmp_path / "work" / "x"
    )
    with pytest.raises(FileNotFoundError, match="run s10-caption first"):
        s10_caption.load_captions(context)
    with pytest.raises(FileNotFoundError, match="run s10-caption first"):
        s10_caption.captions_path(context)


def test_a_video_with_no_shots_writes_empty_outputs_without_loading_a_model(
    single_shot_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Building the captioner loads tens of gigabytes; doing that to describe
    nothing is the one cost in this stage worth guarding."""

    context = _staged(
        single_shot_video, tmp_path, shots=[], tracks=[], identities={}
    )

    def explode(_: object) -> object:
        raise AssertionError("the captioner should not be built for an empty video")

    monkeypatch.setattr(s10_caption, "build_captioner", explode)
    result = s10_caption.run(context)

    assert result.summary["shots"] == 0
    assert s10_caption.load_captions(context) == ()
    assert s10_caption.load_global_caption(context).caption == ""
