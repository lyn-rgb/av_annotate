"""Tests for the deliverable's assembly and its serialisation.

This is the join point: ten stages' records become one document, and a mistake
here is not an exception -- it is a transcript under the wrong name, or a person
missing from the face list, in a file that reads as finished.  So the tests are
about what ends up in the output rather than about whether it crashes.

The other half is the serialisation contract.  ``annotation.json`` is written by
one function and read by another, and they have to agree exactly; the round trip
is asserted directly, and the writer checks its own output through the reader
before returning it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from avannotate.annotation import (
    annotation_from_dict,
    annotation_to_dict,
    parse_script,
)
from avannotate.audio.types import SpeakerTurn
from avannotate.compose.collect import face_tracks, shots, utterances
from avannotate.faces.track import TrackDetection, Tracklet, TrackQuality
from avannotate.interval import Interval
from avannotate.schema import Annotation, FaceTrack, Language, Utterance, VideoMeta, Word
from avannotate.segment import SpeechSegment
from avannotate.stages import (
    s0_preprocess,
    s2_tracks,
    s3_cluster,
    s4_diarize,
    s6_associate,
    s7_tse,
    s8_asr,
    s9_paralinguistic,
    s10_caption,
    s11_compose,
)
from avannotate.stages.base import StageContext

RATE = 16000


def _tracklet(track_id: int, times: list[float], *, frames: int | None = None) -> Tracklet:
    detections = tuple(
        TrackDetection(
            frame_index=int(round(time * 25)),
            time=time,
            box=(100.0, 50.0, 40.0, 50.0),
            score=0.9,
        )
        for time in times
    )
    count = frames if frames is not None else len(detections)
    return Tracklet(
        track_id=track_id,
        start_frame=detections[0].frame_index,
        end_frame=detections[-1].frame_index,
        hits=len(detections),
        quality=TrackQuality(
            frames=count,
            span_seconds=times[-1] - times[0] if len(times) > 1 else 0.0,
            mean_score=0.9,
            mean_width=40.0,
            mean_height=50.0,
            motion=0.05,
        ),
        detections=detections,
    )


def _row(tracklet: Tracklet) -> dict[str, object]:
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


def _segment(
    name: str = "F001_0000",
    *,
    identity: str = "F001",
    start: float = 0.5,
    end: float = 1.5,
) -> SpeechSegment:
    return SpeechSegment(
        identity=identity,
        name=name,
        start=start,
        end=end,
        audio=f"s7-tse/audio/{identity}/{name}.wav",
    )


def _transcript(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "name": "F001_0000",
        "text": "I was late for work today",
        "words": [
            {"text": " I", "start": 0.6, "end": 0.8},
            {"text": " was", "start": 0.8, "end": 1.0},
        ],
        "avg_logprob": -0.2,
        "flags": [],
    }
    return {**base, **overrides}


# --------------------------------------------------------------------------- #
# the serialisation contract
# --------------------------------------------------------------------------- #


def _annotation() -> Annotation:
    return Annotation(
        video=VideoMeta(
            video_id="clip",
            path="/data/clip.mp4",
            duration=31.0,
            fps=25.0,
            width=1920,
            height=1080,
        ),
        utterances=(
            Utterance(
                face_id="F001",
                start=0.5,
                end=2.0,
                text="I was late for work today",
                tag="whispering",
                audio_path="s7-tse/audio/F001/F001_0000.wav",
                words=(Word(text="I", start=0.6, end=0.8),),
                confidence=-0.2,
                flags=("low_confidence",),
            ),
            Utterance(face_id="F000", start=3.0, end=4.0, text="Off we go"),
        ),
        shots=(
            shots([(1, 0.0, 12.4), (2, 12.4, 31.0)], {1: "A bright living room."})[0],
            shots([(1, 0.0, 12.4), (2, 12.4, 31.0)], {1: "A bright living room."})[1],
        ),
        face_tracks=(
            FaceTrack(
                face_id="F001",
                first_seen=0.4,
                last_seen=20.0,
                speaks=True,
                total_speech=2.5,
                tracklets=("1", "2"),
                quality={"mean_score": 0.9, "motion": 0.05},
            ),
        ),
        global_caption="Two people are discussing in a spacious living room.",
        language=Language(code="en", confidence=0.98, source="segments"),
        stats={"segments": 3},
    )


def test_the_deliverable_round_trips_exactly() -> None:
    """The writer and the reader are two functions that have to agree.

    A field the writer renames and the reader does not is caught here rather
    than by a consumer a month later.
    """

    original = _annotation()
    recovered = annotation_from_dict(annotation_to_dict(original))

    assert recovered == original


def test_the_writer_refuses_to_emit_something_it_cannot_read() -> None:
    """A document this pipeline's own reader rejects would fail at the far end
    of a batch, on the one video nobody is watching."""

    payload = annotation_to_dict(_annotation())

    assert payload["schema_version"] == "av-annotation-v1"
    assert isinstance(payload["utterances"], list)


def test_an_annotation_with_no_language_or_stats_still_reads_back() -> None:
    """The optional blocks are optional in both directions."""

    bare = Annotation(
        video=VideoMeta(video_id="v", path="v.mp4", duration=1.0, fps=25.0, width=2, height=2)
    )
    payload = annotation_to_dict(bare)

    assert "language" not in payload
    assert "stats" not in payload
    assert annotation_from_dict(payload) == bare


# --------------------------------------------------------------------------- #
# assembling the parts
# --------------------------------------------------------------------------- #


def test_a_person_who_never_speaks_is_still_a_face_track() -> None:
    """The face list answers "who is in this video", not "who talked in it"."""

    tracks = face_tracks(
        {"F001": (1,), "F002": (2,)},
        [_tracklet(1, [0.5, 1.0, 1.5]), _tracklet(2, [4.0, 4.5])],
        {"F001": (Interval(0.5, 1.5),)},
    )

    assert [track.face_id for track in tracks] == ["F001", "F002"]
    assert [track.speaks for track in tracks] == [True, False]
    assert tracks[1].total_speech == 0.0


def test_a_persons_span_comes_from_detections_not_tracklet_bounds() -> None:
    """A tracklet can begin before its first detection -- the tracker's Kalman
    prediction is not evidence that anyone was there."""

    tracklet = _tracklet(1, [2.0, 3.0])
    # start_frame comes from the first detection, so push it earlier to prove
    # the bounds are not what is read.
    stretched = Tracklet(
        track_id=tracklet.track_id,
        start_frame=0,
        end_frame=tracklet.end_frame,
        hits=tracklet.hits,
        quality=tracklet.quality,
        detections=tracklet.detections,
    )

    tracks = face_tracks({"F001": (1,)}, [stretched], {"F001": (Interval(2.0, 3.0),)})

    assert tracks[0].first_seen == 2.0


def test_a_face_with_no_detections_is_left_out_rather_than_faked() -> None:
    tracks = face_tracks({"F001": (99,)}, [], {})
    assert tracks == ()


def test_an_empty_transcript_is_not_an_utterance() -> None:
    """A line like ``<F001> <S><E>`` promises speech the pipeline cannot keep."""

    built = utterances(
        [_segment()],
        {"F001_0000": _transcript(text="   ")},
        {},
    )

    assert built == ()


def test_a_segment_with_no_transcript_is_skipped() -> None:
    assert utterances([_segment()], {}, {}) == ()


def test_an_utterance_carries_the_tag_audio_words_and_confidence() -> None:
    built = utterances(
        [_segment()],
        {"F001_0000": _transcript()},
        {"F001_0000": "whispering"},
    )

    assert len(built) == 1
    item = built[0]
    assert item.face_id == "F001"
    assert item.tag == "whispering"
    assert item.audio_path == "s7-tse/audio/F001/F001_0000.wav"
    assert item.confidence == pytest.approx(-0.2)
    assert [word.text for word in item.words] == ["I", "was"]


def test_utterances_come_out_in_timeline_order() -> None:
    """Two runs over the same video have to produce the same file."""

    built = utterances(
        [
            _segment("F002_0000", identity="F002", start=5.0, end=6.0),
            _segment("F001_0000", start=1.0, end=2.0),
        ],
        {
            "F001_0000": _transcript(name="F001_0000"),
            "F002_0000": _transcript(name="F002_0000"),
        },
        {},
    )

    assert [item.face_id for item in built] == ["F001", "F002"]


def test_a_shot_with_no_caption_keeps_its_boundaries() -> None:
    """Dropping it would silently shorten the video: the shot list is the
    timeline, and an uncaptioned shot is still a shot."""

    built = shots([(1, 0.0, 5.0), (2, 5.0, 9.0)], {1: "A room."})

    assert [item.index for item in built] == [1, 2]
    assert built[1].caption == ""
    assert built[1].end == 9.0


# --------------------------------------------------------------------------- #
# the stage
# --------------------------------------------------------------------------- #


def _write_upstream(
    context: StageContext,
    *,
    tracklets: list[Tracklet],
    identities: dict[str, list[int]],
    turns: list[tuple[str, float, float]],
    speech: dict[str, list[list[float]]],
    segments: list[SpeechSegment],
    transcripts: list[dict[str, object]],
    tags: list[dict[str, object]],
    captions: list[dict[str, object]],
    global_caption: dict[str, object],
    language: dict[str, object],
) -> None:
    """Nine stages' artifacts, written by hand: S11 only reads them."""

    s2_tracks_path = context.output(s2_tracks.STAGE, s2_tracks.TRACKS_NAME)
    s2_tracks_path.write_text(
        "".join(json.dumps(_row(tracklet)) + "\n" for tracklet in tracklets),
        encoding="utf-8",
    )
    context.output(s3_cluster.STAGE, s3_cluster.IDENTITIES_NAME).write_text(
        json.dumps(
            {
                "schema_version": "x",
                "identities": [
                    {"face_id": face_id, "track_ids": ids}
                    for face_id, ids in sorted(identities.items())
                ],
            }
        ),
        encoding="utf-8",
    )
    context.output(s4_diarize.STAGE, s4_diarize.TURNS_NAME).write_text(
        "".join(
            json.dumps({"speaker": speaker, "start": start, "end": end}) + "\n"
            for speaker, start, end in turns
        ),
        encoding="utf-8",
    )
    context.output(s6_associate.STAGE, s6_associate.ASSIGNMENTS_NAME).write_text(
        json.dumps(
            {
                "schema_version": "x",
                "assignments": [
                    {"speaker": "spk_00", "face_id": "F001", "score": 0.9}
                ],
                "identities": [
                    {"face_id": face_id, "track_ids": [], "speaking_intervals": intervals}
                    for face_id, intervals in sorted(speech.items())
                ],
                "offscreen": {"speakers": [], "seconds": 0.0},
            }
        ),
        encoding="utf-8",
    )
    context.output(s7_tse.STAGE, s7_tse.SEGMENTS_NAME).write_text(
        json.dumps(
            {
                "schema_version": "x",
                "segments": [
                    {**segment.to_dict(), "sample_rate": RATE, "silent": False}
                    for segment in segments
                ],
            }
        ),
        encoding="utf-8",
    )
    context.output(s8_asr.STAGE, s8_asr.TRANSCRIPTS_NAME).write_text(
        json.dumps(
            {
                "schema_version": "x",
                "language": language,
                "transcripts": transcripts,
            }
        ),
        encoding="utf-8",
    )
    context.output(s9_paralinguistic.STAGE, s9_paralinguistic.TAGS_NAME).write_text(
        json.dumps({"schema_version": "x", "segments": tags}),
        encoding="utf-8",
    )
    context.output(s10_caption.STAGE, s10_caption.CAPTIONS_NAME).write_text(
        json.dumps({"schema_version": "x", "shots": captions, "global": global_caption}),
        encoding="utf-8",
    )


def _staged(source: Path, root: Path, **config: object) -> StageContext:
    context = StageContext(
        video_id=source.stem,
        source=source,
        work_dir=root / "work" / source.stem,
        config=config,
    )
    s0_preprocess.run(context)
    _write_upstream(
        context,
        tracklets=[_tracklet(1, [0.4, 0.8, 1.2, 1.6, 2.0])],
        identities={"F001": [1]},
        turns=[("spk_00", 0.4, 2.0)],
        speech={"F001": [[0.5, 1.5]]},
        segments=[_segment()],
        transcripts=[_transcript()],
        tags=[
            {
                "name": "F001_0000",
                "tag": "whispering",
                "eligible": True,
                "flags": [],
                "choice": {"candidates": []},
            }
        ],
        captions=[
            {"index": 1, "start": 0.0, "end": 3.0, "caption": "A living room.", "flags": []}
        ],
        global_caption={"caption": "A bright living room.", "frames": 1, "flags": []},
        language={"code": "en", "confidence": 0.98, "source": "segments"},
    )
    return context


def test_the_stage_writes_the_script_the_json_and_the_report(
    single_shot_video: Path, tmp_path: Path
) -> None:
    context = _staged(single_shot_video, tmp_path)

    result = s11_compose.run(context)

    assert not result.skipped
    assert result.summary["utterances"] == 1
    assert result.summary["faces"] == 1
    assert s11_compose.script_path(context).is_file()
    assert s11_compose.annotation_path(context).is_file()
    assert s11_compose.report_path(context).is_file()


def test_the_written_deliverable_parses_back_into_what_it_came_from(
    single_shot_video: Path, tmp_path: Path
) -> None:
    """The end-to-end form of the format's whole promise."""

    context = _staged(single_shot_video, tmp_path)
    s11_compose.run(context)

    text = s11_compose.load_script(context)
    recovered = s11_compose.load_annotation(context)

    assert "<F001> whispering: <S>I was late for work today<E>" in text
    assert "[GLOBAL]" in text
    assert recovered.utterances[0].text == "I was late for work today"
    assert recovered.global_caption == "A bright living room."

    parsed = parse_script(text)
    assert len(parsed.utterances) == 1
    assert parsed.utterances[0].tag == "whispering"


def test_the_rendered_script_survives_its_own_parser(
    single_shot_video: Path, tmp_path: Path
) -> None:
    """The gate that would invalidate a whole corpus at once if it failed."""

    context = _staged(single_shot_video, tmp_path)
    s11_compose.run(context)
    report = json.loads(s11_compose.report_path(context).read_text())

    gates = {gate["name"]: gate for gate in report["gates"]}
    assert gates["render_round_trip"]["passed"], gates["render_round_trip"]["detail"]
    assert gates["format_hygiene"]["passed"], gates["format_hygiene"]["detail"]
    assert gates["id_integrity"]["passed"], gates["id_integrity"]["detail"]
    assert gates["timeline_sanity"]["passed"], gates["timeline_sanity"]["detail"]


def test_the_accounting_gate_compares_against_the_diarizers_own_total(
    single_shot_video: Path, tmp_path: Path
) -> None:
    """Attributed speech has to add up to the speech the diarizer heard."""

    context = _staged(single_shot_video, tmp_path)
    s11_compose.run(context)
    report = json.loads(s11_compose.report_path(context).read_text())

    gate = next(item for item in report["gates"] if item["name"] == "speech_accounting")
    # 1.0s attributed against a 1.6s turn: a 0.6s gap, inside the tolerance.
    assert gate["passed"], gate["detail"]
    assert "1.60" in gate["detail"]


def test_the_report_says_what_the_upstream_stages_found(
    single_shot_video: Path, tmp_path: Path
) -> None:
    """Notes are read from the stages' own records so the report cannot
    disagree with the stage that produced the data."""

    context = _staged(single_shot_video, tmp_path)
    s11_compose.run(context)
    report = json.loads(s11_compose.report_path(context).read_text())

    assert report["notes"]["offscreen_speakers"] == 0
    assert report["notes"]["caption_frames"] == 1
    assert report["metrics"]["utterance_count"] == 1.0
    assert report["metrics"]["speaking_face_count"] == 1.0


def test_second_run_skips(single_shot_video: Path, tmp_path: Path) -> None:
    context = _staged(single_shot_video, tmp_path)

    assert not s11_compose.run(context).skipped
    assert s11_compose.run(context).skipped


def test_loaders_fail_loudly_when_the_stage_has_not_run(tmp_path: Path) -> None:
    context = StageContext(
        video_id="x", source=tmp_path / "x.mp4", work_dir=tmp_path / "work" / "x"
    )
    with pytest.raises(FileNotFoundError, match="run s11-compose first"):
        s11_compose.script_path(context)
    with pytest.raises(FileNotFoundError, match="run s11-compose first"):
        s11_compose.annotation_path(context)
    with pytest.raises(FileNotFoundError, match="run s11-compose first"):
        s11_compose.report_path(context)


def test_an_offscreen_speaker_makes_the_accounting_gate_fail(
    single_shot_video: Path, tmp_path: Path
) -> None:
    """The gap the pipeline currently has, asserted rather than assumed.

    Off-screen speech produces no segment, because S7 skips any identity without
    tracklets and an off-screen speaker has none by definition.  So the speech
    the diarizer heard is larger than the speech attributed to anyone, and the
    accounting gate says so.  This is the gate working: it is the measurement
    that decides whether off-screen support is needed, and it is better read
    from a report than assumed either way.
    """

    context = _staged(single_shot_video, tmp_path)
    # 2.9 s of speech heard, 1.0 s attributed: a gap well outside the 1.5 s
    # tolerance, so the verdict cannot turn on the exact margin.
    context.output(s4_diarize.STAGE, s4_diarize.TURNS_NAME).write_text(
        json.dumps({"speaker": "spk_00", "start": 0.0, "end": 2.0}) + "\n"
        + json.dumps({"speaker": "spk_01", "start": 2.0, "end": 2.9}) + "\n",
        encoding="utf-8",
    )

    result = s11_compose.run(context, force=True)
    report = json.loads(s11_compose.report_path(context).read_text())
    gate = next(item for item in report["gates"] if item["name"] == "speech_accounting")

    assert not gate["passed"]
    assert "speech_accounting" in result.summary["failing"]
    # And the number is in the report rather than only the verdict, because the
    # size of the gap is what a decision would be made from.
    assert "off by" in gate["detail"]


def test_a_video_with_no_speech_reports_rather_than_fails(
    single_shot_video: Path, tmp_path: Path
) -> None:
    """A video with no speech is legitimate; failing it would be wrong."""

    context = _staged(single_shot_video, tmp_path)
    context.output(s4_diarize.STAGE, s4_diarize.TURNS_NAME).write_text("", encoding="utf-8")
    context.output(s6_associate.STAGE, s6_associate.ASSIGNMENTS_NAME).write_text(
        json.dumps({"schema_version": "x", "identities": [], "assignments": []}),
        encoding="utf-8",
    )

    s11_compose.run(context, force=True)
    report = json.loads(s11_compose.report_path(context).read_text())
    gate = next(item for item in report["gates"] if item["name"] == "speech_accounting")

    assert gate["passed"]
    assert "not checked" in gate["detail"]


def test_the_turn_total_is_a_union_not_a_sum() -> None:
    """Two people talking at once produce two turns and one stretch of time."""

    overlapping = [
        SpeakerTurn(speaker="spk_00", start=0.0, end=2.0),
        SpeakerTurn(speaker="spk_01", start=1.0, end=3.0),
    ]

    assert s11_compose._vad_seconds(overlapping) == pytest.approx(3.0)
    assert s11_compose._vad_seconds([]) is None


def test_an_utterance_is_clamped_to_the_video() -> None:
    """A segment's times are rounded to four decimals on their way to disk, and
    a value rounded *up* lands just past an unrounded duration.

    On the corpus's second video that was 37 microseconds: a real overshoot,
    small enough that the gate's old two-decimal message could not show it, and
    large enough that the gate was right to refuse it.  The video does not have
    those microseconds, so the span does not either.
    """

    built = utterances(
        [_segment(start=6.383, end=8.7170004)],
        {"F001_0000": _transcript()},
        {},
        limit=8.717,
    )

    assert len(built) == 1
    assert built[0].end == 8.717
    assert built[0].start == 6.383


def test_a_span_inside_the_video_is_left_exactly_where_it_is() -> None:
    """Guards the guard: a bound, not a nudge."""

    built = utterances(
        [_segment(start=0.5, end=1.5)],
        {"F001_0000": _transcript()},
        {},
        limit=10.0,
    )

    assert (built[0].start, built[0].end) == (0.5, 1.5)
