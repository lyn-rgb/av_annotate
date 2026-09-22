"""Stage S11: the deliverable.

Nine stages left records behind; this one reads them all and writes the two
files the rest of the world sees -- ``annotation.txt`` and ``annotation.json``
-- plus the quality report that says how much of it to believe.

It runs no model.  Every number in the output was computed by a stage that could
be tested without one, which is the property the whole design was arranged
around: this stage is where a mistake becomes permanent, and it is also the
stage with nothing in it that can fail unpredictably.

What it checks, and what it refuses to hide
-------------------------------------------

The QA gates are run here and their results are written whether they pass or
fail.  The one that matters most is the speech accounting: attributed speech has
to add up to the speech the diarizer heard.  It is the check that catches speech
being lost between the diarizer and the utterance list, and on this pipeline it
is expected to fail for any video where somebody speaks off camera -- see the
note on off-screen speech in the README.  The report says so in the numbers
rather than passing quietly.

The deliverable's own round trip is a gate too: the script is rendered, parsed
back, and compared to what it was rendered from.  A format that cannot survive
its own parser is the one failure that would invalidate a whole corpus at once.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from avannotate.annotation import annotation_to_dict, render_script
from avannotate.audio.types import SpeakerTurn
from avannotate.caption.types import GlobalCaption
from avannotate.compose.collect import face_tracks, shots, utterances
from avannotate.interval import merge, total_duration
from avannotate.qa import build_report
from avannotate.schema import (
    SCHEMA_VERSION,
    Annotation,
    Language,
    VideoMeta,
)
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
)
from avannotate.stages.base import (
    Artifact,
    StageContext,
    StageRecord,
    StageRun,
    StageState,
    config_float,
    hash_file,
    hash_payload,
    write_json,
)

STAGE = "s11-compose"
#: v2: the timeline_sanity message printed its spans to two decimals, which
#: rounded away the overshoot it was reporting -- a span at 8.7201 in an
#: 8.715 s video read as "8.72 outside 8.72".  The gate was right and the
#: message was useless.  Bumped because the report is S11's artifact and
#: nothing else about the stage changed: without this the stage would skip
#: and reprint the old one.
VERSION = "s11-v3"

SCRIPT_NAME = "annotation.txt"
ANNOTATION_NAME = "annotation.json"
REPORT_DIR = "qa"
REPORT_NAME = "report.json"

#: How far the attributed speech may differ from what the diarizer heard before
#: the accounting gate fails.  A second and a half allows for the padding the
#: segmenter adds and the edges it trims; it does not allow for a whole speaker.
DEFAULT_ACCOUNTING_TOLERANCE = 1.5


class ComposeError(RuntimeError):
    """The deliverable could not be assembled from what the stages left."""


def _vad_seconds(turns: Sequence[SpeakerTurn]) -> float | None:
    """How much speech the diarizer heard, as a union rather than a sum.

    The union matters: two people talking at once produce two overlapping turns
    and one stretch of time, and summing them would make the target the
    accounting gate compares against larger than the video.

    ``None`` when there are no turns at all, which the gate reads as "not
    supplied" and reports rather than failing -- a video with no speech is a
    legitimate thing to hand this pipeline, and failing it would be wrong.
    """

    intervals = [turn.interval for turn in turns]
    if not intervals:
        return None
    return total_duration(merge(intervals))


def build_annotation(
    context: StageContext,
    *,
    timeline: s0_preprocess.Timeline,
    shots_data: Mapping[int, str],
    global_caption: str,
    language: Language | None,
) -> Annotation:
    """Everything ten stages know, as one document."""

    identity_tracks = s3_cluster.load_identity_tracks(context)
    tracklets = s2_tracks.load_tracklets(context)
    speech = s6_associate.load_identity_speech(context)

    segments = [SpeechSegment.from_dict(item) for item in s7_tse.load_segments(context)]
    transcripts = {
        str(item.get("name")): item for item in s8_asr.load_transcripts(context)
    }
    tags = {
        str(item.get("name")): str(item["tag"])
        for item in s9_paralinguistic.load_tags(context)
        if item.get("tag")
    }

    stats: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "segments": len(segments),
        "transcribed": len(transcripts),
        "tagged": len(tags),
    }
    return Annotation(
        video=VideoMeta(
            video_id=context.video_id,
            path=str(context.source),
            duration=timeline.duration,
            fps=timeline.fps,
            width=timeline.width,
            height=timeline.height,
        ),
        utterances=utterances(
            segments, transcripts, tags, limit=timeline.duration
        ),
        shots=shots(s0_preprocess.load_shots(context), shots_data),
        face_tracks=face_tracks(identity_tracks, tracklets, speech),
        global_caption=global_caption,
        language=language,
        stats=stats,
    )


def run(context: StageContext, *, force: bool = False) -> StageRun:
    config = S11Config.from_mapping(context.config)
    input_hash = _input_hash(context, config)

    state = StageState(context.work_dir)
    reason = state.reason_to_run(STAGE, code_version=VERSION, input_hash=input_hash)
    if reason is None and not force:
        return StageRun(stage=STAGE, skipped=True, reason="outputs present and unchanged")
    trigger = "forced" if reason is None else reason

    timeline = s0_preprocess.load_timeline(context)
    turns = s4_diarize.load_turns(context)
    captions = {item.index: item.caption for item in s10_caption.load_captions(context)}
    overall = s10_caption.load_global_caption(context)

    annotation = build_annotation(
        context,
        timeline=timeline,
        shots_data=captions,
        global_caption=overall.caption,
        language=s8_asr.load_language(context),
    )

    script = render_script(annotation)
    report = build_report(
        annotation,
        vad_speech_seconds=_vad_seconds(turns),
        accounting_tolerance=config.accounting_tolerance,
        notes=_notes(context, annotation, overall),
    )

    script_path = context.output(STAGE, SCRIPT_NAME)
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script, encoding="utf-8")

    annotation_path = write_json(
        context.output(STAGE, ANNOTATION_NAME), annotation_to_dict(annotation)
    )
    report_path = write_json(
        context.output(STAGE, REPORT_DIR) / REPORT_NAME, report.to_dict()
    )

    artifacts = tuple(
        Artifact.capture(context.work_dir, path)
        for path in (script_path, annotation_path, report_path)
    )
    state.save(
        StageRecord(
            stage=STAGE,
            status="ok",
            code_version=VERSION,
            config_hash=hash_payload(config.to_dict()),
            input_hash=input_hash,
            artifacts=artifacts,
        )
    )

    failing = [gate.name for gate in report.failing]
    return StageRun(
        stage=STAGE,
        skipped=False,
        reason=trigger,
        summary={
            "utterances": len(annotation.utterances),
            "faces": len(annotation.face_tracks),
            "shots": len(annotation.shots),
            "gates_failed": len(failing),
            "failing": failing,
        },
    )


def _notes(
    context: StageContext,
    annotation: Annotation,
    caption: GlobalCaption,
) -> dict[str, object]:
    """What the report should say about how the numbers were arrived at.

    Assembled from the upstream stages' own summaries rather than recomputed, so
    the report cannot disagree with the stage that produced the data.
    """

    assignments = s6_associate.load_assignments(context)
    offscreen_speakers = [item for item in assignments if item.get("face_id") == "F000"]
    silent = sum(
        1 for item in s7_tse.load_segments(context) if item.get("silent") is True
    )

    return {
        "offscreen_speakers": len(offscreen_speakers),
        "silent_extractions": silent,
        "caption_names_dropped": list(caption.dropped),
        "caption_flags": list(caption.flags),
        "caption_frames": caption.frames,
        "unclaimed_faces": sum(
            1 for track in annotation.face_tracks if not track.speaks
        ),
    }


@dataclass(frozen=True)
class S11Config:
    """Only the accounting tolerance, because everything else is a stage's own.

    The one knob here is how far the attributed speech may differ from what the
    diarizer heard before the report says something has been lost.  It has a
    default that is right for this pipeline and is worth raising for exactly one
    reason: a corpus with a lot of off-screen speech, where the gap is expected
    and the gate is being used as a measurement rather than a check.
    """

    accounting_tolerance: float = DEFAULT_ACCOUNTING_TOLERANCE

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> S11Config:
        return cls(
            accounting_tolerance=config_float(
                mapping, "accounting_tolerance", DEFAULT_ACCOUNTING_TOLERANCE
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {"accounting_tolerance": self.accounting_tolerance}


def _input_hash(context: StageContext, config: S11Config) -> str:
    """Every upstream artifact, so the deliverable is rebuilt when any changes.

    One hash over all of them rather than a chain of dependencies: this stage is
    the join point, and a hash that missed one input would leave a stale
    deliverable that looks current.
    """

    return hash_payload(
        {
            "config": config.to_dict(),
            "probe": hash_file(context.work_dir / s0_preprocess.STAGE / s0_preprocess.PROBE_NAME),
            "audio": hash_file(s0_preprocess.audio_path(context)),
            "shots": hash_file(s0_preprocess.shots_path(context)),
            "tracks": hash_file(s2_tracks.tracks_path(context)),
            "identities": hash_file(s3_cluster.identities_path(context)),
            "turns": hash_file(s4_diarize.turns_path(context)),
            "assignments": hash_file(s6_associate.assignments_path(context)),
            "segments": hash_file(s7_tse.segments_path(context)),
            "transcripts": hash_file(s8_asr.transcripts_path(context)),
            "tags": hash_file(s9_paralinguistic.tags_path(context)),
            "captions": hash_file(s10_caption.captions_path(context)),
        }
    )


def load_script(context: StageContext) -> str:
    path = script_path(context)
    return path.read_text(encoding="utf-8")


def script_path(context: StageContext) -> Path:
    path = context.work_dir / STAGE / SCRIPT_NAME
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; run {STAGE} first")
    return path


def annotation_path(context: StageContext) -> Path:
    path = context.work_dir / STAGE / ANNOTATION_NAME
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; run {STAGE} first")
    return path


def load_annotation(context: StageContext) -> Annotation:
    """The deliverable, read back through the same reader a consumer uses."""

    from avannotate.annotation import annotation_from_dict

    payload = json.loads(annotation_path(context).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ComposeError(f"{annotation_path(context)} is not a JSON object")
    return annotation_from_dict(payload)


def report_path(context: StageContext) -> Path:
    path = context.work_dir / STAGE / REPORT_DIR / REPORT_NAME
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; run {STAGE} first")
    return path
