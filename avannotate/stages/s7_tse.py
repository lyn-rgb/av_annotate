"""Stage S7: pull each person's voice out of the mix.

S6 decided who is who; this gives each of them their own audio.  One file per
speaking segment, named after the person, holding only their voice and only
while they were talking -- which is the deliverable the whole pipeline was
built to produce.

How it works around the model's API
-----------------------------------

``AV_MossFormer2_TSE_16K`` is face-conditioned, which is why it was chosen: a
voice-conditioned extractor would need a clean enrolment clip, and producing
those is what this stage is for.  But its public interface takes a video and
runs its *own* face detection and lip-motion scoring to choose the speaker --
so asking it for F001 and getting F002 is a real possibility.

The way out is to make the choice before the model sees anything: cut a video
containing only F001's face and hand that over.  With one candidate it cannot
pick wrong.  That is what :mod:`avannotate.tse.crop_video` writes, and the crop
sequence it follows is the track S2 built and S3 clustered.

Cost
----

Extraction runs per segment, not per person, so the cost tracks speaking time
rather than screen time -- a person visible for a minute and talking for five
seconds costs five seconds, not sixty.  Each segment is given half a second of
context on each side, because a crop starting exactly at the first phoneme
starts with a mouth already moving, and the model has no baseline to compare
against.  The extra audio is trimmed off afterwards.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from avannotate.audio.wav import read_info, read_window, slice_samples, write_pcm16
from avannotate.coercion import coerce_number
from avannotate.faces.track import Tracklet
from avannotate.segment import SegmentationConfig
from avannotate.stages import s0_preprocess, s2_tracks, s3_cluster, s6_associate
from avannotate.stages.base import (
    Artifact,
    StageContext,
    StageRecord,
    StageRun,
    StageState,
    config_float,
    config_int,
    config_optional_str,
    config_str,
    hash_file,
    hash_payload,
    write_json,
)
from avannotate.tse.crop_video import DEFAULT_CROP_SIZE, DEFAULT_MARGIN, write_crop_video
from avannotate.tse.model import (
    MODEL_NAME,
    TargetSpeakerExtractor,
    TseError,
    build_extractor,
)
from avannotate.tse.plan import (
    DEFAULT_CONTEXT_SECONDS,
    ExtractionSegment,
    context_window,
    identity_boxes,
    plan_extractions,
)

STAGE = "s7-tse"
VERSION = "s7-v1"

SEGMENTS_NAME = "segments.json"
SUMMARY_NAME = "summary.json"
AUDIO_DIR = "audio"
CROPS_DIR = "crops"
SCRATCH_DIR = "scratch"

#: Below this RMS a segment came out effectively silent.  Not a gate -- the
#: model can legitimately return silence for a face that never spoke -- but it
#: is the one extraction failure that can be seen without a reference signal.
SILENT_RMS = 1e-4

#: Why a segment was skipped rather than extracted: the crop it is conditioned
#: on holds no face the extractor can find.  A track can be a false positive,
#: and a segment that cannot yield speech is not a stage failure.
NO_FACE = "no face in the crop"


@dataclass(frozen=True)
class S7Config:
    backend: str = "clearvoice"
    model: str = MODEL_NAME
    device: str | None = None
    # Segment rules, shared with the stage that will transcribe these files.
    max_gap: float = 0.20
    min_duration: float = 0.30
    low_confidence_duration: float = 0.50
    pad: float = 0.05
    crop_size: int = DEFAULT_CROP_SIZE
    margin: float = DEFAULT_MARGIN
    context_seconds: float = DEFAULT_CONTEXT_SECONDS
    #: Crop videos are a debugging aid, not an artifact: a thousand segments
    #: is a hundred thousand files if they are kept.
    keep_crops: bool = False

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> S7Config:
        return cls(
            backend=config_str(mapping, "backend", "clearvoice"),
            model=config_str(mapping, "model", MODEL_NAME),
            device=config_optional_str(mapping, "device"),
            max_gap=config_float(mapping, "max_gap", 0.20),
            min_duration=config_float(mapping, "min_duration", 0.30),
            low_confidence_duration=config_float(
                mapping, "low_confidence_duration", 0.50
            ),
            pad=config_float(mapping, "pad", 0.05),
            crop_size=config_int(mapping, "crop_size", DEFAULT_CROP_SIZE),
            margin=config_float(mapping, "margin", DEFAULT_MARGIN),
            context_seconds=config_float(
                mapping, "context_seconds", DEFAULT_CONTEXT_SECONDS
            ),
            keep_crops=bool(mapping.get("keep_crops", False)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "model": self.model,
            "device": self.device,
            "max_gap": self.max_gap,
            "min_duration": self.min_duration,
            "low_confidence_duration": self.low_confidence_duration,
            "pad": self.pad,
            "crop_size": self.crop_size,
            "margin": self.margin,
            "context_seconds": self.context_seconds,
            "keep_crops": self.keep_crops,
        }

    def segmentation(self) -> SegmentationConfig:
        return SegmentationConfig(
            max_gap=self.max_gap,
            min_duration=self.min_duration,
            low_confidence_duration=self.low_confidence_duration,
            pad=self.pad,
        )


def _absent_tracks(members: list[Tracklet], identity: str) -> str | None:
    if members:
        return None
    return f"{identity} has no usable tracklets"


def extract_segment(
    context: StageContext,
    extractor: TargetSpeakerExtractor,
    members: list[Tracklet],
    segment: ExtractionSegment,
    config: S7Config,
    *,
    width: int,
    height: int,
    fps: float,
    frame_count: int,
    sample_rate: int,
    crops_dir: Path,
    scratch_dir: Path,
) -> dict[str, object] | None:
    """Extract one segment, or ``None`` if the extractor found no face in it."""

    start_frame, end_frame = context_window(
        segment, fps=fps, frame_count=frame_count, context_seconds=config.context_seconds
    )
    crop_path = crops_dir / f"{segment.name}.mp4"
    write_crop_video(
        context.source,
        crop_path,
        width=width,
        height=height,
        start_time=start_frame / fps,
        frame_count=end_frame - start_frame,
        boxes=identity_boxes(members, start_frame=start_frame, end_frame=end_frame),
        fps=fps,
        size=config.crop_size,
        margin=config.margin,
    )

    raw_path = extractor.extract(crop_path, scratch_dir)
    if raw_path is None:
        # The extractor ran and found no face in the crop.  A segment that
        # cannot yield speech is skipped rather than failed; see the note in
        # ``ClearerVoiceExtractor.extract``.  The crop stays on disk for whoever
        # asks why, unless the config says not to keep crops.
        if not config.keep_crops:
            crop_path.unlink(missing_ok=True)
        return None

    raw_rate, _ = read_info(raw_path)
    raw_samples = read_window(
        raw_path, start_seconds=0.0, duration_seconds=1e9
    )

    # The extraction covers the crop window, not the segment: the context has to
    # come off both ends or every file would start half a second early.
    offset = segment.start - start_frame / fps
    trimmed = slice_samples(
        raw_samples,
        sample_rate=raw_rate,
        start=offset,
        end=offset + segment.duration,
    )

    target = context.stage_dir(STAGE) / AUDIO_DIR / segment.identity / f"{segment.name}.wav"
    write_pcm16(target, trimmed, sample_rate=raw_rate)

    if not config.keep_crops:
        crop_path.unlink(missing_ok=True)

    rms = float(np.sqrt(np.mean(np.square(trimmed)))) if len(trimmed) else 0.0
    return {
        **segment.to_dict(),
        "audio": str(target.relative_to(context.work_dir)),
        "sample_rate": raw_rate,
        "samples": int(len(trimmed)),
        "rms": round(rms, 6),
        "silent": rms < SILENT_RMS,
    }


def run(context: StageContext, *, force: bool = False) -> StageRun:
    config = S7Config.from_mapping(context.config)
    input_hash = _input_hash(context, config)

    state = StageState(context.work_dir)
    reason = state.reason_to_run(STAGE, code_version=VERSION, input_hash=input_hash)
    if reason is None and not force:
        return StageRun(stage=STAGE, skipped=True, reason="outputs present and unchanged")
    trigger = "forced" if reason is None else reason

    timeline = s0_preprocess.load_timeline(context)
    identity_tracks = s3_cluster.load_identity_tracks(context)
    tracklets = s2_tracks.load_tracklets(context)
    speech = s6_associate.load_identity_speech(context)
    by_id = {tracklet.track_id: tracklet for tracklet in tracklets}

    plan = plan_extractions(
        speech, duration=timeline.duration, config=config.segmentation()
    )
    total = sum(len(segments) for segments in plan.values())

    try:
        extractor = build_extractor(
            {"backend": config.backend, "model": config.model, "device": config.device}
        )
    except TseError as error:
        state.save(
            StageRecord(
                stage=STAGE,
                status="failed",
                code_version=VERSION,
                config_hash=hash_payload(config.to_dict()),
                input_hash=input_hash,
                error=str(error),
            )
        )
        raise

    crops_dir = context.stage_dir(STAGE) / CROPS_DIR
    scratch_dir = context.stage_dir(STAGE) / SCRATCH_DIR
    segments: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []

    for identity in sorted(plan):
        members = [
            by_id[track_id]
            for track_id in identity_tracks.get(identity, ())
            if track_id in by_id
        ]
        absent = _absent_tracks(members, identity)
        if absent is not None:
            skipped.extend(
                {**segment.to_dict(), "reason": absent} for segment in plan[identity]
            )
            continue

        for segment in plan[identity]:
            extracted = extract_segment(
                context,
                extractor,
                members,
                segment,
                config,
                width=timeline.width,
                height=timeline.height,
                fps=timeline.fps,
                frame_count=timeline.frame_count,
                sample_rate=timeline.sample_rate,
                crops_dir=crops_dir,
                scratch_dir=scratch_dir,
            )
            if extracted is None:
                skipped.append({**segment.to_dict(), "reason": NO_FACE})
                continue
            segments.append(extracted)

    segments_path = write_json(
        context.output(STAGE, SEGMENTS_NAME),
        {
            "schema_version": "avannotate-segments-v1",
            "config": config.to_dict(),
            "backend": {"name": extractor.name, "model": config.model},
            "segments": segments,
            "skipped": skipped,
        },
    )
    summary_path = write_json(
        context.output(STAGE, SUMMARY_NAME),
        {
            "schema_version": "avannotate-extraction-summary-v1",
            "planned": total,
            "written": len(segments),
            "skipped": len(skipped),
            "silent": sum(1 for item in segments if bool(item.get("silent"))),
            "seconds": round(
                sum(coerce_number(item["duration"], "duration") for item in segments), 3
            ),
            "identities": sorted({str(item["identity"]) for item in segments}),
        },
    )

    artifacts = tuple(
        Artifact.capture(context.work_dir, path)
        for path in (segments_path, summary_path)
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

    return StageRun(
        stage=STAGE,
        skipped=False,
        reason=trigger,
        summary={
            "written": len(segments),
            "skipped": len(skipped),
            "silent": sum(1 for item in segments if bool(item.get("silent"))),
        },
    )


def _input_hash(context: StageContext, config: S7Config) -> str:
    stat = context.source.stat()
    return hash_payload(
        {
            "config": config.to_dict(),
            "assignments": hash_file(s6_associate.assignments_path(context)),
            "tracks": hash_file(context.work_dir / s2_tracks.STAGE / s2_tracks.TRACKS_NAME),
            "video": {"size": stat.st_size, "mtime": stat.st_mtime},
        }
    )


def load_segments(context: StageContext) -> tuple[dict[str, object], ...]:
    path = segments_path(context)
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("segments")
    if not isinstance(raw, list):
        raise ValueError(f"{path} has no segments list; re-run {STAGE}")
    return tuple(item for item in raw if isinstance(item, dict))


def segments_path(context: StageContext) -> Path:
    path = context.work_dir / STAGE / SEGMENTS_NAME
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; run {STAGE} first")
    return path
