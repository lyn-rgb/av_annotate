"""Stage S1: detect faces in sampled frames.

Detection only.  Tracking lives in S2 and identity clustering in S3, because the
three have different tuning lifetimes: a detector is swapped once per corpus,
tracking parameters are tuned against a review sample, and clustering thresholds
change whenever the corpus's cast does.

On background faces
-------------------

A detector finds faces in framed photographs, posters, and reflections, and it
finds them *persistently* -- a gold record on a wall scored 0.64 across most of
a sample clip here.  Tracking does not filter these: a wall object is static, so
it forms a long, stable, entirely spurious track.  Only later stages can, using
signals that do not exist yet at S1:

* a real face moves -- breathing and head turns give a bounding box nonzero
  variance, a wall object gives almost none;
* a real face is usually larger, and usually scores higher;
* a face that never speaks and never moves is not a person in the scene.

So S1 records the score and the landmarks and drops nothing.  Discarding a
detection here would destroy the evidence the filtering stage needs, and a
threshold blunt enough to remove the gold record also removes genuinely
profile-turned faces.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from avannotate.faces.detect import Detector, DetectorError, build_detector
from avannotate.faces.frames import FrameSampling, iter_frames
from avannotate.faces.types import Detection
from avannotate.stages import s0_preprocess
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
    hash_payload,
    resolve_config_path,
    write_json,
)

STAGE = "s1-faces"
VERSION = "s1-v1"

DETECTIONS_NAME = "detections.jsonl"
SUMMARY_NAME = "summary.json"

#: Reported, not enforced: a corpus whose real faces score here is not broken,
#: it is a corpus to tune the threshold against.
_LOW_SCORE = 0.5


@dataclass(frozen=True)
class S1Config:
    stride: int = 3
    backend: str = "insightface"
    model_path: str | None = None
    model_name: str = "buffalo_l"
    model_root: str | None = None
    score_threshold: float = 0.6
    max_frames: int | None = None

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> S1Config:
        """Paths are resolved here, once, so the value that builds the detector
        and the value that keys the cache are the same string."""

        model_path = config_optional_str(mapping, "model_path")
        model_root = config_optional_str(mapping, "model_root")
        max_frames = mapping.get("max_frames")
        return cls(
            stride=config_int(mapping, "stride", 3),
            backend=config_str(mapping, "backend", "insightface"),
            model_path=(
                str(resolve_config_path(model_path, mapping))
                if model_path is not None
                else None
            ),
            model_name=config_str(mapping, "model_name", "buffalo_l"),
            model_root=(
                str(resolve_config_path(model_root, mapping))
                if model_root is not None
                else None
            ),
            score_threshold=config_float(mapping, "score_threshold", 0.6),
            max_frames=(
                config_int(mapping, "max_frames", 0) if max_frames is not None else None
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stride": self.stride,
            "backend": self.backend,
            "model_path": self.model_path,
            "model_name": self.model_name,
            "model_root": self.model_root,
            "score_threshold": self.score_threshold,
            "max_frames": self.max_frames,
        }

    def detector_config(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "model_path": self.model_path,
            "model_name": self.model_name,
            "model_root": self.model_root,
            "score_threshold": self.score_threshold,
        }


@dataclass
class _Tally:
    """Counts accumulated while decoding, for the summary."""

    sampled_frames: int = 0
    frames_with_faces: int = 0
    detections: int = 0
    scores: list[float] = field(default_factory=list)
    faces_per_frame: dict[int, int] = field(default_factory=dict)

    def observe(self, detections: tuple[Detection, ...]) -> None:
        self.sampled_frames += 1
        self.detections += len(detections)
        if detections:
            self.frames_with_faces += 1
        self.faces_per_frame[len(detections)] = self.faces_per_frame.get(len(detections), 0) + 1
        self.scores.extend(detection.score for detection in detections)

    def summary(self) -> dict[str, object]:
        scores = np.asarray(self.scores, dtype=np.float64)
        payload: dict[str, object] = {
            "sampled_frames": self.sampled_frames,
            "frames_with_faces": self.frames_with_faces,
            "detections": self.detections,
            "faces_per_frame": {
                str(key): value for key, value in sorted(self.faces_per_frame.items())
            },
            "detections_per_frame": (
                self.detections / self.sampled_frames if self.sampled_frames else 0.0
            ),
        }
        if scores.size:
            payload["score"] = {
                "min": float(scores.min()),
                "median": float(np.median(scores)),
                "max": float(scores.max()),
                "below_0.7": int((scores < 0.7).sum()),
                "below_0.5": int((scores < _LOW_SCORE).sum()),
            }
        return payload


def _run_detection(
    context: StageContext,
    config: S1Config,
    detector: Detector,
    *,
    timeline: s0_preprocess.Timeline,
) -> _Tally:
    sampling = FrameSampling(stride=config.stride)
    frame_count = timeline.frame_count
    if config.max_frames is not None:
        frame_count = min(frame_count, config.max_frames * config.stride)

    tally = _Tally()
    destination = context.output(STAGE, DETECTIONS_NAME)
    temporary = destination.with_name(destination.name + ".tmp")

    with temporary.open("w", encoding="utf-8") as handle:
        for index, frame in iter_frames(
            context.source,
            width=timeline.width,
            height=timeline.height,
            sampling=sampling,
            frame_count=frame_count,
        ):
            detections = detector.detect(frame)
            tally.observe(detections)
            handle.write(
                json.dumps(
                    {
                        "frame": index,
                        "time": round(index / timeline.fps, 4) if timeline.fps else 0.0,
                        "faces": [detection.to_dict() for detection in detections],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    temporary.replace(destination)
    return tally


def run(context: StageContext, *, force: bool = False) -> StageRun:
    config = S1Config.from_mapping(context.config)
    input_hash = _input_hash(context, config)

    state = StageState(context.work_dir)
    reason = state.reason_to_run(STAGE, code_version=VERSION, input_hash=input_hash)
    if reason is None and not force:
        return StageRun(stage=STAGE, skipped=True, reason="outputs present and unchanged")
    trigger = "forced" if reason is None else reason

    timeline = s0_preprocess.load_timeline(context)

    try:
        detector = build_detector(config.detector_config())
    except DetectorError as error:
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

    tally = _run_detection(context, config, detector, timeline=timeline)

    summary_path = write_json(
        context.output(STAGE, SUMMARY_NAME),
        {
            "schema_version": "avannotate-faces-summary-v1",
            "detector": detector.name,
            "config": config.to_dict(),
            "timeline": {"duration": timeline.duration, "fps": timeline.fps},
            "detection": tally.summary(),
        },
    )

    artifacts = tuple(
        Artifact.capture(context.work_dir, path)
        for path in (context.output(STAGE, DETECTIONS_NAME), summary_path)
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
            "detector": detector.name,
            "sampled": tally.sampled_frames,
            "detections": tally.detections,
            "frames_with_faces": tally.frames_with_faces,
        },
    )


def _input_hash(context: StageContext, config: S1Config) -> str:
    """S0's outputs plus this stage's config.

    Hashing S0's probe ties this stage to the demux it was run against: if S0
    re-runs because the source changed, S1's cache is invalidated with it rather
    than surviving against a stale frame count.
    """

    probe = context.work_dir / s0_preprocess.STAGE / s0_preprocess.PROBE_NAME
    if not probe.is_file():
        raise FileNotFoundError(
            f"{probe} is missing; run {s0_preprocess.STAGE} before {STAGE}"
        )
    return hash_payload(
        {
            "probe": probe.read_text(encoding="utf-8"),
            "config": config.to_dict(),
        }
    )


def load_detections(context: StageContext) -> tuple[dict[str, object], ...]:
    """Read the per-frame detections back.

    A list rather than a generator: every consumer so far (tracking, clustering)
    needs the whole thing, and an hour of video is tens of thousands of rows.
    """

    path = context.work_dir / STAGE / DETECTIONS_NAME
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; run {STAGE} first")
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return tuple(rows)


def detections_path(context: StageContext) -> Path:
    path = context.work_dir / STAGE / DETECTIONS_NAME
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; run {STAGE} first")
    return path


__all__ = [
    "DETECTIONS_NAME",
    "S1Config",
    "STAGE",
    "VERSION",
    "detections_path",
    "load_detections",
    "run",
]
