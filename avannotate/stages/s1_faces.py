"""Stage S1: detect faces in sampled frames, with identity vectors.

Detection and embedding, no tracking.  Tracking lives in S2 and identity
clustering in S3, because the three have different tuning lifetimes: a detector
is swapped once per corpus, tracking parameters are tuned against a review
sample, and clustering thresholds change whenever the corpus's cast does.

The vectors go in a sidecar array rather than in the rows; each detection that
has one carries an ``emb`` index into it.  Only some detectors produce them --
``Detection.embedding`` is ``None`` under YuNet -- so the summary records
whether the backend does, and S3 refuses to pretend a corpus without vectors
can be clustered.

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
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from avannotate.faces.detect import Detector, DetectorError, build_detector
from avannotate.faces.frames import FrameSampling, iter_frames
from avannotate.faces.types import Detection, FrameDetections, coerce_number
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
EMBEDDINGS_NAME = "embeddings.npy"
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
    #: How often to keep identity vectors, in seconds of video.
    #:
    #: A disk decision, not a compute one: insightface computes the embedding as
    #: part of its own pipeline, so detecting on a frame produces one whether or
    #: not it is written.  At 512 float16 values per face, an hour of two-person
    #: video is roughly 25 MB per second of interval -- keeping every sampled
    #: frame would be tens of gigabytes across a thousand-hour batch, while one
    #: per second still gives a ten-second track ten vectors to average.
    embedding_interval_seconds: float = 1.0

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
            embedding_interval_seconds=config_float(
                mapping, "embedding_interval_seconds", 1.0
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
            "embedding_interval_seconds": self.embedding_interval_seconds,
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
    embeddings: int = 0
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
            "embeddings": self.embeddings,
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


def _embedding_every(config: S1Config, timeline: s0_preprocess.Timeline) -> int:
    """Sampled-frame interval between kept embeddings.

    A non-positive interval means "keep none", expressed as an interval so large
    the modulo never fires, rather than a second branch in the loop.
    """

    if config.embedding_interval_seconds <= 0.0:
        return sys.maxsize
    steps = timeline.fps * config.embedding_interval_seconds / max(1, config.stride)
    return max(1, int(round(steps)))


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

    keeping = detector.provides_embeddings
    every = _embedding_every(config, timeline)

    tally = _Tally()
    destination = context.output(STAGE, DETECTIONS_NAME)
    temporary = destination.with_name(destination.name + ".tmp")
    vectors: list[tuple[float, ...]] = []

    with temporary.open("w", encoding="utf-8") as handle:
        for step, (index, frame) in enumerate(
            iter_frames(
                context.source,
                width=timeline.width,
                height=timeline.height,
                sampling=sampling,
                frame_count=frame_count,
            )
        ):
            detections = detector.detect(frame)
            tally.observe(detections)

            store = keeping and step % every == 0
            faces: list[dict[str, object]] = []
            for detection in detections:
                if store and detection.embedding is not None:
                    # Take the vector out of the detection before writing the row
                    # -- 512 floats per detection would dominate the file -- and
                    # record where it went.  The append has to happen first: it
                    # is the only place the vector still exists.
                    vectors.append(detection.embedding)
                    detection = replace(
                        detection, embedding=None, embedding_index=len(vectors) - 1
                    )
                    tally.embeddings += 1
                faces.append(detection.to_dict())

            handle.write(
                json.dumps(
                    {
                        "frame": index,
                        "time": round(index / timeline.fps, 4) if timeline.fps else 0.0,
                        "faces": faces,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    temporary.replace(destination)
    if vectors:
        np.save(context.output(STAGE, EMBEDDINGS_NAME), np.asarray(vectors, dtype=np.float16))
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
            "provides_embeddings": detector.provides_embeddings,
            "config": config.to_dict(),
            "timeline": {"duration": timeline.duration, "fps": timeline.fps},
            "detection": tally.summary(),
        },
    )

    produced = [context.output(STAGE, DETECTIONS_NAME), summary_path]
    embeddings_file = context.output(STAGE, EMBEDDINGS_NAME)
    if embeddings_file.is_file():
        produced.append(embeddings_file)
    artifacts = tuple(Artifact.capture(context.work_dir, path) for path in produced)
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


def load_frame_detections(context: StageContext) -> tuple[FrameDetections, ...]:
    """The detections as typed objects, not raw rows.

    This stage owns the file format, so this is where it gets parsed.  A
    consumer reaching into ``detections.jsonl`` itself would be coupled to a
    schema it has no reason to know.
    """

    frames: list[FrameDetections] = []
    for row in load_detections(context):
        raw = row.get("faces") or []
        if not isinstance(raw, list):
            raise ValueError(f"faces must be a list, got {type(raw).__name__}")
        frames.append(
            FrameDetections(
                frame_index=int(coerce_number(row["frame"], "frame")),
                time=coerce_number(row["time"], "time"),
                detections=tuple(
                    Detection.from_dict(face) for face in raw if isinstance(face, dict)
                ),
            )
        )
    return tuple(frames)


def load_embeddings(context: StageContext) -> NDArray[np.float32] | None:
    """The identity vectors, or ``None`` when the backend produced none.

    float16 on disk, widened to float32 here: the comparison that matters is a
    cosine similarity between vectors that are ~0.09 apart, and float16 carries
    about three decimal digits, which is not enough headroom to be comfortable
    about a threshold near 0.4.
    """

    path = context.work_dir / STAGE / EMBEDDINGS_NAME
    if not path.is_file():
        return None
    vectors = np.load(path)
    if vectors.size == 0:
        return None
    return np.asarray(vectors, dtype=np.float32)


def load_config(context: StageContext) -> S1Config:
    """The config this stage actually ran with.

    Read back from the summary rather than from the caller's mapping: a later
    stage needs the settings that produced the detections on disk, which are not
    necessarily the ones in today's config file.
    """

    path = context.work_dir / STAGE / SUMMARY_NAME
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; run {STAGE} first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"{path} has no config block; re-run {STAGE}")
    return S1Config.from_mapping(config)


__all__ = [
    "DETECTIONS_NAME",
    "S1Config",
    "STAGE",
    "VERSION",
    "detections_path",
    "load_detections",
    "run",
]
