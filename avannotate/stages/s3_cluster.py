"""Stage S3: rejoin tracklets into people.

S2's output is runs of frames containing the same face.  A camera pan breaks
those runs, and no amount of tracking logic reconnects them, because the
reconnection needs appearance rather than position.  This stage supplies it:
measured on the sample corpus, two fragments of one person 211 px apart after a
pan score 0.78 cosine against each other and 0.08-0.12 against everyone else.

The identities produced here are the ``F001``, ``F002`` ... that the script
renders, so their numbering is part of the deliverable.  Assignment is by total
screen time descending, ties by first appearance -- deterministic, so two runs
of one video compare directly, and the people who matter come first.

Tracklets that cannot be clustered are dropped with a reason, never silently.  A
one-detection tracklet has no appearance to match on, and attaching a stray box
to a real person's identity is worse than admitting there is nothing there.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from avannotate.faces.cluster import DEFAULT_MAX_DISTANCE, cluster_vectors
from avannotate.faces.track import TrackQuality
from avannotate.stages import s1_faces, s2_tracks
from avannotate.stages.base import (
    Artifact,
    StageContext,
    StageRecord,
    StageRun,
    StageState,
    config_float,
    config_int,
    hash_file,
    hash_payload,
    write_json,
)
from avannotate.stages.s2_tracks import Tracklet

STAGE = "s3-cluster"
VERSION = "s3-v1"

IDENTITIES_NAME = "identities.json"
SUMMARY_NAME = "summary.json"

#: Below this normalised motion a tracklet is flagged as possibly set dressing.
#: Flagged, not dropped: during a camera pan a wall object moves in frame too, so
#: the signal is not reliable enough to act on by itself.
DEFAULT_STATIC_MOTION = 0.005


@dataclass(frozen=True)
class S3Config:
    #: Hard noise filters.  One detection has no appearance to cluster on.
    min_frames: int = 3
    min_span_seconds: float = 0.3
    min_mean_score: float = 0.6
    min_mean_height: float = 16.0
    #: Cosine distance ceiling for two tracklets being the same person.
    max_distance: float = DEFAULT_MAX_DISTANCE
    static_motion_threshold: float = DEFAULT_STATIC_MOTION

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> S3Config:
        return cls(
            min_frames=config_int(mapping, "min_frames", 3),
            min_span_seconds=config_float(mapping, "min_span_seconds", 0.3),
            min_mean_score=config_float(mapping, "min_mean_score", 0.6),
            min_mean_height=config_float(mapping, "min_mean_height", 16.0),
            max_distance=config_float(mapping, "max_distance", DEFAULT_MAX_DISTANCE),
            static_motion_threshold=config_float(
                mapping, "static_motion_threshold", DEFAULT_STATIC_MOTION
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "min_frames": self.min_frames,
            "min_span_seconds": self.min_span_seconds,
            "min_mean_score": self.min_mean_score,
            "min_mean_height": self.min_mean_height,
            "max_distance": self.max_distance,
            "static_motion_threshold": self.static_motion_threshold,
        }


@dataclass(frozen=True)
class TrackVectors:
    """One tracklet's identity evidence."""

    track_id: int
    #: Mean of its detection vectors, re-normalised.
    centroid: NDArray[np.float32]
    #: Mean pairwise cosine among its own vectors.  A tracklet that merged two
    #: people has a low one, because its members disagree with each other.
    cohesion: float
    samples: int


def collect_vectors(tracklet: Tracklet, embeddings: NDArray[np.float32]) -> TrackVectors | None:
    """Gather a tracklet's vectors, or ``None`` when it has none."""

    rows: list[NDArray[np.float32]] = []
    for detection in tracklet.detections:
        index = detection.embedding_index
        if index is None or not 0 <= index < len(embeddings):
            continue
        rows.append(embeddings[index])

    if not rows:
        return None

    matrix = np.asarray(rows, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.where(norms > 0.0, norms, 1.0)

    centroid = np.asarray(matrix.mean(axis=0), dtype=np.float32)
    norm = float(np.linalg.norm(centroid))
    if norm > 0.0:
        centroid = np.asarray(centroid / norm, dtype=np.float32)

    cohesion = 1.0
    if len(matrix) > 1:
        similarity = matrix @ matrix.T
        upper = similarity[np.triu_indices(len(matrix), k=1)]
        cohesion = float(upper.mean()) if upper.size else 1.0

    return TrackVectors(
        track_id=tracklet.track_id, centroid=centroid, cohesion=cohesion, samples=len(rows)
    )


def rejection_reason(tracklet: Tracklet, config: S3Config) -> str | None:
    """Why this tracklet cannot become a person, or ``None``."""

    quality = tracklet.quality
    if quality.frames < config.min_frames:
        return f"only {quality.frames} detections"
    if quality.span_seconds < config.min_span_seconds:
        return f"visible for {quality.span_seconds:.2f}s"
    if quality.mean_score < config.min_mean_score:
        return f"mean detection score {quality.mean_score:.2f}"
    if quality.mean_height < config.min_mean_height:
        return f"face only {quality.mean_height:.0f}px tall"
    return None


def _weighted(qualities: Sequence[TrackQuality], attribute: str) -> float:
    """Frame-weighted mean of a quality field, so a long tracklet counts more."""

    total = 0.0
    weight = 0.0
    for quality in qualities:
        value = float(getattr(quality, attribute))
        total += value * quality.frames
        weight += quality.frames
    return total / weight if weight > 0.0 else 0.0


def build_identity(
    face_id: str,
    members: Sequence[Tracklet],
    vectors: Sequence[TrackVectors],
    config: S3Config,
) -> dict[str, object]:
    qualities = [tracklet.quality for tracklet in members]
    motion = _weighted(qualities, "motion")
    return {
        "face_id": face_id,
        "track_ids": [tracklet.track_id for tracklet in members],
        "first_seen": round(min(t.detections[0].time for t in members if t.detections), 4)
        if any(t.detections for t in members)
        else 0.0,
        "last_seen": round(max(t.detections[-1].time for t in members if t.detections), 4)
        if any(t.detections for t in members)
        else 0.0,
        "total_screen_time": round(sum(q.span_seconds for q in qualities), 4),
        "quality": {
            "frames": sum(q.frames for q in qualities),
            "tracks": len(members),
            "mean_score": round(_weighted(qualities, "mean_score"), 4),
            "mean_height": round(_weighted(qualities, "mean_height"), 2),
            "motion": round(motion, 5),
            "cohesion": round(sum(v.cohesion for v in vectors) / len(vectors), 4),
        },
        "suspect_static": motion < config.static_motion_threshold,
    }


def cluster_tracklets(
    kept: Sequence[Tracklet],
    vectors: Mapping[int, TrackVectors],
    config: S3Config,
) -> list[dict[str, object]]:
    """Group tracklets into people and number them.

    Separated from the stage so the ordering rule can be tested without files:
    the numbering is the deliverable, and it must not depend on cluster order.
    """

    if not kept:
        return []

    matrix = np.asarray([vectors[t.track_id].centroid for t in kept], dtype=np.float32)
    groups = cluster_vectors(matrix, max_distance=config.max_distance)

    def rank(cluster: tuple[int, ...]) -> tuple[float, float]:
        members = [kept[index] for index in cluster]
        screen_time = sum(t.quality.span_seconds for t in members)
        first = min(
            (t.detections[0].time for t in members if t.detections), default=0.0
        )
        # Longer first; ties to whoever appeared earlier.
        return (-screen_time, first)

    identities: list[dict[str, object]] = []
    for ordinal, cluster in enumerate(sorted(groups, key=rank), start=1):
        members = [kept[index] for index in cluster]
        identities.append(
            build_identity(
                f"F{ordinal:03d}",
                members,
                [vectors[t.track_id] for t in members],
                config,
            )
        )
    return identities


def require_embeddings(context: StageContext) -> NDArray[np.float32]:
    """The identity vectors, or a message naming the fix.

    Checked before anything else in :func:`run`, including the input hash, which
    also needs the file.  Otherwise the caller gets "file not found" for a
    missing sidecar and never sees which backend to change.
    """

    embeddings = s1_faces.load_embeddings(context)
    if embeddings is not None:
        return embeddings
    backend = s1_faces.load_config(context).backend
    raise ValueError(
        f"{s1_faces.STAGE} produced no identity vectors, so tracklets cannot be "
        f"rejoined into people.  The detector backend was {backend!r}; insightface "
        "provides vectors and YuNet does not.  Re-run S1 with an embedding-capable "
        "backend, or accept per-tracklet ids by stopping here."
    )


def run(context: StageContext, *, force: bool = False) -> StageRun:
    config = S3Config.from_mapping(context.config)
    embeddings = require_embeddings(context)
    input_hash = _input_hash(context, config, embeddings)

    state = StageState(context.work_dir)
    reason = state.reason_to_run(STAGE, code_version=VERSION, input_hash=input_hash)
    if reason is None and not force:
        return StageRun(stage=STAGE, skipped=True, reason="outputs present and unchanged")
    trigger = "forced" if reason is None else reason

    tracklets = s2_tracks.load_tracklets(context)

    kept: list[Tracklet] = []
    vectors: dict[int, TrackVectors] = {}
    dropped: list[dict[str, object]] = []

    for tracklet in tracklets:
        rejection = rejection_reason(tracklet, config)
        if rejection is not None:
            dropped.append({"track_id": tracklet.track_id, "reason": rejection})
            continue
        collected = collect_vectors(tracklet, embeddings)
        if collected is None:
            dropped.append({"track_id": tracklet.track_id, "reason": "no identity vector"})
            continue
        kept.append(tracklet)
        vectors[tracklet.track_id] = collected

    identities = cluster_tracklets(kept, vectors, config)

    identities_path = write_json(
        context.output(STAGE, IDENTITIES_NAME),
        {
            "schema_version": "avannotate-identities-v1",
            "config": config.to_dict(),
            "identities": identities,
            "dropped": dropped,
        },
    )

    summary_path = write_json(
        context.output(STAGE, SUMMARY_NAME),
        {
            "schema_version": "avannotate-identities-summary-v1",
            "tracklets": len(tracklets),
            "identities": len(identities),
            "clustered_tracklets": len(kept),
            "dropped": len(dropped),
            # Tracklets per identity: whether S2's fragmentation was recovered,
            # or whether the corpus is simply full of short-lived detections.
            "tracklets_per_identity": {
                "min": min((len(i["track_ids"]) for i in identities), default=0),  # type: ignore[arg-type]
                "max": max((len(i["track_ids"]) for i in identities), default=0),  # type: ignore[arg-type]
            },
            "suspect_static": sum(1 for i in identities if i["suspect_static"]),
        },
    )

    artifacts = tuple(
        Artifact.capture(context.work_dir, path)
        for path in (identities_path, summary_path)
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
            "identities": len(identities),
            "from_tracklets": len(tracklets),
            "dropped": len(dropped),
        },
    )


def _input_hash(
    context: StageContext, config: S3Config, embeddings: NDArray[np.float32]
) -> str:
    tracks = s2_tracks.tracks_path(context)
    # The embeddings array is hashed from memory rather than from the file: it
    # has already been read, and the loader widens float16 to float32, so the
    # two would not agree anyway.
    return hash_payload(
        {
            "config": config.to_dict(),
            "tracks": hash_file(tracks),
            "embeddings": hash_payload(embeddings.tolist()),
        }
    )


def load_identities(context: StageContext) -> tuple[dict[str, object], ...]:
    path = identities_path(context)
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("identities")
    if not isinstance(raw, list):
        raise ValueError(f"{path} has no identities list; re-run {STAGE}")
    return tuple(item for item in raw if isinstance(item, dict))


def identities_path(context: StageContext) -> Path:
    path = context.work_dir / STAGE / IDENTITIES_NAME
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; run {STAGE} first")
    return path
