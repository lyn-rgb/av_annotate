"""Stage S2: link per-frame detections into tracklets.

The output is **tracklets, not identities**.  One person may produce several:
this stage associates by overlap, and overlap stops working the moment the
camera moves far enough between two sampled frames that a face no longer
overlaps where it just was.

Measured on the sample corpus: during a fast pan one face moved 176 px in
0.125 s, more than its own width, and the track broke.  Sampling every frame
instead of every third held that track together, but it cannot fix the general
case -- after the pan the same person reappears 211 px away, and no amount of
sampling makes two positions that far apart overlap.  Recovering identity there
needs appearance, not position, which is what S3's embedding clustering is for.

So the honest reading of this output is "runs of frames containing the same
face", and the summary reports how fragmented they are.  A corpus whose tracks
are mostly one or two hits is a corpus whose sampling stride is too coarse or
whose camera moves too much -- both worth knowing before blaming the clustering
stage downstream.

The quality numbers matter as much as the boxes.  A detector finds faces in wall
art, and those become long, stable tracklets -- a picture on a wall holds still
better than a person does, so tracking coherence is not the test.  What
separates them is that a person's box moves and a picture's does not, which is
what ``motion`` records.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from avannotate.faces.track import (
    Track,
    TrackDetection,
    TrackerConfig,
    Tracklet,
    TrackQuality,
    track_detections,
)
from avannotate.stages import s0_preprocess, s1_faces
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

STAGE = "s2-tracks"
VERSION = "s2-v1"

TRACKS_NAME = "tracks.jsonl"
SUMMARY_NAME = "summary.json"


@dataclass(frozen=True)
class S2Config:
    track_thresh: float = 0.5
    match_thresh: float = 0.8
    second_match_thresh: float = 0.5
    det_thresh: float = 0.6
    low_score_floor: float = 0.1
    min_hits: int = 1
    #: How long a track survives with no detections, in seconds.  In seconds
    #: rather than update steps because "it survives a second of occlusion" is
    #: a statement about the video, and the step count depends on the sampling
    #: stride, which is an S1 setting this stage should not have to know.
    max_time_lost_seconds: float = 1.0

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> S2Config:
        return cls(
            track_thresh=config_float(mapping, "track_thresh", 0.5),
            match_thresh=config_float(mapping, "match_thresh", 0.8),
            second_match_thresh=config_float(mapping, "second_match_thresh", 0.5),
            det_thresh=config_float(mapping, "det_thresh", 0.6),
            low_score_floor=config_float(mapping, "low_score_floor", 0.1),
            min_hits=config_int(mapping, "min_hits", 1),
            max_time_lost_seconds=config_float(mapping, "max_time_lost_seconds", 1.0),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "track_thresh": self.track_thresh,
            "match_thresh": self.match_thresh,
            "second_match_thresh": self.second_match_thresh,
            "det_thresh": self.det_thresh,
            "low_score_floor": self.low_score_floor,
            "min_hits": self.min_hits,
            "max_time_lost_seconds": self.max_time_lost_seconds,
        }

    def tracker_config(self, *, fps: float, stride: int) -> TrackerConfig:
        return TrackerConfig(
            track_thresh=self.track_thresh,
            match_thresh=self.match_thresh,
            second_match_thresh=self.second_match_thresh,
            det_thresh=self.det_thresh,
            low_score_floor=self.low_score_floor,
            min_hits=self.min_hits,
        ).with_max_time_lost_seconds(
            self.max_time_lost_seconds, fps=fps, stride=stride
        )


def _track_payload(track: Track, quality: TrackQuality) -> dict[str, object]:
    return {
        "track_id": track.track_id,
        "start_frame": track.start_frame,
        "end_frame": track.end_frame,
        "hits": track.hits,
        "quality": quality.to_dict(),
        "detections": [item.to_dict() for item in track.detections],
    }


def run(context: StageContext, *, force: bool = False) -> StageRun:
    config = S2Config.from_mapping(context.config)
    input_hash = _input_hash(context, config)

    state = StageState(context.work_dir)
    reason = state.reason_to_run(STAGE, code_version=VERSION, input_hash=input_hash)
    if reason is None and not force:
        return StageRun(stage=STAGE, skipped=True, reason="outputs present and unchanged")
    trigger = "forced" if reason is None else reason

    timeline = s0_preprocess.load_timeline(context)
    frames = s1_faces.load_frame_detections(context)
    # The stride comes from S1's recorded config, not from ours: it is a
    # detection setting, and duplicating it here would let the two disagree
    # about how many update steps a second contains.
    stride = s1_faces.load_config(context).stride

    tracker_config = config.tracker_config(fps=timeline.fps, stride=stride)
    tracks = track_detections(frames, tracker_config)

    destination = context.output(STAGE, TRACKS_NAME)
    temporary = destination.with_name(destination.name + ".tmp")
    qualities: list[TrackQuality] = []
    with temporary.open("w", encoding="utf-8") as handle:
        for track in tracks:
            quality = TrackQuality.from_track(track, fps=timeline.fps)
            qualities.append(quality)
            handle.write(json.dumps(_track_payload(track, quality), ensure_ascii=False) + "\n")
    temporary.replace(destination)

    summary_path = write_json(
        context.output(STAGE, SUMMARY_NAME),
        {
            "schema_version": "avannotate-tracks-summary-v1",
            "config": config.to_dict(),
            "resolved": {
                "stride": stride,
                "fps": timeline.fps,
                "max_time_lost_steps": tracker_config.max_time_lost,
            },
            "tracks": _summarize(tracks, qualities),
        },
    )

    artifacts = tuple(
        Artifact.capture(context.work_dir, path) for path in (destination, summary_path)
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
            "tracks": len(tracks),
            "hits": sum(track.hits for track in tracks),
            "longest": max((q.frames for q in qualities), default=0),
        },
    )


def _summarize(tracks: tuple[Track, ...], qualities: list[TrackQuality]) -> dict[str, object]:
    if not tracks:
        return {"count": 0, "quality": {}, "note": "no track survived the association"}

    motions = sorted(quality.motion for quality in qualities)
    hits = sorted(track.hits for track in tracks)
    return {
        "count": len(tracks),
        "total_detections": sum(quality.frames for quality in qualities),
        "hits_per_track": {
            "min": hits[0],
            "median": hits[len(hits) // 2],
            "max": hits[-1],
        },
        "span_seconds": {
            "min": round(min(q.span_seconds for q in qualities), 3),
            "max": round(max(q.span_seconds for q in qualities), 3),
        },
        "mean_score": round(sum(q.mean_score for q in qualities) / len(qualities), 4),
        # Fragmentation.  A track of one or two detections is what a camera pan
        # or an over-coarse sampling stride looks like from here, and this count
        # is the cheapest way to see it without opening tracks.jsonl.
        "fragmentation": {
            "single_hit": sum(1 for value in hits if value == 1),
            "at_most_three": sum(1 for value in hits if value <= 3),
            "share_single_hit": round(sum(1 for value in hits if value == 1) / len(hits), 3),
        },
        # The wall-art discriminator, reported so a corpus can be judged without
        # the caller having to open tracks.jsonl at all.
        "motion": {
            "min": round(motions[0], 5),
            "median": round(motions[len(motions) // 2], 5),
            "max": round(motions[-1], 5),
            "near_static": sum(1 for value in motions if value < 0.005),
        },
    }


def _input_hash(context: StageContext, config: S2Config) -> str:
    detections = s1_faces.detections_path(context)
    stride = s1_faces.load_config(context).stride
    # Content, not size: a same-size edit to a JSONL file is exactly the case a
    # size check cannot see, and S0's resume check was written for the same
    # reason.  The stride joins it because the same detections tracked against a
    # different sampling stride are a different set of tracks.
    return hash_payload(
        {
            "config": config.to_dict(),
            "detections": hash_file(detections),
            "stride": stride,
        }
    )


def load_tracks(context: StageContext) -> tuple[dict[str, object], ...]:
    """Raw rows, for callers that only want to look at the JSON."""

    path = tracks_path(context)
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                payload = json.loads(stripped)
                if isinstance(payload, dict):
                    rows.append(payload)
    return tuple(rows)


def tracklet_from_row(row: Mapping[str, object]) -> Tracklet:
    """Rebuild one typed tracklet from its JSON row."""

    raw_quality = row.get("quality")
    if not isinstance(raw_quality, dict):
        raise ValueError(f"track {row.get('track_id')} has no quality block")

    raw_detections = row.get("detections")
    if raw_detections is not None and not isinstance(raw_detections, list):
        raise ValueError(f"track {row.get('track_id')} detections must be a list")

    return Tracklet(
        track_id=int(float(str(row["track_id"]))),
        start_frame=int(float(str(row["start_frame"]))),
        end_frame=int(float(str(row["end_frame"]))),
        hits=int(float(str(row["hits"]))),
        quality=TrackQuality.from_dict(raw_quality),
        detections=tuple(
            TrackDetection.from_dict(item)
            for item in (raw_detections or [])
            if isinstance(item, dict)
        ),
    )


def load_tracklets(context: StageContext) -> tuple[Tracklet, ...]:
    """The typed form, which is what the stages after this one consume."""

    return tuple(tracklet_from_row(row) for row in load_tracks(context))


def tracks_path(context: StageContext) -> Path:
    path = context.work_dir / STAGE / TRACKS_NAME
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; run {STAGE} first")
    return path
