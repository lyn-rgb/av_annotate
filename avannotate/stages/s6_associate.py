"""Stage S6: which face each diarization speaker is.

Where the pipeline converges.  S3 produced people, S4 produced turn-taking, S5
produced per-frame evidence of who was talking -- this decides who is who, and
every stage after it reads that decision.

The algorithm is in ``avannotate.associate`` and was written and tested before
any of the stages that feed it existed.  What is left here is the adaptation:
turning three stages' outputs into the two shapes it takes, and turning its
verdicts into something the rest of the pipeline can use.

Three things worth stating plainly about the outcome:

* **A speaker can come out as off-screen.**  A diarizer hears a voice; no face
  ever matches it.  That is narration, a phone call, or someone out of frame,
  and it becomes ``F000`` rather than being pinned to the least-bad face.
* **A speaker can share a face with another.**  That is what diarization
  over-segmentation looks like -- one person split into two clusters -- and
  sending the spare cluster off-screen would relabel real speech as narration.
  It is reported as ``merged``.
* **An assignment can be ambiguous.**  Two faces explain a speaker comparably
  well, and the margin says so.  The assignment is still reported; a reviewer
  wants the best guess and its competition, not a blank.

What comes out, beyond the mapping, is each identity's speaking intervals --
the turns of the speakers assigned to it.  That is what gives a face a voice
timeline, and it is the input S7 extracts audio for.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from avannotate import associate as algorithm
from avannotate.asd.types import AsdResult
from avannotate.associate import AssociationConfig, FaceObservation, SpeakerAssignment
from avannotate.audio.types import SpeakerTurn
from avannotate.coercion import coerce_number
from avannotate.faces.track import Tracklet
from avannotate.interval import Interval, merge, total_duration
from avannotate.stages import (
    s2_tracks,
    s3_cluster,
    s4_diarize,
    s5_asd,
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

STAGE = "s6-associate"
#: arithmetic over the tracks; no model at all.
USES_GPU = False
VERSION = "s6-v1"

ASSIGNMENTS_NAME = "assignments.json"
SUMMARY_NAME = "summary.json"

#: Visibility spans closer than this are one presence.  A face that leaves and
#: returns within half a second was not really gone, and treating the gap as an
#: absence would split its coverage against a speaker's turn.
DEFAULT_PRESENCE_GAP = 0.5


@dataclass(frozen=True)
class S6Config:
    min_score: float = algorithm.DEFAULT_MIN_SCORE
    min_margin: float = algorithm.DEFAULT_MIN_MARGIN
    min_coverage: float = algorithm.DEFAULT_MIN_COVERAGE
    contrast_penalty: float = algorithm.DEFAULT_CONTRAST_PENALTY
    presence_gap_seconds: float = DEFAULT_PRESENCE_GAP

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> S6Config:
        return cls(
            min_score=config_float(mapping, "min_score", algorithm.DEFAULT_MIN_SCORE),
            min_margin=config_float(mapping, "min_margin", algorithm.DEFAULT_MIN_MARGIN),
            min_coverage=config_float(
                mapping, "min_coverage", algorithm.DEFAULT_MIN_COVERAGE
            ),
            contrast_penalty=config_float(
                mapping, "contrast_penalty", algorithm.DEFAULT_CONTRAST_PENALTY
            ),
            presence_gap_seconds=config_float(
                mapping, "presence_gap_seconds", DEFAULT_PRESENCE_GAP
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "min_score": self.min_score,
            "min_margin": self.min_margin,
            "min_coverage": self.min_coverage,
            "contrast_penalty": self.contrast_penalty,
            "presence_gap_seconds": self.presence_gap_seconds,
        }

    def association(self) -> AssociationConfig:
        return AssociationConfig(
            min_score=self.min_score,
            min_margin=self.min_margin,
            min_coverage=self.min_coverage,
            contrast_penalty=self.contrast_penalty,
        )


def face_observations(
    identity_tracks: Mapping[str, Sequence[int]],
    tracklets: Sequence[Tracklet],
    speaking: AsdResult,
    config: S6Config,
) -> tuple[FaceObservation, ...]:
    """Build the algorithm's view of each person from three stages' outputs.

    Visibility comes from the tracklets' own presence; the speaking trace is
    their ASD samples merged in time order.  Both are per *identity*, not per
    tracklet, because that is the unit the assignment is about -- a person split
    across two tracklets by a camera pan is still one face to match.
    """

    by_id = {tracklet.track_id: tracklet for tracklet in tracklets}
    observations: list[FaceObservation] = []

    for face_id in sorted(identity_tracks):
        spans: list[Interval] = []
        samples: list[algorithm.SpeechSample] = []
        for track_id in identity_tracks[face_id]:
            tracklet = by_id.get(track_id)
            if tracklet is None:
                continue
            spans.extend(tracklet.presence())
            trace = speaking.for_track(track_id)
            if trace is not None:
                samples.extend(
                    algorithm.SpeechSample(time=sample.time, probability=sample.probability)
                    for sample in trace.samples
                )

        observations.append(
            FaceObservation(
                face_id=face_id,
                visible=merge(spans, max_gap=config.presence_gap_seconds),
                speech=tuple(sorted(samples, key=lambda sample: sample.time)),
            )
        )
    return tuple(observations)


def algorithm_turns(turns: Sequence[SpeakerTurn]) -> tuple[algorithm.SpeakerTurn, ...]:
    """Adapt the diarizer's turns to the algorithm's own narrow type.

    Two shapes with the same fields, kept apart on purpose: the algorithm should
    not depend on a provider's type, and a diarizer should not have to satisfy
    an algorithm's.  The mapping is asserted in the tests so it cannot drift.
    """

    return tuple(
        algorithm.SpeakerTurn(speaker=turn.speaker, start=turn.start, end=turn.end)
        for turn in turns
    )


def speaking_intervals(
    face_id: str, assignments: Sequence[SpeakerAssignment], turns: Sequence[SpeakerTurn],
    *, gap: float,
) -> tuple[Interval, ...]:
    """When the person behind ``face_id`` was talking.

    The union of their speakers' turns.  An identity can hold more than one
    speaker -- that is the over-segmentation case -- and the intervals are merged
    so a cluster boundary inside one utterance does not show up as a pause.
    """

    speakers = {item.speaker for item in assignments if item.face_id == face_id}
    if not speakers:
        return ()
    return merge(
        [turn.interval for turn in turns if turn.speaker in speakers], max_gap=gap
    )


def run(context: StageContext, *, force: bool = False) -> StageRun:
    config = S6Config.from_mapping(context.config)
    input_hash = _input_hash(context, config)

    state = StageState(context.work_dir)
    reason = state.reason_to_run(STAGE, code_version=VERSION, input_hash=input_hash)
    if reason is None and not force:
        return StageRun(stage=STAGE, skipped=True, reason="outputs present and unchanged")
    trigger = "forced" if reason is None else reason

    identity_tracks = s3_cluster.load_identity_tracks(context)
    tracklets = s2_tracks.load_tracklets(context)
    turns = s4_diarize.load_turns(context)
    speaking = s5_asd.load_result(context)

    assignments = algorithm.assign_speakers(
        face_observations(identity_tracks, tracklets, speaking, config),
        algorithm_turns(turns),
        config.association(),
    )

    identities: list[dict[str, object]] = []
    for face_id in sorted(identity_tracks):
        intervals = speaking_intervals(
            face_id, assignments, turns, gap=config.presence_gap_seconds
        )
        identities.append(
            {
                "face_id": face_id,
                "track_ids": list(identity_tracks[face_id]),
                "speakers": sorted(item.speaker for item in assignments if item.face_id == face_id),
                "speaking_intervals": [
                    [round(interval.start, 4), round(interval.end, 4)] for interval in intervals
                ],
                "speaking_seconds": round(total_duration(intervals), 4),
            }
        )

    offscreen = [item for item in assignments if item.is_offscreen]
    assignments_path = write_json(
        context.output(STAGE, ASSIGNMENTS_NAME),
        {
            "schema_version": "avannotate-assignments-v1",
            "config": config.to_dict(),
            "assignments": [
                {
                    "speaker": item.speaker,
                    "face_id": item.face_id,
                    "score": round(item.score, 4),
                    "margin": round(item.margin, 4),
                    "coverage": round(item.coverage, 4),
                    "candidate": item.candidate,
                    "ambiguous": item.ambiguous,
                    "merged": item.merged,
                    "reason": item.reason,
                }
                for item in assignments
            ],
            "identities": identities,
            "offscreen": {
                "speakers": [item.speaker for item in offscreen],
                "seconds": round(
                    sum(turn.duration for turn in turns
                        if turn.speaker in {item.speaker for item in offscreen}),
                    4,
                ),
            },
        },
    )

    matched = [item for item in assignments if not item.is_offscreen]
    summary_path = write_json(
        context.output(STAGE, SUMMARY_NAME),
        {
            "schema_version": "avannotate-assignments-summary-v1",
            "counts": {
                "speakers": len(assignments),
                "assigned": len(matched),
                "offscreen": len(offscreen),
                "ambiguous": sum(1 for item in assignments if item.ambiguous),
                "merged": sum(1 for item in assignments if item.merged),
                # People nobody claimed: on screen, never matched to a voice.
                "unclaimed_faces": sum(1 for item in identities if not item["speakers"]),
            },
            "scores": {
                "min": round(min((item.score for item in matched), default=0.0), 4),
                "mean": round(
                    sum(item.score for item in matched) / len(matched), 4
                ) if matched else 0.0,
            },
        },
    )

    artifacts = tuple(
        Artifact.capture(context.work_dir, path)
        for path in (assignments_path, summary_path)
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
            "speakers": len(assignments),
            "assigned": len(matched),
            "offscreen": len(offscreen),
            "ambiguous": sum(1 for item in assignments if item.ambiguous),
            "merged": sum(1 for item in assignments if item.merged),
        },
    )


def _input_hash(context: StageContext, config: S6Config) -> str:
    return hash_payload(
        {
            "config": config.to_dict(),
            "identities": hash_file(s3_cluster.identities_path(context)),
            "turns": hash_file(s4_diarize.turns_path(context)),
            "speaking": hash_file(s5_asd.speaking_path(context)),
            "tracks": hash_file(s2_tracks.tracks_path(context)),
        }
    )


def load_assignments(context: StageContext) -> tuple[dict[str, object], ...]:
    path = assignments_path(context)
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("assignments")
    if not isinstance(raw, list):
        raise ValueError(f"{path} has no assignments list; re-run {STAGE}")
    return tuple(item for item in raw if isinstance(item, dict))


def load_identity_speech(context: StageContext) -> dict[str, tuple[Interval, ...]]:
    """``face_id -> when that person was talking``, which S7 consumes."""

    path = assignments_path(context)
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("identities")
    if not isinstance(raw, list):
        raise ValueError(f"{path} has no identities list; re-run {STAGE}")

    speech: dict[str, tuple[Interval, ...]] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        face_id = str(item["face_id"])
        intervals = item.get("speaking_intervals") or []
        speech[face_id] = tuple(
            Interval(
                coerce_number(pair[0], "start"),
                coerce_number(pair[1], "end"),
            )
            for pair in intervals
            if isinstance(pair, list) and len(pair) == 2
        )
    return speech


def assignments_path(context: StageContext) -> Path:
    path = context.work_dir / STAGE / ASSIGNMENTS_NAME
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; run {STAGE} first")
    return path
