"""Stage S5: which face is talking, frame by frame.

The stage that decides *who*, as opposed to S4's *when*.  A diarizer says two
people spoke between 3 s and 5 s; this says which of the faces on screen was
which.  S6 puts the two together.

The work is a loop over windows and, inside each, over targets -- because
LoCoNet scores a face in the company of its neighbours, and the same window is a
different company depending on who is being scored.  Everything that decides
*what* goes into a pass, and what comes out, lives in ``avannotate.asd`` and is
tested without a network; what is left here is the loop.

Output is a probability per tracklet per frame, deliberately unthresholded: S6
wants the number, because a face at 0.45 during someone else's turn is evidence
about who was not talking.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from avannotate.asd.batch import boxes_for_window, read_faces, stack_speakers
from avannotate.asd.model import (
    AsdError,
    AsdModel,
    build_asd_model,
    features_for_window,
)
from avannotate.asd.stitch import stitch_predictions
from avannotate.asd.types import AsdResult, TrackSpeaking, Window
from avannotate.asd.window import (
    DEFAULT_MAX_SPEAKERS,
    DEFAULT_MAX_WINDOW_FRAMES,
    groups_per_window,
    plan_windows,
)
from avannotate.faces.frames import iter_gray_window
from avannotate.faces.track import Tracklet
from avannotate.model_cache import model_for
from avannotate.stages import s0_preprocess, s2_tracks
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
    resolve_config_path,
    write_json,
)

STAGE = "s5-asd"
#: LoCoNet, on torch.
USES_GPU = True
#: v2: the audio frontend was a generic log-mel rather than VGGish's, which
#: made every probability this stage produced meaningless.  The bump is what
#: makes the stage re-run -- ``reason_to_run`` compares this string, and the
#: fix changed neither the config nor any input, so a v1 record would have
#: been considered current and the bad numbers reused.  See
#: :func:`avannotate.asd.model.log_mel`.
VERSION = "s5-v2"

SPEAKING_NAME = "speaking.jsonl"
SUMMARY_NAME = "summary.json"


@dataclass(frozen=True)
class S5Config:
    backend: str = "loconet"
    checkpoint: str | None = None
    repo: str | None = None
    device: str | None = None
    window_seconds: float = 8.0
    overlap_seconds: float = 1.0
    max_window_frames: int = DEFAULT_MAX_WINDOW_FRAMES
    max_speakers: int = DEFAULT_MAX_SPEAKERS
    #: TalkNet's cropScale, inherited by LoCoNet's preprocessing.
    margin: float = 0.40
    crop_size: int = 112

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> S5Config:
        checkpoint = config_optional_str(mapping, "checkpoint")
        repo = config_optional_str(mapping, "repo")
        return cls(
            backend=config_str(mapping, "backend", "loconet"),
            checkpoint=(
                str(resolve_config_path(checkpoint, mapping)) if checkpoint else None
            ),
            repo=str(resolve_config_path(repo, mapping)) if repo else None,
            device=config_optional_str(mapping, "device"),
            window_seconds=config_float(mapping, "window_seconds", 8.0),
            overlap_seconds=config_float(mapping, "overlap_seconds", 1.0),
            max_window_frames=config_int(
                mapping, "max_window_frames", DEFAULT_MAX_WINDOW_FRAMES
            ),
            max_speakers=config_int(mapping, "max_speakers", DEFAULT_MAX_SPEAKERS),
            margin=config_float(mapping, "margin", 0.40),
            crop_size=config_int(mapping, "crop_size", 112),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "checkpoint": self.checkpoint,
            "repo": self.repo,
            "device": self.device,
            "window_seconds": self.window_seconds,
            "overlap_seconds": self.overlap_seconds,
            "max_window_frames": self.max_window_frames,
            "max_speakers": self.max_speakers,
            "margin": self.margin,
            "crop_size": self.crop_size,
        }


def _score_window(
    model: AsdModel,
    context: StageContext,
    timeline: s0_preprocess.Timeline,
    tracklets: tuple[Tracklet, ...],
    window: Window,
    config: S5Config,
) -> dict[int, NDArray[np.float32]]:
    """One prediction array per track active in the window."""

    frames = list(
        iter_gray_window(
            context.source,
            width=timeline.width,
            height=timeline.height,
            start_time=window.start,
            count=window.frame_count,
        )
    )
    if len(frames) != window.frame_count:
        raise ValueError(
            f"window {window.index} asked for {window.frame_count} frames and got "
            f"{len(frames)}; the crops would not align with the audio"
        )

    audio = s0_preprocess.load_audio_window(
        context,
        start_seconds=window.start,
        duration_seconds=window.frame_count / timeline.fps,
    )
    features = features_for_window(
        audio, video_frames=window.frame_count, fps=timeline.fps
    )

    by_id = {tracklet.track_id: tracklet for tracklet in tracklets}
    results: dict[int, NDArray[np.float32]] = {}
    for group in groups_per_window(tracklets, window, max_speakers=config.max_speakers):
        tiles = [
            read_faces(
                frames,
                boxes_for_window(by_id[track_id], window),
                size=config.crop_size,
                margin=config.margin,
            )
            for track_id in group
        ]
        crops = stack_speakers(tiles)
        probabilities = model.score(crops, features)
        # The target is first by construction in `context_speakers`, and only
        # its row is meaningful: the others were context, not questions.
        results[group[0]] = np.asarray(probabilities[0], dtype=np.float32)
    return results


def run(context: StageContext, *, force: bool = False) -> StageRun:
    config = S5Config.from_mapping(context.config)
    input_hash = _input_hash(context, config)

    state = StageState(context.work_dir)
    reason = state.reason_to_run(STAGE, code_version=VERSION, input_hash=input_hash)
    if reason is None and not force:
        return StageRun(stage=STAGE, skipped=True, reason="outputs present and unchanged")
    trigger = "forced" if reason is None else reason

    timeline = s0_preprocess.load_timeline(context)
    tracklets = s2_tracks.load_tracklets(context)

    plan = plan_windows(
        timeline.duration,
        fps=timeline.fps,
        window_seconds=config.window_seconds,
        overlap_seconds=config.overlap_seconds,
        max_frames=config.max_window_frames,
    )

    if not tracklets:
        # Nothing to score.  Written as an empty result rather than skipped, so
        # S6 can tell "ran and found no faces" from "has not run".
        result = AsdResult(metadata={"backend": config.backend, "windows": 0})
        model_name = "none"
    else:
        try:
            model = model_for(
                STAGE,
                {
                    "backend": config.backend,
                    "checkpoint": config.checkpoint,
                    "repo": config.repo,
                    "device": config.device,
                },
                build_asd_model,
            )
        except AsdError as error:
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

        collected: dict[int, list[tuple[Window, NDArray[np.float32]]]] = defaultdict(list)
        for window in plan.windows:
            for track_id, probabilities in _score_window(
                model, context, timeline, tracklets, window, config
            ).items():
                collected[track_id].append((window, probabilities))

        result = AsdResult(
            tracks=tuple(
                TrackSpeaking(
                    track_id=track_id,
                    samples=stitch_predictions(collected[track_id], fps=timeline.fps),
                )
                for track_id in sorted(collected)
            ),
            metadata={
                "backend": config.backend,
                "model": model.name,
                "device": getattr(model, "device", None),
                "windows": len(plan.windows),
                # How much of the checkpoint the encoder recognised.  Recorded
                # by the adapter since the beginning and never written down
                # anywhere: if the weights do not belong to this network,
                # ``missing`` is most of them and every score below is noise
                # from an untrained encoder, which looks exactly like a model
                # that is merely unimpressed by the video.
                "checkpoint_load": getattr(model, "load_report", None),
            },
        )
        model_name = model.name

    speaking_path = context.output(STAGE, SPEAKING_NAME)
    temporary = speaking_path.with_name(speaking_path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for track in result.tracks:
            handle.write(json.dumps(track.to_dict(), ensure_ascii=False) + "\n")
    temporary.replace(speaking_path)

    summary_path = write_json(
        context.output(STAGE, SUMMARY_NAME),
        {
            "schema_version": "avannotate-asd-summary-v1",
            "config": config.to_dict(),
            "backend": dict(result.metadata),
            "windows": [window.to_dict() for window in plan.windows],
            "window_plan": {
                "window_frames": plan.window_frames,
                "step_frames": plan.step_frames,
                "dropped_tail_frames": plan.dropped_tail_frames,
            },
            "tracks": len(result.tracks),
            "samples": result.sample_count,
            "model": model_name,
        },
    )

    artifacts = tuple(
        Artifact.capture(context.work_dir, path)
        for path in (speaking_path, summary_path)
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
            "tracks": len(result.tracks),
            "samples": result.sample_count,
            "windows": len(plan.windows),
            "model": model_name,
        },
    )


def _input_hash(context: StageContext, config: S5Config) -> str:
    tracks = s2_tracks.tracks_path(context)
    timeline = s0_preprocess.load_timeline(context)
    # The video is identified by size and mtime rather than content: it is
    # gigabytes, and S0 already re-runs when it changes.
    stat = context.source.stat()
    return hash_payload(
        {
            "config": config.to_dict(),
            "tracks": hash_file(tracks),
            "video": {"size": stat.st_size, "mtime": stat.st_mtime},
            "timeline": timeline.duration,
        }
    )


def load_result(context: StageContext) -> AsdResult:
    path = speaking_path(context)
    tracks: list[TrackSpeaking] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                tracks.append(TrackSpeaking.from_dict(json.loads(stripped)))

    summary_path = context.work_dir / STAGE / SUMMARY_NAME
    metadata: dict[str, object] = {}
    if summary_path.is_file():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        backend = payload.get("backend")
        if isinstance(backend, dict):
            metadata = dict(backend)
    return AsdResult(tracks=tuple(tracks), metadata=metadata)


def speaking_path(context: StageContext) -> Path:
    path = context.work_dir / STAGE / SPEAKING_NAME
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; run {STAGE} first")
    return path
