"""Stage S10: what the video looks like.

Two levels, as the deliverable asks for: one sentence for the whole video and
one for each shot.  Both are purely visual -- the format's own example has no
dialogue in either -- so this stage reads no audio and does not depend on S4
through S9 at all.  It can run first, last, or beside them.

The one thing it does depend on is tracking, and not for the pictures: the model
is told which tracked people are in the frames it is shown, so that when it has
to refer to a person it uses the same identifier the rest of the annotation
does.  Nothing in a still says which face is F001, so asking it to work that out
would be asking for a guess; telling it who is present turns an unanswerable
question into an answerable one.

And then the answer is checked.  A name the model uses that was not on the
roster it was given is removed, because a name the tracker never saw is a claim
this pipeline cannot support -- see :mod:`avannotate.caption.verify`.  The
removal is recorded, so a caption that reads oddly can be traced to it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from avannotate.caption.model import Captioner, CaptionError, build_captioner
from avannotate.caption.plan import (
    DEFAULT_MAX_FRAMES,
    DEFAULT_MIN_FRAMES,
    DEFAULT_MIN_PRESENCE,
    DEFAULT_SECONDS_PER_FRAME,
    global_sample,
    plan_shot_samples,
)
from avannotate.caption.prompt import global_prompt, shot_prompt
from avannotate.caption.types import GlobalCaption, ShotCaption
from avannotate.caption.verify import check_references, flags_for
from avannotate.faces.frames import iter_sampled_frames, read_frame
from avannotate.faces.types import Frame
from avannotate.ffmpeg import FFmpegError
from avannotate.model_cache import model_for
from avannotate.stages import s0_preprocess, s2_tracks, s3_cluster
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

STAGE = "s10-caption"
VERSION = "s10-v1"

CAPTIONS_NAME = "captions.json"
SUMMARY_NAME = "summary.json"

#: How much of the video's width the model is shown.  A description of a room
#: and its furniture does not need 1080p, and the token cost of an image scales
#: with its area -- this is the cheapest lever on the stage's runtime that does
#: not change what is being asked.
DEFAULT_MAX_EDGE = 448


@dataclass(frozen=True)
class S10Config:
    backend: str = "qwen3-vl"
    model: str = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    device: str | None = None
    #: Model precision.  The 30B mixture-of-experts checkpoint is about 61 GB in
    #: bf16, which does not fit a 48 GB card; fp8 or a 4-bit quantisation does.
    dtype: str | None = None
    device_map: str | None = None
    max_new_tokens: int = 128
    #: Frames per shot, and how the count scales with the shot's length.
    seconds_per_frame: float = DEFAULT_SECONDS_PER_FRAME
    min_frames: int = DEFAULT_MIN_FRAMES
    max_frames: int = DEFAULT_MAX_FRAMES
    #: Frames for the global description: one per shot, thinned to this.
    global_max_frames: int = 12
    #: How long someone must be visible in a shot to be named in its roster.
    min_presence: float = DEFAULT_MIN_PRESENCE
    presence_gap: float = 0.5
    max_edge: int = DEFAULT_MAX_EDGE
    seed: int = 0

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> S10Config:
        return cls(
            backend=config_str(mapping, "backend", "qwen3-vl"),
            model=config_str(mapping, "model", "Qwen/Qwen3-VL-30B-A3B-Instruct"),
            device=config_optional_str(mapping, "device"),
            dtype=config_optional_str(mapping, "dtype"),
            device_map=config_optional_str(mapping, "device_map"),
            max_new_tokens=config_int(mapping, "max_new_tokens", 128),
            seconds_per_frame=config_float(
                mapping, "seconds_per_frame", DEFAULT_SECONDS_PER_FRAME
            ),
            min_frames=config_int(mapping, "min_frames", DEFAULT_MIN_FRAMES),
            max_frames=config_int(mapping, "max_frames", DEFAULT_MAX_FRAMES),
            global_max_frames=config_int(mapping, "global_max_frames", 12),
            min_presence=config_float(mapping, "min_presence", DEFAULT_MIN_PRESENCE),
            presence_gap=config_float(mapping, "presence_gap", 0.5),
            max_edge=config_int(mapping, "max_edge", DEFAULT_MAX_EDGE),
            seed=config_int(mapping, "seed", 0),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "model": self.model,
            "device": self.device,
            "dtype": self.dtype,
            "device_map": self.device_map,
            "max_new_tokens": self.max_new_tokens,
            "seconds_per_frame": self.seconds_per_frame,
            "min_frames": self.min_frames,
            "max_frames": self.max_frames,
            "global_max_frames": self.global_max_frames,
            "min_presence": self.min_presence,
            "presence_gap": self.presence_gap,
            "max_edge": self.max_edge,
            "seed": self.seed,
        }


def _scaled(width: int, height: int, *, max_edge: int) -> tuple[int, int]:
    """The frame size to decode at, preserving the aspect ratio.

    Rounded to even numbers because the decoder resizes the *decoded* frame, and
    an odd width in the middle of a chroma-subsampled stream is a source of
    off-by-one complaints that have nothing to do with what is being asked.
    """

    if max_edge <= 0 or max(width, height) <= max_edge:
        return width, height
    scale = max_edge / max(width, height)
    return max(2, int(width * scale) // 2 * 2), max(2, int(height * scale) // 2 * 2)


def _frames_at(
    context: StageContext,
    times: Sequence[float],
    *,
    width: int,
    height: int,
    window: tuple[float, float],
) -> list[Frame]:
    """Decode a shot's chosen frames in one ffmpeg pass.

    The pass covers the whole span and samples within it, so the timestamps are
    a description of the sampling rather than a list of seeks, and the frames
    that are not wanted are never decoded.  One pass per shot is why a shot is
    the unit here at all.
    """

    start, end = window
    if not times or end <= start:
        return []
    return list(
        iter_sampled_frames(
            context.source,
            width=width,
            height=height,
            start_time=start,
            duration=end - start,
            count=len(times),
        )
    )


def _caption_frames(
    captioner: Captioner,
    frames: list[Frame],
    *,
    prompt: str,
    allowed: tuple[str, ...],
    known: set[str],
) -> tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """One request, one checked answer.

    A shot with no decodable frames is not sent: an empty image list is not a
    request any of these models answers usefully, and the honest result is an
    absent caption with a flag rather than whatever the model says about
    nothing.
    """

    if not frames:
        return "", (), (), ("no_frames",)
    checked = check_references(captioner.caption(frames, prompt=prompt), allowed=allowed)
    return checked.text, checked.referenced, checked.dropped, flags_for(checked, known=known)


def run(context: StageContext, *, force: bool = False) -> StageRun:
    config = S10Config.from_mapping(context.config)
    input_hash = _input_hash(context, config)

    state = StageState(context.work_dir)
    reason = state.reason_to_run(STAGE, code_version=VERSION, input_hash=input_hash)
    if reason is None and not force:
        return StageRun(stage=STAGE, skipped=True, reason="outputs present and unchanged")
    trigger = "forced" if reason is None else reason

    timeline = s0_preprocess.load_timeline(context)
    shots = s0_preprocess.load_shots(context)
    tracklets = s2_tracks.load_tracklets(context)
    identity_tracks = s3_cluster.load_identity_tracks(context)
    known = set(identity_tracks)

    samples = plan_shot_samples(
        shots,
        identity_tracks,
        tracklets,
        duration=timeline.duration,
        seconds_per_frame=config.seconds_per_frame,
        min_frames=config.min_frames,
        max_frames=config.max_frames,
        presence_gap=config.presence_gap,
        min_presence=config.min_presence,
    )
    width, height = _scaled(timeline.width, timeline.height, max_edge=config.max_edge)

    if not samples:
        # Nothing to describe.  Returning here rather than below matters:
        # building the captioner loads tens of gigabytes of checkpoint, and
        # doing that to caption no shots at all is the one cost in this stage
        # worth avoiding with a guard.
        state.save(
            StageRecord(
                stage=STAGE,
                status="ok",
                code_version=VERSION,
                config_hash=hash_payload(config.to_dict()),
                input_hash=input_hash,
                artifacts=(
                    Artifact.capture(
                        context.work_dir,
                        write_json(
                            context.output(STAGE, CAPTIONS_NAME),
                            {
                                "schema_version": "avannotate-captions-v1",
                                "config": config.to_dict(),
                                "backend": {"name": "none", "model": config.model},
                                "shots": [],
                                "global": GlobalCaption(caption="").to_dict(),
                            },
                        ),
                    ),
                ),
            )
        )
        return StageRun(
            stage=STAGE, skipped=False, reason=trigger, summary={"shots": 0, "captioned": 0}
        )

    try:
        captioner = model_for(
            STAGE,
            {
                "backend": config.backend,
                "model": config.model,
                "device": config.device,
                "dtype": config.dtype,
                "device_map": config.device_map,
                "max_new_tokens": config.max_new_tokens,
                "seed": config.seed,
            },
            build_captioner,
        )
        captioned: list[ShotCaption] = []
        for shot in samples:
            frames = _frames_at(
                context,
                shot.times,
                width=width,
                height=height,
                window=(shot.start, shot.end),
            )
            text, referenced, dropped, flags = _caption_frames(
                captioner,
                frames,
                prompt=shot_prompt(shot, total=len(samples)),
                allowed=shot.identities,
                known=known,
            )
            captioned.append(
                ShotCaption(
                    index=shot.index,
                    start=shot.start,
                    end=shot.end,
                    caption=text,
                    identities=shot.identities,
                    referenced=referenced,
                    dropped=dropped,
                    flags=flags,
                )
            )

        times, everyone = global_sample(samples, max_frames=config.global_max_frames)
        global_frames = _global_frames(
            context, times, width=width, height=height, duration=timeline.duration
        )
        overall, referenced, dropped, flags = _caption_frames(
            captioner,
            global_frames,
            prompt=global_prompt(
                identities=everyone, frames=len(global_frames), shots=len(samples)
            ),
            allowed=everyone,
            known=known,
        )
    except (CaptionError, FFmpegError) as error:
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

    result = GlobalCaption(
        caption=overall,
        identities=everyone,
        referenced=referenced,
        dropped=dropped,
        flags=flags,
        frames=len(global_frames),
    )
    captions_path = write_json(
        context.output(STAGE, CAPTIONS_NAME),
        {
            "schema_version": "avannotate-captions-v1",
            "config": config.to_dict(),
            "backend": {"name": captioner.name, "model": config.model},
            "shots": [item.to_dict() for item in captioned],
            "global": result.to_dict(),
        },
    )
    summary_path = write_json(
        context.output(STAGE, SUMMARY_NAME),
        {
            "schema_version": "avannotate-captions-summary-v1",
            "shots": len(captioned),
            "captioned": sum(1 for item in captioned if item.caption),
            "empty": sum(1 for item in captioned if not item.caption),
            "global_frames": result.frames,
            "named": sum(1 for item in captioned if item.referenced),
            "referenced": _counts(
                [name for item in captioned for name in item.referenced]
            ),
            "dropped": _counts([name for item in captioned for name in item.dropped]),
            "flags": _counts([flag for item in captioned for flag in item.flags])
            | ({"global:" + flag: 1 for flag in result.flags}),
            "identities": sorted(known),
        },
    )

    artifacts = tuple(
        Artifact.capture(context.work_dir, path)
        for path in (captions_path, summary_path)
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
            "shots": len(captioned),
            "captioned": sum(1 for item in captioned if item.caption),
            "dropped_names": sum(len(item.dropped) for item in captioned) + len(dropped),
        },
    )


def _global_frames(
    context: StageContext,
    times: Sequence[float],
    *,
    width: int,
    height: int,
    duration: float,
) -> list[Frame]:
    """One frame at each time, in a single seek per frame.

    These are spread across the whole video by construction, so there is no
    span to decode through and no gain in pretending otherwise: one pass per
    picked shot is the cheapest correct thing, and the number of them is capped
    by ``global_max_frames``.
    """

    frames: list[Frame] = []
    for time in times:
        if time >= duration:
            continue
        frame = read_frame(
            context.source, width=width, height=height, time=time
        )
        if frame is not None:
            frames.append(frame)
    return frames


def _counts(values: Sequence[object]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _input_hash(context: StageContext, config: S10Config) -> str:
    stat = context.source.stat()
    return hash_payload(
        {
            "config": config.to_dict(),
            "shots": hash_file(s0_preprocess.shots_path(context)),
            "tracks": hash_file(s2_tracks.tracks_path(context)),
            "identities": hash_file(s3_cluster.identities_path(context)),
            "video": {"size": stat.st_size, "mtime": stat.st_mtime},
        }
    )


def captions_path(context: StageContext) -> Path:
    path = context.work_dir / STAGE / CAPTIONS_NAME
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; run {STAGE} first")
    return path


def load_captions(context: StageContext) -> tuple[ShotCaption, ...]:
    payload = json.loads(captions_path(context).read_text(encoding="utf-8"))
    raw = payload.get("shots")
    if not isinstance(raw, list):
        raise ValueError(f"{captions_path(context)} has no shots list; re-run {STAGE}")
    return tuple(
        ShotCaption.from_dict(item) for item in raw if isinstance(item, dict)
    )


def load_global_caption(context: StageContext) -> GlobalCaption:
    payload = json.loads(captions_path(context).read_text(encoding="utf-8"))
    raw = payload.get("global")
    if not isinstance(raw, dict):
        raise ValueError(f"{captions_path(context)} has no global caption; re-run {STAGE}")
    return GlobalCaption(
        caption=str(raw.get("caption", "")),
        identities=tuple(str(item) for item in raw.get("identities", []) or []),
        referenced=tuple(str(item) for item in raw.get("referenced", []) or []),
        dropped=tuple(str(item) for item in raw.get("dropped", []) or []),
        flags=tuple(str(item) for item in raw.get("flags", []) or []),
        frames=int(raw.get("frames", 0) or 0),
    )
