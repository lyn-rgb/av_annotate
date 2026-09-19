"""Stage S0: probe the container, demux the audio, and find the shot boundaries.

Cheap, entirely local, and a prerequisite for everything else -- so it is also
where the timing question gets settled once, rather than being re-guessed by
every later stage.

The question is which duration is authoritative.  A demuxed track is routinely
tens of milliseconds longer than the video: AAC codes in fixed-size frames, so
the last frame's padding survives the round trip.  On the sample corpus the
audio runs 23-51 ms long.  Cutting audio by a timestamp derived from the video
is then off by up to a frame and a half, which is enough to slice a word in half
before target-speaker extraction ever sees it.

The video wins.  Its duration is an exact multiple of its frame rate and its
frames are what the face stages index into; the surplus audio is padding, and
:func:`load_timeline` is how later stages learn to clamp to the video's end
rather than the file's.
"""

from __future__ import annotations

import wave
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from avannotate.ffmpeg import (
    DEFAULT_CUT_THRESHOLD,
    TARGET_CHANNELS,
    TARGET_SAMPLE_RATE,
    MediaInfo,
    detect_cuts,
    extract_audio,
    probe_media,
    shots_from_cuts,
)
from avannotate.stages.base import (
    Artifact,
    StageContext,
    StageRecord,
    StageRun,
    StageState,
    hash_payload,
    read_json,
    write_json,
)

STAGE = "s0-preprocess"
VERSION = "s0-v2"

PROBE_NAME = "probe.json"
AUDIO_NAME = "mix.wav"
SHOTS_NAME = "shots.json"


@dataclass(frozen=True)
class S0Config:
    """S0's knobs, typed.

    A plain dict would carry ``object`` and force a coercion at every use; this
    also gives the config hash a stable shape to serialize.
    """

    sample_rate: int = TARGET_SAMPLE_RATE
    channels: int = TARGET_CHANNELS
    cut_threshold: float = DEFAULT_CUT_THRESHOLD

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> S0Config:
        return cls(
            sample_rate=int(mapping.get("sample_rate", TARGET_SAMPLE_RATE)),
            channels=int(mapping.get("channels", TARGET_CHANNELS)),
            cut_threshold=float(mapping.get("cut_threshold", DEFAULT_CUT_THRESHOLD)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "cut_threshold": self.cut_threshold,
        }


@dataclass(frozen=True)
class Timeline:
    """The settled timing for one video.

    ``duration`` is the video's, and is the one every later stage must clamp to.
    ``audio_delta`` is reported rather than hidden because a large or negative
    value means something is wrong with the file, not merely with the encoder.
    """

    duration: float
    audio_duration: float
    fps: float
    frame_count: int
    width: int
    height: int
    sample_rate: int

    @property
    def audio_delta(self) -> float:
        return self.audio_duration - self.duration

    def to_dict(self) -> dict[str, object]:
        return {
            "duration": self.duration,
            "audio_duration": self.audio_duration,
            "audio_delta": self.audio_delta,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
            "sample_rate": self.sample_rate,
        }


def _audio_duration(path: Path) -> float:
    """Length of a PCM WAV from its header, without decoding it.

    The stdlib ``wave`` module is enough: S0 writes the file itself as PCM16, so
    there is no format guessing to do.
    """

    with wave.open(str(path)) as handle:
        frames = int(handle.getnframes())
        rate = int(handle.getframerate())
    if rate <= 0:
        raise ValueError(f"{path} reports a zero sample rate")
    return frames / rate


def run(context: StageContext, *, force: bool = False) -> StageRun:
    config = S0Config.from_mapping(context.config)
    input_hash = _input_hash(context, config)

    state = StageState(context.work_dir)
    reason = state.reason_to_run(STAGE, code_version=VERSION, input_hash=input_hash)
    if reason is None and not force:
        probe = read_json(context.work_dir / STAGE / PROBE_NAME)
        return StageRun(
            stage=STAGE,
            skipped=True,
            reason="outputs present and unchanged",
            summary={"duration": probe["timeline"]["duration"]},
        )
    # Carry the trigger into the result.  "Why did this video re-run?" is the
    # first question asked of a batch log, and the answer exists here already.
    trigger = "forced" if reason is None else reason

    info: MediaInfo = probe_media(context.source)
    if not info.has_audio:
        raise ValueError(f"{context.source.name} has no audio track; S0 cannot proceed")

    audio_path = extract_audio(
        context.source,
        context.output(STAGE, AUDIO_NAME),
        sample_rate=config.sample_rate,
        channels=config.channels,
    )
    audio_duration = _audio_duration(audio_path)

    cuts = detect_cuts(context.source, threshold=config.cut_threshold)
    shots = shots_from_cuts(cuts, info.duration)

    timeline = Timeline(
        duration=info.duration,
        audio_duration=audio_duration,
        fps=info.fps,
        frame_count=info.frame_count,
        width=info.width,
        height=info.height,
        sample_rate=config.sample_rate,
    )

    probe_path = write_json(
        context.output(STAGE, PROBE_NAME),
        {
            "schema_version": "avannotate-probe-v1",
            "source": str(context.source),
            "timeline": timeline.to_dict(),
            "container": {
                "duration": info.duration,
                "fps": info.fps,
                "width": info.width,
                "height": info.height,
                "frame_count": info.frame_count,
                "sample_rate": info.sample_rate,
                "channels": info.channels,
            },
        },
    )
    shots_path = write_json(
        context.output(STAGE, SHOTS_NAME),
        {
            "schema_version": "avannotate-shots-v1",
            "cut_threshold": config.cut_threshold,
            "cuts": list(cuts),
            "shots": [
                {"index": index, "start": start, "end": end}
                for index, (start, end) in enumerate(shots, start=1)
            ],
        },
    )

    artifacts = tuple(
        Artifact.capture(context.work_dir, path)
        for path in (probe_path, audio_path, shots_path)
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
            "duration": timeline.duration,
            "audio_delta": timeline.audio_delta,
            "shots": len(shots),
            "cuts": len(cuts),
        },
    )


def _input_hash(context: StageContext, config: S0Config) -> str:
    """The source file's identity: path, size, and mtime.

    Hashing a multi-gigabyte video on every stage invocation would dominate the
    runtime, and size plus mtime already changes whenever the file does.
    """

    stat = context.source.stat()
    return hash_payload(
        {
            "source": str(context.source.resolve()),
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "config": config.to_dict(),
        }
    )


def load_timeline(context: StageContext) -> Timeline:
    """Read S0's verdict.  Raises if S0 has not run."""

    path = context.work_dir / STAGE / PROBE_NAME
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; run {STAGE} first")
    payload = read_json(path)["timeline"]
    return Timeline(
        duration=float(payload["duration"]),
        audio_duration=float(payload["audio_duration"]),
        fps=float(payload["fps"]),
        frame_count=int(payload["frame_count"]),
        width=int(payload["width"]),
        height=int(payload["height"]),
        sample_rate=int(payload["sample_rate"]),
    )


def load_shots(context: StageContext) -> tuple[tuple[int, float, float], ...]:
    """Shot boundaries as ``(index, start, end)``, 1-based to match the script."""

    path = context.work_dir / STAGE / SHOTS_NAME
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; run {STAGE} first")
    payload = read_json(path)
    return tuple(
        (int(item["index"]), float(item["start"]), float(item["end"]))
        for item in payload["shots"]
    )


def audio_path(context: StageContext) -> Path:
    path = context.work_dir / STAGE / AUDIO_NAME
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; run {STAGE} first")
    return path
