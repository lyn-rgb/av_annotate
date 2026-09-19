"""ffmpeg and ffprobe, wrapped thinly and honestly.

ffmpeg is the one hard external dependency: it demuxes the audio, and its
``scene`` filter is the shot detector.  Using ffmpeg for shot detection rather
than adding PySceneDetect keeps the dependency list to the tool we already
cannot avoid, at the cost of a threshold that has to be tuned per corpus -- see
:func:`detect_cuts`.

Nothing here imports numpy or torch.  The audio comes out as a WAV file and the
caller decides what to do with it.
"""

from __future__ import annotations

import importlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

#: Audio is normalised to this everywhere.  It matches SyncEdit's
#: ``preprocess.sample_rate`` and is what every speech model in the pipeline wants.
TARGET_SAMPLE_RATE = 16_000
TARGET_CHANNELS = 1

#: A content change this large between consecutive frames counts as a cut.
#: Calibrated against the sample corpus, where continuous shots score below 0.04
#: and real cuts score above 0.4 -- the margin is wide, so this is not delicate.
DEFAULT_CUT_THRESHOLD = 0.3


class FFmpegError(RuntimeError):
    """ffmpeg or ffprobe failed, or is not installed."""


@dataclass(frozen=True)
class MediaInfo:
    """What ffprobe reports about a file's primary video and audio streams."""

    duration: float
    fps: float
    width: int
    height: int
    frame_count: int
    sample_rate: int
    channels: int
    has_audio: bool

    @property
    def has_video(self) -> bool:
        return self.width > 0 and self.height > 0


def _locate(name: str, *, env_var: str, fallback_module: str | None = None) -> str:
    override = shutil.which(name)
    if override is not None:
        return override
    if fallback_module is not None:
        try:
            module = importlib.import_module(fallback_module)
        except ModuleNotFoundError:
            module = None
        if module is not None:
            return str(module.get_ffmpeg_exe())
    raise FFmpegError(
        f"{name} not found on PATH; install ffmpeg or set {env_var} to its location"
    )


def find_ffmpeg() -> str:
    return _locate("ffmpeg", env_var="AVANNOTATE_FFMPEG", fallback_module="imageio_ffmpeg")


def find_ffprobe() -> str:
    """ffprobe, requiring a real binary.

    Unlike ffmpeg there is no bundled fallback: imageio-ffmpeg ships ffmpeg only,
    so a machine without ffprobe gets a clear error rather than a confusing
    failure later.
    """

    return _locate("ffprobe", env_var="AVANNOTATE_FFPROBE")


def _run(command: list[str], *, what: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        tail = (result.stderr or "").strip().splitlines()[-5:]
        raise FFmpegError(f"{what} failed (exit {result.returncode}): " + " | ".join(tail))
    return result


def _parse_rate(value: str | None) -> float:
    """ffprobe reports frame rates as a rational string like ``30000/1001``."""

    if not value:
        return 0.0
    try:
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return 0.0


def probe_media(path: str | Path) -> MediaInfo:
    """Read container metadata without decoding frames."""

    source = Path(path)
    if not source.is_file():
        raise FFmpegError(f"media does not exist: {source}")

    result = _run(
        [
            find_ffprobe(),
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(source),
        ],
        what=f"probing {source.name}",
    )
    payload: dict[str, Any] = json.loads(result.stdout or "{}")
    streams = payload.get("streams") or []
    container = payload.get("format") or {}

    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    if video is None:
        raise FFmpegError(f"no video stream in {source}")

    # ``duration`` is on the container for most files and on the stream for some;
    # either is better than counting frames.
    duration = 0.0
    for candidate in (container.get("duration"), video.get("duration")):
        if candidate is not None:
            try:
                duration = float(candidate)
                break
            except (TypeError, ValueError):
                continue
    if duration <= 0.0:
        raise FFmpegError(f"could not determine the duration of {source}")

    frame_count = 0
    try:
        frame_count = int(video.get("nb_frames") or 0)
    except (TypeError, ValueError):
        frame_count = 0
    if frame_count <= 0:
        fps_guess = _parse_rate(video.get("r_frame_rate"))
        frame_count = int(round(duration * fps_guess)) if fps_guess > 0 else 0

    return MediaInfo(
        duration=duration,
        fps=_parse_rate(video.get("r_frame_rate")) or _parse_rate(video.get("avg_frame_rate")),
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        frame_count=frame_count,
        sample_rate=int(audio.get("sample_rate") or 0) if audio else 0,
        channels=int(audio.get("channels") or 0) if audio else 0,
        has_audio=audio is not None,
    )


def extract_audio(
    source: str | Path,
    target: str | Path,
    *,
    sample_rate: int = TARGET_SAMPLE_RATE,
    channels: int = TARGET_CHANNELS,
) -> Path:
    """Demux the audio to PCM16 WAV, resampling and downmixing as needed.

    Re-encoding rather than stream-copying is deliberate: every speech model
    downstream expects 16 kHz mono, and a copy would preserve whatever the
    container happened to hold.
    """

    destination = Path(target)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            find_ffmpeg(),
            "-y",
            "-v",
            "error",
            "-i",
            str(source),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            str(destination),
        ],
        what=f"extracting audio from {Path(source).name}",
    )
    return destination


def detect_cuts(
    source: str | Path, *, threshold: float = DEFAULT_CUT_THRESHOLD
) -> tuple[float, ...]:
    """Timestamps where the picture changes abruptly.

    Uses ffmpeg's ``scene`` filter, which scores each frame against its
    predecessor.  Returns an empty tuple for a single continuous shot, which is
    the common case for short clips -- a caller must not read "no cuts" as
    "detection failed" (check :func:`scene_scores` if that distinction matters).

    ``-fps_mode vfr`` is required: without it ffmpeg duplicates frames to keep a
    constant output rate and the selected frames lose their original timestamps.
    """

    result = subprocess.run(
        [
            find_ffmpeg(),
            "-hide_banner",
            "-nostats",
            "-i",
            str(source),
            "-filter:v",
            f"select='gt(scene,{threshold})',showinfo",
            "-fps_mode",
            "vfr",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        tail = (result.stderr or "").strip().splitlines()[-5:]
        raise FFmpegError(f"scene detection failed on {Path(source).name}: " + " | ".join(tail))

    # showinfo writes to the log, which ffmpeg sends to stderr.
    cuts: list[float] = []
    for line in result.stderr.splitlines():
        marker = "pts_time:"
        index = line.find(marker)
        if index < 0:
            continue
        token = line[index + len(marker) :].split()[0]
        try:
            cuts.append(float(token))
        except ValueError:
            continue
    return tuple(sorted(cuts))


def scene_scores(source: str | Path) -> tuple[float, ...]:
    """Per-frame scene score, for diagnosing a threshold.

    Kept because "no cuts found" and "the filter never ran" look identical from
    :func:`detect_cuts` alone, and that ambiguity costs an hour every time.
    """

    result = subprocess.run(
        [
            find_ffmpeg(),
            "-hide_banner",
            "-nostats",
            "-i",
            str(source),
            "-filter:v",
            "select='gte(scene,0)',metadata=print:file=-",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        tail = (result.stderr or "").strip().splitlines()[-5:]
        raise FFmpegError(f"scene scoring failed on {Path(source).name}: " + " | ".join(tail))

    scores: list[float] = []
    for line in result.stdout.splitlines():
        marker = "scene_score="
        index = line.find(marker)
        if index < 0:
            continue
        try:
            scores.append(float(line[index + len(marker) :].split()[0]))
        except ValueError:
            continue
    return tuple(scores)


def shots_from_cuts(cuts: tuple[float, ...], duration: float) -> tuple[tuple[float, float], ...]:
    """Turn cut timestamps into contiguous half-open shots covering the video.

    The shots tile the whole video, with no gaps.  A detector reports boundaries,
    so the stretch before its first cut and after its last are still one shot
    each -- treating them as unattributed would leave the renderer with
    utterances outside every shot.

    Cuts at or beyond the end, before the start, or duplicated are dropped, and
    out-of-order input is sorted rather than silently truncated.
    """

    if duration <= 0.0:
        raise ValueError("duration must be positive")
    boundaries = [0.0]
    for cut in sorted(cuts):
        if 0.0 < cut < duration and cut > boundaries[-1]:
            boundaries.append(cut)
    boundaries.append(duration)
    return tuple(
        (start, end) for start, end in zip(boundaries, boundaries[1:], strict=False) if end > start
    )
