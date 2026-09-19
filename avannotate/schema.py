"""The annotation deliverable's data contract.

``annotation.json`` is the only artifact downstream code reads.  Every
intermediate the pipeline writes is private to the stage that produced it, so a
change to the two-level script format is a change to ``annotation.py`` alone and
never re-runs a model.

Times are seconds from the start of the video.  Paths are relative to the
annotation file so a produced directory can be moved or handed off intact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION = "av-annotation-v1"

#: Reserved for speech with no visible face -- narration, a phone call, someone
#: off camera.  The utterance grammar requires a face tag, so off-screen speech
#: needs an id rather than an omitted tag; see ``annotation.parse_script``.
OFFSCREEN_FACE_ID = "F000"

#: A tag is rendered inside the utterance line, and the parser recovers it with
#: ``[\w-]+``.  Anything outside that charset would silently become part of the
#: text, so the vocabulary is closed and validated at construction.
TAG_PATTERN = r"[\w-]+"


@dataclass(frozen=True)
class Word:
    """One word with its own timestamps, as the aligner reported it."""

    text: str
    start: float
    end: float


@dataclass(frozen=True)
class Utterance:
    """One person's speech between two silence boundaries.

    ``tag`` is the paralinguistic label rendered before the spoken words
    (``whispering``, ``surprised``).  It is optional: a segment too short for the
    tagger to be trustworthy, or one the tagger scored as neutral, renders
    without it.
    """

    face_id: str
    start: float
    end: float
    text: str
    tag: str | None = None
    audio_path: str | None = None
    words: tuple[Word, ...] = ()
    confidence: float | None = None
    flags: tuple[str, ...] = ()

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def is_offscreen(self) -> bool:
        return self.face_id == OFFSCREEN_FACE_ID


@dataclass(frozen=True)
class FaceTrack:
    """One person in one video.

    A track is an identity, not a detector output: it may be assembled from
    several tracklets that the tracker split and clustering rejoined.
    """

    face_id: str
    first_seen: float
    last_seen: float
    speaks: bool
    total_speech: float = 0.0
    tracklets: tuple[str, ...] = ()
    quality: dict[str, float] = field(default_factory=dict)

    @property
    def screen_time(self) -> float:
        return self.last_seen - self.first_seen


@dataclass(frozen=True)
class Shot:
    """One camera shot, with the caption describing its visuals.

    Captions are visual only -- the example this format was specified from has
    no dialogue in either the global or the per-shot caption.  That is what lets
    the captioning stage run independently of the whole audio chain.
    """

    index: int
    start: float
    end: float
    caption: str = ""


@dataclass(frozen=True)
class VideoMeta:
    video_id: str
    path: str
    duration: float
    fps: float
    width: int
    height: int


@dataclass(frozen=True)
class Language:
    code: str
    confidence: float | None = None
    source: str = "unknown"


@dataclass(frozen=True)
class Annotation:
    """Everything known about one video, and the input to the renderer."""

    video: VideoMeta
    utterances: tuple[Utterance, ...] = ()
    shots: tuple[Shot, ...] = ()
    face_tracks: tuple[FaceTrack, ...] = ()
    global_caption: str = ""
    language: Language | None = None
    stats: dict[str, Any] = field(default_factory=dict)

    def face_track(self, face_id: str) -> FaceTrack | None:
        for track in self.face_tracks:
            if track.face_id == face_id:
                return track
        return None

    def utterances_for(self, face_id: str) -> tuple[Utterance, ...]:
        return tuple(item for item in self.utterances if item.face_id == face_id)


def new_face_id(ordinal: int) -> str:
    """Render the nth on-screen person's id.

    Ordinals are 1-based and zero-padded to three digits, matching the example
    format.  ``F000`` is deliberately unreachable: assigning it here would
    collide with the off-screen bucket.
    """

    if ordinal < 1:
        raise ValueError(f"face ordinals start at 1, got {ordinal}")
    return f"F{ordinal:03d}"


def parse_face_id(face_id: str) -> int:
    """Inverse of :func:`new_face_id`, for sorting and validation."""

    if not face_id.startswith("F") or not face_id[1:].isdigit():
        raise ValueError(f"not a face id: {face_id!r}")
    return int(face_id[1:])
