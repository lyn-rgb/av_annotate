"""What a caption is, what it is given, and what has to be true about it.

The visual description is the one part of the deliverable that does not come
from the audio chain at all -- the format's own example has no dialogue in
either the global or the per-shot caption.  That decoupling is why this stage
can run on any schedule, and why its record is kept separate from the
transcript's.
"""

from __future__ import annotations

from dataclasses import dataclass

from avannotate.coercion import coerce_number


@dataclass(frozen=True)
class ShotSample:
    """One shot, the frames chosen to describe it, and who is in it.

    ``identities`` comes from tracking and never from the model.  That is the
    whole point of it: the model is *told* who is present so it can refer to
    them by name, and that roster is the thing its output is checked against.
    """

    index: int
    start: float
    end: float
    times: tuple[float, ...]
    #: Empty means tracking saw nobody, which is a state the model is told about
    #: rather than one it is left to guess at.
    identities: tuple[str, ...] = ()

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "start": round(self.start, 4),
            "end": round(self.end, 4),
            "duration": round(self.duration, 4),
            "frames": [round(time, 4) for time in self.times],
            "identities": list(self.identities),
        }


@dataclass(frozen=True)
class ShotCaption:
    """One shot's caption, and the audit of the names in it.

    ``referenced`` is what survived checking, ``dropped`` is what did not.  Both
    are recorded because a caption whose names were stripped still reads as
    prose, and a reader looking at it cannot tell that anything was removed.
    """

    index: int
    start: float
    end: float
    caption: str
    identities: tuple[str, ...] = ()
    referenced: tuple[str, ...] = ()
    dropped: tuple[str, ...] = ()
    flags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "start": round(self.start, 4),
            "end": round(self.end, 4),
            "caption": self.caption,
            "identities": list(self.identities),
            "referenced": list(self.referenced),
            "dropped": list(self.dropped),
            "flags": list(self.flags),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> ShotCaption:
        def strings(key: str) -> tuple[str, ...]:
            raw = payload.get(key)
            return tuple(str(item) for item in raw) if isinstance(raw, list) else ()

        return cls(
            index=int(coerce_number(payload.get("index", 0), "index")),
            start=coerce_number(payload.get("start", 0.0), "start"),
            end=coerce_number(payload.get("end", 0.0), "end"),
            caption=str(payload.get("caption", "")),
            identities=strings("identities"),
            referenced=strings("referenced"),
            dropped=strings("dropped"),
            flags=strings("flags"),
        )


@dataclass(frozen=True)
class GlobalCaption:
    """The video's overall description, and the same audit."""

    caption: str
    identities: tuple[str, ...] = ()
    referenced: tuple[str, ...] = ()
    dropped: tuple[str, ...] = ()
    flags: tuple[str, ...] = ()
    #: How many frames the description was written from, so a reader can tell a
    #: caption written from a full spread from one written from a single frame.
    frames: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "caption": self.caption,
            "identities": list(self.identities),
            "referenced": list(self.referenced),
            "dropped": list(self.dropped),
            "flags": list(self.flags),
            "frames": self.frames,
        }
