"""The shapes active speaker detection works in.

A detector answers "is *this* face talking right now", per frame.  That is a
dense series rather than a segmentation, and it is deliberately not thresholded
here: the pipeline's association stage wants the probability, because a face at
0.45 during another speaker's turn is evidence about who was *not* talking, and
a hard label would throw that away.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from avannotate.coercion import coerce_number
from avannotate.interval import Interval


@dataclass(frozen=True)
class Window:
    """One slice of the video the network is asked about.

    Frame counts, not just seconds: the network has a hard ceiling (200 frames
    for LoCoNet, past which it runs out of memory), so a window is planned in
    frames and its duration follows from the frame rate.
    """

    index: int
    start_frame: int
    end_frame: int
    start: float
    end: float

    @property
    def frame_count(self) -> int:
        return max(0, self.end_frame - self.start_frame)

    @property
    def interval(self) -> Interval:
        return Interval(self.start, self.end)

    def contains(self, time: float) -> bool:
        return self.start <= time < self.end

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start": round(self.start, 4),
            "end": round(self.end, 4),
        }


@dataclass(frozen=True)
class SpeakingSample:
    """The network's answer for one face at one instant."""

    time: float
    probability: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(f"probability out of range: {self.probability}")

    def to_dict(self) -> dict[str, object]:
        return {"time": round(self.time, 4), "probability": round(self.probability, 4)}

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> SpeakingSample:
        return cls(
            time=coerce_number(payload["time"], "time"),
            probability=coerce_number(payload["probability"], "probability"),
        )


@dataclass(frozen=True)
class TrackSpeaking:
    """One tracklet's speaking trace, in time order."""

    track_id: int
    samples: tuple[SpeakingSample, ...] = ()

    @property
    def span(self) -> Interval | None:
        if not self.samples:
            return None
        return Interval(self.samples[0].time, self.samples[-1].time)

    def probability_at(self, time: float, *, tolerance: float = 0.05) -> float | None:
        """The sample nearest ``time``, or ``None`` when none is close enough.

        Nearest rather than interpolated: a probability is not a quantity that
        means anything between two frames, and averaging across a turn boundary
        would invent a 0.5 that neither frame reported.
        """

        best: SpeakingSample | None = None
        for sample in self.samples:
            distance = abs(sample.time - time)
            if distance > tolerance:
                continue
            if best is None or distance < abs(best.time - time):
                best = sample
        return best.probability if best is not None else None

    def mean_probability(self, intervals: tuple[Interval, ...]) -> float | None:
        """Mean over samples inside ``intervals``, or ``None`` if none fall there."""

        selected = [
            sample.probability
            for sample in self.samples
            if any(interval.start <= sample.time < interval.end for interval in intervals)
        ]
        return sum(selected) / len(selected) if selected else None

    def to_dict(self) -> dict[str, object]:
        return {
            "track_id": self.track_id,
            "samples": [sample.to_dict() for sample in self.samples],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> TrackSpeaking:
        raw = payload.get("samples") or []
        if not isinstance(raw, list):
            raise ValueError(f"samples must be a list, got {type(raw).__name__}")
        return cls(
            track_id=int(coerce_number(payload["track_id"], "track_id")),
            samples=tuple(
                SpeakingSample.from_dict(item) for item in raw if isinstance(item, dict)
            ),
        )


@dataclass(frozen=True)
class AsdResult:
    """Every tracklet's speaking trace, plus what produced them."""

    tracks: tuple[TrackSpeaking, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def for_track(self, track_id: int) -> TrackSpeaking | None:
        for track in self.tracks:
            if track.track_id == track_id:
                return track
        return None

    @property
    def sample_count(self) -> int:
        return sum(len(track.samples) for track in self.tracks)

    def to_dict(self) -> dict[str, object]:
        return {
            "tracks": [track.to_dict() for track in self.tracks],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> AsdResult:
        raw = payload.get("tracks") or []
        if not isinstance(raw, list):
            raise ValueError(f"tracks must be a list, got {type(raw).__name__}")
        metadata = payload.get("metadata")
        return cls(
            tracks=tuple(
                TrackSpeaking.from_dict(item) for item in raw if isinstance(item, dict)
            ),
            metadata=dict(metadata) if isinstance(metadata, dict) else {},
        )
