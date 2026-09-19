"""Diarization turns, and the geometry that follows from them.

A diarizer reports per-speaker activity, and those intervals are allowed to
overlap -- that is what simultaneous speech looks like.  The interesting
quantity for this pipeline is exactly that overlap: two people talking at once
is the case the whole design exists to handle, and it is what target-speaker
extraction in S7 has to undo.

Times are seconds from the start of the audio.  A caller passing turns into the
video timeline must clamp them first; S0 established that a demuxed track runs
tens of milliseconds longer than its video, so a turn near the end of the file
can sit past the end of the picture.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from avannotate.interval import Interval, merge, overlap_duration, total_duration


@dataclass(frozen=True)
class SpeakerTurn:
    """One speaker active over one span."""

    speaker: str
    start: float
    end: float

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(
                f"turn ends before it starts: {self.speaker} {self.start} > {self.end}"
            )

    @property
    def interval(self) -> Interval:
        return Interval(self.start, self.end)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict[str, object]:
        return {
            "speaker": self.speaker,
            "start": round(self.start, 4),
            "end": round(self.end, 4),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> SpeakerTurn:
        return cls(
            speaker=str(payload["speaker"]),
            start=float(payload["start"]),  # type: ignore[arg-type]
            end=float(payload["end"]),  # type: ignore[arg-type]
        )


def overlap_intervals(turns: Sequence[SpeakerTurn]) -> tuple[Interval, ...]:
    """Spans where two or more distinct speakers are active.

    A sweep rather than an all-pairs scan: an hour of meeting audio has
    thousands of turns, and the quadratic form would dominate the stage's cost
    for no benefit.

    Distinct speakers, not distinct turns: one person's two adjacent turns that
    happen to touch are not an overlap, and counting them as one would report
    simultaneous speech where there is none.
    """

    events: list[tuple[float, int, str]] = []
    for turn in turns:
        events.append((turn.start, 1, turn.speaker))
        events.append((turn.end, -1, turn.speaker))
    if not events:
        return ()
    # Ends before starts at the same timestamp, so touching turns are adjacent
    # rather than overlapping.
    events.sort(key=lambda event: (event[0], event[1]))

    active: dict[str, int] = {}
    spans: list[Interval] = []
    span_start: float | None = None
    previous = events[0][0]

    index = 0
    while index < len(events):
        time = events[index][0]
        if time > previous:
            if len(active) >= 2 and span_start is None:
                span_start = previous
            elif len(active) < 2 and span_start is not None:
                spans.append(Interval(span_start, previous))
                span_start = None

        while index < len(events) and events[index][0] == time:
            _, delta, speaker = events[index]
            active[speaker] = active.get(speaker, 0) + delta
            if active[speaker] <= 0:
                del active[speaker]
            index += 1
        previous = time

    if span_start is not None:
        spans.append(Interval(span_start, previous))
    return tuple(spans)


@dataclass(frozen=True)
class DiarizationResult:
    """Everything a diarizer reports about one file."""

    turns: tuple[SpeakerTurn, ...] = ()
    #: Free-form provenance: model name, revision, device, whatever the backend
    #: wants recorded.  Kept opaque so a backend change does not change the type.
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def speakers(self) -> tuple[str, ...]:
        """Distinct speaker labels, sorted.

        Sorted because downstream matching builds a matrix indexed by this list,
        and a set iteration order would make the assignment depend on nothing.
        """

        return tuple(sorted({turn.speaker for turn in self.turns}))

    @property
    def speech_seconds(self) -> float:
        """Time with any speech, counting simultaneous speakers once."""

        return total_duration([turn.interval for turn in self.turns])

    @property
    def speaker_seconds(self) -> float:
        """Total speaker-time, counting simultaneous speakers once each."""

        return sum(turn.duration for turn in self.turns)

    @property
    def overlap_seconds(self) -> float:
        return total_duration(overlap_intervals(self.turns))

    @property
    def overlap_ratio(self) -> float:
        """Overlap against speech, not against the file.

        A quiet clip with two words spoken over each other is not 40% overlapped
        in any sense that matters; this measures the fraction of speech that is
        contested.
        """

        speech = self.speech_seconds
        return self.overlap_seconds / speech if speech > 0.0 else 0.0

    def clamped(self, duration: float) -> DiarizationResult:
        """Fit the turns inside ``[0, duration]``, dropping what falls outside.

        Needed because the audio is longer than the video: a diarizer reports
        against the file it was handed, and its last turn can extend past the
        end of the picture.
        """

        fitted: list[SpeakerTurn] = []
        for turn in self.turns:
            start = min(max(turn.start, 0.0), duration)
            end = min(max(turn.end, 0.0), duration)
            if end > start:
                fitted.append(SpeakerTurn(speaker=turn.speaker, start=start, end=end))
        return DiarizationResult(turns=tuple(fitted), metadata=dict(self.metadata))

    def merged(self, *, max_gap: float) -> DiarizationResult:
        """Join each speaker's turns that sit closer than ``max_gap``.

        Diarization emits many short turns for one continuous utterance; the
        segmentation stage downstream cares about speaker changes, not about
        where the sliding window happened to land.
        """

        by_speaker: dict[str, list[SpeakerTurn]] = {}
        for turn in self.turns:
            by_speaker.setdefault(turn.speaker, []).append(turn)

        joined: list[SpeakerTurn] = []
        for speaker, turns in by_speaker.items():
            for interval in merge([turn.interval for turn in turns], max_gap=max_gap):
                joined.append(
                    SpeakerTurn(speaker=speaker, start=interval.start, end=interval.end)
                )
        joined.sort(key=lambda turn: (turn.start, turn.end, turn.speaker))
        return DiarizationResult(turns=tuple(joined), metadata=dict(self.metadata))

    def to_dict(self) -> dict[str, object]:
        return {
            "turns": [
                turn.to_dict()
                for turn in sorted(
                    self.turns, key=lambda t: (t.start, t.end, t.speaker)
                )
            ],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> DiarizationResult:
        raw = payload.get("turns") or []
        if not isinstance(raw, list):
            raise ValueError(f"turns must be a list, got {type(raw).__name__}")
        metadata = payload.get("metadata")
        return cls(
            turns=tuple(
                SpeakerTurn.from_dict(item) for item in raw if isinstance(item, dict)
            ),
            metadata=dict(metadata) if isinstance(metadata, dict) else {},
        )

    def summary(self) -> dict[str, object]:
        return {
            "turns": len(self.turns),
            "speakers": len(self.speakers),
            "speech_seconds": round(self.speech_seconds, 3),
            "speaker_seconds": round(self.speaker_seconds, 3),
            "overlap_seconds": round(self.overlap_seconds, 3),
            "overlap_ratio": round(self.overlap_ratio, 4),
        }


def speaker_overlap_seconds(
    turns: Sequence[SpeakerTurn], first: str, second: str
) -> float:
    """Shared time between two named speakers."""

    return overlap_duration(
        [turn.interval for turn in turns if turn.speaker == first],
        [turn.interval for turn in turns if turn.speaker == second],
    )
