"""Turning a speaking trace into trim, silence-free utterances.

Two sources feed this.  The ASD trace says roughly where a face was talking;
the VAD says where there is speech at all.  Intersecting them removes the ASD
false positives (a face working its mouth over music), and gating on the VAD
after target-speaker extraction removes whatever the extractor failed to
suppress.

The thresholds are policy, not physics, so they live in one dataclass a caller
can override and every choice is explained where it is made.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from avannotate.coercion import coerce_number
from avannotate.interval import Interval, intersect, merge


@dataclass(frozen=True)
class SegmentationConfig:
    """When to join, trim and drop."""

    #: Two runs closer than this are one utterance.  Whisper transcribes a
    #: mid-sentence breath badly, and a 0.1 s gap is a breath, not a turn.
    max_gap: float = 0.20

    #: Runs shorter than this are dropped.  Below roughly a syllable there is no
    #: text to transcribe and no timbre to clone, so the row would only add noise.
    min_duration: float = 0.30

    #: Extend each end by this much.  Both the VAD and the extractor shave
    #: onsets and codas, so the recovered span is systematically short; this
    #: gives back the plosive and the final consonant.
    pad: float = 0.05

    #: Emitted, but flagged: long enough to keep, too short to trust.
    low_confidence_duration: float = 0.50

    #: Below this, no paralinguistic tag is rendered.  Emotion and style
    #: classifiers are unreliable on sub-second clips, and a wrong
    #: ``whispering:`` is worse than none -- the format makes the tag optional
    #: precisely so it can be withheld.
    min_tag_duration: float = 0.50


@dataclass(frozen=True)
class Segment:
    """One emitted utterance span, before any transcript is attached."""

    start: float
    end: float
    flags: tuple[str, ...] = ()

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def interval(self) -> Interval:
        return Interval(self.start, self.end)

    @property
    def is_low_confidence(self) -> bool:
        return "short" in self.flags


@dataclass(frozen=True)
class SpeechSegment:
    """One person's speech between two silences, and the file holding it.

    Shared by three stages, which is why it is here rather than in a stage or a
    package: S7 writes these, S8 transcribes them, S9 tags them.  A type two
    packages both need does not belong to either, and leaving it in the
    transcription package would make the tagger import the recogniser in order
    to learn what a segment is.

    ``start`` and ``end`` are video time.  ``audio`` is S7's extraction for this
    segment -- one person, and only while they were talking -- relative to the
    work directory, so a produced directory can be moved or handed off.
    """

    identity: str
    name: str
    start: float
    end: float
    audio: str

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def interval(self) -> Interval:
        return Interval(self.start, self.end)

    def to_dict(self) -> dict[str, object]:
        return {
            "identity": self.identity,
            "name": self.name,
            "start": round(self.start, 4),
            "end": round(self.end, 4),
            "duration": round(self.duration, 4),
            "audio": self.audio,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> SpeechSegment:
        identity = str(payload.get("identity", "")).strip()
        if not identity:
            raise ValueError("a segment needs an identity")
        return cls(
            identity=identity,
            name=str(payload.get("name", "")).strip(),
            start=coerce_number(payload.get("start", 0.0), "start"),
            end=coerce_number(payload.get("end", 0.0), "end"),
            audio=str(payload.get("audio", "")),
        )


def segment_speech(
    speaking: Sequence[Interval],
    speech_activity: Sequence[Interval] | None = None,
    config: SegmentationConfig | None = None,
    *,
    limit: float | None = None,
) -> tuple[Segment, ...]:
    """Produce the utterance spans for one speaker.

    ``speaking`` is the per-face ASD trace; ``speech_activity`` is the VAD (or
    the VAD of the extracted audio) and gates it when supplied.  ``limit`` is the
    media duration, so padding cannot run past the end of the video.
    """

    active = config or SegmentationConfig()
    if active.pad < 0.0:
        raise ValueError("pad cannot be negative")

    candidates = merge(speaking)
    if speech_activity is not None:
        candidates = intersect(candidates, speech_activity)
    candidates = merge(candidates, max_gap=active.max_gap)

    padded: list[Interval] = []
    for interval in candidates:
        start = max(0.0, interval.start - active.pad)
        end = interval.end + active.pad
        if limit is not None:
            end = min(end, limit)
        if end > start:
            padded.append(Interval(start, end))

    # Coalesce only the spans padding has actually made touch.  A gap that
    # survives padding is a real turn boundary: ``max_gap`` was already applied to
    # the unpadded speech, and padding exists to recover shaved onsets, not to
    # revise where the segmentation decided one utterance ends.
    padded = list(merge(padded))

    segments: list[Segment] = []
    for interval in padded:
        duration = interval.duration
        if duration < active.min_duration:
            continue
        flags: tuple[str, ...] = ()
        if duration < active.low_confidence_duration:
            flags = ("short",)
        segments.append(Segment(start=interval.start, end=interval.end, flags=flags))
    return tuple(segments)


class _Spanned(Protocol):
    """Anything with a duration, which is all this rule reads."""

    @property
    def duration(self) -> float: ...


def tag_eligible(segment: _Spanned, config: SegmentationConfig | None = None) -> bool:
    """Whether a paralinguistic tag should be rendered for this segment.

    Takes either kind of span: a bare :class:`Segment` before a transcript is
    attached, or a :class:`SpeechSegment` after extraction has given it audio.
    The rule is about length and nothing else, so it does not care which.
    """

    active = config or SegmentationConfig()
    return segment.duration >= active.min_tag_duration


def merge_adjacent(
    segments: Sequence[Segment], config: SegmentationConfig | None = None
) -> tuple[Segment, ...]:
    """Join segments that overlap or sit closer than ``max_gap``.

    Needed after per-segment extraction: the extractor runs on clipped spans and
    can return a slightly different length than it was given, which would
    otherwise leave hairline overlaps in the timeline.
    """

    active = config or SegmentationConfig()
    if not segments:
        return ()

    ordered = sorted(segments, key=lambda item: (item.start, item.end))
    merged: list[Segment] = [ordered[0]]
    for segment in ordered[1:]:
        last = merged[-1]
        if segment.start - last.end <= active.max_gap:
            flags = tuple(dict.fromkeys(last.flags + segment.flags))
            merged[-1] = Segment(start=last.start, end=max(last.end, segment.end), flags=flags)
        else:
            merged.append(segment)
    return tuple(merged)


def total_speech(segments: Sequence[Segment]) -> float:
    return sum(segment.duration for segment in segments)
