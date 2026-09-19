"""Which audio each segment's transcript should come from.

Two sources are available for every segment and they are good at different
things.

The **original mix** is the best audio there is -- it is what the microphone
recorded, with no model between the speaker and the recogniser.  But it holds
everyone at once, so it is only usable while one person is talking.

**S7's extraction** holds exactly one person, so it stays usable when two
people are talking at once -- which is precisely when the mix stops being
usable.  The price is that it has been through a separation model, and a
separated voice is not a clean one.

So the rule is: read the mix when this person is the only one speaking and the
extraction when they are not.  That is a decision about a time span, not about
audio, so it is made here from interval arithmetic and can be tested without
either file existing.

The one thing this must not do is guess.  A segment read from the mix while
someone else was also talking produces a transcript that is fluent, confident,
and half the other person's words -- and nothing downstream can tell.  So
overlap is measured against every *other* identity's speech, not against the
same person's, and the threshold is on the overlap's duration rather than on
whether the two intervals touch.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from avannotate.asr.types import (
    SOURCE_EXTRACTED,
    SOURCE_MIX,
    SegmentSource,
    SpeechSegment,
)
from avannotate.interval import Interval


def overlap_seconds(
    segment: SpeechSegment,
    speech: Mapping[str, Sequence[Interval]],
    *,
    exclude: str,
) -> float:
    """How long somebody other than ``exclude`` was speaking during the segment.

    Total, not pairwise: three people overlapping at once is one number, because
    what the mix sounds like is one thing.
    """

    span = Interval(segment.start, segment.end)
    total = 0.0
    for identity, intervals in speech.items():
        if identity == exclude:
            continue
        for interval in intervals:
            total += span.overlap(interval)
    return total


def is_overlapping(
    segment: SpeechSegment,
    speech: Mapping[str, Sequence[Interval]],
    *,
    min_seconds: float,
) -> bool:
    return overlap_seconds(segment, speech, exclude=segment.identity) >= min_seconds


def window(
    segment: SpeechSegment, *, context_seconds: float, duration: float
) -> tuple[float, float]:
    """The span of the mix to read for a segment: the segment plus context.

    Clamped to the video rather than the audio file.  The demuxed track runs
    longer than the video -- AAC pads its last frame -- and a window that
    reached into that padding would hand the recogniser samples that are not in
    the video, under timestamps that claim they are.
    """

    start = max(0.0, segment.start - context_seconds)
    end = min(duration, segment.end + context_seconds)
    if end <= start:
        # A segment entirely past the end of the video: S6 can produce one when
        # the diarizer's last turn runs past the video's duration.  Reading a
        # zero-length window would fail inside the recogniser; reading the
        # segment's own extraction is what S7 would have written anyway.
        return segment.start, segment.end
    return start, end


def plan_sources(
    segments: Sequence[SpeechSegment],
    speech: Mapping[str, Sequence[Interval]],
    *,
    mix_audio: str,
    duration: float,
    context_seconds: float,
    min_overlap_seconds: float,
) -> tuple[SegmentSource, ...]:
    """Choose a source for every segment, in timeline order.

    Ordered so the stage's output does not depend on the order a plan happened
    to iterate in -- two runs of the same video have to produce byte-identical
    files or the resume record is worthless.
    """

    sources: list[SegmentSource] = []
    for segment in sorted(segments, key=lambda item: (item.start, item.identity, item.name)):
        overlapping = is_overlapping(segment, speech, min_seconds=min_overlap_seconds)
        if overlapping:
            sources.append(
                SegmentSource(
                    segment=segment,
                    path=segment.audio,
                    origin=segment.start,
                    file_origin=segment.start,
                    duration=segment.duration,
                    source=SOURCE_EXTRACTED,
                    overlapping=True,
                )
            )
            continue

        start, end = window(segment, context_seconds=context_seconds, duration=duration)
        sources.append(
            SegmentSource(
                segment=segment,
                path=mix_audio,
                origin=start,
                file_origin=0.0,
                duration=end - start,
                source=SOURCE_MIX,
                overlapping=False,
            )
        )
    return tuple(sources)
