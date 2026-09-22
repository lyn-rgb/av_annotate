"""Deciding what to extract, and which face crops to condition it on.

Two kinds of unit to keep apart:

* A **segment** is one contiguous stretch of one person talking.  It is what the
  deliverable is made of -- one audio file per segment -- and it comes out of
  S6's speaking intervals run through the same trimming rules S8 will use, so
  the file on disk and the transcript written for it describe the same span.

* A **crop window** is the run of frames handed to the extractor.  It is longer
  than the segment at both ends, because an extractor given a hard cut at the
  first phoneme has no context for it, and the file it returns is trimmed back
  afterwards.

Running the extractor per segment rather than per person is what keeps the cost
proportional to speech rather than to video: a person on screen for a minute and
talking for five seconds costs five seconds of extraction, not sixty.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from avannotate.faces.track import Tracklet
from avannotate.interval import Interval
from avannotate.matching import Box
from avannotate.segment import SegmentationConfig, segment_speech

#: Frames of context the extractor is given on each side of a segment.
#:
#: The model is conditioned on lip motion, and a crop that starts exactly at the
#: first phoneme starts with a mouth already moving -- the encoder has no
#: baseline to compare against.  Half a second is enough to establish one and
#: short enough not to drag a neighbouring speaker into the crop.
DEFAULT_CONTEXT_SECONDS = 0.5


@dataclass(frozen=True)
class ExtractionSegment:
    """One stretch of one person talking, and the file it becomes."""

    identity: str
    index: int
    start: float
    end: float
    flags: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        return segment_name(self.identity, self.index)

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def interval(self) -> Interval:
        return Interval(self.start, self.end)

    def to_dict(self) -> dict[str, object]:
        return {
            "identity": self.identity,
            "index": self.index,
            "name": self.name,
            "start": round(self.start, 4),
            "end": round(self.end, 4),
            "duration": round(self.duration, 4),
            "flags": list(self.flags),
        }


def segment_name(identity: str, index: int) -> str:
    """``F001`` and 0 to ``F001_0000``.

    Four digits because a long video can hold thousands of segments for one
    person, and a name that sorts lexically the same as it sorts numerically
    keeps a directory listing in time order.
    """

    if index < 0:
        raise ValueError(f"segment index cannot be negative, got {index}")
    return f"{identity}_{index:04d}"


def plan_extractions(
    speech: Mapping[str, Sequence[Interval]],
    *,
    duration: float,
    config: SegmentationConfig | None = None,
) -> dict[str, tuple[ExtractionSegment, ...]]:
    """Turn each person's speaking intervals into the segments to extract.

    Uses S8's trimming rules deliberately: the segments written here are the
    ones the transcript will be written for, and two definitions of where an
    utterance starts would put the text and the audio out of step.
    """

    active = config or SegmentationConfig()
    planned: dict[str, tuple[ExtractionSegment, ...]] = {}

    for identity in sorted(speech):
        spans = segment_speech(list(speech[identity]), config=active, limit=duration)
        planned[identity] = tuple(
            ExtractionSegment(
                identity=identity,
                index=index,
                start=span.start,
                end=span.end,
                flags=span.flags,
            )
            for index, span in enumerate(spans)
        )
    return planned


def identity_boxes(
    tracklets: Sequence[Tracklet],
    *,
    start_frame: int,
    end_frame: int,
    fill: bool = True,
) -> tuple[Box | None, ...]:
    """One box per frame for a person, drawn from whichever of their tracklets
    has a sighting nearest that frame.

    An identity can hold several tracklets -- a camera pan splits them -- so a
    per-tracklet answer is not enough: the crop has to follow the person across
    the split, and a frame where the tracker had not yet re-acquired them still
    needs a box or the extractor is handed a blank tile.
    """

    if end_frame < start_frame:
        raise ValueError(f"end_frame {end_frame} precedes start_frame {start_frame}")

    sightings: dict[int, Box] = {}
    for tracklet in tracklets:
        for detection in tracklet.detections:
            if start_frame <= detection.frame_index < end_frame:
                sightings.setdefault(detection.frame_index, detection.box)

    boxes: list[Box | None] = []
    held: Box | None = None
    for frame_index in range(start_frame, end_frame):
        found = sightings.get(frame_index)
        if found is not None:
            held = found
        boxes.append(found if found is not None else (held if fill else None))

    if not fill:
        return tuple(boxes)

    # Leading frames, before the first sighting, take the first box: the
    # alternative is a black tile the extractor has to interpret.
    first = next((box for box in boxes if box is not None), None)
    return tuple(first if box is None else box for box in boxes)


def context_window(
    segment: ExtractionSegment, *, fps: float, frame_count: int, context_seconds: float
) -> tuple[int, int]:
    """The frame range the extractor actually sees, wider than the segment.

    Clamped to ``frame_count`` -- the frames the container holds -- and not to
    ``duration * fps``, which is what this used to do and is a different number.
    Eight and a bit seconds at 25 fps rounds to more frames than a file that
    ends on frame 214 has, so the window asked for one that does not exist, and
    the decoder refused the whole window rather than its last frame:

        s7-tse: FFmpegError: decoded 67 frames ... starting at 5.881s, expected 68

    The video was fine.  ``Timeline.frame_count`` is the authority and has been
    all along: S0's docstring says every later stage clamps to it, and this one
    was clamping to the duration instead.
    """

    if fps <= 0.0:
        raise ValueError(f"fps must be positive, got {fps}")
    if frame_count < 0:
        raise ValueError(f"frame_count cannot be negative, got {frame_count}")
    total = max(1, frame_count)
    start = max(0, int(round((segment.start - context_seconds) * fps)))
    end = min(total, int(round((segment.end + context_seconds) * fps)))
    return start, max(start + 1, end)
