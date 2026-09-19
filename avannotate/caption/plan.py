"""Choosing which frames to show the model, and telling it who is in them.

Two decisions, both about what the model is allowed to know.

**Which frames.**  A shot is described from a handful of stills rather than from
the video, because the request is a scene description and the frames' job is to
cover the shot rather than to be any particular frame.  The count follows the
shot's length up to a cap: a one-second shot needs one frame, a one-minute shot
does not need sixty times as many, and the cost of a request is dominated by the
number of images in it.

Sampling from the middle of each interval rather than from its start is the
smaller half of the same idea -- the first frame of a shot is the one most likely
to be a transition, and the last is the one most likely to be a cut.

**Who is in them.**  The roster is computed from tracking and handed to the
model as a constraint.  The model cannot tell which face is F001 -- nothing in a
still says so -- so asking it to *identify* people would be asking for a guess.
Telling it who is present and asking it to use those names when it has to refer
to someone turns an unanswerable question into an answerable one, and gives the
stage a roster to check the answer against afterwards.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from avannotate.caption.types import ShotSample
from avannotate.faces.track import Tracklet
from avannotate.interval import Interval, merge, total_duration

#: A shot is sampled about this often, before the cap.
DEFAULT_SECONDS_PER_FRAME = 4.0

#: How many frames one shot may have.  The floor because a scene cannot be
#: described from nothing; the ceiling because request cost grows with the
#: number of images and the tenth frame of a shot adds less than the third.
DEFAULT_MIN_FRAMES = 1
DEFAULT_MAX_FRAMES = 8

#: How long an identity has to be visible in a shot to be listed as present.
#: Longer than zero because the tracker produces single-frame tracklets, and a
#: face detected once behind a shoulder should not put a name in the roster that
#: the model is then asked to use.
DEFAULT_MIN_PRESENCE = 0.2


def frames_for(duration: float, *, seconds_per_frame: float, low: int, high: int) -> int:
    """How many stills to describe a shot of this length with."""

    if duration <= 0.0:
        return low
    if seconds_per_frame <= 0.0:
        return high
    wanted = int(round(duration / seconds_per_frame))
    return max(low, min(high, wanted))


def sample_times(start: float, end: float, *, count: int) -> tuple[float, ...]:
    """``count`` times spread across a span.

    Midpoints of equal sub-intervals, which is the same as even spacing except
    that it never lands exactly on either edge.  The first of them is therefore
    early in the span without being its first frame -- which is what the global
    caption wants, since it takes one frame per shot and that frame should be
    from the start of each shot but not from its transition.
    """

    if count <= 0:
        return ()
    if count == 1:
        return (start,)
    step = (end - start) / count
    return tuple(round(start + (index + 0.5) * step, 4) for index in range(count))


def identity_presence(
    identity_tracks: Mapping[str, Sequence[int]],
    tracklets: Sequence[Tracklet],
    *,
    window: Interval,
    max_gap: float,
    min_presence: float = DEFAULT_MIN_PRESENCE,
) -> tuple[str, ...]:
    """Which identities are on screen in ``window``, for long enough to name.

    An identity can have several tracklets -- the tracker split them and S3
    rejoined them -- so presence is the union across the identity's tracklets
    rather than any one of them.  Missing that would drop a person from the
    roster whenever their time in a shot happened to be split across a cut in
    the tracking, and a missing name is a name the model cannot use.
    """

    by_id = {tracklet.track_id: tracklet for tracklet in tracklets}
    present: list[str] = []
    for face_id, track_ids in identity_tracks.items():
        spans: list[Interval] = []
        for track_id in track_ids:
            tracklet = by_id.get(track_id)
            if tracklet is None:
                continue
            for interval in tracklet.presence(max_gap=max_gap):
                if interval.overlap(window) > 0.0:
                    spans.append(
                        Interval(
                            max(interval.start, window.start),
                            min(interval.end, window.end),
                        )
                    )
        if total_duration(merge(spans)) >= min_presence:
            present.append(face_id)
    return tuple(sorted(present))


def plan_shot_samples(
    shots: Sequence[tuple[int, float, float]],
    identity_tracks: Mapping[str, Sequence[int]],
    tracklets: Sequence[Tracklet],
    *,
    duration: float,
    seconds_per_frame: float,
    min_frames: int,
    max_frames: int,
    presence_gap: float,
    min_presence: float,
) -> tuple[ShotSample, ...]:
    """Every shot, with its frames chosen and its roster resolved."""

    samples: list[ShotSample] = []
    for index, start, end in shots:
        # Clamped to the video: a shot detector can report a last shot that runs
        # past the end, and sampling there would ask ffmpeg for frames that do
        # not exist and get nothing.
        stop = min(end, duration) if duration > 0.0 else end
        if stop <= start:
            stop = start
        count = frames_for(
            stop - start,
            seconds_per_frame=seconds_per_frame,
            low=min_frames,
            high=max_frames,
        )
        samples.append(
            ShotSample(
                index=index,
                start=start,
                end=stop,
                times=sample_times(start, stop, count=count),
                identities=identity_presence(
                    identity_tracks,
                    tracklets,
                    window=Interval(start, stop),
                    max_gap=presence_gap,
                    min_presence=min_presence,
                ),
            )
        )
    return tuple(samples)


def global_sample(
    shots: Sequence[ShotSample], *, max_frames: int
) -> tuple[tuple[float, ...], tuple[str, ...]]:
    """One frame from the start of each shot, thinned to a budget, and everyone.

    Taking the first frame *chosen* for each shot -- which is early in the shot,
    not its first frame, since that is the one most likely to be a transition.
    An even spread over the timeline would land wherever the cuts happen to
    fall, while one frame per shot covers every distinct scene the video
    contains.

    When there are more shots than the budget allows they are thinned evenly, so
    the sample still spans the whole video rather than its first half.
    """

    everyone = sorted({face_id for shot in shots for face_id in shot.identities})
    starts = [shot.times[0] for shot in shots if shot.times]
    if len(starts) <= max_frames:
        return tuple(starts), tuple(everyone)

    # Even thinning: floor((i + 0.5) * n / max) walks the whole list rather than
    # stopping partway, and never repeats an index.
    chosen = [
        starts[min(len(starts) - 1, int((index + 0.5) * len(starts) / max_frames))]
        for index in range(max_frames)
    ]
    return tuple(dict.fromkeys(chosen)), tuple(everyone)
