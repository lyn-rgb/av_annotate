"""Slicing the video into windows, and choosing who shares each one.

LoCoNet does not score a face in isolation.  It scores a face *in the company of
the other faces on screen at the time*, because the discriminative signal for
one person talking is often that another person is not.  So a forward pass
covers a group, and the target's output is one channel of it.

That has three consequences this module implements:

* **The window length is bounded by memory, not preference.**  LoCoNet's own
  ablation puts 200 frames (~8 s) at the balance point and 400 frames out of
  memory, so the ceiling is planned in frames and the duration follows.
* **Several people per pass, but not many.**  The model was trained on AVA,
  where 99% of clips have three or fewer faces.  A scene with six gets its
  target plus the two who overlap it most, and the rest are dropped -- a
  documented limitation rather than a silent one.
* **Every target needs its own pass**, because the same window is a different
  group depending on who the target is.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from avannotate.asd.types import Window
from avannotate.faces.track import Tracklet
from avannotate.interval import Interval, merge, overlap_duration

#: Frames per forward pass.  LoCoNet's ablation: 20 frames is worst, 100 gives
#: about +5% mAP, 200 is the memory/accuracy balance, 400 does not fit.
DEFAULT_MAX_WINDOW_FRAMES = 200

#: Faces per pass, including the target.  AVA has three or fewer in 99% of its
#: clips, so the model has no reason to have learned a fourth.
DEFAULT_MAX_SPEAKERS = 3


@dataclass(frozen=True)
class WindowPlan:
    """The windows covering a video, and the parameters that produced them."""

    windows: tuple[Window, ...]
    window_frames: int
    step_frames: int
    dropped_tail_frames: int

    @property
    def covers_everything(self) -> bool:
        return self.dropped_tail_frames == 0


def plan_windows(
    duration: float,
    *,
    fps: float,
    window_seconds: float = 8.0,
    overlap_seconds: float = 1.0,
    max_frames: int = DEFAULT_MAX_WINDOW_FRAMES,
) -> WindowPlan:
    """Cover ``[0, duration)`` with overlapping windows no longer than ``max_frames``.

    Overlap is not optional: a face's speaking state is decided per frame from
    surrounding context, so a prediction at a window boundary is made with half
    the evidence.  Overlapping windows let the stitcher keep the interior of
    each window and discard its edges.
    """

    if duration <= 0.0:
        raise ValueError(f"duration must be positive, got {duration}")
    if fps <= 0.0:
        raise ValueError(f"fps must be positive, got {fps}")
    if max_frames < 1:
        raise ValueError(f"max_frames must be at least 1, got {max_frames}")

    total_frames = max(1, int(round(duration * fps)))
    window_frames = min(max(1, int(round(window_seconds * fps))), max_frames)
    step_frames = window_frames - max(0, int(round(overlap_seconds * fps)))

    if step_frames < 1:
        # An overlap as long as the window would never advance.  Clamping to a
        # single-frame step keeps the plan valid and makes the mistake visible
        # in the reported step rather than as a hang.
        step_frames = 1

    windows: list[Window] = []
    start = 0
    while start < total_frames:
        end = min(start + window_frames, total_frames)
        windows.append(
            Window(
                index=len(windows),
                start_frame=start,
                end_frame=end,
                start=start / fps,
                end=end / fps,
            )
        )
        if end >= total_frames:
            break
        start += step_frames

    covered = windows[-1].end_frame if windows else 0
    return WindowPlan(
        windows=tuple(windows),
        window_frames=window_frames,
        step_frames=step_frames,
        dropped_tail_frames=max(0, total_frames - covered),
    )


#: A tracklet seen once contributes a span this long, so that a brief
#: appearance can still be compared with others.  Zero-length spans never
#: overlap, which would make every such tracklet look equally unrelated.
_SINGLE_DETECTION_SECONDS = 0.1


def activity(tracklet: Tracklet, window: Window) -> tuple[Interval, ...]:
    """The spans a tracklet is present for, inside one window.

    A detection is an instant, not a span: the face is present from one
    detection until the next.  Building the spans that way is what makes
    "do these two overlap in time" a question with an answer -- treating each
    detection as a zero-length interval makes every pair disjoint.
    """

    times = sorted(
        detection.time
        for detection in tracklet.detections
        if window.start <= detection.time < window.end
    )
    if not times:
        return ()
    if len(times) == 1:
        return (Interval(times[0], times[0] + _SINGLE_DETECTION_SECONDS),)

    # The last detection has no successor to bound it, so it is extended by the
    # median spacing between detections -- the best available estimate of how
    # long this tracklet persists between sightings.  Measured from the times
    # rather than the spans: deriving it from span starts gives a different
    # answer for two detections than for three at the same spacing.
    gaps = sorted(times[index + 1] - times[index] for index in range(len(times) - 1))
    tail = gaps[len(gaps) // 2]

    spans = [Interval(times[index], times[index + 1]) for index in range(len(times) - 1)]
    spans.append(Interval(times[-1], times[-1] + tail))
    return merge(spans)


def tracks_in_window(tracklets: Sequence[Tracklet], window: Window) -> tuple[int, ...]:
    """Ids of the tracklets with at least one detection inside the window."""

    return tuple(sorted(t.track_id for t in tracklets if activity(t, window)))


def context_speakers(
    target_id: int,
    tracklets: Sequence[Tracklet],
    window: Window,
    *,
    max_speakers: int = DEFAULT_MAX_SPEAKERS,
) -> tuple[int, ...]:
    """The ids sharing a forward pass with ``target_id``, target first.

    Chosen by how much they overlap the target's own presence in the window,
    because that is the signal the model uses: someone on screen while the
    target talks is what makes the target's lip motion discriminative.  Ties
    break to the lower track id, so a rerun sees the same group.
    """

    if max_speakers < 1:
        raise ValueError(f"max_speakers must be at least 1, got {max_speakers}")

    by_id = {tracklet.track_id: tracklet for tracklet in tracklets}
    if target_id not in by_id:
        return ()

    target_spans = activity(by_id[target_id], window)
    if not target_spans:
        return (target_id,)

    scored: list[tuple[float, int]] = []
    for tracklet in tracklets:
        if tracklet.track_id == target_id:
            continue
        other_spans = activity(tracklet, window)
        if not other_spans:
            continue
        scored.append((overlap_duration(target_spans, other_spans), tracklet.track_id))

    scored.sort(key=lambda item: (-item[0], item[1]))
    chosen = [other_id for _, other_id in scored[: max_speakers - 1]]
    return (target_id, *chosen)


def groups_per_window(
    tracklets: Sequence[Tracklet],
    window: Window,
    *,
    max_speakers: int = DEFAULT_MAX_SPEAKERS,
) -> tuple[tuple[int, ...], ...]:
    """One group per target active in the window.

    Every target needs its own pass: the same window is a different company
    depending on who is being scored, because the target is always included and
    the context is drawn from its overlap partners.
    """

    return tuple(
        context_speakers(target_id, tracklets, window, max_speakers=max_speakers)
        for target_id in tracks_in_window(tracklets, window)
    )
