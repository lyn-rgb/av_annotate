"""Stage S2's algorithm: ByteTrack over per-frame detections.

The two-stage association is the whole point.  Detections below the tracking
threshold are not discarded; they are held back and offered to tracks that the
first pass could not match.  A face turning away from the camera scores lower
for a few frames, and a tracker that drops those detections loses the track
exactly when the face is hardest to re-find.  Keeping them is what makes
ByteTrack hold identity through an occlusion that a plain SORT loses.

Detections are indexed in *update steps*, not video frames: one call to
:meth:`ByteTracker.update` per sampled frame.  ``max_time_lost`` is therefore in
steps, and the stage converts its seconds-based config before constructing the
tracker, because "this track survives one second of occlusion" is a statement a
human can tune and "thirty update steps" is not.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from avannotate.faces.kalman import KalmanFilter, Matrix, Vector
from avannotate.faces.types import Detection, FrameDetections
from avannotate.matching import Box, iou_distance, linear_assignment


class TrackState(Enum):
    NEW = "new"
    TRACKED = "tracked"
    LOST = "lost"
    REMOVED = "removed"


def _to_measurement(box: Box) -> Vector:
    """``(x, y, w, h)`` to the filter's ``(center_x, center_y, aspect, height)``."""

    x, y, width, height = box
    safe_height = max(height, 1e-6)
    return np.array([x + width / 2.0, y + height / 2.0, width / safe_height, safe_height])


def _to_box(mean: Vector) -> Box:
    height = float(mean[3])
    width = float(mean[2]) * height
    return (float(mean[0]) - width / 2.0, float(mean[1]) - height / 2.0, width, height)


@dataclass(frozen=True)
class TrackDetection:
    """One detection accepted into a track."""

    frame_index: int
    time: float
    box: Box
    score: float

    def to_dict(self) -> dict[str, object]:
        x, y, width, height = self.box
        return {
            "frame": self.frame_index,
            "time": round(self.time, 4),
            "x": round(x, 2),
            "y": round(y, 2),
            "w": round(width, 2),
            "h": round(height, 2),
            "score": round(self.score, 4),
        }


@dataclass
class Track:
    """A face followed across frames, with the detections that built it."""

    track_id: int
    mean: Vector
    covariance: Matrix
    state: TrackState = TrackState.NEW
    detections: list[TrackDetection] = field(default_factory=list)
    start_frame: int = 0
    end_frame: int = 0
    #: Calls to :meth:`update` that accepted a detection -- the track's evidence.
    hits: int = 0
    #: Update steps since the track began, including the ones it was missed.
    age: int = 0
    #: Steps since the last accepted detection: reset on every hit, incremented
    #: on every prediction.  Not ``age - hits``, which is the *cumulative* miss
    #: count -- a track that misses one frame in ten would accumulate misses
    #: until ``max_time_lost`` retired it, while it was being tracked the whole
    #: time.
    missed: int = 0
    last_score: float = 0.0
    kalman: KalmanFilter = field(default_factory=KalmanFilter, repr=False)

    @property
    def box(self) -> Box:
        """The current estimate, in ``(x, y, w, h)``."""

        return _to_box(self.mean)

    @property
    def time_since_update(self) -> int:
        return self.missed

    def predict(self) -> None:
        self.mean, self.covariance = self.kalman.predict(self.mean, self.covariance)
        self.age += 1
        self.missed += 1

    def _observe(self, detection: Detection, frame_index: int, time: float) -> None:
        self.mean, self.covariance = self.kalman.update(
            self.mean,
            self.covariance,
            _to_measurement((detection.x, detection.y, detection.width, detection.height)),
        )
        self.detections.append(
            TrackDetection(
                frame_index=frame_index,
                time=time,
                box=(detection.x, detection.y, detection.width, detection.height),
                score=detection.score,
            )
        )
        self.last_score = detection.score
        self.end_frame = frame_index
        self.hits += 1
        self.missed = 0

    def update(self, detection: Detection, frame_index: int, time: float) -> None:
        self._observe(detection, frame_index, time)
        self.state = TrackState.TRACKED

    def re_activate(self, detection: Detection, frame_index: int, time: float) -> None:
        """A lost track picked up again.

        The filter keeps its accumulated state rather than restarting from the
        new box: the whole reason the track survived is that its prediction was
        still meaningful.
        """

        self._observe(detection, frame_index, time)
        self.state = TrackState.TRACKED

    def mark_lost(self) -> None:
        self.state = TrackState.LOST

    def mark_removed(self) -> None:
        self.state = TrackState.REMOVED


@dataclass(frozen=True)
class TrackerConfig:
    """ByteTrack's knobs, with the meanings a face corpus gives them."""

    #: Splits detections into the confident set and the held-back set.  A face
    #: in profile scores lower than the same face frontally, so this is set
    #: below a frontal-face score on purpose.
    track_thresh: float = 0.5

    #: IoU distance ceiling for the first association.  0.8 allows a match at
    #: IoU 0.2, which is what a fast head turn produces between sampled frames.
    match_thresh: float = 0.8

    #: Stricter for the held-back detections: they are weaker evidence, so they
    #: may only rescue a track they overlap well.
    second_match_thresh: float = 0.5

    #: A new track needs at least this score.  Below it, detections still rescue
    #: existing tracks but do not start one -- that is what stops a wall object
    #: at 0.6 from becoming a long-lived spurious track.
    det_thresh: float = 0.6

    #: How long a lost track is kept before removal, in update steps.
    max_time_lost: int = 30

    #: Held-back detections below this are ignored entirely.
    low_score_floor: float = 0.1

    #: Reported on the track, not used to filter: a caller deciding what is a
    #: person looks at hits, motion and score together.
    min_hits: int = 1

    def with_max_time_lost_seconds(
        self, seconds: float, *, fps: float, stride: int
    ) -> TrackerConfig:
        """Translate "occluded for this many seconds" into update steps."""

        steps_per_second = fps / max(stride, 1)
        steps = max(1, int(round(seconds * steps_per_second)))
        return TrackerConfig(
            track_thresh=self.track_thresh,
            match_thresh=self.match_thresh,
            second_match_thresh=self.second_match_thresh,
            det_thresh=self.det_thresh,
            max_time_lost=steps,
            low_score_floor=self.low_score_floor,
            min_hits=self.min_hits,
        )


class ByteTracker:
    """Assigns stable ids to detections across frames."""

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self.kalman = KalmanFilter()
        self.tracked: list[Track] = []
        self.lost: list[Track] = []
        self.removed: list[Track] = []
        self._next_id = 1

    def _new_track(self, detection: Detection, frame_index: int, time: float) -> Track:
        measurement = _to_measurement(
            (detection.x, detection.y, detection.width, detection.height)
        )
        mean, covariance = self.kalman.initiate(measurement)
        track = Track(
            track_id=self._next_id,
            mean=mean,
            covariance=covariance,
            start_frame=frame_index,
            end_frame=frame_index,
            kalman=self.kalman,
        )
        self._next_id += 1
        track.update(detection, frame_index, time)
        return track

    def update(self, detections: Sequence[Detection], frame_index: int, time: float) -> None:
        config = self.config

        confident = [d for d in detections if d.score > config.track_thresh]
        held_back = [
            d for d in detections if config.low_score_floor < d.score <= config.track_thresh
        ]

        pool = self.tracked + self.lost
        for track in pool:
            track.predict()

        activated: list[Track] = []

        # First pass: confident detections against everything still alive.
        matches, unmatched_pool, unmatched_confident = linear_assignment(
            iou_distance([track.box for track in pool], _boxes(confident)),
            threshold=config.match_thresh,
            # Explicit: with an empty pool every detection is unmatched, and a
            # caller that lets the width be inferred from a rowless matrix gets
            # no detections back and silently creates no tracks at all.
            column_count=len(confident),
        )
        for track_index, detection_index in matches:
            track = pool[track_index]
            if track.state is TrackState.TRACKED:
                track.update(confident[detection_index], frame_index, time)
            else:
                track.re_activate(confident[detection_index], frame_index, time)
            activated.append(track)

        # Second pass: the held-back detections rescue tracks the first pass left
        # unmatched.  Only tracks that were *tracked* are eligible -- a lost
        # track needs confident evidence to come back, or a low-scoring wall
        # object could revive it indefinitely.
        remaining = [pool[i] for i in unmatched_pool if pool[i].state is TrackState.TRACKED]
        matches, unmatched_remaining, _ = linear_assignment(
            iou_distance([track.box for track in remaining], _boxes(held_back)),
            threshold=config.second_match_thresh,
            column_count=len(held_back),
        )
        for track_index, detection_index in matches:
            track = remaining[track_index]
            track.update(held_back[detection_index], frame_index, time)
            activated.append(track)

        for index in unmatched_remaining:
            track = remaining[index]
            track.mark_lost()
            self.lost.append(track)

        # New tracks, but only from detections confident enough to deserve one.
        for index in unmatched_confident:
            detection = confident[index]
            if detection.score < config.det_thresh:
                continue
            activated.append(self._new_track(detection, frame_index, time))

        for track in list(self.lost):
            if track.time_since_update > config.max_time_lost:
                track.mark_removed()
                self.removed.append(track)
                self.lost.remove(track)

        # A track matched in the first pass is already in `tracked` and is also
        # in `activated`, so the union needs deduplicating by identity.
        self.tracked = [t for t in _unique([*self.tracked, *activated])
                        if t.state is TrackState.TRACKED]
        self.lost = [t for t in _unique(self.lost) if t.state is TrackState.LOST]

    def finalize(self) -> None:
        """Close out the tracks still open when the video ends.

        Without this the last faces of a clip sit in ``tracked`` and never reach
        ``removed``, which is where a caller looks for tracks that have ended.
        """

        for track in self.tracked:
            if track.state is TrackState.TRACKED:
                track.mark_lost()
                self.lost.append(track)
        self.tracked = []

    @property
    def all_tracks(self) -> tuple[Track, ...]:
        """Every track that ever held a detection, oldest first."""

        everything = [*self.tracked, *self.lost, *self.removed]
        return tuple(sorted(_unique(everything), key=lambda item: item.track_id))


def _unique(tracks: Sequence[Track]) -> list[Track]:
    """Deduplicate by identity, preserving order."""

    seen: set[int] = set()
    result: list[Track] = []
    for track in tracks:
        if id(track) not in seen:
            seen.add(id(track))
            result.append(track)
    return result


def _boxes(detections: Sequence[Detection]) -> list[Box]:
    return [(d.x, d.y, d.width, d.height) for d in detections]


@dataclass(frozen=True)
class TrackQuality:
    """The evidence a later stage needs to decide whether a track is a person.

    ``motion`` is the load-bearing one.  A detector finds faces in wall art and
    framed photographs, and those form long, stable tracks that no amount of
    tracking logic rejects -- a static object tracks *better* than a person.  What
    separates them is that a person's bounding box moves: breathing and head
    turns give a real face a few percent of its own height of displacement per
    sampled frame, and a picture on a wall gives essentially none.  Normalising
    by height keeps the number comparable between a close-up and a distant face.
    """

    frames: int
    span_seconds: float
    mean_score: float
    mean_width: float
    mean_height: float
    #: Mean centre displacement per step, as a fraction of the face height.
    motion: float

    @classmethod
    def from_track(cls, track: Track, *, fps: float) -> TrackQuality:
        del fps  # times are already seconds; kept for a caller that wants frame units
        detections = track.detections
        if not detections:
            return cls(0, 0.0, 0.0, 0.0, 0.0, 0.0)

        count = len(detections)
        mean_height = sum(item.box[3] for item in detections) / count
        mean_width = sum(item.box[2] for item in detections) / count

        displacements: list[float] = []
        for previous, current in zip(detections, detections[1:], strict=False):
            dx = (current.box[0] + current.box[2] / 2) - (previous.box[0] + previous.box[2] / 2)
            dy = (current.box[1] + current.box[3] / 2) - (previous.box[1] + previous.box[3] / 2)
            displacements.append((dx * dx + dy * dy) ** 0.5)

        motion = (
            sum(displacements) / len(displacements) / mean_height
            if displacements and mean_height > 0.0
            else 0.0
        )
        return cls(
            frames=count,
            span_seconds=max(0.0, detections[-1].time - detections[0].time),
            mean_score=sum(item.score for item in detections) / count,
            mean_width=mean_width,
            mean_height=mean_height,
            motion=motion,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "frames": self.frames,
            "span_seconds": round(self.span_seconds, 4),
            "mean_score": round(self.mean_score, 4),
            "mean_width": round(self.mean_width, 2),
            "mean_height": round(self.mean_height, 2),
            "motion": round(self.motion, 5),
        }


def track_detections(
    frames: Sequence[FrameDetections], config: TrackerConfig | None = None
) -> tuple[Track, ...]:
    """Run the tracker over a whole video's sampled frames."""

    tracker = ByteTracker(config)
    for frame in frames:
        tracker.update(frame.detections, frame.frame_index, frame.time)
    tracker.finalize()
    return tracker.all_tracks
