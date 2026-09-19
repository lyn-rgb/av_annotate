"""Geometry the face stages pass around.

Coordinates are pixels in the *source* frame, never in a resized copy: the
detector may run on a downscaled image for speed, and mapping back is the
detector's job.  Everything downstream -- cropping for ASD, cropping for
target-speaker extraction, the reference still -- crops the original frame, so a
bbox that silently refers to a different resolution is a bug that shows up as a
misaligned mouth three stages later.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from avannotate.coercion import coerce_number

#: An RGB frame, as :mod:`avannotate.faces.frames` decodes it and
#: :mod:`avannotate.faces.detect` consumes it.
Frame = NDArray[np.uint8]


def _points(value: object, field: str) -> tuple[tuple[float, float], ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list, got {type(value).__name__}")
    points: list[tuple[float, float]] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"{field} entries must be [x, y] pairs, got {item!r}")
        points.append((coerce_number(item[0], field), coerce_number(item[1], field)))
    return tuple(points)


@dataclass(frozen=True)
class Detection:
    """One face in one frame.

    ``landmarks`` is the five-point set (both eyes, nose, both mouth corners)
    when the detector provides it.  It is not decoration: the mouth corners give
    the lip crop target-speaker extraction wants, and the eye line gives the
    roll angle that decides whether a crop is worth extracting at all.

    ``embedding`` is the identity vector, present only when the detector
    produces one -- insightface does, YuNet does not.  It is what lets S3 rejoin
    the tracklets that a camera pan splits: measured on the sample corpus, two
    fragments of one person after a 211 px pan score 0.78 cosine against each
    other and 0.08-0.12 against everyone else.
    """

    x: float
    y: float
    width: float
    height: float
    score: float
    landmarks: tuple[tuple[float, float], ...] = ()
    embedding: tuple[float, ...] | None = None
    #: Row in the sidecar array holding this detection's vector.  Carried
    #: separately from the vector so the JSON stays small, and carried at all so
    #: S2 does not have to re-link by box when it copies detections into tracks.
    embedding_index: int | None = None

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def mouth(self) -> tuple[tuple[float, float], tuple[float, float]] | None:
        """The two mouth corners, if the detector reported landmarks."""

        if len(self.landmarks) < 5:
            return None
        return (self.landmarks[3], self.landmarks[4])

    def to_dict(self) -> dict[str, object]:
        """JSON without the embedding.

        A 512-float vector per detection would dominate the file -- hundreds of
        megabytes an hour.  S1 writes them to a sidecar array and links each row
        to its vector by position, so this stays readable.
        """

        payload: dict[str, object] = {
            "x": round(self.x, 2),
            "y": round(self.y, 2),
            "w": round(self.width, 2),
            "h": round(self.height, 2),
            "score": round(self.score, 4),
        }
        if self.landmarks:
            payload["landmarks"] = [[round(px, 2), round(py, 2)] for px, py in self.landmarks]
        if self.embedding_index is not None:
            payload["emb"] = self.embedding_index
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> Detection:
        return cls(
            x=coerce_number(payload["x"], "x"),
            y=coerce_number(payload["y"], "y"),
            width=coerce_number(payload["w"], "w"),
            height=coerce_number(payload["h"], "h"),
            score=coerce_number(payload["score"], "score"),
            landmarks=_points(payload.get("landmarks"), "landmarks"),
            embedding_index=(
                int(coerce_number(payload["emb"], "emb"))
                if payload.get("emb") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class FrameDetections:
    """Every face found in one sampled frame."""

    frame_index: int
    time: float
    detections: tuple[Detection, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "frame": self.frame_index,
            "time": round(self.time, 4),
            "faces": [detection.to_dict() for detection in self.detections],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> FrameDetections:
        raw = payload.get("faces")
        if raw is not None and not isinstance(raw, list):
            raise ValueError(f"faces must be a list, got {type(raw).__name__}")
        return cls(
            frame_index=int(coerce_number(payload["frame"], "frame")),
            time=coerce_number(payload["time"], "time"),
            detections=tuple(
                Detection.from_dict(face)
                for face in (raw or [])
                if isinstance(face, dict)
            ),
        )
