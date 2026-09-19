"""Face detection, tracking, and identity clustering.

The detector is pluggable (:mod:`avannotate.faces.detect`); everything after it
-- association across frames, identity clustering, quality filtering -- is plain
arithmetic over boxes and vectors, and is tested without any model at all.
"""

from avannotate.faces.detect import (
    Detector,
    DetectorError,
    InsightFaceDetector,
    YuNetDetector,
    build_detector,
    fetch_yunet,
)
from avannotate.faces.frames import FrameSampling, iter_frames, read_frame
from avannotate.faces.types import Detection, FrameDetections

__all__ = [
    "Detection",
    "Detector",
    "DetectorError",
    "FrameDetections",
    "FrameSampling",
    "InsightFaceDetector",
    "YuNetDetector",
    "build_detector",
    "fetch_yunet",
    "iter_frames",
    "read_frame",
]
