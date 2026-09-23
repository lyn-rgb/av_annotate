"""Face detectors behind one interface.

Two implementations, because they answer different questions.  InsightFace's
``buffalo_l`` (SCRFD for boxes, ArcFace for embeddings) is what the pipeline is
built around: it is the accuracy choice, and it hands back the 512-d embedding
that identity clustering needs.  OpenCV's YuNet is the one that runs anywhere
with a single ``pip install``, needs no GPU, and is enough to exercise every
stage after this one.

The detector returns boxes in **source-frame pixels**.  A detector that
downscales internally must map back before returning; nothing downstream knows
the difference.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from avannotate import threads
from avannotate.faces.types import Detection, Frame

#: Where OpenCV publishes the YuNet ONNX model.  Pinned to a dated release
#: rather than ``main`` so a rerun next month sees the same weights.
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)

DEFAULT_YUNET_SCORE = 0.6


class DetectorError(RuntimeError):
    """The detector could not be built or run."""


class Detector(Protocol):
    """What the stage needs from a detector, and nothing more."""

    name: str

    #: Whether ``Detection.embedding`` will be populated.  Declared rather than
    #: discovered: S3's identity clustering cannot run at all without vectors,
    #: and a corpus detected with YuNet should say so before an hour of
    #: clustering produces nothing.
    provides_embeddings: bool

    def detect(self, frame: Frame) -> tuple[Detection, ...]: ...


class YuNetDetector:
    """OpenCV's YuNet through ``cv2.FaceDetectorYN``.

    Takes RGB frames, because that is what :mod:`avannotate.faces.frames`
    decodes, and converts to BGR internally: YuNet was trained on OpenCV's
    channel order, and feeding it RGB costs accuracy for no reason.
    """

    name = "yunet"
    provides_embeddings = False

    def __init__(
        self,
        model_path: str | Path,
        *,
        score_threshold: float = DEFAULT_YUNET_SCORE,
        nms_threshold: float = 0.3,
        top_k: int = 50,
        input_size: tuple[int, int] = (320, 320),
    ) -> None:
        path = Path(model_path).expanduser()
        if not path.is_file():
            raise DetectorError(
                f"YuNet model not found at {path}. Fetch it from {YUNET_URL} "
                f"(see avannotate.faces.detect.fetch_yunet)."
            )
        try:
            import cv2
            threads.cap_opencv()
        except ModuleNotFoundError as error:
            raise DetectorError(
                "OpenCV is required for the YuNet detector: pip install opencv-python-headless"
            ) from error

        self._cv2 = cv2
        self._detector = cv2.FaceDetectorYN.create(
            str(path), "", input_size, score_threshold, nms_threshold, top_k
        )
        self._size: tuple[int, int] | None = None

    def detect(self, frame: Frame) -> tuple[Detection, ...]:
        height, width = frame.shape[:2]
        if self._size != (width, height):
            self._detector.setInputSize((width, height))
            self._size = (width, height)

        bgr = self._cv2.cvtColor(frame, self._cv2.COLOR_RGB2BGR)
        _, raw = self._detector.detect(bgr)
        if raw is None:
            return ()

        # cv2 returns (N, 15): x, y, w, h, then five (x, y) landmark pairs in
        # ArcFace's order -- right eye, left eye, nose, right then left mouth
        # corner -- then the score.  Through asarray because the binding types
        # the result as a scalar union, which is not indexable to a checker.
        faces = np.asarray(raw, dtype=np.float64).reshape(-1, 15)

        results: list[Detection] = []
        for face in faces:
            landmarks = tuple(
                (float(face[4 + 2 * index]), float(face[5 + 2 * index])) for index in range(5)
            )
            results.append(
                Detection(
                    x=float(face[0]),
                    y=float(face[1]),
                    width=float(face[2]),
                    height=float(face[3]),
                    score=float(face[14]),
                    landmarks=landmarks,
                )
            )
        return tuple(results)


class InsightFaceDetector:
    """SCRFD boxes plus ArcFace embeddings, through the ``insightface`` package.

    Kept behind a lazy import: the package pulls in onnxruntime, which is a
    heavyweight dependency for a machine that only ever runs the stages before
    clustering.
    """

    name = "insightface"
    provides_embeddings = True

    def __init__(
        self,
        *,
        model_name: str = "buffalo_l",
        root: str | Path | None = None,
        det_size: tuple[int, int] = (640, 640),
        providers: tuple[str, ...] = ("CUDAExecutionProvider", "CPUExecutionProvider"),
    ) -> None:
        try:
            from insightface.app import FaceAnalysis
        except ModuleNotFoundError as error:
            raise DetectorError(
                "insightface is required for this detector: pip install insightface onnxruntime"
            ) from error

        # ``buffalo_l`` bundles five models; two of them are unused here.
        # Landmarks come from the detector itself, and nothing downstream wants
        # gender or age, so loading them is pure cost -- and on a corpus this
        # size the cost is the stage's runtime.
        self._app = FaceAnalysis(
            name=model_name,
            root=str(root) if root is not None else None,
            providers=list(providers),
            allowed_modules=["detection", "recognition"],
        )
        self._app.prepare(ctx_id=0, det_size=det_size)

    def detect(self, frame: Frame) -> tuple[Detection, ...]:
        # insightface expects BGR, like OpenCV.
        bgr = frame[:, :, ::-1]
        results: list[Detection] = []
        for face in self._app.get(bgr):
            keypoints = getattr(face, "kps", None)
            landmarks: tuple[tuple[float, float], ...] = ()
            if keypoints is not None:
                landmarks = tuple((float(x), float(y)) for x, y in np.asarray(keypoints))

            # ``bbox`` is [x1, y1, x2, y2] -- corners, not a corner and a size.
            # Reading it as (x, y, w, h) yields a box the size of the frame and
            # an IoU near zero, which looks like a matching failure rather than
            # a convention mistake.
            left, top, right, bottom = (float(value) for value in face.bbox)

            raw_embedding = getattr(face, "normed_embedding", None)
            embedding = (
                tuple(float(value) for value in np.asarray(raw_embedding))
                if raw_embedding is not None
                else None
            )
            results.append(
                Detection(
                    x=left,
                    y=top,
                    width=right - left,
                    height=bottom - top,
                    score=float(face.det_score),
                    landmarks=landmarks,
                    embedding=embedding,
                )
            )
        return tuple(results)


def fetch_yunet(destination: str | Path, *, timeout: float = 120.0) -> Path:
    """Download the YuNet weights, explicitly.

    Not called from the stage: a batch job that silently reaches the network
    mid-run is one that fails on an air-gapped server at hour nine.  The server
    fetches once, and the config points at the file.
    """

    target = Path(destination).expanduser()
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(YUNET_URL, timeout=timeout) as response:
            payload = response.read()
    except (urllib.error.URLError, TimeoutError) as error:
        raise DetectorError(
            f"could not download the YuNet model from {YUNET_URL}: {error}"
        ) from error
    if len(payload) < 100_000:
        raise DetectorError(
            f"the downloaded YuNet model is only {len(payload)} bytes; "
            "the URL probably returned an error page"
        )
    target.write_bytes(payload)
    return target


def build_detector(config: Mapping[str, Any]) -> Detector:
    """Construct the detector a stage's config asks for.

    Defaults to insightface when the package is present, because that is the
    accuracy choice, and falls back to YuNet only when a model path is given --
    a silent downgrade in detection quality would be invisible in the output.
    """

    backend = str(config.get("backend", "insightface"))
    if backend == "yunet":
        model_path = config.get("model_path")
        if model_path is None:
            raise DetectorError("the yunet backend needs a model_path")
        return YuNetDetector(
            model_path,
            score_threshold=float(config.get("score_threshold", DEFAULT_YUNET_SCORE)),
        )
    if backend == "insightface":
        model_root = config.get("model_root")
        return InsightFaceDetector(
            model_name=str(config.get("model_name", "buffalo_l")),
            root=str(model_root) if model_root is not None else None,
        )
    raise DetectorError(f"unknown detector backend {backend!r}; expected yunet or insightface")
