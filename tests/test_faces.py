"""Tests for face detection plumbing: types, frame decoding, detector wiring.

Detector *accuracy* is not tested here -- that needs real faces, and it lives in
the sample-corpus tests.  What is tested is everything that goes wrong silently:
a bbox in the wrong coordinate space, a frame index off by one, a detector that
returns boxes for an image it never saw.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from avannotate.faces.detect import (
    DetectorError,
    YuNetDetector,
    build_detector,
)
from avannotate.faces.frames import (
    FrameSampling,
    iter_frames,
    iter_sampled_frames,
    iter_window_frames,
    read_frame,
)
from avannotate.faces.types import Detection, FrameDetections
from avannotate.ffmpeg import FFmpegError, probe_media

# --------------------------------------------------------------------------- #
# types
# --------------------------------------------------------------------------- #


def _detection(**overrides: object) -> Detection:
    base: dict[str, object] = {
        "x": 10.0,
        "y": 20.0,
        "width": 40.0,
        "height": 50.0,
        "score": 0.9,
    }
    base.update(overrides)
    return Detection(**base)  # type: ignore[arg-type]


def test_detection_center_and_area() -> None:
    detection = _detection()
    assert detection.center == (30.0, 45.0)
    assert detection.area == pytest.approx(2000.0)


def test_mouth_requires_five_landmarks() -> None:
    assert _detection(landmarks=((0.0, 0.0),) * 4).mouth is None

    points = ((1.0, 1.0), (2.0, 1.0), (1.5, 2.0), (1.0, 3.0), (2.0, 3.0))
    mouth = _detection(landmarks=points).mouth
    assert mouth == ((1.0, 3.0), (2.0, 3.0))


def test_detection_round_trips_through_json() -> None:
    detection = _detection(landmarks=((1.0, 2.0),) * 5)
    assert Detection.from_dict(detection.to_dict()) == detection


def test_detection_without_landmarks_round_trips() -> None:
    detection = _detection()
    payload = detection.to_dict()
    assert "landmarks" not in payload
    assert Detection.from_dict(payload) == detection


def test_frame_detections_round_trip() -> None:
    frame = FrameDetections(
        frame_index=12, time=0.4, detections=(_detection(), _detection(x=99.0))
    )
    assert FrameDetections.from_dict(frame.to_dict()) == frame


# --------------------------------------------------------------------------- #
# frame sampling
# --------------------------------------------------------------------------- #


def test_sampling_indices_cover_the_video_at_the_stride() -> None:
    assert FrameSampling(stride=3).indices(10) == (0, 3, 6, 9)
    assert FrameSampling(stride=1).indices(3) == (0, 1, 2)
    assert FrameSampling(stride=10).indices(5) == (0,)


def test_sampling_rejects_a_bad_stride() -> None:
    with pytest.raises(ValueError):
        FrameSampling(stride=0)
    with pytest.raises(ValueError):
        FrameSampling(stride=-1)


def test_sampling_indices_reject_a_negative_count() -> None:
    with pytest.raises(ValueError):
        FrameSampling().indices(-1)


# --------------------------------------------------------------------------- #
# decoding
# --------------------------------------------------------------------------- #


def test_iter_frames_yields_exactly_the_sampled_indices(
    single_shot_video: Path,
) -> None:
    info = probe_media(single_shot_video)
    sampling = FrameSampling(stride=5)
    expected = sampling.indices(info.frame_count)

    seen: list[int] = []
    shapes: set[tuple[int, int, int]] = set()
    for index, frame in iter_frames(
        single_shot_video,
        width=info.width,
        height=info.height,
        sampling=sampling,
        frame_count=info.frame_count,
    ):
        seen.append(index)
        shapes.add(frame.shape)

    assert tuple(seen) == expected
    assert shapes == {(info.height, info.width, 3)}


def test_iter_frames_yields_writable_arrays(single_shot_video: Path) -> None:
    """A read-only view of the decode buffer would break every consumer."""

    info = probe_media(single_shot_video)
    for _, frame in iter_frames(
        single_shot_video,
        width=info.width,
        height=info.height,
        sampling=FrameSampling(stride=25),
        frame_count=info.frame_count,
    ):
        assert frame.flags.writeable
        frame[0, 0] = (1, 2, 3)
        break


def test_iter_frames_raises_when_the_frame_count_is_wrong(single_shot_video: Path) -> None:
    """The check that stops a short read from shifting every later index."""

    info = probe_media(single_shot_video)
    with pytest.raises(FFmpegError, match="frame indices would be wrong"):
        list(
            iter_frames(
                single_shot_video,
                width=info.width,
                height=info.height,
                sampling=FrameSampling(stride=1),
                frame_count=info.frame_count + 500,
            )
        )


def test_iter_frames_rejects_a_zero_sized_frame(single_shot_video: Path) -> None:
    with pytest.raises(ValueError, match="invalid frame size"):
        list(
            iter_frames(
                single_shot_video,
                width=0,
                height=0,
                sampling=FrameSampling(),
                frame_count=1,
            )
        )


def test_read_frame_returns_an_image_and_none_past_the_end(single_shot_video: Path) -> None:
    info = probe_media(single_shot_video)
    frame = read_frame(single_shot_video, width=info.width, height=info.height, time=1.0)
    assert frame is not None
    assert frame.shape == (info.height, info.width, 3)

    assert read_frame(single_shot_video, width=info.width, height=info.height, time=999.0) is None


@pytest.mark.parametrize("decoder", ["read", "sampled", "window", "frames"])
def test_a_scaled_decode_is_not_the_top_strip_of_the_frame(
    single_shot_video: Path, decoder: str
) -> None:
    """The regression: a smaller frame used to come back as the frame's top strip.

    None of these three ever told ffmpeg to scale, so it emitted the video at
    its native size while the caller read ``width * height * 3`` bytes per frame
    and reshaped them.  For a smaller request that is the top strip of the
    picture wrapped into the requested width -- and it raised nothing.

    It survived every test in this file because they all pass the native
    dimensions, where the strip *is* the frame.  Captioning is the one caller
    that asks for less, and a run of it described a wooden wall as "a close-up
    of a woven basket, no people visible" with two faces tracked in the very
    same frames.

    The assertion is exact rather than a tolerance: with the scale missing, the
    output is those bytes verbatim, so comparing against them catches the bug
    and needs no threshold to tune.
    """

    def decode(width: int, height: int) -> np.ndarray:
        if decoder == "frames":
            # The decoder S1 uses, and the one this test did not cover -- which
            # is the whole reason its missing `scale` went unnoticed.  One frame
            # is enough: the strip is visible in the first one.
            frames = list(
                iter_frames(
                    single_shot_video,
                    width=width,
                    height=height,
                    sampling=FrameSampling(stride=1),
                    frame_count=1,
                )
            )
            return np.asarray(frames[0][1])
        if decoder == "read":
            frame = read_frame(single_shot_video, width=width, height=height, time=1.0)
            assert frame is not None
            return np.asarray(frame)
        if decoder == "sampled":
            frames = list(
                iter_sampled_frames(
                    single_shot_video,
                    width=width,
                    height=height,
                    start_time=1.0,
                    duration=1.0,
                    count=1,
                )
            )
            return np.asarray(frames[0])
        frames = list(
            iter_window_frames(
                single_shot_video, width=width, height=height, start_time=1.0, count=1
            )
        )
        return np.asarray(frames[0])

    full = decode(320, 240)
    small = decode(160, 120)

    assert small.shape == (120, 160, 3)
    strip = full.reshape(-1)[: 160 * 120 * 3].reshape(120, 160, 3)
    assert not np.array_equal(small, strip)


# --------------------------------------------------------------------------- #
# detector construction
# --------------------------------------------------------------------------- #


def test_missing_model_file_is_named_in_the_error(tmp_path: Path) -> None:
    with pytest.raises(DetectorError, match="YuNet model not found"):
        YuNetDetector(tmp_path / "absent.onnx")


def test_unknown_backend_is_rejected() -> None:
    with pytest.raises(DetectorError, match="unknown detector backend"):
        build_detector({"backend": "magic"})


def test_yunet_detector_returns_boxes_in_frame_coordinates(sample_videos: tuple[Path, ...]) -> None:
    """Detection quality is not the point; coordinate space is.

    A detector that downscales internally and forgets to map back produces boxes
    that look plausible and crop the wrong region three stages later.
    """

    model = Path(__file__).resolve().parents[1] / "models" / "yunet.onnx"
    if not model.is_file():
        pytest.skip(f"YuNet weights not present at {model}")

    detector = YuNetDetector(model, score_threshold=0.6)
    source = sample_videos[0]
    info = probe_media(source)
    frame = read_frame(source, width=info.width, height=info.height, time=1.0)
    assert frame is not None

    detections = detector.detect(frame)
    assert detections, "expected at least one face in a two-person clip"
    for detection in detections:
        assert 0.0 <= detection.score <= 1.0
        assert detection.x >= -5.0
        assert detection.y >= -5.0
        assert detection.x + detection.width <= info.width + 5.0
        assert detection.y + detection.height <= info.height + 5.0
        assert detection.width > 0.0 and detection.height > 0.0


def test_yunet_detector_handles_a_face_having_image(
    single_shot_video: Path,
) -> None:
    """A test pattern has no faces; the detector must return empty, not crash."""

    model = Path(__file__).resolve().parents[1] / "models" / "yunet.onnx"
    if not model.is_file():
        pytest.skip(f"YuNet weights not present at {model}")

    info = probe_media(single_shot_video)
    frame = read_frame(single_shot_video, width=info.width, height=info.height, time=1.0)
    assert frame is not None
    assert YuNetDetector(model).detect(frame) == ()


def test_yunet_detector_handles_a_changed_frame_size(
    single_shot_video: Path,
) -> None:
    """The input size is renegotiated, not assumed to stay constant."""

    model = Path(__file__).resolve().parents[1] / "models" / "yunet.onnx"
    if not model.is_file():
        pytest.skip(f"YuNet weights not present at {model}")

    detector = YuNetDetector(model)
    detector.detect(np.zeros((240, 320, 3), dtype=np.uint8))
    detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))


def test_a_window_running_past_the_end_repeats_the_last_frame(
    single_shot_video: Path,
) -> None:
    """Running off the end of the file is ordinary, and used to raise.

    S7 asked for one frame more than a video had and lost the whole video to it.
    The frames that do arrive are the right ones in the right order, and the
    audio they pair with was computed for exactly this many frames -- so the
    tail is padded with the last real frame and both of those stay true.
    """

    info = probe_media(single_shot_video)
    frames = list(
        iter_window_frames(
            single_shot_video,
            width=64,
            height=48,
            start_time=info.duration - 0.2,  # about five frames from the end
            count=100,                       # and a hundred asked for
        )
    )

    assert len(frames) == 100
    # The tail is constant, because it is one frame repeated...
    assert np.array_equal(frames[-1], frames[-2])
    # ...and it is a tail: the first frames are real ones, and they move.
    assert not np.array_equal(frames[0], frames[-1])


def test_a_window_inside_the_file_is_not_padded(single_shot_video: Path) -> None:
    """Guards the guard: the padding must not creep into ordinary windows."""

    frames = list(
        iter_window_frames(single_shot_video, width=64, height=48, start_time=0.0, count=10)
    )

    assert len(frames) == 10
    # Ten frames off an animated pattern, not one frame repeated ten times.
    assert len({frame.tobytes() for frame in frames}) == 10
