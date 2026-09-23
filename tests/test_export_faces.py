"""Tests for the crop export.

An export that is subtly wrong is worse than one that fails: it is 8,716 videos
of somebody's face cropped to the wrong place, discovered later by whoever reads
them.  So what is pinned here is the geometry and the timing -- how long the crop
is, which frames have a face in them, and what a gap looks like -- rather than
that a file appeared.

The fixture is a work directory built the way the stages build one, through the
same dataclasses they write with, because the thing most likely to break this is
a format drifting rather than the arithmetic.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from avannotate.annotation import annotation_to_dict
from avannotate.faces.track import TrackDetection, Tracklet, TrackQuality
from avannotate.schema import Annotation, VideoMeta
from scripts.export_faces import FACES_DIR, export_video

FPS = 25.0
WIDTH, HEIGHT = 320, 240


def _tracklet(track_id: int, frames: list[int], *, box_x: float = 40.0) -> Tracklet:
    detections = [
        TrackDetection(
            frame_index=index, time=index / FPS, box=(box_x + index, 60.0, 60.0, 60.0), score=0.9
        )
        for index in frames
    ]
    return Tracklet(
        track_id=track_id,
        start_frame=frames[0],
        end_frame=frames[-1] + 1,
        hits=len(detections),
        quality=TrackQuality(
            frames=len(detections),
            span_seconds=len(detections) / FPS,
            mean_score=0.9,
            mean_width=60.0,
            mean_height=60.0,
            motion=0.01,
        ),
        detections=tuple(detections),
    )


def _work_dir(tmp_path: Path, video: Path, *, tracks, identities: dict[str, list[int]]) -> Path:
    work = tmp_path / "work" / "clip"
    for stage in ("s2-tracks", "s3-cluster", "s11-compose"):
        (work / stage).mkdir(parents=True, exist_ok=True)

    with (work / "s2-tracks" / "tracks.jsonl").open("w", encoding="utf-8") as handle:
        for track in tracks:
            handle.write(
                json.dumps(
                    {
                        "track_id": track.track_id,
                        "start_frame": track.start_frame,
                        "end_frame": track.end_frame,
                        "hits": track.hits,
                        "quality": track.quality.to_dict(),
                        "detections": [item.to_dict() for item in track.detections],
                    }
                )
                + "\n"
            )

    (work / "s3-cluster" / "identities.json").write_text(
        json.dumps(
            {
                "schema_version": "avannotate-identities-v1",
                "identities": [
                    {"face_id": face_id, "track_ids": ids} for face_id, ids in identities.items()
                ],
                "dropped": [],
            }
        ),
        encoding="utf-8",
    )

    annotation = Annotation(
        video=VideoMeta(
            video_id="clip",
            path=str(video),
            duration=3.0,
            fps=FPS,
            width=WIDTH,
            height=HEIGHT,
        )
    )
    (work / "s11-compose" / "annotation.json").write_text(
        json.dumps(annotation_to_dict(annotation)), encoding="utf-8"
    )
    return work


def _probe(path: Path) -> tuple[int, float, int, int]:
    """(frames, duration, width, height) of a written crop."""

    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries",
            "stream=nb_frames,width,height:format=duration",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True, check=True,
    ).stdout
    payload = json.loads(out)
    stream = payload["streams"][0]
    return (
        int(stream["nb_frames"]),
        float(payload["format"]["duration"]),
        int(stream["width"]),
        int(stream["height"]),
    )


def _frame(path: Path, index: int) -> np.ndarray:
    """One frame of the crop, as greyscale pixels."""

    raw = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(path),
            "-vf", f"select=eq(n\\,{index})", "-frames:v", "1",
            "-f", "rawvideo", "-pix_fmt", "gray", "-",
        ],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.uint8)


def test_one_crop_per_identity(tmp_path: Path, single_shot_video: Path) -> None:
    work = _work_dir(
        tmp_path,
        single_shot_video,
        tracks=[_tracklet(0, list(range(0, 40))), _tracklet(1, list(range(0, 40)))],
        identities={"F001": [0], "F002": [1]},
    )

    export_video(work, size=224, margin=0.4, force=False, fill_gaps=False)

    assert sorted(item.name for item in (work / FACES_DIR).iterdir()) == [
        "F001.mp4",
        "F002.mp4",
    ]


def test_the_crop_is_as_long_as_the_identity_is_on_screen(
    tmp_path: Path, single_shot_video: Path
) -> None:
    """Frames 10 to 50 of a 25 fps video is 40 frames and 1.6 seconds."""

    work = _work_dir(
        tmp_path,
        single_shot_video,
        tracks=[_tracklet(0, list(range(10, 50)))],
        identities={"F001": [0]},
    )

    export_video(work, size=224, margin=0.4, force=False, fill_gaps=False)

    frames, duration, width, height = _probe(work / FACES_DIR / "F001.mp4")

    assert (frames, width, height) == (40, 224, 224)
    assert duration == pytest.approx(40 / FPS, abs=0.05)


def test_a_gap_is_black_by_default(tmp_path: Path, single_shot_video: Path) -> None:
    """A frame the tracker had no sighting in is a frame with no face in it.

    Holding the last known box instead fills the gap with whatever happens to be
    where the person used to be, which reads as content and is not.
    """

    work = _work_dir(
        tmp_path,
        single_shot_video,
        tracks=[_tracklet(0, list(range(0, 10))), _tracklet(1, list(range(60, 70)))],
        identities={"F001": [0, 1]},
    )

    export_video(work, size=224, margin=0.4, force=False, fill_gaps=False)

    crop = work / FACES_DIR / "F001.mp4"
    frames, duration, _, _ = _probe(crop)

    assert frames == 70, "the gap keeps its timing rather than being cut out"
    assert duration == pytest.approx(70 / FPS, abs=0.05)
    assert _frame(crop, 5).max() > 0, "the first sighting is a real picture"
    assert _frame(crop, 35).max() == 0, "the gap is black"
    assert _frame(crop, 65).max() > 0, "and the second sighting is too"


def test_filling_gaps_holds_the_last_box(tmp_path: Path, single_shot_video: Path) -> None:
    """What S7 hands the extractor -- a sequence it must not have holes in."""

    work = _work_dir(
        tmp_path,
        single_shot_video,
        tracks=[_tracklet(0, list(range(0, 10))), _tracklet(1, list(range(60, 70)))],
        identities={"F001": [0, 1]},
    )

    export_video(work, size=224, margin=0.4, force=False, fill_gaps=True)

    assert _frame(work / FACES_DIR / "F001.mp4", 35).max() > 0


def test_a_crop_that_is_already_there_is_left_alone(
    tmp_path: Path, single_shot_video: Path
) -> None:
    work = _work_dir(
        tmp_path,
        single_shot_video,
        tracks=[_tracklet(0, list(range(0, 40)))],
        identities={"F001": [0]},
    )
    export_video(work, size=224, margin=0.4, force=False, fill_gaps=False)
    crop = work / FACES_DIR / "F001.mp4"
    before = crop.stat().st_mtime_ns

    result = export_video(work, size=224, margin=0.4, force=False, fill_gaps=False)

    assert (result.written, result.skipped) == (0, 1)
    assert crop.stat().st_mtime_ns == before


def test_a_video_with_no_deliverable_is_reported_not_raised(
    tmp_path: Path, single_shot_video: Path
) -> None:
    """One video missing its annotation must not stop an export over thousands.

    Which is also the only videos this can run on: the deliverable is the one
    place a video's source path is written down.
    """

    work = tmp_path / "work" / "empty"
    work.mkdir(parents=True)

    result = export_video(work, size=224, margin=0.4, force=False, fill_gaps=False)

    assert result.written == 0
    assert result.error, "it has to say why, not just produce nothing"
