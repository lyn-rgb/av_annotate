#!/usr/bin/env python
"""One crop video per person, from what the corpus already recorded.

    scripts/export_faces.py --output /results
    scripts/export_faces.py --output /results --videos some.txt --limit 20
    scripts/export_faces.py --output /results --workers 16

**Why this is not a stage.** The plan drew ``faces/F001/crops/`` as an S2
artifact, and nothing ever wrote it: S2 records boxes and this pipeline crops
from the source on demand -- S5 for its windows, S7 for its extractor, each into
scratch and gone. So the crops were never missing from any *run*; they were
missing from the *result*, and nothing compared the plan to the code.

Putting it back costs no model. Everything needed is already on disk: the boxes
in ``s2-tracks/tracks.jsonl``, who they belong to in
``s3-cluster/identities.json``, and the source's path, size and frame rate in the
deliverable's own ``annotation.json``. So this reads those and re-cuts with
ffmpeg, using the same two helpers S7 uses
(:func:`avannotate.tse.plan.identity_boxes` and
:func:`avannotate.tse.crop_video.write_crop_video`) -- so the face is followed
across a camera pan the same way, and the crop is built the same way.

**One video per identity, over the whole span it appears in.** Gaps keep their
timing rather than being cut out, so a seek in the crop means the same instant as
a seek in the source. What a gap *looks* like is the one choice here, and the
default is black: a frame the tracker had no sighting in is a frame with no
face in it, and thirty seconds of the last known position reads as content when
it is nothing of the sort. ``--fill-gaps`` holds the last known box instead,
which is what S7 hands the extractor -- a fixed-length sequence it must not have
holes in -- and is the right choice when the crop is going into another model
rather than being looked at.

**Re-running is cheap and safe.** A crop that is already there is left alone
unless ``--force``. Nothing here reads or writes a stage record, so ``--force``
is not needed to keep a stage honest and dropping it costs only the files it
rewrites.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from multiprocessing import get_context
from pathlib import Path

from avannotate import threads
from avannotate.faces.track import Tracklet
from avannotate.progress import Progress
from avannotate.stages import s2_tracks, s3_cluster, s11_compose
from avannotate.stages.base import StageContext
from avannotate.tse.crop_video import DEFAULT_CROP_SIZE, DEFAULT_MARGIN, write_crop_video
from avannotate.tse.plan import identity_boxes

#: Where the exports go, under each video's work directory.  Not a stage
#: directory: nothing here has a version, a resume record, or a say in whether
#: any stage should re-run, and putting them beside S3's reference stills would
#: invite somebody to think otherwise.
FACES_DIR = "faces"


def _span(tracklets: Sequence[Tracklet]) -> tuple[int, int]:
    """The first and last frame this identity is seen in, end-exclusive."""

    frames = [item.frame_index for tracklet in tracklets for item in tracklet.detections]
    return min(frames), max(frames) + 1


@dataclass(frozen=True)
class _Export:
    """What one video produced, or why it produced nothing."""

    video_id: str
    written: int = 0
    skipped: int = 0
    error: str = ""


def export_video(
    work_dir: Path, *, size: int, margin: float, force: bool, fill_gaps: bool
) -> _Export:
    """Every identity of one video, as a crop video each."""

    try:
        context = StageContext(
            video_id=work_dir.name,
            # Filled in below.  The deliverable is the only place a video's
            # source path is written down, and reading it needs a context --
            # so this reads it with the one field it does not need, and then
            # corrects that field.
            source=Path(),
            work_dir=work_dir,
        )
        annotation = s11_compose.load_annotation(context)
        context = replace(context, source=Path(annotation.video.path))
        tracklets = s2_tracks.load_tracklets(context)
        members = s3_cluster.load_identity_tracks(context)
    except Exception as error:  # noqa: BLE001 - one video must not stop the rest
        return _Export(work_dir.name, 0, 0, f"{type(error).__name__}: {error}")

    if not tracklets or not members:
        return _Export(work_dir.name, 0, 0)

    by_id = {tracklet.track_id: tracklet for tracklet in tracklets}
    fps = annotation.video.fps
    if fps <= 0:
        return _Export(work_dir.name, 0, 0, f"the annotation says fps={fps}")

    written = skipped = 0
    for face_id, track_ids in sorted(members.items()):
        mine = [by_id[item] for item in track_ids if item in by_id]
        if not mine:
            continue

        target = work_dir / FACES_DIR / f"{face_id}.mp4"
        if target.is_file() and target.stat().st_size > 0 and not force:
            skipped += 1
            continue

        start, end = _span(mine)
        try:
            write_crop_video(
                context.source,
                target,
                width=annotation.video.width,
                height=annotation.video.height,
                start_time=start / fps,
                frame_count=end - start,
                boxes=identity_boxes(
                    mine, start_frame=start, end_frame=end, fill=fill_gaps
                ),
                fps=fps,
                size=size,
                margin=margin,
            )
        except Exception as error:  # noqa: BLE001 - one identity must not stop the rest
            return _Export(
                work_dir.name, written, skipped, f"{face_id}: {type(error).__name__}: {error}"
            )
        written += 1

    return _Export(work_dir.name, written, skipped)


def _work_dirs(output: Path, only: Sequence[str] | None) -> list[Path]:
    work = output / "work"
    if not work.is_dir():
        raise SystemExit(f"no work directory under {output}")
    wanted = set(only) if only else None
    found = [
        item
        for item in sorted(work.iterdir())
        if item.is_dir() and (wanted is None or item.name in wanted)
    ]
    if wanted:
        missing = wanted - {item.name for item in found}
        if missing:
            raise SystemExit(f"{len(missing)} named videos have no work directory: "
                             + ", ".join(sorted(missing)[:5]))
    return found


def _read_names(path: Path) -> list[str]:
    """Video ids from a list file -- the same rule the batch uses, its stem."""

    return [
        Path(line.strip()).stem
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


def _worker(job: tuple[str, int, float, bool, int, bool]) -> tuple[str, int, int, str]:
    directory, size, margin, force, workers, fill_gaps = job
    # Before anything imports cv2: the numeric libraries read these once, and a
    # pool that already exists ignores the environment.  Same rule and the same
    # reason as the batch's workers -- this is ffmpeg plus one resize per frame,
    # and giving each worker its share is what makes N of them N times faster.
    #
    # `workers` travels in the job rather than being read from a module global,
    # because under spawn a child re-imports this file and would find the
    # default there instead of the operator's -- one thread per core, times N.
    threads.cap(workers)
    result = export_video(
        Path(directory), size=size, margin=margin, force=force, fill_gaps=fill_gaps
    )
    return result.video_id, result.written, result.skipped, result.error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="One crop video per person, from what the corpus recorded.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="One crop video per identity, over the whole span it appears in, "
        "with gaps kept at their real timing.",
    )
    parser.add_argument("--output", required=True, type=Path, help="the corpus output")
    parser.add_argument(
        "--videos", type=Path, help="a list file; only these videos are exported"
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="videos at once (default: 8)"
    )
    parser.add_argument("--limit", type=int, help="stop after this many videos")
    parser.add_argument("--force", action="store_true", help="rewrite crops that exist")
    parser.add_argument(
        "--fill-gaps",
        action="store_true",
        help="hold the last known box through a gap, as S7 does, instead of black",
    )
    parser.add_argument("--size", type=int, default=DEFAULT_CROP_SIZE, help="crop edge in pixels")
    parser.add_argument(
        "--margin", type=float, default=DEFAULT_MARGIN, help="padding around the face"
    )
    args = parser.parse_args(argv)

    workers = max(1, args.workers)
    output = args.output.expanduser().resolve()
    only = _read_names(args.videos.expanduser()) if args.videos else None
    directories = _work_dirs(output, only)
    if args.limit:
        directories = directories[: args.limit]
    if not directories:
        print("nothing to export")
        return 0

    jobs = [
        (str(item), args.size, args.margin, args.force, workers, args.fill_gaps)
        for item in directories
    ]
    print(f"videos    {len(jobs)}")
    print(f"workers   {workers}")
    print(f"crop      {args.size}px, margin {args.margin}")
    print(f"gaps      {'held at the last known box' if args.fill_gaps else 'black'}")
    print(f"output    {output}/work/<video>/{FACES_DIR}/<F00x>.mp4")
    print(flush=True)

    written = skipped = failed = 0
    failures: list[tuple[str, str]] = []

    progress = Progress(title="faces", total=len(jobs), lines=sys.stdout)
    context = get_context("spawn")
    with context.Pool(processes=workers) as pool:
        for video_id, made, kept, error in pool.imap_unordered(_worker, jobs):
            written += made
            skipped += kept
            if error:
                failed += 1
                failures.append((video_id, error))
            progress.advance(ok=not error)
            if error:
                # The bar says how many; only a line says which and why.
                progress.clear()
                print(f"  FAIL {video_id}: {error}", flush=True)
    progress.close()

    print()
    print(f"crops     {written} written, {skipped} already there")
    print(f"videos    {len(jobs) - failed} ok, {failed} failed")
    if failures:
        print("failures:")
        for video_id, error in failures[:10]:
            print(f"  {video_id}: {error}")
        if len(failures) > 10:
            print(f"  ... and {len(failures) - 10} more")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
