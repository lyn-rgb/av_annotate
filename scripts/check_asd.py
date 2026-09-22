#!/usr/bin/env python3
"""What LoCoNet actually said, per video and per track.

    python scripts/check_asd.py <output-root> [--video ID] [--threshold 0.5]

S5 writes an unthresholded probability per tracklet per frame, and the number
that decides whether a run can work at all is the largest of them.  S6 assigns
a face to a speaker only above ``DEFAULT_MIN_SCORE``, so a run whose best score
is 0.011 has no assignment to make and reports ``utt=0`` however well every
other stage did.

That was the symptom for a whole round of runs, and it was produced by the
audio frontend being a generic log-mel rather than VGGish's: every batch went
into the network outside the distribution its checkpoint was trained on, and it
answered "not speaking" confidently everywhere.  Reading the maximum is what
tells that apart from a detector that is merely unimpressed by a quiet clip --
a broken frontend gives small numbers *everywhere*, a working detector on a
clip with no speech on camera gives small numbers too, and only the shape of
the distribution separates them.

It also prints each video's checkpoint load counts, which is the same symptom
arriving by a different road: if the encoder did not load, every score is noise
from untrained weights, and no amount of staring at the scores will say so.

Exits non-zero when no track in the whole root reaches the association
threshold, so it can be used as a gate rather than only read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from avannotate.associate import DEFAULT_MIN_SCORE  # noqa: E402

SPEAKING_NAME = "speaking.jsonl"
SUMMARY_NAME = "summary.json"


def read_tracks(path: Path) -> list[tuple[int, float, float, float, int]]:
    """``(track_id, max, mean, time of max, frames)`` per line of one trace."""

    tracks: list[tuple[int, float, float, float, int]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            samples = payload.get("samples") or []
            if not samples:
                # A track with no frames is written rather than omitted, and it
                # is not a zero: it is a face the window planner never scored.
                continue
            probabilities = [float(sample["probability"]) for sample in samples]
            times = [float(sample["time"]) for sample in samples]
            best = max(range(len(probabilities)), key=probabilities.__getitem__)
            tracks.append(
                (
                    int(payload["track_id"]),
                    probabilities[best],
                    sum(probabilities) / len(probabilities),
                    times[best],
                    len(probabilities),
                )
            )
    return sorted(tracks)


def read_checkpoint_load(path: Path) -> dict[str, int] | None:
    """The encoder load counts this run recorded, if it recorded any.

    Written from the beginning and reported nowhere until now, which is how a
    checkpoint that half-loaded could go unnoticed through nine smoke runs.
    """

    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    backend = payload.get("backend")
    if not isinstance(backend, dict):
        return None
    load = backend.get("checkpoint_load")
    return dict(load) if isinstance(load, dict) else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path, help="the --output directory a batch wrote to")
    parser.add_argument("--video", help="only this video id")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="what counts as a confident score (default: 0.5)",
    )
    args = parser.parse_args(argv)

    if not args.root.is_dir():
        print(f"no such directory: {args.root}", file=sys.stderr)
        return 2

    found = sorted(args.root.glob(f"work/*/s5-asd/{SPEAKING_NAME}"))
    if args.video:
        found = [path for path in found if path.parent.parent.name == args.video]
    if not found:
        print(f"no S5 output under {args.root}; has a batch run s5-asd yet?")
        return 2

    videos = 0
    tracks = 0
    confident = 0
    best_overall = 0.0
    weak_loads: list[str] = []

    for path in found:
        video = path.parent.parent.name
        videos += 1
        rows = read_tracks(path)
        print(f"\n  {video}")
        if not rows:
            print("    no tracks were scored")
            continue
        for track_id, peak, mean, at, frames in rows:
            tracks += 1
            best_overall = max(best_overall, peak)
            if peak >= args.threshold:
                confident += 1
            print(
                f"    track {track_id:<4d} max {peak:.4f} @ {at:7.2f}s"
                f"   mean {mean:.4f}   {frames} frames"
            )

        load = read_checkpoint_load(path.with_name(SUMMARY_NAME))
        if load is None:
            print("    checkpoint  (not recorded by this run)")
        else:
            print(f"    checkpoint  missing {load.get('missing', 0)}, "
                  f"unexpected {load.get('unexpected', 0)}")
            # A handful of misses is normal -- buffers, unused heads.  Hundreds
            # is a state dict that does not belong to this network.
            if load.get("missing", 0) > 20:
                weak_loads.append(video)

    print(f"\n  {videos} videos, {tracks} tracks, {confident} peak above {args.threshold}")
    print(f"  best score anywhere: {best_overall:.4f}")

    for video in weak_loads:
        print(
            f"\n  WARNING  {video}: the encoder load is missing many keys."
            "\n           Every score below it is noise from untrained weights."
        )

    if best_overall < DEFAULT_MIN_SCORE:
        print(
            f"\n  nothing reached {DEFAULT_MIN_SCORE}, the score S6 assigns at --"
            "\n  so S6 has no assignment to make and will report utt=0.  That is"
            "\n  the frontend-or-checkpoint symptom, not a quiet clip."
        )
        return 1

    if confident == 0:
        print(
            "\n  scores clear the association threshold but nothing is confident."
            "\n  S6 will assign, on thin evidence -- check the QA report's margins."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
