"""Command line entry point.

Deliberately small: ``run`` executes one stage over one video or a list of them,
and that is enough to drive a batch from a shell loop or a job scheduler.  The
per-stage resume record, not the CLI, is what makes a rerun cheap.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from avannotate import requirements
from avannotate.stages import STAGE_ORDER, available_stages, get_stage
from avannotate.stages.base import StageContext, StageError, load_config_file

VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"})


def _read_inputs(value: str, *, base: Path) -> list[Path]:
    """Accept a video, a directory of them, or a text file listing them.

    A list's entries are resolved against the working directory first and the
    list's own directory second, because both conventions are in use and a
    wrong guess would otherwise be an unhelpful "file not found".
    """

    candidate = Path(value).expanduser()
    if candidate.is_dir():
        found = sorted(
            path for path in candidate.iterdir() if path.suffix.lower() in VIDEO_SUFFIXES
        )
        if not found:
            raise SystemExit(f"no video files in {candidate}")
        return found

    if candidate.suffix.lower() in VIDEO_SUFFIXES:
        return [candidate.resolve()]

    if not candidate.is_file():
        raise SystemExit(f"input is not a video, directory, or list: {candidate}")

    resolved: list[Path] = []
    missing: list[str] = []
    for line in candidate.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        for root in (base, candidate.parent):
            attempt = (root / entry).expanduser()
            if attempt.is_file():
                resolved.append(attempt.resolve())
                break
        else:
            missing.append(entry)
    if missing:
        raise SystemExit(
            "these entries were not found relative to "
            f"{base} or {candidate.parent}: {', '.join(missing[:5])}"
        )
    if not resolved:
        raise SystemExit(f"no entries in {candidate}")
    return resolved


def _load_config(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    return load_config_file(path)


def _cmd_stages(_: argparse.Namespace) -> int:
    implementable = set(available_stages())
    for name in STAGE_ORDER:
        print(f"  {'implemented' if name in implementable else 'not yet    '}  {name}")
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    """Report what each stage needs that this machine does not have."""

    stages = tuple(args.stage) if args.stage else available_stages()
    unknown = [name for name in stages if name not in available_stages()]
    if unknown:
        raise SystemExit(f"unknown stage(s): {', '.join(unknown)}")

    statuses = requirements.check(stages, configs_dir=args.configs_dir)

    for status in statuses:
        mark = "ready  " if status.ready else "MISSING"
        print(f"{mark}  {status.stage:22} {status.detail}")
        for item in status.todo:
            print(f"         {item}")

    missing = [status.stage for status in statuses if not status.ready]
    if missing:
        print()
        print(f"{len(missing)} of {len(statuses)} stages cannot run here: {', '.join(missing)}")
        return 1
    print()
    print(f"all {len(statuses)} stages can run here")
    return 0


def _cmd_batch(args: argparse.Namespace) -> int:
    """The whole pipeline over a corpus, several videos at a time."""

    from avannotate import batch as batch_module

    sources = batch_module.read_video_list(args.input, base=Path.cwd())
    output = args.output.expanduser().resolve()
    config_root = (
        args.configs.expanduser().resolve()
        if args.configs
        else (Path(__file__).resolve().parents[1] / "configs")
    )

    stages = tuple(args.stage) if args.stage else STAGE_ORDER
    unknown = [name for name in stages if name not in STAGE_ORDER]
    if unknown:
        raise SystemExit(f"unknown stage(s): {', '.join(unknown)}")

    jobs = batch_module.plan_jobs(sources, output=output)
    gpus = batch_module.parse_gpu_list(args.gpus)
    if gpus is None:
        gpus = batch_module.detect_gpus()
    devices, workers = batch_module.plan_workers(gpus=gpus, requested=args.workers)

    if args.only_missing:
        jobs = tuple(job for job in jobs if not batch_module.is_complete(job))
        if not jobs:
            print("every video in the list already has a deliverable")
            return 0

    print(f"videos    {len(jobs)}")
    print(f"stages    {', '.join(stages)}")
    print(f"configs   {config_root}")
    print(f"output    {output}")
    print(
        f"workers   {workers}"
        + (f"  on GPUs {', '.join(str(item) for item in devices)}" if devices else "  on CPU")
    )
    print(flush=True)

    started = time.monotonic()
    stage_events: list[str] = []

    def on_stage(video_id: str, stage: str, skipped: bool) -> None:
        # One line per stage, so a batch that is stuck inside a long stage is
        # visibly inside a long stage rather than apparently dead.
        mark = "skip" if skipped else "run "
        elapsed = batch_module.format_duration(time.monotonic() - started)
        line = f"        {elapsed:>9}  {mark}  {video_id}  {stage}"
        stage_events.append(line)
        if args.verbose:
            print(line, flush=True)

    results = batch_module.run_corpus(
        jobs,
        stages=stages,
        config_root=config_root,
        workers=workers,
        gpus=tuple(devices),
        force=args.force,
        on_stage=on_stage,
    )

    index_path = output / "index.jsonl"
    failures_path = output / "failures.jsonl"
    failures = [item for item in results if not item.ok]
    batch_module.write_jsonl(index_path, (item.to_dict() for item in results))
    batch_module.write_jsonl(failures_path, (item.to_dict() for item in failures))

    summary = batch_module.summarise(results)
    print()
    print(f"videos   {summary['succeeded']} ok, {summary['failed']} failed")
    print(f"time     {batch_module.format_duration(time.monotonic() - started)}")
    slowest = str(summary["slowest_stage"])
    if slowest:
        per_stage = summary["seconds_per_stage"]
        total = per_stage[slowest] if isinstance(per_stage, dict) else "?"
        print(f"slowest  {slowest} ({total}s total)")
    print(f"index    {index_path}")
    if failures:
        print(f"failures {failures_path}")
        for item in failures[:5]:
            print(f"         {item.video_id}: {item.failed_stage}: {item.error}")
    return 1 if failures else 0


def _cmd_run(args: argparse.Namespace) -> int:
    module = get_stage(args.stage)
    inputs = _read_inputs(args.input, base=Path.cwd())
    root = args.output.expanduser().resolve()
    config = _load_config(args.config)

    failures = 0
    for source in inputs:
        context = StageContext(
            video_id=source.stem,
            source=source,
            work_dir=root / "work" / source.stem,
            config=config,
        )
        try:
            result = module.run(context, force=args.force)
        except (StageError, ValueError, FileNotFoundError) as error:
            failures += 1
            print(f"FAIL  {source.name}: {error}", file=sys.stderr)
            continue

        detail = " ".join(f"{key}={value}" for key, value in result.summary.items())
        verb = "skip" if result.skipped else "ok  "
        print(f"{verb}  {source.name}: {result.reason} {detail}")

    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="avannotate",
        description="Offline annotation pipeline for multi-person video.",
    )
    subparsers = parser.add_subparsers(dest="command")

    stages = subparsers.add_parser("stages", help="list the pipeline's stages")
    stages.set_defaults(handler=_cmd_stages)

    doctor = subparsers.add_parser(
        "doctor",
        help="report whether this machine can run each stage",
        description=(
            "Checks for the packages, checkouts and weights each stage needs, "
            "so a missing one is found before a batch rather than during it."
        ),
    )
    doctor.add_argument(
        "--configs-dir",
        type=Path,
        default=Path("configs"),
        help="where the stage config files live (default: ./configs)",
    )
    doctor.add_argument(
        "--stage",
        action="append",
        help="check only this stage; repeatable. Default: every stage.",
    )
    doctor.set_defaults(handler=_cmd_doctor)

    run = subparsers.add_parser("run", help="run one stage over one or more videos")
    run.add_argument("--stage", required=True, help="stage name, e.g. s0-preprocess")
    run.add_argument(
        "--input",
        required=True,
        help="a video file, a directory of videos, or a text file listing them",
    )
    run.add_argument("--output", required=True, type=Path, help="output root directory")
    run.add_argument("--config", type=Path, help="stage configuration JSON")
    run.add_argument(
        "--force",
        action="store_true",
        help="re-run even when the previous outputs are present and unchanged",
    )
    run.set_defaults(handler=_cmd_run)

    batch = subparsers.add_parser(
        "batch",
        help="run every stage over a list of videos, several at a time",
        description=(
            "The whole pipeline over a corpus. One video is the unit of work, "
            "so each is finished or not; a failure is recorded and the batch "
            "carries on."
        ),
    )
    batch.add_argument(
        "--input", required=True, type=Path, help="a text file listing the videos"
    )
    batch.add_argument("--output", required=True, type=Path, help="output root directory")
    batch.add_argument(
        "--configs",
        type=Path,
        help="directory of stage configs (default: the repository's configs/)",
    )
    batch.add_argument(
        "--gpus",
        help="comma-separated device indices, e.g. 0,1,2 (default: detect them)",
    )
    batch.add_argument(
        "--workers",
        type=int,
        help="videos at once (default: one per GPU, or 1 with no GPU)",
    )
    batch.add_argument(
        "--stage",
        action="append",
        help="run only this stage; repeatable. Default: all of them, in order.",
    )
    batch.add_argument(
        "--only-missing",
        action="store_true",
        help="skip videos that already have a deliverable",
    )
    batch.add_argument(
        "--force", action="store_true", help="re-run even when outputs are unchanged"
    )
    batch.add_argument(
        "-v", "--verbose", action="store_true", help="print a line per stage, not per video"
    )
    batch.set_defaults(handler=_cmd_batch)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 0
    return int(handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
