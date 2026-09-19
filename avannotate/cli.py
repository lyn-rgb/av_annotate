"""Command line entry point.

Deliberately small: ``run`` executes one stage over one video or a list of them,
and that is enough to drive a batch from a shell loop or a job scheduler.  The
per-stage resume record, not the CLI, is what makes a rerun cheap.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from avannotate.stages import STAGE_ORDER, available_stages, get_stage
from avannotate.stages.base import StageContext, StageError

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
    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"config must be a JSON object: {path}")
    from avannotate.stages.base import CONFIG_ROOT_KEY

    payload.pop(CONFIG_ROOT_KEY, None)
    payload[CONFIG_ROOT_KEY] = str(resolved.parent)
    return payload


def _cmd_stages(_: argparse.Namespace) -> int:
    implementable = set(available_stages())
    for name in STAGE_ORDER:
        print(f"  {'implemented' if name in implementable else 'not yet    '}  {name}")
    return 0


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
