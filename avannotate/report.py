"""What a corpus run produced, read back off the disk.

A thousand-video run ends with a question that is not "did the script exit
zero": which videos are done, which are not, and of those, which stage stopped
them.  The answer is not in the batch's output -- that scrolls past, and in a
stage-major run it scrolls past twelve times -- so this reads it back from what
the stages left behind.

**The state files, not the batch's own bookkeeping.**  Every video's work
directory holds a ``stage_state.json`` naming, for each stage that ran, whether
it succeeded and what it said when it did not.  That is the durable record: it
is written by the stage that did the work, it survives a restart, and it is the
same file the resume logic reads.  Anything assembled from the driver's stdout
would be a second account of the same events, and the two would drift.

**Why the report reads this and not ``failures.jsonl``.**  That file is
overwritten by every batch invocation, and a stage-major run makes twelve of
them -- so after a full run it holds S11's failures and nothing else.  A video
that died at S7 is not in it, which is precisely the video somebody is looking
for.  (``failures.jsonl`` is still the right thing to read mid-run, for the
stage that is running now.)

**A video fails once, at the first stage that stopped it.**  Under stage-major
every later stage is attempted anyway and fails too, because its input is
missing -- so a single S7 failure shows up as a failed record in S7, S8, S9,
S10 and S11.  Counting records would report five failures for one broken video.
The report counts the first, and keeps the rest as a separate column, because
the cascade is worth seeing: it is the difference between "S8 is broken" and
"S8 has nothing to read".
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from avannotate.stages import STAGE_ORDER, s11_compose
from avannotate.stages.base import STATE_FILENAME, read_json

REPORT_NAME = "corpus_report.md"
DATA_NAME = "corpus_report.json"

#: Long enough for a real message, short enough that a table stays a table.
ERROR_LIMIT = 160


@dataclass(frozen=True)
class Outcome:
    """One video, and whether the corpus got a deliverable out of it."""

    video_id: str
    ok: bool
    #: The first stage that failed, or ``""`` when none did -- including the
    #: case of a video nothing ever ran on.
    failed_stage: str
    error: str
    stages_ran: int

    def to_dict(self) -> dict[str, object]:
        return {
            "video_id": self.video_id,
            "ok": self.ok,
            "failed_stage": self.failed_stage,
            "error": self.error,
            "stages_ran": self.stages_ran,
        }


@dataclass(frozen=True)
class Report:
    output: Path
    generated_at: str
    expected: int
    outcomes: tuple[Outcome, ...]
    #: stage -> how many videos it stopped first.
    stopped: dict[str, int] = field(default_factory=dict)
    #: stage -> how many videos recorded a failure there at all, cascades
    #: included.
    touched: dict[str, int] = field(default_factory=dict)

    @property
    def succeeded(self) -> tuple[Outcome, ...]:
        return tuple(item for item in self.outcomes if item.ok)

    @property
    def failed(self) -> tuple[Outcome, ...]:
        return tuple(item for item in self.outcomes if not item.ok)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "avannotate-corpus-report-v1",
            "output": str(self.output),
            "generated_at": self.generated_at,
            "totals": {
                "expected": self.expected,
                "seen": len(self.outcomes),
                "succeeded": len(self.succeeded),
                "failed": len(self.failed),
                "never_started": sum(1 for item in self.failed if not item.failed_stage),
            },
            "stopped_at": dict(sorted(self.stopped.items())),
            "failed_at_any_stage": dict(sorted(self.touched.items())),
            "failures": [item.to_dict() for item in self.failed],
        }


def _expected_ids(listing: Path | None) -> tuple[str, ...]:
    """The video ids a list file names, according to the same rule the batch uses.

    ``plan_jobs`` keys a video by its file's stem, and an entry naming a bare id
    stems to itself -- so the id is the entry's last path component minus any
    extension, without needing the files to exist.  Which is the point: this has
    to work for a video whose file was never found, because that is one of the
    failures worth reporting.
    """

    if listing is None or not listing.is_file():
        return ()
    ids: list[str] = []
    for line in listing.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        video_id = Path(entry).stem
        if video_id and video_id not in ids:
            ids.append(video_id)
    return tuple(ids)


def _read_state(video_dir: Path) -> dict[str, tuple[str, str]]:
    """stage -> (status, error), for every stage that recorded one.

    Read once per video and kept, because the caller wants both the status and
    -- for the stage that stopped it -- the message, and a thousand videos is
    not the place to open the same file twice.
    """

    path = video_dir / STATE_FILENAME
    if not path.is_file():
        return {}
    try:
        payload = read_json(path)
    except (OSError, json.JSONDecodeError):
        # The same bargain the resume logic makes: a state file that cannot be
        # read is a video whose fate is unknown, not a reason to stop.
        return {}
    if not isinstance(payload, dict):
        return {}
    return {
        str(item["stage"]): (str(item.get("status", "")), str(item.get("error") or ""))
        for item in (payload.get("stages") or [])
        if isinstance(item, dict) and "stage" in item
    }


def _first_failure(state: dict[str, tuple[str, str]]) -> str:
    """The earliest stage in pipeline order that did not succeed.

    Pipeline order and not the order they happened to finish.  A stage whose
    record is *absent* was never reached, so it cannot be the cause; a stage
    that failed late because an early one did is a consequence.
    """

    for stage in STAGE_ORDER:
        if state.get(stage, ("", ""))[0] == "failed":
            return stage
    return ""


def read_video(video_dir: Path) -> tuple[Outcome, tuple[str, ...]]:
    """(what happened here, every stage that recorded a failure)."""

    state = _read_state(video_dir)
    failed = tuple(stage for stage, (status, _) in state.items() if status == "failed")

    if s11_compose.deliverable_path(video_dir).is_file():
        return Outcome(video_dir.name, True, "", "", len(state)), failed

    stage = _first_failure(state)
    if not stage:
        # No deliverable and nothing failed on the record.  Worded for what is
        # actually known: the batch now writes a failed record for any stage
        # that raises, so this is "nothing ran" in the ordinary case -- but a
        # worker killed outright leaves a directory it never got to write to,
        # and that is not the same claim.
        return (
            Outcome(
                video_dir.name,
                False,
                "",
                "no stage recorded anything for this video",
                len(state),
            ),
            failed,
        )

    return (
        Outcome(video_dir.name, False, stage, state[stage][1], len(state)),
        failed,
    )


def build(output: Path, *, listing: Path | None = None) -> Report:
    """Read a corpus run back off the disk."""

    work = output / "work"
    outcomes: list[Outcome] = []
    stopped: Counter[str] = Counter()
    touched: Counter[str] = Counter()

    if work.is_dir():
        for video_dir in sorted(work.iterdir()):
            if not video_dir.is_dir():
                continue
            outcome, failed = read_video(video_dir)
            outcomes.append(outcome)
            for stage in failed:
                touched[stage] += 1
            if not outcome.ok and outcome.failed_stage:
                stopped[outcome.failed_stage] += 1

    # Anything the list named that has no directory at all.  Counted, not
    # listed one by one: at that point the run is wrong in a way a thousand
    # rows would not explain any better than the number does.
    known = {item.video_id for item in outcomes}
    unseen = [video_id for video_id in _expected_ids(listing) if video_id not in known]
    outcomes.extend(
        Outcome(video_id, False, "", "no stage ever ran for this video", 0)
        for video_id in unseen
    )

    return Report(
        output=output,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        expected=len(_expected_ids(listing)),
        outcomes=tuple(outcomes),
        stopped=dict(stopped),
        touched=dict(touched),
    )


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #


def _share(count: int, total: int) -> str:
    return f"{count / total * 100:.1f}%" if total else "n/a"


def _clip(text: str) -> str:
    flat = " ".join(text.split())
    if not flat:
        return ""
    if len(flat) <= ERROR_LIMIT:
        return flat
    return flat[: ERROR_LIMIT - 1] + "…"


def _cell(text: str) -> str:
    """A table cell, with the one character that would end it escaped.

    Not hypothetical: ffmpeg's messages contain ``|`` -- ffprobe separates its
    stream summary with them -- so the first real corpus that failed at S0
    produced a failure table whose rows had two extra columns in the middle of
    them.
    """

    return text.replace("|", "\\|")


def _table(rows: Iterable[Sequence[str]], headers: Sequence[str]) -> list[str]:
    lines = ["| " + " | ".join(_cell(item) for item in headers) + " |"]
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        lines.append("| " + " | ".join(_cell(item) for item in row) + " |")
    return lines


def render(report: Report) -> str:
    """The report as a document somebody reads and acts on."""

    total = len(report.outcomes)
    succeeded = len(report.succeeded)
    failed = report.failed
    share = _share(succeeded, total)

    lines = [
        "# Corpus report",
        "",
        f"- output: `{report.output}`",
        f"- generated: {report.generated_at}",
        f"- videos: {total}" + (f" (list names {report.expected})" if report.expected else ""),
        "",
        "A video counts as succeeded when S11 wrote its `annotation.json`. A",
        "video fails once, at the first stage that stopped it -- later stages",
        "are attempted anyway under a stage-major run and fail for want of",
        "input, so counting records would report five failures for one video.",
        "",
        "## Summary",
        "",
        *_table(
            [
                ["succeeded", str(succeeded), share],
                ["failed", str(len(failed)), _share(len(failed), total)],
            ],
            ["", "videos", "share"],
        ),
        "",
    ]

    never = sum(1 for item in failed if not item.failed_stage)
    if never:
        lines += [
            f"{never} of the failures name no stage: nothing recorded a failure for",
            "them, so the batch did not reach them, or a worker died before it could",
            "write one. Neither is a fault in the video.",
            "",
        ]

    lines += ["## Where they stopped", ""]
    if report.stopped:
        lines += _table(
            [
                [stage, str(report.stopped.get(stage, 0)), str(report.touched.get(stage, 0))]
                for stage in STAGE_ORDER
                if report.stopped.get(stage) or report.touched.get(stage)
            ],
            ["stage", "stopped here", "also failed here"],
        )
    else:
        lines.append("Nothing failed.")
    lines.append("")

    lines += [f"## Failures ({len(failed)})", ""]
    if not failed:
        lines.append("None.")
        return "\n".join(lines) + "\n"

    # Grouped first: when three hundred videos die the same way, the useful
    # statement is "three hundred die this way", not three hundred rows.
    common = Counter((item.failed_stage, _clip(item.error)) for item in failed)
    if len(common) < len(failed):
        lines += ["Most common first, before the full list:", ""]
        for (stage, error), count in common.most_common(10):
            where = stage or "never started"
            lines.append(f"- **{count}×** `{where}` — {error or 'no message'}")
        lines.append("")

    lines += _table(
        (
            [item.video_id, item.failed_stage or "never started", _clip(item.error)]
            for item in failed
        ),
        ["video_id", "stage", "error"],
    )
    return "\n".join(lines) + "\n"


def write(report: Report) -> tuple[Path, Path]:
    """Write the document and the machine-readable copy; return both paths."""

    report.output.mkdir(parents=True, exist_ok=True)
    document = report.output / REPORT_NAME
    document.write_text(render(report), encoding="utf-8")

    data = report.output / DATA_NAME
    data.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return document, data
