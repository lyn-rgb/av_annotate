"""Tests for the corpus report.

The report is read back off the stage state files, so these write real ones --
through ``StageState``, not by hand -- because the thing most likely to break it
is the format of a file it does not own.

The load-bearing claim is *which* failure a video is reported under.  Under a
stage-major run one broken video leaves a failed record in every stage after
the one that broke, and a report that picks the last of those, or counts them
all, sends somebody to fix a stage that is working.
"""

from __future__ import annotations

import json
from pathlib import Path

from avannotate import report as report_module
from avannotate.report import build, render, write
from avannotate.stages import STAGE_ORDER
from avannotate.stages.base import StageRecord, StageState

COMPOSE = "s11-compose"


def _video(
    output: Path,
    name: str,
    statuses: dict[str, tuple[str, str | None]],
    *,
    deliverable: bool = False,
) -> Path:
    """A work directory with real stage records in it."""

    work_dir = output / "work" / name
    work_dir.mkdir(parents=True, exist_ok=True)
    state = StageState(work_dir)
    for stage, (status, error) in statuses.items():
        state.save(
            StageRecord(
                stage=stage,
                status=status,
                code_version="v1",
                config_hash="cfg",
                input_hash="in",
                error=error,
            )
        )
    if deliverable:
        path = work_dir / COMPOSE / "annotation.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    return work_dir


def _all_ok() -> dict[str, tuple[str, str | None]]:
    return {stage: ("ok", None) for stage in STAGE_ORDER}


def _stopped_at(stage: str, error: str) -> dict[str, tuple[str, str | None]]:
    """Everything fine up to ``stage``, which failed; the rest failed for want of it."""

    index = STAGE_ORDER.index(stage)
    statuses = {item: ("ok", None) for item in STAGE_ORDER[:index]}
    statuses[stage] = ("failed", error)
    for later in STAGE_ORDER[index + 1 :]:
        statuses[later] = ("failed", f"{stage}/output.json is missing")
    return statuses


# --------------------------------------------------------------------------- #
# who counts as what
# --------------------------------------------------------------------------- #


def test_a_deliverable_is_what_makes_a_video_succeed(tmp_path: Path) -> None:
    _video(tmp_path, "good", _all_ok(), deliverable=True)

    summary = build(tmp_path)

    assert [item.video_id for item in summary.succeeded] == ["good"]
    assert summary.failed == ()


def test_records_alone_are_not_enough(tmp_path: Path) -> None:
    """Every stage can say ok and the video still not be finished.

    S11's own record is written before its artifact in some failure modes, and
    the thing the corpus is for is the file.
    """

    _video(tmp_path, "half", _all_ok(), deliverable=False)

    summary = build(tmp_path)

    assert summary.succeeded == ()
    assert [item.video_id for item in summary.failed] == ["half"]


def test_a_video_is_reported_under_the_stage_that_stopped_it(tmp_path: Path) -> None:
    """And under *that* stage's message, not the cascade's.

    Every stage after S7 fails too, each saying its input is missing -- which is
    true, and useless.  The cause is the one that says why the input never
    appeared.
    """

    _video(
        tmp_path,
        "broken",
        _stopped_at("s7-tse", "TseError: the extractor wrote nothing"),
    )

    (outcome,) = build(tmp_path).failed

    assert outcome.failed_stage == "s7-tse"
    assert outcome.error == "TseError: the extractor wrote nothing"


def test_the_cascade_is_counted_separately_from_the_cause(tmp_path: Path) -> None:
    """``stopped`` says where to look; ``touched`` says how far it spread.

    The gap between them is the whole reason both are reported: a stage with a
    high ``touched`` and a zero ``stopped`` has nothing wrong with it.
    """

    _video(tmp_path, "broken", _stopped_at("s4-diarize", "MemoryError: out of memory"))

    summary = build(tmp_path)

    assert summary.stopped == {"s4-diarize": 1}
    assert summary.touched["s4-diarize"] == 1
    assert summary.touched["s11-compose"] == 1, "the cascade should be visible"
    assert "s11-compose" not in summary.stopped


def test_a_video_nothing_ran_on_is_a_failure_with_no_stage(tmp_path: Path) -> None:
    """Absent from the report is the one answer that helps nobody."""

    _video(tmp_path, "untouched", {})

    (outcome,) = build(tmp_path).failed

    assert outcome.failed_stage == ""
    assert "no stage" in outcome.error


def test_a_video_the_list_names_but_nothing_ran_on_is_counted(tmp_path: Path) -> None:
    """The batch never reached it, which is a fault in the run."""

    listing = tmp_path / "list.txt"
    listing.write_text("part_001/43/be/aaaa\npart_001/43/be/bbbb\n", encoding="utf-8")
    _video(tmp_path / "out", "aaaa", _all_ok(), deliverable=True)

    summary = build(tmp_path / "out", listing=listing)

    assert summary.expected == 2
    assert [item.video_id for item in summary.failed] == ["bbbb"]


def test_an_empty_output_directory_is_not_a_crash(tmp_path: Path) -> None:
    summary = build(tmp_path / "nowhere")

    assert summary.outcomes == ()
    document = render(summary)
    assert "Nothing failed." in document
    assert "## Failures (0)" in document


def test_a_state_file_that_cannot_be_read_is_not_a_crash(tmp_path: Path) -> None:
    """The bargain the resume logic makes, made again: unknown is not fatal."""

    work_dir = tmp_path / "work" / "mangled"
    work_dir.mkdir(parents=True)
    (work_dir / "stage_state.json").write_text("{not json", encoding="utf-8")

    (outcome,) = build(tmp_path).outcomes

    assert not outcome.ok
    assert outcome.failed_stage == ""


# --------------------------------------------------------------------------- #
# the document
# --------------------------------------------------------------------------- #


def test_the_document_lists_every_failure(tmp_path: Path) -> None:
    for name in ("a1", "b2", "c3"):
        _video(tmp_path, name, _stopped_at("s3-cluster", "no faces were tracked"))
    _video(tmp_path, "good", _all_ok(), deliverable=True)

    document = render(build(tmp_path))

    for name in ("a1", "b2", "c3"):
        assert name in document, f"{name} is missing from the failure list"
    assert "| 3 |" in document, "the summary should count the failures"


def test_the_document_repeats_an_identical_error_once(tmp_path: Path) -> None:
    """Three hundred videos dying the same way is one sentence, not three hundred."""

    for index in range(30):
        _video(tmp_path, f"v{index}", _stopped_at("s7-tse", "the extractor wrote nothing"))

    document = render(build(tmp_path))

    assert "**30×**" in document
    assert "the extractor wrote nothing" in document


def test_writing_produces_both_copies(tmp_path: Path) -> None:
    _video(tmp_path, "good", _all_ok(), deliverable=True)

    document, data = write(build(tmp_path))

    assert document.name == report_module.REPORT_NAME
    assert data.name == report_module.DATA_NAME
    assert document.read_text(encoding="utf-8").startswith("# Corpus report")
    payload = json.loads(data.read_text(encoding="utf-8"))
    assert payload["totals"]["succeeded"] == 1
    assert payload["totals"]["failed"] == 0


def test_the_data_copy_keeps_the_whole_error(tmp_path: Path) -> None:
    """The document clips for the sake of the table; this one does not."""

    long_error = "RuntimeError: " + ("x" * 400)
    _video(tmp_path, "broken", _stopped_at("s5-asd", long_error))

    _, data = write(build(tmp_path))

    payload = json.loads(data.read_text(encoding="utf-8"))
    assert payload["failures"][0]["error"] == long_error


def test_an_error_containing_a_pipe_does_not_break_the_table(tmp_path: Path) -> None:
    """ffmpeg separates its stream summary with ``|``, so this is not exotic.

    Unescaped, the row grows two columns in the middle and the table stops
    being a table -- for the one stage most likely to have failed on a corpus
    nobody has probed before.
    """

    _video(
        tmp_path,
        "bad",
        _stopped_at(
            "s0-preprocess",
            "FFmpegError: probing failed: [mov,mp4 @ 0x1] moov atom not found | /data/x.mp4",
        ),
    )

    document = render(build(tmp_path))
    row = next(line for line in document.splitlines() if line.startswith("| bad "))

    assert "\\|" in row, "the pipe was not escaped"
    assert row.count(" | ") == 2, "the row has more columns than the table"
