"""Tests for the batch driver's planning and reporting.

The pool itself is exercised by running it; what is tested here is everything
that decides *what* runs and *what the operator sees* -- because those are the
mistakes that stay invisible until a corpus is half-processed.  A video list
that quietly drops entries produces a short corpus that looks like a complete
one, and two videos sharing a stem share a work directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from avannotate import batch
from avannotate.stages import STAGE_ORDER
from avannotate.stages.base import StageError, StageRun


def _video(directory: Path, name: str) -> Path:
    path = directory / name
    path.write_bytes(b"\x00")
    return path


# --------------------------------------------------------------------------- #
# the video list
# --------------------------------------------------------------------------- #


def test_the_list_keeps_its_order(tmp_path: Path) -> None:
    for name in ("c.mp4", "a.mp4", "b.mp4"):
        _video(tmp_path, name)
    listing = tmp_path / "list.txt"
    listing.write_text("c.mp4\na.mp4\nb.mp4\n", encoding="utf-8")

    found = batch.read_video_list(listing, base=tmp_path)

    assert [item.name for item in found] == ["c.mp4", "a.mp4", "b.mp4"]


def test_blanks_and_comments_are_skipped(tmp_path: Path) -> None:
    _video(tmp_path, "a.mp4")
    listing = tmp_path / "list.txt"
    listing.write_text("# a comment\n\na.mp4\n\n   \n", encoding="utf-8")

    assert [item.name for item in batch.read_video_list(listing, base=tmp_path)] == ["a.mp4"]


def test_a_repeated_entry_is_listed_once(tmp_path: Path) -> None:
    """Running the same video twice is wasted work, and the second run would
    find everything already done."""

    _video(tmp_path, "a.mp4")
    listing = tmp_path / "list.txt"
    listing.write_text("a.mp4\n./a.mp4\na.mp4\n", encoding="utf-8")

    assert len(batch.read_video_list(listing, base=tmp_path)) == 1


def test_entries_resolve_against_the_working_directory_first(tmp_path: Path) -> None:
    """Both conventions are in use, and a wrong guess is "file not found" for
    every line at once."""

    working = tmp_path / "work"
    working.mkdir()
    _video(working, "a.mp4")
    listing = tmp_path / "list.txt"
    listing.write_text("a.mp4\n", encoding="utf-8")

    found = batch.read_video_list(listing, base=working)

    assert found[0] == (working / "a.mp4").resolve()


def test_entries_fall_back_to_the_lists_own_directory(tmp_path: Path) -> None:
    _video(tmp_path, "a.mp4")
    listing = tmp_path / "list.txt"
    listing.write_text("a.mp4\n", encoding="utf-8")

    found = batch.read_video_list(listing, base=tmp_path / "elsewhere")

    assert found[0] == (tmp_path / "a.mp4").resolve()


def test_a_missing_entry_is_an_error_not_a_skip(tmp_path: Path) -> None:
    """A silently short corpus is the failure that looks like success."""

    _video(tmp_path, "a.mp4")
    listing = tmp_path / "list.txt"
    listing.write_text("a.mp4\nnot-there.mp4\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="not-there.mp4"):
        batch.read_video_list(listing, base=tmp_path)


# --------------------------------------------------------------------------- #
# planning
# --------------------------------------------------------------------------- #


def test_every_video_becomes_one_job_under_the_output(tmp_path: Path) -> None:
    sources = tuple(_video(tmp_path, name) for name in ("a.mp4", "b.mp4"))

    jobs = batch.plan_jobs(sources, output=tmp_path / "out")

    assert [job.video_id for job in jobs] == ["a", "b"]
    assert jobs[0].work_dir == tmp_path / "out" / "work" / "a"


def test_two_videos_with_one_stem_is_an_error(tmp_path: Path) -> None:
    """The stem names the work directory and the video_id in the deliverable,
    so a collision would silently mix two videos into one annotation."""

    first = tmp_path / "one" / "clip.mp4"
    second = tmp_path / "two" / "clip.mp4"
    for path in (first, second):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x00")

    with pytest.raises(ValueError, match="same stem"):
        batch.plan_jobs((first, second), output=tmp_path / "out")


def test_a_video_is_complete_when_it_has_a_deliverable(tmp_path: Path) -> None:
    """Not when its last stage ran: a video whose stages all ran but whose
    compose failed is not done."""

    job = batch.Job(video_id="a", source=tmp_path / "a.mp4", work_dir=tmp_path / "work" / "a")

    assert not batch.is_complete(job)
    (job.work_dir / "s11-compose").mkdir(parents=True)
    assert not batch.is_complete(job)
    (job.work_dir / "s11-compose" / "annotation.json").write_text("{}", encoding="utf-8")
    assert batch.is_complete(job)


# --------------------------------------------------------------------------- #
# devices and workers
# --------------------------------------------------------------------------- #


def test_a_gpu_list_is_parsed() -> None:
    assert batch.parse_gpu_list("0,1,2") == (0, 1, 2)
    assert batch.parse_gpu_list(" 3 , 1 ") == (3, 1)
    assert batch.parse_gpu_list(None) is None
    assert batch.parse_gpu_list("") is None


def test_a_malformed_gpu_list_says_what_it_wanted() -> None:
    with pytest.raises(ValueError, match="indices"):
        batch.parse_gpu_list("cuda:0")


def test_one_worker_per_gpu_by_default() -> None:
    devices, workers = batch.plan_workers(gpus=(0, 1, 3), requested=None)

    assert devices == (0, 1, 3)
    assert workers == 3


def test_a_machine_with_no_gpu_still_runs() -> None:
    devices, workers = batch.plan_workers(gpus=(), requested=None)

    assert devices == ()
    assert workers == 1


def test_workers_can_be_asked_for_explicitly() -> None:
    """Occasionally right -- a stage waiting on the CPU, a card with room --
    which is why it is allowed and not why it is the default."""

    devices, workers = batch.plan_workers(gpus=(0, 1), requested=5)

    assert devices == (0, 1)
    assert workers == 5


def test_zero_workers_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        batch.plan_workers(gpus=(0,), requested=0)


# --------------------------------------------------------------------------- #
# what the operator sees
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(9, "9s"), (59, "59s"), (60, "1m 00s"), (3599, "59m 59s"), (3600, "1h 00m 00s")],
)
def test_durations_read_the_way_a_watcher_thinks(seconds: int, expected: str) -> None:
    assert batch.format_duration(seconds) == expected


def test_a_progress_line_carries_what_a_watcher_needs(tmp_path: Path) -> None:
    job = batch.Job(video_id="clip", source=tmp_path / "clip.mp4", work_dir=tmp_path / "w")

    line = batch.format_progress(
        done=3, total=10, job=job, ok=True, seconds=12.0, detail="utt=48", elapsed=36.0
    )

    assert "[   3/10]" in line
    assert "30.0%" in line
    assert "ok" in line
    assert "clip" in line
    assert "12s" in line
    assert "utt=48" in line
    # 36s for 3 leaves 84s for 7.
    assert "1m 24s" in line


def test_a_failed_line_says_so_and_carries_no_estimate(tmp_path: Path) -> None:
    job = batch.Job(video_id="clip", source=tmp_path / "clip.mp4", work_dir=tmp_path / "w")

    line = batch.format_progress(
        done=1, total=10, job=job, ok=False, seconds=2.0, detail="s1: boom", elapsed=2.0
    )

    assert "FAIL" in line
    assert "s1: boom" in line
    assert "eta" not in line


def test_the_last_line_has_no_estimate(tmp_path: Path) -> None:
    job = batch.Job(video_id="clip", source=tmp_path / "clip.mp4", work_dir=tmp_path / "w")

    line = batch.format_progress(
        done=10, total=10, job=job, ok=True, seconds=1.0, detail="", elapsed=100.0
    )

    assert "eta" not in line


def test_a_long_error_is_clipped_for_the_progress_line() -> None:
    """A connection failure arrives as several hundred characters of urllib
    wrapping; a thousand of those is a log nobody reads."""

    clipped = batch._clip("x" * 500)

    assert len(clipped) <= 110
    assert clipped.endswith("…")


def test_a_short_error_is_left_alone() -> None:
    assert batch._clip("StageError: nothing to read") == "StageError: nothing to read"


# --------------------------------------------------------------------------- #
# the corpus summary
# --------------------------------------------------------------------------- #


def _result(video_id: str, *, ok: bool, stages: list[tuple[str, float, bool]]) -> batch.VideoResult:
    return batch.VideoResult(
        video_id=video_id,
        ok=ok,
        seconds=sum(item[1] for item in stages),
        stages=[
            batch.StageLog(stage=name, skipped=skipped, seconds=seconds, summary={})
            for name, seconds, skipped in stages
        ],
        error="" if ok else "boom",
        failed_stage="" if ok else stages[-1][0],
    )


def test_the_summary_counts_what_a_reader_wants() -> None:
    results = [
        _result("a", ok=True, stages=[("s0-preprocess", 1.0, False), ("s1-faces", 4.0, False)]),
        _result("b", ok=True, stages=[("s0-preprocess", 2.0, False), ("s1-faces", 6.0, False)]),
        _result("c", ok=False, stages=[("s0-preprocess", 3.0, False)]),
    ]

    summary = batch.summarise(results)

    assert summary["videos"] == 3
    assert summary["succeeded"] == 2
    assert summary["failed"] == 1
    assert summary["slowest_stage"] == "s1-faces"
    assert summary["seconds_per_stage"]["s1-faces"] == 10.0


def test_the_summary_records_where_stages_were_skipped() -> None:
    """That is what tells an operator a rerun cost nothing, rather than that
    the pipeline is fast."""

    results = [
        _result("a", ok=True, stages=[("s0-preprocess", 0.0, True)]),
        _result("b", ok=True, stages=[("s0-preprocess", 0.0, True)]),
    ]

    summary = batch.summarise(results)

    assert summary["skipped_per_stage"] == {"s0-preprocess": 2}


def test_an_empty_corpus_summarises_without_dividing_by_zero() -> None:
    summary = batch.summarise([])

    assert summary["videos"] == 0
    assert summary["slowest_stage"] == ""


# --------------------------------------------------------------------------- #
# what is written
# --------------------------------------------------------------------------- #


def test_jsonl_appends_one_record_per_line(tmp_path: Path) -> None:
    path = tmp_path / "index.jsonl"

    written = batch.write_jsonl(path, [{"a": 1}, {"a": 2}])
    written += batch.write_jsonl(path, [{"a": 3}])

    assert written == 3
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [item["a"] for item in lines] == [1, 2, 3]


def test_a_video_result_round_trips_through_its_dict() -> None:
    result = _result("a", ok=True, stages=[("s0-preprocess", 1.5, False)])
    payload = json.loads(json.dumps(result.to_dict()))

    rebuilt = batch._rebuild(payload)

    assert rebuilt.video_id == "a"
    assert rebuilt.ok
    assert [item.stage for item in rebuilt.stages] == ["s0-preprocess"]
    assert rebuilt.stages[0].seconds == pytest.approx(1.5)


def test_a_failed_video_keeps_the_stages_it_did_finish() -> None:
    """What is already on disk stays there, and the record says how far it got
    -- that is what makes a rerun cheap."""

    result = _result(
        "a", ok=False, stages=[("s0-preprocess", 1.0, False), ("s1-faces", 2.0, False)]
    )

    rebuilt = batch._rebuild(json.loads(json.dumps(result.to_dict())))

    assert not rebuilt.ok
    assert rebuilt.failed_stage == "s1-faces"
    assert len(rebuilt.stages) == 2


def test_every_stage_has_a_config_named() -> None:
    """A stage missing from the table would run with an empty config and
    silently take every default -- which is a different run, not a failure."""

    for stage in STAGE_ORDER:
        assert stage in batch.STAGE_CONFIGS, f"{stage} has no config in the batch table"


def test_the_named_configs_all_exist() -> None:
    root = Path(__file__).resolve().parents[1] / "configs"

    for stage, name in batch.STAGE_CONFIGS.items():
        if name is None:
            continue
        assert (root / name).is_file(), f"{stage} names {name}, which is not in configs/"


# --------------------------------------------------------------------------- #
# what a watcher sees while it runs
# --------------------------------------------------------------------------- #
#
# The line used to be emitted only once a stage had returned, so a stage that
# never returned emitted nothing at all -- and a batch printing nothing while
# working is indistinguishable from one that has stopped.  When S7 was killing
# its worker, the corpus sat silent for minutes with nothing on screen saying
# where it was.  Where it is, is the entire question.


class _StubStage:
    """A stage that reports when it was called, and what it hands back."""

    def __init__(self, events: list[str], *, error: Exception | None, skipped: bool) -> None:
        self._events = events
        self._error = error
        self._skipped = skipped

    def run(self, context: object, *, force: bool = False) -> object:
        self._events.append("work")
        if self._error is not None:
            raise self._error
        return StageRun(stage="s0-preprocess", skipped=self._skipped, reason="test")


def _announcements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: _StubStage
) -> list[str]:
    events: list[str] = []
    stage._events = events

    monkeypatch.setattr(batch, "get_stage", lambda _: stage)
    job = batch.Job(
        video_id="v1", source=tmp_path / "v1.mp4", work_dir=tmp_path / "work" / "v1"
    )
    batch.run_video(
        job,
        stages=("s0-preprocess",),
        config_root=Path(__file__).resolve().parents[1] / "configs",
        on_stage=lambda _job, _stage, event: events.append(event),
    )
    return events


def test_a_stage_is_announced_before_it_does_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one that makes a hang visible.

    Announcing afterwards is announcing everything except the case that
    matters: a stage that hangs never afterwards.
    """

    events = _announcements(
        tmp_path, monkeypatch, _StubStage([], error=StageError("never returns"), skipped=False)
    )

    assert events == ["start", "work"]


@pytest.mark.parametrize(("skipped", "expected"), [(False, "run"), (True, "skip")])
def test_a_finished_stage_says_which_way_it_went(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, skipped: bool, expected: str
) -> None:
    events = _announcements(
        tmp_path, monkeypatch, _StubStage([], error=None, skipped=skipped)
    )

    # "work" is the stub's own marker, standing in for the stage doing something.
    assert events == ["start", "work", expected]
