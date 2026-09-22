"""Running the whole pipeline over a corpus, several videos at a time.

Every other part of this project runs one stage over one video; this is the
loop around them, and it is where a hundred-hour corpus either makes progress
or does not.

Three decisions shape it.

**A video is the unit of work, not a stage.**  Every stage is resumable on its
own, so the alternative -- sweep the corpus stage by stage -- would be equally
correct and much worse to watch: nothing is finished until everything is, and a
failure at S7 leaves a hundred videos half-annotated.  Running a video end to
end means each one is either done or not, and a corpus can be stopped at any
point and still hold complete deliverables.

**One worker per GPU.**  The stage adapters each load their own model, so two
workers sharing a card would swap weights in and out of memory for every video
and finish slower than one.  The parallelism that matters is across videos, and
the number of cards is what bounds it.

**A failure is recorded, not raised.**  One unreadable video should not end a
batch that has nineteen hours left to run.  Failures go to ``failures.jsonl``
next to the outputs, and the driver exits non-zero at the end if there were
any -- so a scheduler sees the failure without the corpus paying for it.

The pure parts -- planning the work, choosing how many workers, formatting the
progress line -- are separate from the running, because those are the parts
whose mistakes are invisible until a corpus is half-processed.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import queue
import subprocess
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from avannotate.progress import format_duration
from avannotate.stages import STAGE_ORDER, get_stage
from avannotate.stages.base import StageContext, StageError, load_config_file

#: The config each stage is normally run with.  S1 is the one with a real
#: choice: insightface produces the identity vectors S3 needs, YuNet does not.
STAGE_CONFIGS: dict[str, str | None] = {
    "s0-preprocess": None,
    "s1-faces": "s1.insightface.json",
    "s2-tracks": "s2.default.json",
    "s3-cluster": "s3.default.json",
    "s4-diarize": "s4.diarizen.json",
    "s5-asd": "s5.loconet.json",
    "s6-associate": "s6.associate.json",
    "s7-tse": "s7.clearvoice.json",
    "s8-asr": "s8.whisper.json",
    "s9-paralinguistic": "s9.paralinguistic.json",
    "s10-caption": "s10.caption.json",
    "s11-compose": "s11.compose.json",
}


# A second ``VIDEO_SUFFIXES`` used to sit here, unused by anything.  It is gone
# rather than left: the list it duplicates is the one that is *not* enough on
# its own to decide whether a file is a video -- see ``cli.is_media_file`` -- and
# a spare copy of it lying around is an invitation to write the check that
# trusts it.


# --------------------------------------------------------------------------- #
# planning
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Job:
    """One video, and where its work goes."""

    video_id: str
    source: Path
    work_dir: Path

    def to_dict(self) -> dict[str, object]:
        return {
            "video_id": self.video_id,
            "source": str(self.source),
            "work_dir": str(self.work_dir),
        }


def read_video_list(path: Path, *, base: Path) -> tuple[Path, ...]:
    """The videos named by a list file, in order, without duplicates.

    Entries are resolved against the working directory first and the list's own
    directory second, because both conventions are in use and a wrong guess
    would be an unhelpful "file not found" for every line at once.

    A missing file is an error rather than a skip.  A silently short corpus is
    the failure that looks like success.
    """

    resolved: list[Path] = []
    seen: set[Path] = set()
    missing: list[str] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        for root in (base, path.parent):
            attempt = (root / entry).expanduser()
            if attempt.is_file():
                candidate = attempt.resolve()
                if candidate not in seen:
                    seen.add(candidate)
                    resolved.append(candidate)
                break
        else:
            missing.append(entry)

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} entries of {path} were not found, relative to {base} "
            f"or {path.parent}: {', '.join(missing[:5])}"
            + (" ..." if len(missing) > 5 else "")
        )
    return tuple(resolved)


def plan_jobs(sources: Sequence[Path], *, output: Path) -> tuple[Job, ...]:
    """One job per video, keyed by the file's stem.

    The stem is the video's identity everywhere downstream -- it names the work
    directory and the ``video_id`` in the deliverable -- so two files with the
    same stem would silently share one work directory and one set of outputs.
    That is caught here rather than discovered as a deliverable that describes
    the wrong video.
    """

    jobs: list[Job] = []
    by_id: dict[str, Path] = {}
    for source in sources:
        video_id = source.stem
        if video_id in by_id:
            raise ValueError(
                f"{source} and {by_id[video_id]} have the same stem ({video_id}), and "
                "the stem is the video's identity: they would share one work "
                "directory. Rename one of them."
            )
        by_id[video_id] = source
        jobs.append(Job(video_id=video_id, source=source, work_dir=output / "work" / video_id))
    return tuple(jobs)


def is_complete(job: Job) -> bool:
    """Whether this video already has a deliverable.

    The test is S11's ``annotation.json`` and not "the last stage ran": a
    deliverable is what the corpus is for, and a video whose stages all ran but
    whose compose failed is not done.  Everything the stages wrote is still
    there, so rerunning one costs the compose step and not the models.
    """

    return (job.work_dir / "s11-compose" / "annotation.json").is_file()


def parse_gpu_list(raw: str | None) -> tuple[int, ...] | None:
    """``"0,2,3"`` as ``(0, 2, 3)``; ``None`` means "detect them"."""

    if raw is None or not raw.strip():
        return None
    devices: list[int] = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        try:
            devices.append(int(piece))
        except ValueError as error:
            raise ValueError(f"--gpus takes indices like '0,1,2', not {piece!r}") from error
    if not devices:
        raise ValueError("--gpus was given but named no devices")
    return tuple(devices)


def detect_gpus() -> tuple[int, ...]:
    """Which CUDA devices this machine appears to have.

    ``nvidia-smi`` first: it answers before torch is imported, and it is right
    about a machine whose torch build has no CUDA support even though the
    driver is there.  torch second, because it is the authority on what the
    frameworks will actually see.  Neither is required -- a corpus can be run on
    the CPU, slowly.
    """

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if result.returncode == 0:
            indices = [
                int(line.strip())
                for line in result.stdout.splitlines()
                if line.strip().isdigit()
            ]
            if indices:
                return tuple(indices)
    except (OSError, subprocess.SubprocessError, ValueError):
        pass

    try:
        import torch

        count = torch.cuda.device_count()
        if count:
            return tuple(range(count))
    except Exception:  # noqa: BLE001 - any failure here means "no torch GPUs"
        pass

    return ()


def plan_workers(
    *, gpus: tuple[int, ...], requested: int | None
) -> tuple[tuple[int, ...], int]:
    """How many videos to run at once, and on which devices.

    One per GPU unless told otherwise.  More than one per GPU is allowed
    because it is occasionally right -- a stage that is waiting on the CPU, or
    a card with room to spare -- but it is not the default: every adapter loads
    its own model, so two workers on one card swap weights for every video.
    """

    if requested is not None:
        if requested < 1:
            raise ValueError(f"--workers must be at least 1, got {requested}")
        return gpus, requested
    if gpus:
        return gpus, len(gpus)
    return (), 1


# --------------------------------------------------------------------------- #
# progress
# --------------------------------------------------------------------------- #


def format_progress(
    *,
    done: int,
    total: int,
    job: Job,
    ok: bool,
    seconds: float,
    detail: str,
    elapsed: float,
) -> str:
    """One line per finished video, carrying what a watcher needs.

    Percentage, the running tally, the video, how long it took and — for a
    corpus that runs for hours — how long the rest will take.  The estimate is
    naive (finished videos times remaining) and it is still the first thing
    anybody looks at.
    """

    share = (done / total * 100.0) if total else 100.0
    mark = "ok  " if ok else "FAIL"
    line = (
        f"[{done:>4}/{total}] {share:5.1f}%  {mark}  {job.video_id}  "
        f"{format_duration(seconds):>9}  {detail}"
    )
    if ok and 0 < done < total and elapsed > 0:
        remaining = (elapsed / done) * (total - done)
        line += f"  eta {format_duration(remaining)}"
    return line


# --------------------------------------------------------------------------- #
# running
# --------------------------------------------------------------------------- #


@dataclass
class StageLog:
    stage: str
    skipped: bool
    seconds: float
    summary: dict[str, object] = field(default_factory=dict)


@dataclass
class VideoResult:
    video_id: str
    ok: bool
    seconds: float
    stages: list[StageLog] = field(default_factory=list)
    error: str = ""
    failed_stage: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "video_id": self.video_id,
            "ok": self.ok,
            "seconds": round(self.seconds, 2),
            "failed_stage": self.failed_stage,
            "error": self.error,
            "stages": {
                item.stage: {
                    "skipped": item.skipped,
                    "seconds": round(item.seconds, 2),
                    **item.summary,
                }
                for item in self.stages
            },
        }


def run_video(
    job: Job,
    *,
    stages: Sequence[str],
    config_root: Path,
    force: bool = False,
    on_stage: Any = None,
) -> VideoResult:
    """Every stage, in order, for one video.

    Stops at the first stage that raises: everything after it reads that
    stage's output, so continuing would produce a deliverable built from
    missing pieces.  What is already written stays written, and rerunning skips
    to the stage that failed.
    """

    started = time.monotonic()
    result = VideoResult(video_id=job.video_id, ok=False, seconds=0.0)

    for stage in stages:
        module = get_stage(stage)
        config_name = STAGE_CONFIGS.get(stage)
        config = load_config_file(config_root / config_name) if config_name else {}
        context = StageContext(
            video_id=job.video_id,
            source=job.source,
            work_dir=job.work_dir,
            config=config,
        )
        # Announced on the way *in*, not only on the way out.
        #
        # The line used to be emitted once the stage had returned, so a stage
        # that never returned emitted nothing at all -- and "nothing at all" is
        # indistinguishable from a batch that is working quietly.  When S7 was
        # killing its worker, the corpus sat silent for minutes and the only
        # visible fact was that nothing was happening.  *Where* it is happening
        # is the entire question, and only a line written before the work
        # answers it.
        if on_stage is not None:
            on_stage(job, stage, "start")

        step = time.monotonic()
        try:
            run = module.run(context, force=force)
        except (StageError, ValueError, FileNotFoundError, RuntimeError, OSError) as error:
            result.error = f"{type(error).__name__}: {error}"
            result.failed_stage = stage
            result.seconds = time.monotonic() - started
            return result

        result.stages.append(
            StageLog(
                stage=stage,
                skipped=bool(run.skipped),
                seconds=time.monotonic() - step,
                summary=dict(run.summary),
            )
        )
        if on_stage is not None:
            on_stage(job, stage, "skip" if run.skipped else "run")

    result.ok = True
    result.seconds = time.monotonic() - started
    return result


def summarise(results: Sequence[VideoResult]) -> dict[str, object]:
    """The corpus-level numbers, for the index and the closing summary."""

    ok = [item for item in results if item.ok]
    per_stage: dict[str, float] = {stage: 0.0 for stage in STAGE_ORDER}
    skipped: dict[str, int] = {stage: 0 for stage in STAGE_ORDER}
    for result in results:
        for log in result.stages:
            per_stage[log.stage] = per_stage.get(log.stage, 0.0) + log.seconds
            if log.skipped:
                skipped[log.stage] = skipped.get(log.stage, 0) + 1

    return {
        "videos": len(results),
        "succeeded": len(ok),
        "failed": len(results) - len(ok),
        "seconds": round(sum(item.seconds for item in results), 2),
        "seconds_per_stage": {k: round(v, 2) for k, v in per_stage.items() if v},
        "skipped_per_stage": {k: v for k, v in skipped.items() if v},
        "slowest_stage": (
            max(per_stage, key=lambda name: per_stage[name]) if any(per_stage.values()) else ""
        ),
    }


def write_jsonl(path: Path, records: Iterable[dict[str, object]]) -> int:
    """Append-only records, one JSON object per line.  Returns how many."""

    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


# --------------------------------------------------------------------------- #
# the pool
# --------------------------------------------------------------------------- #
#
# `spawn`, not fork.  A forked worker inherits whatever the parent has already
# initialised -- and `detect_gpus` may have imported torch, which hands every
# worker the parent's CUDA context no matter what CUDA_VISIBLE_DEVICES says.
# Spawning costs a few seconds of imports per worker and makes the device
# assignment mean what it says.


@dataclass
class _Worker:
    queue: Any
    config_root: Path
    stages: tuple[str, ...]
    force: bool


_WORKER: _Worker | None = None


def _worker_init(
    assignments: tuple[int | None, ...],
    counter: Any,
    lock: Any,
    queue_: Any,
    config_root: str,
    stages: tuple[str, ...],
    force: bool,
) -> None:
    """Give this worker a card, once, before anything imports torch.

    A worker keeps its card for its whole life rather than per video: a CUDA
    context cannot be moved once it exists, so `CUDA_VISIBLE_DEVICES` set after
    the first video would be silently ignored -- and everything would run on
    the first card while the others sat idle.  The parent cannot tell a worker
    its index, so the workers take one in turn.
    """

    global _WORKER
    with lock:
        index = counter.value
        counter.value = index + 1
    gpu = assignments[index % len(assignments)] if assignments else None
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    _WORKER = _Worker(
        queue=queue_, config_root=Path(config_root), stages=stages, force=force
    )


def _worker_run(payload: dict[str, object]) -> dict[str, object]:
    """One video, in a worker process, reporting as it goes."""

    assert _WORKER is not None, "worker was not initialised"
    job = Job(
        video_id=str(payload["video_id"]),
        source=Path(str(payload["source"])),
        work_dir=Path(str(payload["work_dir"])),
    )

    def announce(_job: Job, stage: str, event: str) -> None:
        """``event`` is ``start``, ``run`` or ``skip`` -- see :func:`run_video`."""

        _WORKER.queue.put(("stage", job.video_id, stage, event, time.monotonic()))

    result = run_video(
        job,
        stages=_WORKER.stages,
        config_root=_WORKER.config_root,
        force=_WORKER.force,
        on_stage=announce,
    )
    _WORKER.queue.put(("done", job.video_id, result.to_dict(), time.monotonic()))
    return result.to_dict()


#: How often the watchdog looks at the workers, in seconds.
#:
#: A module constant rather than an argument so a test can shorten it: the
#: behaviour below is only observable by killing a process, and a test that has
#: to wait ten seconds for that is a test that gets skipped.
WORKER_WATCH_SECONDS = 10.0

#: What a video's failure says when the process running it died.
WORKER_DIED = (
    "the worker process running this video died, so the batch was stopped "
    "rather than left waiting for it; re-running resumes from the stages that "
    "did finish"
)


def watch_workers(
    pool: Any,
    *,
    stop: threading.Event,
    seconds: float,
    on_dead: Callable[[tuple[Any, ...]], None],
) -> None:
    """Stop the pool waiting for a worker that is never coming back.

    ``Pool`` does not notice a worker dying.  The task it was running never
    returns and the parent blocks on a result that will never arrive -- forever,
    because nothing here is an error, it is only silence.  A batch that should
    have reported a failure sits all night instead.

    Liveness of the *process* and not of progress.  A stage can legitimately
    take minutes -- S10 is much the slowest -- and a watchdog that watched the
    clock would kill exactly the long videos it exists to protect.

    Terminating takes the other workers' videos with them, and that is the right
    trade: ``Pool`` cannot replace one worker, so the choice is between losing
    the videos in flight and losing the whole night.

    Returns when it has stopped the pool, or when ``stop`` is set.
    """

    while not stop.wait(seconds):
        dead = tuple(worker for worker in _worker_processes(pool) if not worker.is_alive())
        if dead:
            on_dead(dead)
            pool.terminate()
            return


def account_for_missing(
    results: list[VideoResult], jobs: Sequence[Job]
) -> list[VideoResult]:
    """Every job in the answer, whether or not a worker returned it.

    A worker that dies takes its video with it and the iteration over the pool
    simply *ends* -- no exception, no sentinel, nothing to catch.  So without
    this the batch reports the videos that happened to finish as though they
    were all there was, which is the same silence one layer further out: the
    watchdog stops the waiting, and this is what stops the lying.

    Ordered by the job list rather than by who finished first, so the index
    reads the same way the corpus was given.
    """

    reported = {item.video_id for item in results}
    for job in jobs:
        if job.video_id not in reported:
            results.append(
                VideoResult(
                    video_id=job.video_id,
                    ok=False,
                    seconds=0.0,
                    failed_stage="(unknown)",
                    error=WORKER_DIED,
                )
            )

    order = {job.video_id: index for index, job in enumerate(jobs)}
    results.sort(key=lambda item: order[item.video_id])
    return results


def _worker_processes(pool: Any) -> tuple[Any, ...]:
    """The pool's worker processes, or an empty tuple if they cannot be had.

    ``Pool._pool`` is private and there is no public accessor for it -- which
    matters, because "is a worker still alive" is exactly the question a parent
    blocked on a result needs answered and ``Pool`` exposes nothing that answers
    it.  Read through ``getattr`` so a Python that renames it degrades to no
    watchdog rather than to an AttributeError in a thread.
    """

    return tuple(getattr(pool, "_pool", ()) or ())


def run_corpus(
    jobs: Sequence[Job],
    *,
    stages: Sequence[str],
    config_root: Path,
    workers: int = 1,
    gpus: tuple[int, ...] = (),
    force: bool = False,
    on_line: Any = None,
    on_stage: Any = None,
    on_done: Any = None,
) -> list[VideoResult]:
    """Every video, ``workers`` at a time, reporting as they finish.

    Progress arrives on a queue that a reader thread drains, so a stage line
    appears while the batch is working rather than in a burst after the next
    video ends.  A batch of a thousand videos is watched for hours; output that
    only moves between videos reads as a hang.

    ``on_done(ok)`` is called once per finished video, and is how a caller
    replaces the per-video line with something else -- a bar, a counter, a
    notification.  When it is given, only failures still get a line: the bar
    says how many, and only a line can say which video and why.
    """

    import threading

    context = multiprocessing.get_context("spawn")
    progress: Any = context.Queue()
    assignments: tuple[int | None, ...] = (
        tuple(gpus[index % len(gpus)] for index in range(workers))
        if gpus
        else (None,) * workers
    )

    pool = context.Pool(
        processes=workers,
        initializer=_worker_init,
        initargs=(
            assignments,
            context.Value("i", 0),
            context.Lock(),
            progress,
            str(config_root),
            tuple(stages),
            force,
        ),
    )

    lock = threading.Lock()

    def emit(line: str) -> None:
        with lock:
            if on_line is not None:
                on_line(line)
            else:
                print(line, flush=True)

    started = time.monotonic()
    total = len(jobs)
    seen = 0
    stop = threading.Event()

    def drain() -> None:
        """Print stage events until told to stop or the queue runs dry."""

        while not stop.is_set():
            try:
                message = progress.get(timeout=0.2)
            except queue.Empty:
                continue
            except (OSError, EOFError):  # pool shut down under us
                return
            if message[0] == "stage" and on_stage is not None:
                on_stage(message[1], message[2], message[3])

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()

    def report_dead(dead: tuple[Any, ...]) -> None:
        emit(
            f"  a worker process died (pid {dead[0].pid}) with "
            f"{total - seen} video(s) unfinished; stopping the batch"
        )

    if _worker_processes(pool):
        threading.Thread(
            target=watch_workers,
            args=(pool,),
            kwargs={
                "stop": stop,
                "seconds": WORKER_WATCH_SECONDS,
                "on_dead": report_dead,
            },
            daemon=True,
        ).start()
    else:
        # Said once, loudly, rather than leaving a batch that looks watched and
        # is not.
        emit(
            "  note: this Python does not expose the pool's workers, so a dead "
            "worker will hang the batch instead of stopping it"
        )

    results: list[VideoResult] = []
    try:
        for payload in pool.imap_unordered(
            _worker_run, [job.to_dict() for job in jobs]
        ):
            job = next(item for item in jobs if item.video_id == payload["video_id"])
            seen += 1
            seconds = _number(payload.get("seconds"))
            ok = bool(payload["ok"])
            detail = (
                _brief(payload)
                if ok
                else f"{payload['failed_stage']}: {_clip(str(payload['error']))}"
            )
            # Failures get their line first, then the bar is redrawn over
            # nothing -- the other order leaves the line erased and the bar
            # gone until the next video ends.
            if not ok or on_done is None:
                emit(
                    format_progress(
                        done=seen,
                        total=total,
                        job=job,
                        ok=ok,
                        seconds=seconds,
                        detail=detail,
                        elapsed=time.monotonic() - started,
                    )
                )
            if on_done is not None:
                on_done(ok)
            results.append(_rebuild(payload))
    finally:
        stop.set()
        reader.join(timeout=2.0)
        pool.terminate()
        pool.join()

    return account_for_missing(results, jobs)


def _clip(text: str, *, limit: int = 110) -> str:
    """An error message short enough to sit on a progress line.

    A connection failure arrives as several hundred characters of urllib
    wrapping, and a batch of a thousand videos each printing that is a log
    nobody reads.  The whole thing is in ``failures.jsonl``; the line only has
    to say which stage and roughly what.
    """

    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 1] + "…"


def _number(value: object) -> float:
    """A float from whatever a worker sent back, without trusting its type.

    The payload crosses a process boundary as plain data, so mypy is right to
    call it `object` and the coercion has to happen somewhere.
    """

    return float(value) if isinstance(value, (int, float)) else 0.0


def _brief(payload: dict[str, object]) -> str:
    """The one-line gist of a finished video, from its last stage."""

    stages = payload.get("stages")
    if not isinstance(stages, dict) or not stages:
        return ""
    final = stages.get("s11-compose")
    if isinstance(final, dict):
        return (
            f"utt={final.get('utterances', '?')} faces={final.get('faces', '?')} "
            f"gates={final.get('gates_failed', '?')}"
        )
    return f"{len(stages)} stages"


def _rebuild(payload: dict[str, object]) -> VideoResult:
    """A VideoResult from the dict a worker sent back."""

    stages = payload.get("stages")
    logs = [
        StageLog(
            stage=str(name),
            skipped=bool(item.get("skipped", False)),
            seconds=_number(item.get("seconds")),
            summary={k: v for k, v in item.items() if k not in {"skipped", "seconds"}},
        )
        for name, item in (stages.items() if isinstance(stages, dict) else [])
        if isinstance(item, dict)
    ]
    return VideoResult(
        video_id=str(payload["video_id"]),
        ok=bool(payload["ok"]),
        seconds=_number(payload.get("seconds")),
        stages=logs,
        error=str(payload.get("error", "")),
        failed_stage=str(payload.get("failed_stage", "")),
    )
