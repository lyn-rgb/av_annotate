"""Tests for the stage-major driver, `scripts/run_corpus.sh`.

The models are not run here.  What is run is the script itself against a stub
that stands in for `avannotate.cli`, because the mistakes this driver can make
are all in its control flow and none of them are visible from a successful run:

* the stage list is derived from the package, so a stage that is added and then
  silently never scheduled shows up only as a missing file much later;
* a stage that fails for *every* video must stop the run -- otherwise the night
  is spent running later stages over videos that have no input, each reporting
  its own version of the same failure;
* and the log a failure points at has to be the log that failure is in.

The stub also keeps the test off the real stages: this driver is invoked as a
subprocess, so a real one would load real models.

`--data` is always passed and points into `tmp_path`.  `run_batch.sh` links the
clearvoice checkpoint into whatever directory it ends up in, and that directory
should not be the checkout.
"""

from __future__ import annotations

import os
import re
import subprocess
import textwrap
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_corpus.sh"

#: Two implemented, one not, so the exclusion is exercised rather than assumed.
_STAGES = """\
  implemented  s0-preprocess
  implemented  s1-faces
  not yet      s2-tracks
  implemented  s3-cluster
  implemented  s4-diarize
"""

#: Substituted with `replace` rather than `%` or `.format`: the stub is shell,
#: so it is full of `$stage` and `${FAIL_STAGE:-...}` and either formatter would
#: eat one or die on the other.
_STUB = """\
#!/usr/bin/env bash
for argument in "$@"; do
    if [[ "$argument" == "stages" ]]; then
        cat <<'LIST'
@STAGES@LIST
        exit 0
    fi
    if [[ "$argument" == "report" ]]; then
        printf 'videos    3 (3 ok, 0 failed)\n'
        exit "${REPORT_STATUS:-0}"
    fi
done

stage=""
previous=""
for argument in "$@"; do
    [[ "$previous" == "--stage" ]] && stage="$argument"
    previous="$argument"
done

printf 'videos   3\\nstages    %s\\n\\n' "$stage"
echo "       0s  start  clipA  $stage"

if [[ "$stage" == "${FAIL_STAGE:-__none__}" ]]; then
    printf 'videos   0 ok, 3 failed\\ntime     1s\\n'
    exit 1
fi
if [[ "$stage" == "${PARTIAL_STAGE:-__none__}" ]]; then
    printf 'videos   2 ok, 1 failed\\ntime     1s\\n'
    exit 1
fi
if [[ "$stage" == "${CRASH_STAGE:-__none__}" ]]; then
    echo "Traceback (most recent call last):" >&2
    exit 7
fi
printf 'videos   3 ok, 0 failed\\ntime     1s\\n'
""".replace("@STAGES@", _STAGES)

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    return _ANSI.sub("", text)


class _Corpus:
    """A stub `avannotate.cli` and the directories to run it against."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.data = root / "videos"
        self.output = root / "out"
        self.data.mkdir(parents=True, exist_ok=True)
        self.output.mkdir(parents=True, exist_ok=True)
        (self.data / "list.txt").write_text("clipA\nclipB\n", encoding="utf-8")

        self.stub = root / "bin" / "stubpy"
        self.stub.parent.mkdir(parents=True, exist_ok=True)
        self.stub.write_text(textwrap.dedent(_STUB), encoding="utf-8")
        self.stub.chmod(0o755)

    def run(self, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
        command = [
            "bash",
            str(SCRIPT),
            "--data",
            str(self.data),
            "--list",
            str(self.data / "list.txt"),
            "--output",
            str(self.output),
            *args,
        ]
        return subprocess.run(
            command,
            cwd=self.root,
            env=os.environ | {"PYTHON": str(self.stub)} | env,
            capture_output=True,
            text=True,
        )

    def plan(self, *args: str, **env: str) -> list[str]:
        result = self.run("--dry-run", *args, **env)
        assert result.returncode == 0, result.stderr
        return re.findall(r"^\s+\d+/\d+\s+(\S+)$", _plain(result.stdout), re.MULTILINE)

    def logs(self) -> set[str]:
        directory = self.output / "logs"
        return {path.name for path in directory.glob("*.log")} if directory.is_dir() else set()


@pytest.fixture
def corpus(tmp_path: Path) -> _Corpus:
    return _Corpus(tmp_path)


def test_plan_comes_from_the_package_in_order(corpus: _Corpus) -> None:
    assert corpus.plan() == ["s0-preprocess", "s1-faces", "s3-cluster", "s4-diarize"]


def test_unimplemented_stages_are_not_scheduled(corpus: _Corpus) -> None:
    assert "s2-tracks" not in corpus.plan()


def test_explicit_stages_keep_the_order_given(corpus: _Corpus) -> None:
    assert corpus.plan("--stages", "s4-diarize,s0-preprocess") == ["s4-diarize", "s0-preprocess"]


def test_from_skips_the_stages_before_it(corpus: _Corpus) -> None:
    assert corpus.plan("--from", "s3-cluster") == ["s3-cluster", "s4-diarize"]


def test_from_rejects_a_stage_that_is_not_in_the_plan(corpus: _Corpus) -> None:
    result = corpus.run("--from", "s9-nope", "--dry-run")

    assert result.returncode == 2
    assert "is not one of" in _plain(result.stderr)


def test_a_run_that_works_reports_every_stage(corpus: _Corpus) -> None:
    result = corpus.run()

    assert result.returncode == 0
    assert corpus.logs() == {
        "s0-preprocess.log",
        "s1-faces.log",
        "s3-cluster.log",
        "s4-diarize.log",
    }
    assert "4 stages" in _plain(result.stdout)


def test_a_stage_that_fails_every_video_stops_the_run(corpus: _Corpus) -> None:
    """The one that saves the night: no later stage is attempted."""

    result = corpus.run(FAIL_STAGE="s3-cluster")

    assert result.returncode == 1
    assert "s3-cluster failed for every video" in _plain(result.stderr)
    assert "s4-diarize.log" not in corpus.logs(), "the run should have stopped at s3"


def test_a_crash_without_a_summary_also_stops_the_run(corpus: _Corpus) -> None:
    result = corpus.run(CRASH_STAGE="s1-faces")

    assert result.returncode == 1
    assert "s1-faces exited 7" in _plain(result.stderr)
    assert "s3-cluster.log" not in corpus.logs()


def test_the_abort_points_at_the_failing_stage_s_log(corpus: _Corpus) -> None:
    """Not at the last stage in the plan, which is what the loop would hold."""

    result = corpus.run(FAIL_STAGE="s1-faces")

    assert "logs/s1-faces.log" in _plain(result.stderr)


def test_partial_failure_does_not_stop_the_run(corpus: _Corpus) -> None:
    """Some videos failing is normal; every video failing is a broken stage.

    This is the distinction the abort rests on, and it is the one that is easy
    to get wrong in the other direction: `run_batch.sh` exits nonzero whenever
    *anything* failed, so keying the abort on the exit status would stop a
    thousand-video run because one video could not get through S3.
    """

    result = corpus.run(PARTIAL_STAGE="s3-cluster")

    assert "s3-cluster" not in _plain(result.stderr), "one failed video is not a broken stage"
    assert "s4-diarize.log" in corpus.logs(), "the run should have carried on"


def test_the_combined_log_survives_a_second_run(corpus: _Corpus) -> None:
    """Per-stage logs are replaced; run.log is the record of the whole night."""

    corpus.run("--stages", "s0-preprocess")
    corpus.run("--stages", "s1-faces")

    combined = (corpus.output / "run.log").read_text(encoding="utf-8")

    assert combined.count("run started") == 2
    assert "videos   3 ok, 0 failed" in combined
    # And the per-stage log holds only the latest attempt, not both.
    assert (corpus.output / "logs" / "s0-preprocess.log").read_text().count("start  clipA") == 1


# --------------------------------------------------------------------------- #
# the written account
# --------------------------------------------------------------------------- #


def test_the_corpus_report_is_written_at_the_end(corpus: _Corpus) -> None:
    """Even a run that stopped early gets one -- that is the run that needs it."""

    result = corpus.run()

    assert "videos    3 (3 ok, 0 failed)" in result.stdout


def test_the_report_is_written_even_when_the_run_aborted(corpus: _Corpus) -> None:
    result = corpus.run(FAIL_STAGE="s3-cluster")

    assert result.returncode == 1
    assert "videos    3 (3 ok, 0 failed)" in result.stdout, "the report was skipped"


def test_a_corpus_with_failures_fails_the_script(corpus: _Corpus) -> None:
    """The exit status is the report's, because that is the one that counts
    videos rather than stages.

    Every batch invocation exits nonzero when anything failed, so the driver
    cannot use those -- it would stop at the first stage that lost a video.  The
    report is asked at the end, once, and answers for the corpus.
    """

    result = corpus.run(REPORT_STATUS="1")

    assert result.returncode == 1


def test_a_clean_corpus_exits_zero(corpus: _Corpus) -> None:
    assert corpus.run().returncode == 0
