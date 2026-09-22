"""Tests for the S5 score reader.

It is a diagnostic, and the thing a diagnostic must not do is be wrong in the
direction that looks like good news.  Every assertion here is about a number a
person would act on: a peak read off the wrong field, a threshold applied to
the wrong side, a checkpoint warning that never fires.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_asd.py"


@pytest.fixture(scope="module")
def check_asd() -> ModuleType:
    """The script, loaded from its path -- ``scripts/`` is not a package."""

    spec = importlib.util.spec_from_file_location("check_asd", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_video(
    root: Path,
    video: str,
    tracks: dict[int, list[float]],
    *,
    load: dict[str, int] | None = None,
) -> None:
    stage = root / "work" / video / "s5-asd"
    stage.mkdir(parents=True)
    with (stage / "speaking.jsonl").open("w", encoding="utf-8") as handle:
        for track_id, probabilities in tracks.items():
            handle.write(
                json.dumps(
                    {
                        "track_id": track_id,
                        "samples": [
                            {"time": index * 0.04, "probability": value}
                            for index, value in enumerate(probabilities)
                        ],
                    }
                )
                + "\n"
            )
    if load is not None:
        (stage / "summary.json").write_text(
            json.dumps({"backend": {"model": "loconet", "checkpoint_load": load}}),
            encoding="utf-8",
        )


def test_the_peak_is_the_peak_not_the_last_value(
    tmp_path: Path, check_asd: ModuleType
) -> None:
    """Off by one here reports a working frontend as broken, or the reverse."""

    _write_video(tmp_path, "v1", {1: [0.1, 0.9, 0.2]})
    ((track_id, peak, mean, at, frames),) = check_asd.read_tracks(
        tmp_path / "work/v1/s5-asd/speaking.jsonl"
    )
    assert (track_id, frames) == (1, 3)
    assert peak == pytest.approx(0.9)
    assert at == pytest.approx(0.04)  # the middle sample, not the first
    assert mean == pytest.approx(0.4)


def test_a_track_with_no_samples_is_not_a_zero(
    tmp_path: Path, check_asd: ModuleType
) -> None:
    """A face the window planner never scored is absent, not silent.

    Counting it as a 0.0 would drag the mean down and make a working run look
    like a failing one.
    """

    stage = tmp_path / "work" / "v1" / "s5-asd"
    stage.mkdir(parents=True)
    (stage / "speaking.jsonl").write_text(
        json.dumps({"track_id": 7, "samples": []})
        + "\n"
        + json.dumps({"track_id": 8, "samples": [{"time": 0.0, "probability": 0.9}]})
        + "\n",
        encoding="utf-8",
    )
    rows = check_asd.read_tracks(stage / "speaking.jsonl")
    assert [row[0] for row in rows] == [8]


def test_a_run_that_never_clears_the_association_score_exits_nonzero(
    tmp_path: Path, check_asd: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    """The smoke8 shape: every track under 0.05, which is why S6 assigns nothing."""

    _write_video(tmp_path, "v1", {1: [0.0003] * 50, 2: [0.0004] * 50 + [0.0114]})

    status = check_asd.main([str(tmp_path)])
    output = capsys.readouterr().out
    assert status == 1
    assert "utt=0" in output
    assert "best score anywhere: 0.0114" in output


def test_a_run_with_one_confident_track_exits_zero(
    tmp_path: Path, check_asd: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_video(tmp_path, "v1", {1: [0.02] * 50, 2: [0.02] * 25 + [0.88] + [0.02] * 24})

    assert check_asd.main([str(tmp_path)]) == 0
    assert "1 peak above 0.5" in capsys.readouterr().out


def test_a_half_loaded_checkpoint_is_called_out(
    tmp_path: Path, check_asd: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    """Untrained weights give small scores by a different road, and the scores
    themselves cannot say so."""

    _write_video(tmp_path, "v1", {1: [0.88]}, load={"missing": 311, "unexpected": 311})
    check_asd.main([str(tmp_path)])
    assert "untrained weights" in capsys.readouterr().out


def test_a_clean_checkpoint_load_is_not_flagged(
    tmp_path: Path, check_asd: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    """Guards the guard: the warning has to be able to stay quiet."""

    _write_video(tmp_path, "v1", {1: [0.88]}, load={"missing": 0, "unexpected": 0})
    check_asd.main([str(tmp_path)])
    output = capsys.readouterr().out
    assert "missing 0, unexpected 0" in output
    assert "untrained weights" not in output


def test_a_root_with_no_s5_output_is_not_a_crash(
    tmp_path: Path, check_asd: ModuleType
) -> None:
    assert check_asd.main([str(tmp_path)]) == 2


def test_a_missing_root_is_an_error(tmp_path: Path, check_asd: ModuleType) -> None:
    assert check_asd.main([str(tmp_path / "nope")]) == 2
