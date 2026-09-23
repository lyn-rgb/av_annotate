"""Tests for the resume record.

The resume check is the difference between a batch that survives a crash and one
that silently ships half-written artifacts, so each way an artifact can go bad
gets its own case.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from avannotate.stages import available_stages, get_stage, uses_gpu
from avannotate.stages.base import (
    Artifact,
    StageRecord,
    StageState,
    hash_file,
    hash_payload,
    read_json,
    write_json,
)


def _record(work: Path, stage: str = "s0", *, version: str = "v1", input_hash: str = "abc"):
    artifact_path = work / f"{stage}.txt"
    artifact_path.write_text("payload", encoding="utf-8")
    return StageRecord(
        stage=stage,
        status="ok",
        code_version=version,
        config_hash="cfg",
        input_hash=input_hash,
        artifacts=(Artifact.capture(work, artifact_path),),
    )


def test_hash_file_is_content_addressed(tmp_path: Path) -> None:
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_text("same", encoding="utf-8")
    second.write_text("same", encoding="utf-8")
    assert hash_file(first) == hash_file(second)

    second.write_text("different", encoding="utf-8")
    assert hash_file(first) != hash_file(second)


def test_hash_payload_ignores_key_order() -> None:
    """A rebuilt dict must not look like a config change, or nothing ever skips."""

    assert hash_payload({"a": 1, "b": 2}) == hash_payload({"b": 2, "a": 1})
    assert hash_payload({"a": 1}) != hash_payload({"a": 2})


def test_write_json_is_atomic_and_leaves_no_temp_file(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "payload.json"
    write_json(target, {"k": "v"})
    assert read_json(target) == {"k": "v"}
    assert list(target.parent.iterdir()) == [target]


def test_artifact_round_trip(tmp_path: Path) -> None:
    artifact = _record(tmp_path).artifacts[0]
    assert Artifact.from_dict(artifact.to_dict()) == artifact


def test_record_round_trip(tmp_path: Path) -> None:
    record = _record(tmp_path)
    assert StageRecord.from_dict(record.to_dict()) == record


def test_first_run_has_no_previous_record(tmp_path: Path) -> None:
    state = StageState(tmp_path)
    assert state.reason_to_run("s0", code_version="v1", input_hash="abc") == "no previous run"


def test_an_unchanged_stage_is_skippable(tmp_path: Path) -> None:
    state = StageState(tmp_path)
    state.save(_record(tmp_path))
    assert StageState(tmp_path).reason_to_run("s0", code_version="v1", input_hash="abc") is None


def test_a_failed_run_is_retried(tmp_path: Path) -> None:
    state = StageState(tmp_path)
    state.save(replace(_record(tmp_path), status="failed"))
    reason = StageState(tmp_path).reason_to_run("s0", code_version="v1", input_hash="abc")
    assert reason is not None and "failed" in reason


def test_code_version_change_invalidates(tmp_path: Path) -> None:
    state = StageState(tmp_path)
    state.save(_record(tmp_path, version="v1"))
    reason = StageState(tmp_path).reason_to_run("s0", code_version="v2", input_hash="abc")
    assert reason is not None and "code changed" in reason


def test_input_change_invalidates(tmp_path: Path) -> None:
    state = StageState(tmp_path)
    state.save(_record(tmp_path, input_hash="abc"))
    reason = StageState(tmp_path).reason_to_run("s0", code_version="v1", input_hash="def")
    assert reason == "inputs changed"


def test_missing_artifact_invalidates(tmp_path: Path) -> None:
    state = StageState(tmp_path)
    state.save(_record(tmp_path))
    (tmp_path / "s0.txt").unlink()
    reason = StageState(tmp_path).reason_to_run("s0", code_version="v1", input_hash="abc")
    assert reason is not None and "missing artifact" in reason


def test_resized_artifact_invalidates(tmp_path: Path) -> None:
    state = StageState(tmp_path)
    state.save(_record(tmp_path))
    (tmp_path / "s0.txt").write_text("payload and more", encoding="utf-8")
    reason = StageState(tmp_path).reason_to_run("s0", code_version="v1", input_hash="abc")
    assert reason is not None and "changed size" in reason


def test_same_size_but_different_content_invalidates(tmp_path: Path) -> None:
    """The case a size check alone would miss -- a truncated-then-padded WAV."""

    state = StageState(tmp_path)
    state.save(_record(tmp_path))
    (tmp_path / "s0.txt").write_text("payloaD", encoding="utf-8")
    reason = StageState(tmp_path).reason_to_run("s0", code_version="v1", input_hash="abc")
    assert reason is not None and "changed content" in reason


def test_corrupt_state_file_starts_over_rather_than_crashing(tmp_path: Path) -> None:
    (tmp_path / "stage_state.json").write_text("{not json", encoding="utf-8")
    state = StageState(tmp_path)
    assert state.records == {}
    assert state.reason_to_run("s0", code_version="v1", input_hash="abc") == "no previous run"


def test_state_survives_multiple_stages(tmp_path: Path) -> None:
    state = StageState(tmp_path)
    state.save(_record(tmp_path, "s0"))
    state.save(_record(tmp_path, "s1", input_hash="xyz"))

    reloaded = StageState(tmp_path)
    assert set(reloaded.records) == {"s0", "s1"}
    assert reloaded.reason_to_run("s0", code_version="v1", input_hash="abc") is None
    assert reloaded.reason_to_run("s1", code_version="v1", input_hash="xyz") is None


def test_state_file_records_the_schema_version(tmp_path: Path) -> None:
    StageState(tmp_path).save(_record(tmp_path))
    payload = json.loads((tmp_path / "stage_state.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == "avannotate-stage-state-v1"
    assert payload["package_version"]


@pytest.mark.parametrize("status", ["ok", "failed"])
def test_record_status_round_trips(tmp_path: Path, status: str) -> None:
    record = StageRecord(
        stage="s0", status=status, code_version="v1", config_hash="c", input_hash="i"
    )
    assert StageRecord.from_dict(record.to_dict()).status == status


# --------------------------------------------------------------------------- #
# what each stage runs on
# --------------------------------------------------------------------------- #
#
# `USES_GPU` decides how many videos run at once: one worker per card for a
# stage that uses one, sixteen for a stage that does not.  The failure it
# guards against is silent and expensive -- a stage that forgot to declare
# itself gets the GPU answer by default, and the corpus runs at a quarter of
# the speed it could with nothing to show for it.

#: The stages whose work is arithmetic, ffmpeg, or both.
_CPU_STAGES = frozenset(
    {"s0-preprocess", "s2-tracks", "s3-cluster", "s6-associate", "s11-compose"}
)


def test_every_implemented_stage_declares_what_it_runs_on() -> None:
    for name in available_stages():
        module = get_stage(name)
        assert hasattr(module, "USES_GPU"), f"{name} does not say whether it uses a GPU"
        assert isinstance(module.USES_GPU, bool), f"{name}'s USES_GPU is not a bool"


def test_the_stages_that_use_no_card_are_the_ones_that_use_no_card() -> None:
    """Not a tautology: it is the list the defaults are sized from.

    Flipping a stage here by mistake does not fail anything visibly -- it just
    sizes its pool wrongly, which is four videos at a time instead of sixteen,
    or the other way round.
    """

    cpu = {name for name in available_stages() if not uses_gpu(name)}

    assert cpu == _CPU_STAGES


def test_an_unknown_stage_is_assumed_to_want_a_card() -> None:
    """Nobody has described its work, so nothing should size a pool for it."""

    assert uses_gpu("s99-nonsense") is True
