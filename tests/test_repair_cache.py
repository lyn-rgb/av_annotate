"""Tests for the hub-cache repair.

The whole script is three lines of arithmetic around one write, so what is worth
testing is the write.  ``refs/main`` is not read by huggingface_hub -- it is used
*as a directory name*, byte for byte, which makes the exact bytes the only thing
that matters and makes a trailing newline a second fault rather than a repair.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "repair_cache.py"

#: Arbitrary but distinct revisions, the way a real cache names them.
GOOD = "aea7f62f8b0e8c5c3a68447275893afe3ca0e4c3"
PARTIAL = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"


@pytest.fixture(scope="module")
def repair_cache() -> ModuleType:
    spec = importlib.util.spec_from_file_location("repair_cache", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _repo(root: Path, *, current: str | None, sizes: dict[str, int]) -> Path:
    """A cache directory holding one repo with the given snapshots."""

    repo = root / "models--Qwen--Qwen3-VL-8B-Instruct"
    for revision, size in sizes.items():
        snapshot = repo / "snapshots" / revision
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_bytes(b"x" * size)
    (repo / "refs").mkdir(parents=True, exist_ok=True)
    if current is not None:
        (repo / "refs" / "main").write_bytes(current.encode())
    return repo


def test_the_ref_moves_to_the_snapshot_that_has_the_weights(
    tmp_path: Path, repair_cache: ModuleType
) -> None:
    """The failure this exists for: the small snapshot is the newer revision.

    A run resolved ``main``, fetched the new revision's configs, died part way
    through its weights, and left the ref naming a snapshot with nothing in it.
    """

    repo = _repo(tmp_path, current=PARTIAL, sizes={PARTIAL: 10, GOOD: 5000})

    line = repair_cache.repair(repo, dry_run=False)

    assert line is not None and GOOD[:12] in line
    assert (repo / "refs" / "main").read_bytes() == GOOD.encode()


def test_the_written_ref_has_no_trailing_newline(
    tmp_path: Path, repair_cache: ModuleType
) -> None:
    """The one byte that decides whether any of this works.

    huggingface_hub uses these bytes as the snapshot directory name.  A newline
    makes ``<revision>\\n``, which is not a directory, so every lookup misses --
    and the symptom is the same "couldn't find them in the cached files" that
    the repair was meant to cure.
    """

    repo = _repo(tmp_path, current=PARTIAL, sizes={PARTIAL: 10, GOOD: 5000})
    repair_cache.repair(repo, dry_run=False)

    written = (repo / "refs" / "main").read_bytes()
    assert not written.endswith(b"\n")
    assert written.decode() == GOOD


def test_an_already_correct_ref_is_left_alone(
    tmp_path: Path, repair_cache: ModuleType
) -> None:
    """Guards the guard: it must not rewrite what is already right, or the
    script becomes something you cannot run twice."""

    repo = _repo(tmp_path, current=GOOD, sizes={PARTIAL: 10, GOOD: 5000})

    assert repair_cache.repair(repo, dry_run=False) is None
    assert (repo / "refs" / "main").read_bytes() == GOOD.encode()


def test_a_dry_run_changes_nothing(tmp_path: Path, repair_cache: ModuleType) -> None:
    repo = _repo(tmp_path, current=PARTIAL, sizes={PARTIAL: 10, GOOD: 5000})

    line = repair_cache.repair(repo, dry_run=True)

    assert line is not None
    assert (repo / "refs" / "main").read_bytes() == PARTIAL.encode()


def test_the_root_is_the_one_the_run_uses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, repair_cache: ModuleType
) -> None:
    """Not ``~/.cache/huggingface``, which is huggingface_hub's own default and
    not what a batch reads -- run_batch.sh points HF_HOME at the checkout.

    Repairing one cache while the run reads another is a repair that reports
    success and changes nothing.
    """

    monkeypatch.delenv("HF_HOME", raising=False)
    assert repair_cache.hub_root() == SCRIPT.resolve().parents[1] / "models" / "hf" / "hub"

    monkeypatch.setenv("HF_HOME", str(tmp_path))
    assert repair_cache.hub_root() == tmp_path / "hub"
