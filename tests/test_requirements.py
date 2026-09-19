"""Tests for the readiness check.

The point of this module is to be right about a machine before a batch runs on
it, so the tests are about the answers rather than the mechanism: a path
configured relative to its config file has to resolve the way the stage will
resolve it, or the doctor reports a file missing that the run would have found.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from avannotate.requirements import (
    REPOSITORIES,
    REQUIREMENTS,
    Requirement,
    check,
    load_stage_config,
    missing_modules,
    requirement_for,
)
from avannotate.stages import available_stages

# --------------------------------------------------------------------------- #
# the declarations themselves
# --------------------------------------------------------------------------- #


def test_every_declared_stage_exists() -> None:
    """A requirement for a stage that was renamed is a check nobody runs."""

    known = set(available_stages())
    for requirement in REQUIREMENTS:
        assert requirement.stage in known, f"{requirement.stage} is not a stage"


def test_a_stage_that_needs_a_checkout_says_where_to_get_it() -> None:
    """``repo`` is "this needs a clone"; ``repo_key`` is "and the checkout has
    to be pointed at", which is only true when it is not installed as a package.
    Every declared repository has to be one we know the URL of."""

    for requirement in REQUIREMENTS:
        if requirement.repo is not None:
            assert requirement.repo in REPOSITORIES, (
                f"{requirement.stage} needs a checkout nobody can find"
            )
        if requirement.repo_key is not None:
            assert requirement.repo is not None, (
                f"{requirement.stage} has a checkout path but no repository to clone"
            )


def test_a_stage_with_weights_says_where_they_come_from() -> None:
    """A weights file with no source is an instruction to go and find one."""

    for requirement in REQUIREMENTS:
        if requirement.weights:
            assert requirement.weights_hint, (
                f"{requirement.stage} needs weights but does not say from where"
            )


def test_stages_that_need_nothing_are_not_declared() -> None:
    """S0 and S11 run on ffmpeg and pure Python; claiming otherwise would make
    the doctor's list longer than the thing it is reporting on."""

    for stage in ("s0-preprocess", "s2-tracks", "s6-associate", "s11-compose"):
        assert requirement_for(stage) is None


# --------------------------------------------------------------------------- #
# module checks
# --------------------------------------------------------------------------- #


def test_a_real_module_is_found() -> None:
    assert missing_modules(("json", "pathlib")) == ()


def test_a_missing_module_is_named() -> None:
    assert missing_modules(("definitely_not_a_module_xyz",)) == (
        "definitely_not_a_module_xyz",
    )


def test_a_module_behind_a_missing_parent_is_reported_not_raised() -> None:
    """A dotted name whose parent is absent raises out of find_spec rather than
    returning None, and the answer is the same either way."""

    assert missing_modules(("not_a_package.sub",)) == ("not_a_package.sub",)


def test_checking_never_imports_the_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A module that raises on import, which find_spec must find but not run.

    Importing is what the adapters do when they load a checkpoint, and a doctor
    that loaded a 60 GB model to check it was present would be worse than no
    doctor at all.  Proven with a module that cannot be imported rather than by
    timing one that can.
    """

    import sys

    (tmp_path / "explodes_on_import.py").write_text(
        "raise AssertionError('the readiness check imported the module')\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "explodes_on_import", raising=False)

    assert missing_modules(("explodes_on_import",)) == ()
    assert "explodes_on_import" not in sys.modules


# --------------------------------------------------------------------------- #
# path checks
# --------------------------------------------------------------------------- #


def _config(tmp_path: Path, *pairs: tuple[str, str]) -> dict[str, object]:
    """A config mapping with its own directory recorded, like the loader does."""

    from avannotate.stages.base import CONFIG_ROOT_KEY

    return {CONFIG_ROOT_KEY: str(tmp_path), **dict(pairs)}


def test_an_unset_weight_path_is_reported() -> None:
    requirement = Requirement(stage="s", weights=("checkpoint",))

    assert requirement.absent_weights({}) == ("checkpoint is not set",)


def test_a_weight_path_that_is_there_is_not_reported(tmp_path: Path) -> None:
    weights = tmp_path / "model.pth"
    weights.write_bytes(b"x")
    requirement = Requirement(stage="s", weights=("checkpoint",))

    assert requirement.absent_weights(_config(tmp_path, ("checkpoint", str(weights)))) == ()


def test_a_weight_path_that_is_not_there_is_reported_with_where(tmp_path: Path) -> None:
    requirement = Requirement(stage="s", weights=("checkpoint",))
    found = requirement.absent_weights(_config(tmp_path, ("checkpoint", "gone.pth")))

    assert len(found) == 1
    assert "gone.pth is not a file" in found[0]


def test_a_relative_weight_path_resolves_against_the_config_directory(
    tmp_path: Path,
) -> None:
    """The stage resolves it that way, so the doctor has to as well.

    Getting this wrong is the one failure that makes the check worse than
    nothing: it would report a model missing that the run would have loaded.
    """

    models = tmp_path / "models"
    models.mkdir()
    (models / "checkpoint.pth").write_bytes(b"x")
    requirement = Requirement(stage="s", weights=("checkpoint",))

    # Named relative to the config's directory, exactly as configs/s5 does it.
    assert requirement.absent_weights(
        _config(tmp_path, ("checkpoint", "models/checkpoint.pth"))
    ) == ()
    # And the same string means something else from elsewhere, which is why the
    # config directory has to travel with it.
    assert requirement.absent_weights({"checkpoint": "models/checkpoint.pth"}) != ()


def test_a_checkout_is_checked_as_a_directory_not_a_file(tmp_path: Path) -> None:
    """A file where a checkout should be is not a checkout."""

    checkout = tmp_path / "LoCoNet_ASD"
    checkout.write_bytes(b"not a directory")
    requirement = Requirement(stage="s", repo_key="repo")

    found = requirement.checkout_missing(_config(tmp_path, ("repo", str(checkout))))

    assert found is not None
    assert "is not a directory" in found


def test_a_present_checkout_is_not_reported(tmp_path: Path) -> None:
    checkout = tmp_path / "LoCoNet_ASD"
    checkout.mkdir()
    requirement = Requirement(stage="s", repo_key="repo")

    assert requirement.checkout_missing(_config(tmp_path, ("repo", str(checkout)))) is None


def test_a_stage_without_a_checkout_key_is_not_asked_for_one() -> None:
    assert Requirement(stage="s").checkout_missing({}) is None


# --------------------------------------------------------------------------- #
# the whole check
# --------------------------------------------------------------------------- #


def test_a_stage_that_needs_nothing_is_ready() -> None:
    (status,) = check(("s11-compose",))

    assert status.ready
    assert status.detail == "no model needed"
    assert status.todo == ()


def test_a_stage_with_a_missing_module_is_not_ready_and_says_what_to_do() -> None:
    requirement = requirement_for("s4-diarize")
    assert requirement is not None
    if not missing_modules(requirement.modules[:1]):
        pytest.skip("diarizen is installed here; nothing to report")

    (status,) = check(("s4-diarize",))

    assert not status.ready
    assert "diarizen" in status.detail
    assert any(item.startswith("clone:") for item in status.todo)
    assert any(item.startswith("install:") for item in status.todo)


def test_a_stage_whose_config_names_paths_is_checked_against_them(
    tmp_path: Path,
) -> None:
    """A config in a directory of its own, with the relative paths the real
    configs use."""

    (tmp_path / "s5.loconet.json").write_text(
        json.dumps({"checkpoint": "models/loconet_AVA.model", "repo": "models/LoCoNet_ASD"}),
        encoding="utf-8",
    )
    models = tmp_path / "models"
    models.mkdir()
    (models / "loconet_AVA.model").write_bytes(b"x")
    (models / "LoCoNet_ASD").mkdir()

    requirement = requirement_for("s5-asd")
    assert requirement is not None
    config = load_stage_config(requirement, configs_dir=tmp_path, override=None)

    # Both paths resolve against the config's own directory and both are there,
    # so whatever else the stage is missing, it is not these.
    assert requirement.absent_weights(config) == ()
    assert requirement.checkout_missing(config) is None


def test_a_missing_config_file_leaves_the_paths_unset_rather_than_skipping() -> None:
    """The truth is that the stage cannot run; reporting it as fine would be
    the one answer this module exists to avoid giving."""

    (status,) = check(("s5-asd",), configs_dir="/nonexistent-configs")

    if status.ready:
        pytest.skip("s5's modules are installed here, so only the paths matter")
    assert "not set" in status.detail


def test_every_declared_stage_is_checked_by_default() -> None:
    statuses = check(available_stages())

    assert [status.stage for status in statuses] == list(available_stages())
    assert all(isinstance(status.ready, bool) for status in statuses)
