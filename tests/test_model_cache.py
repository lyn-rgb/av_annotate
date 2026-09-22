"""Tests for the per-process model cache.

The whole value of this module is that it *does not* rebuild, so the tests are
mostly about the two ways it could rebuild when it should not -- a key that
moves when the config has not changed, and a key that fails to move when it
has.  The second is the dangerous one: a cache that keeps serving a model built
from last week's checkpoint computes every score against settings nobody chose,
and does it quietly.

Every test calls :func:`release` first.  The cache is module state shared with
whatever ran before, so a test that did not clear it would pass or fail on the
order pytest happened to collect files in.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from avannotate import model_cache
from avannotate.model_cache import key_for, model_for, release


@pytest.fixture(autouse=True)
def _empty_cache() -> None:
    release()


class _Counter:
    """A builder that records how many times it was asked to build."""

    def __init__(self) -> None:
        self.builds: list[Mapping[str, Any]] = []

    def __call__(self, config: Mapping[str, Any]) -> object:
        self.builds.append(dict(config))
        return object()

    @property
    def count(self) -> int:
        return len(self.builds)


def test_second_call_reuses_the_first_build() -> None:
    build = _Counter()
    config = {"device": "cuda", "threshold": 0.5}

    first = model_for("s5-asd", config, build)
    second = model_for("s5-asd", config, build)

    assert build.count == 1
    assert first is second


def test_key_ignores_dict_ordering() -> None:
    build = _Counter()

    model_for("s5-asd", {"a": 1, "b": 2}, build)
    model_for("s5-asd", {"b": 2, "a": 1}, build)

    assert build.count == 1, "the same config written in another order is the same config"


def test_key_separates_stages() -> None:
    build = _Counter()
    config = {"device": "cuda"}
    other = _Counter()

    model_for("s5-asd", config, build)
    model_for("s8-asr", config, other)

    assert (build.count, other.count) == (1, 1)


def test_a_changed_config_rebuilds() -> None:
    build = _Counter()

    model_for("s5-asd", {"threshold": 0.5}, build)
    model_for("s5-asd", {"threshold": 0.6}, build)

    assert build.count == 2
    assert build.builds[-1] == {"threshold": 0.6}, "the new model must see the new config"


def test_only_one_slot_is_held() -> None:
    """The cache holds one model, not one per config ever seen.

    Holding more would be the difference between a batch that starts and one
    that cannot: the ten stages' models do not fit on the card together.  So
    going back to an earlier config must rebuild, not resurrect.
    """

    build = _Counter()

    model_for("s5-asd", {"threshold": 0.5}, build)
    model_for("s5-asd", {"threshold": 0.6}, build)
    model_for("s5-asd", {"threshold": 0.5}, build)

    assert build.count == 3


def test_failed_build_leaves_nothing_behind() -> None:
    """A half-built model must not be served to the next call."""

    calls: list[int] = []

    def build(config: Mapping[str, Any]) -> object:
        calls.append(1)
        raise RuntimeError("no checkpoint")

    with pytest.raises(RuntimeError):
        model_for("s5-asd", {"threshold": 0.5}, build)

    with pytest.raises(RuntimeError):
        model_for("s5-asd", {"threshold": 0.5}, build)

    assert len(calls) == 2, "the failure must not have been cached"


def test_release_forces_a_rebuild() -> None:
    build = _Counter()
    config = {"device": "cuda"}

    model_for("s5-asd", config, build)
    release()
    model_for("s5-asd", config, build)

    assert build.count == 2


def test_release_clears_whether_or_not_it_is_holding_anything() -> None:
    """The empty case returns early, and it must return early *only* on the
    collection.

    Releasing an empty cache is what makes the first call of every stage cheap;
    releasing a full one is the whole point of the module.  Both directions are
    a line each and only one of them is obvious, so both are exercised here --
    the failure would be an early return placed a line too high, which leaves
    the cache populated and every later build serving a stale model.
    """

    build = _Counter()
    release()  # nothing held
    model_for("s5-asd", {"device": "cuda"}, build)
    release()  # something held
    model_for("s5-asd", {"device": "cuda"}, build)

    assert build.count == 2


def test_key_is_json_not_repr() -> None:
    """``repr`` would put a memory address in the key and never match twice."""

    key = key_for("s5-asd", {"checkpoint": "/models/best.pth"})

    assert key == 's5-asd|{"checkpoint": "/models/best.pth"}'
    assert "0x" not in key


def test_key_survives_a_path() -> None:
    """Stage configs carry ``Path`` objects; ``json`` does not take them raw."""

    module = Path(model_cache.__file__)
    key = key_for("s1-faces", {"model_dir": module})

    assert key.startswith("s1-faces|")
    assert str(module) in key
