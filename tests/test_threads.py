"""Tests for the per-worker thread budget.

The thing being prevented is not a crash and does not show up in any output: a
worker that asks the machine for a hundred threads to run one video looks
exactly like a worker that is busy, right up until you notice the card it is
feeding is at three percent.

So the arithmetic is pinned here, and the two ways of getting it wrong are
pinned too -- overriding an operator who set the variables themselves, and
setting them too late to matter.
"""

from __future__ import annotations

import os

import pytest

from avannotate.threads import VARIABLES, cap, cap_opencv, share_for


@pytest.fixture(autouse=True)
def _no_inherited_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start from a machine that has not been configured, so `setdefault` sets."""

    for name in VARIABLES:
        monkeypatch.delenv(name, raising=False)


def test_the_share_is_the_machine_divided_by_the_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: 128)

    assert share_for(16) == 8
    assert share_for(4) == 32
    assert share_for(1) == 128


def test_the_share_is_never_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """More workers than cores is allowed -- the pool does not refuse it -- and
    a thread count of zero is not a way to say so."""

    monkeypatch.setattr(os, "cpu_count", lambda: 8)

    assert share_for(64) == 1


def test_a_machine_that_will_not_say_gets_one_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: None)

    assert share_for(4) == 1


def test_capping_sets_every_variable_the_stack_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Which one a library honours depends on how it was built."""

    monkeypatch.setattr(os, "cpu_count", lambda: 128)

    assert cap(16) == 8

    for name in VARIABLES:
        assert os.environ[name] == "8", f"{name} was left alone"


def test_an_operator_who_set_one_keeps_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """They know something this does not, and this runs after they said it."""

    monkeypatch.setattr(os, "cpu_count", lambda: 128)
    monkeypatch.setenv("OMP_NUM_THREADS", "3")

    cap(16)

    assert os.environ["OMP_NUM_THREADS"] == "3"
    assert os.environ["MKL_NUM_THREADS"] == "8", "the rest are still divided"


def test_opencv_is_safe_to_cap_without_opencv() -> None:
    """S0 and S11 never import it, and a batch runs whatever stages it is given."""

    cap_opencv()


# --------------------------------------------------------------------------- #
# and that a worker actually gets it
# --------------------------------------------------------------------------- #


class _Counter:
    value = 0


class _Lock:
    def __enter__(self) -> _Lock:
        return self

    def __exit__(self, *_: object) -> bool:
        return False


def test_a_started_worker_is_capped_before_it_imports_anything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Which is the only moment it can be done.

    The libraries read these at import and keep the pool; a variable changed
    afterwards is a variable nothing reads.
    """

    from avannotate import batch

    monkeypatch.setattr(os, "cpu_count", lambda: 128)

    # None rather than an index, so the CUDA_VISIBLE_DEVICES the real thing
    # sets in this process is not set in this one.
    batch._worker_init(
        (None, None, None, None),
        _Counter(),
        _Lock(),
        None,
        "/tmp",
        ("s0-preprocess",),
        False,
    )

    assert os.environ["OMP_NUM_THREADS"] == "32", "four workers on 128 cores"
