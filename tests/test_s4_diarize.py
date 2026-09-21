"""Tests for stage S4.

DiariZen is not installed here and cannot be, so the stage is driven by a stub
diarizer.  What that covers is everything the stage owns: the ordering of clamp,
merge and drop, the resume record, the summary, and the failure modes.  What it
does not cover is the ten-line adapter that calls DiariZen, which is why that
adapter is a separate module with its own docstring naming what to verify.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from avannotate.audio.diarize import DiarizerError, build_diarizer
from avannotate.audio.types import DiarizationResult, SpeakerTurn
from avannotate.stages import s0_preprocess, s4_diarize
from avannotate.stages.base import StageContext


class _StubDiarizer:
    """Reports the turns it was constructed with, ignoring the audio."""

    name = "stub"

    def __init__(self, turns: tuple[SpeakerTurn, ...]) -> None:
        self.turns = turns
        self.calls = 0

    def diarize(self, audio: Path) -> DiarizationResult:
        self.calls += 1
        return DiarizationResult(turns=self.turns, metadata={"backend": self.name})


def _context(source: Path, root: Path, **config: object) -> StageContext:
    return StageContext(
        video_id=source.stem, source=source, work_dir=root / "work" / source.stem,
        config=config,
    )


@pytest.fixture
def staged(single_shot_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """S0 run, and a stub diarizer wired in.  The clip is 3 seconds long."""

    context = _context(single_shot_video, tmp_path)
    s0_preprocess.run(context)

    def install(turns: tuple[SpeakerTurn, ...]) -> _StubDiarizer:
        stub = _StubDiarizer(turns)
        monkeypatch.setattr(s4_diarize, "build_diarizer", lambda _: stub)
        return stub

    return context, install


# --------------------------------------------------------------------------- #
# postprocessing order
# --------------------------------------------------------------------------- #


def test_short_turns_are_merged_before_being_dropped() -> None:
    """The ordering that matters: dropping first would lose the pieces that
    merging was about to join into a turn long enough to keep."""

    config = s4_diarize.S4Config(merge_gap_seconds=0.2, min_turn_seconds=0.5)
    # Four 0.3s turns separated by 0.1s gaps: individually under the floor,
    # together 1.5s of continuous speech.
    raw = DiarizationResult(
        turns=tuple(
            SpeakerTurn("spk_00", index * 0.4, index * 0.4 + 0.3) for index in range(4)
        )
    )
    result, counts = s4_diarize.postprocess(raw, duration=10.0, config=config)

    assert counts["dropped_short"] == 0
    assert len(result.turns) == 1
    assert result.turns[0].duration == pytest.approx(1.5)


def test_short_turns_that_stay_short_are_dropped() -> None:
    config = s4_diarize.S4Config(merge_gap_seconds=0.2, min_turn_seconds=0.5)
    raw = DiarizationResult(turns=(SpeakerTurn("spk_00", 0.0, 0.1),))
    result, counts = s4_diarize.postprocess(raw, duration=10.0, config=config)

    assert result.turns == ()
    assert counts["dropped_short"] == 1


def test_turns_past_the_video_are_clamped_not_kept() -> None:
    """The audio is longer than the video; a turn can sit past the picture."""

    config = s4_diarize.S4Config(min_turn_seconds=0.01)
    raw = DiarizationResult(
        turns=(
            SpeakerTurn("spk_00", 0.0, 1.0),
            SpeakerTurn("spk_01", 2.0, 4.0),  # the clip is 3s
        )
    )
    result, counts = s4_diarize.postprocess(raw, duration=3.0, config=config)

    assert [(t.start, t.end) for t in result.turns] == [(0.0, 1.0), (2.0, 3.0)]
    assert counts["outside_video"] == 0  # trimmed, not discarded


def test_counts_account_for_every_turn() -> None:
    config = s4_diarize.S4Config(merge_gap_seconds=0.2, min_turn_seconds=0.5)
    raw = DiarizationResult(
        turns=(
            SpeakerTurn("spk_00", 0.0, 0.3),
            SpeakerTurn("spk_00", 0.35, 0.65),  # merges with the previous
            SpeakerTurn("spk_01", 5.0, 5.1),  # too short, dropped
            SpeakerTurn("spk_01", 9.0, 20.0),  # clamped to the clip
        )
    )
    _, counts = s4_diarize.postprocess(raw, duration=10.0, config=config)
    assert counts["raw_turns"] == 4
    assert counts["clamped_turns"] == 4
    assert counts["merged_turns"] == 3
    assert counts["dropped_short"] == 1
    assert counts["turns"] == 2


# --------------------------------------------------------------------------- #
# the stage
# --------------------------------------------------------------------------- #


def test_run_writes_turns_and_summary(staged: Any) -> None:
    context, install = staged
    stub = install(
        (
            SpeakerTurn("spk_00", 0.0, 1.0),
            SpeakerTurn("spk_01", 0.5, 1.5),
            SpeakerTurn("spk_00", 2.0, 2.5),
        )
    )
    result = s4_diarize.run(context)

    assert not result.skipped
    assert stub.calls == 1
    assert result.summary["speakers"] == 2

    turns = s4_diarize.load_turns(context)
    assert len(turns) == 3
    assert [t.speaker for t in turns] == ["spk_00", "spk_01", "spk_00"]


def test_summary_reports_the_overlap_ratio(staged: Any) -> None:
    """The number the rest of the design turns on."""

    context, install = staged
    install((SpeakerTurn("spk_00", 0.0, 2.0), SpeakerTurn("spk_01", 1.0, 3.0)))
    s4_diarize.run(context)

    summary = json.loads((context.work_dir / "s4-diarize" / "summary.json").read_text())
    assert summary["diarization"]["overlap_seconds"] == pytest.approx(1.0)
    assert summary["diarization"]["overlap_ratio"] == pytest.approx(1.0 / 3.0, abs=5e-5)
    assert summary["timeline"]["video_duration"] < summary["timeline"]["audio_duration"]


def test_backend_metadata_reaches_the_summary_and_the_loader(staged: Any) -> None:
    context, install = staged
    install((SpeakerTurn("spk_00", 0.0, 1.0),))
    s4_diarize.run(context)

    assert s4_diarize.load_result(context).metadata == {"backend": "stub"}


def test_second_run_skips(staged: Any) -> None:
    context, install = staged
    install((SpeakerTurn("spk_00", 0.0, 1.0),))
    assert not s4_diarize.run(context).skipped
    assert s4_diarize.run(context).skipped


def test_force_reruns(staged: Any) -> None:
    context, install = staged
    install((SpeakerTurn("spk_00", 0.0, 1.0),))
    s4_diarize.run(context)
    assert not s4_diarize.run(context, force=True).skipped


def test_a_config_change_invalidates_the_cache(staged: Any) -> None:
    context, install = staged
    install((SpeakerTurn("spk_00", 0.0, 1.0),))
    s4_diarize.run(context)

    rerun = s4_diarize.run(
        _context(context.source, context.work_dir.parents[1], min_turn_seconds=0.9)
    )
    assert not rerun.skipped
    assert "inputs changed" in rerun.reason


def test_a_diarizer_that_cannot_be_built_marks_the_stage_failed(
    single_shot_video: Path, tmp_path: Path
) -> None:
    """The batch driver must retry it rather than skip it as done."""

    context = _context(single_shot_video, tmp_path, backend="diarizen")
    s0_preprocess.run(context)

    with pytest.raises(DiarizerError):
        s4_diarize.run(context)

    state = json.loads((context.work_dir / "stage_state.json").read_text())
    record = next(item for item in state["stages"] if item["stage"] == "s4-diarize")
    assert record["status"] == "failed"
    assert "DiariZen" in record["error"]


def test_running_before_s0_is_a_clear_error(single_shot_video: Path, tmp_path: Path) -> None:
    context = _context(single_shot_video, tmp_path)
    with pytest.raises(FileNotFoundError, match="run s0-preprocess first"):
        s4_diarize.run(context)


def test_loaders_fail_loudly_when_the_stage_has_not_run(tmp_path: Path) -> None:
    context = StageContext(
        video_id="x", source=tmp_path / "x.mp4", work_dir=tmp_path / "work" / "x"
    )
    with pytest.raises(FileNotFoundError, match="run s4-diarize first"):
        s4_diarize.load_turns(context)
    with pytest.raises(FileNotFoundError, match="run s4-diarize first"):
        s4_diarize.turns_path(context)


# --------------------------------------------------------------------------- #
# the adapter's contract, without the package
# --------------------------------------------------------------------------- #


def test_diarizen_absent_gives_install_instructions(absent_module) -> None:
    """The error path, exercised whether or not this machine has the package.

    It used to rely on DiariZen being absent, which is true on a development
    box and false on any machine that runs S4.
    """

    with absent_module("diarizen"), pytest.raises(
        DiarizerError, match="github.com/BUTSpeechFIT/DiariZen"
    ):
        build_diarizer({"backend": "diarizen"})


def test_an_unknown_backend_is_rejected() -> None:
    with pytest.raises(DiarizerError, match="unknown diarization backend"):
        build_diarizer({"backend": "pyannote"})


def test_there_is_no_token_to_configure() -> None:
    """DiariZen has no token argument and needs none.

    Its checkpoints, and the wespeaker embedding model pulled alongside them,
    are all ungated -- verified by anonymous request. The field used to exist
    here and was passed as ``use_auth_token``, which is not a parameter of
    ``from_pretrained(repo_id, cache_dir=None, rttm_out_dir=None)``: it raised
    ``TypeError`` on every construction and the bare ``except TypeError`` around
    it swallowed that. A knob that cannot be turned is worse than no knob,
    because it still looks like one.
    """

    config = s4_diarize.S4Config.from_mapping({"huggingface_token": "hf_secret"})

    assert "huggingface_token" not in config.to_dict()
    assert "hf_secret" not in json.dumps(config.cache_key())
