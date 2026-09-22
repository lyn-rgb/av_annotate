"""Tests for stage S4.

The stage is driven by a stub diarizer throughout.  That is not a workaround
for DiariZen being missing -- it is installed on any machine that runs S4 -- it
is what makes these tests mean the same thing everywhere.

Two of them used to rely on the package being absent, and so tested something
else entirely on the server: they built a real pipeline, ran it on the fixture's
sine tone, and crashed inside pyannote.  Both now force the failure they name
rather than waiting for the machine to supply it.

The adapter that calls DiariZen is covered here too, with its pipeline replaced.
Its error wrapping is what keeps a third-party crash from reaching the stage
unrecognised -- which is the other half of the same story, since a stage that
does not recognise a failure records none.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from avannotate.audio.diarize import (
    DEFAULT_BATCH_SIZE,
    DiariZenDiarizer,
    DiarizerError,
    build_diarizer,
)
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
    single_shot_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The batch driver must retry it rather than skip it as done.

    The build is made to fail rather than relied on to.  This called
    ``build_diarizer`` for real and counted on DiariZen being absent -- true on
    a development box and false on the server, where it is installed because
    the stage needs it.  There it built fine and the test went on to exercise a
    path it never named.
    """

    context = _context(single_shot_video, tmp_path, backend="diarizen")
    s0_preprocess.run(context)

    def refuse(_: object) -> None:
        raise DiarizerError(
            "DiariZen is required for this stage. It is not on PyPI; install it "
            "from source -- https://github.com/BUTSpeechFIT/DiariZen"
        )

    monkeypatch.setattr(s4_diarize, "build_diarizer", refuse)

    with pytest.raises(DiarizerError):
        s4_diarize.run(context)

    state = json.loads((context.work_dir / "stage_state.json").read_text())
    record = next(item for item in state["stages"] if item["stage"] == "s4-diarize")
    assert record["status"] == "failed"
    assert "DiariZen" in record["error"]


def test_a_diarizer_that_fails_while_running_marks_the_stage_failed_too(
    single_shot_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The half that was missing.

    Only the build was guarded, so a crash while diarizing recorded nothing at
    all -- and the record is exactly what tells the batch driver this video
    failed rather than that it is done.
    """

    context = _context(single_shot_video, tmp_path, backend="diarizen")
    s0_preprocess.run(context)

    class Exploding:
        name = "exploding"

        def diarize(self, audio: Path) -> DiarizationResult:
            raise DiarizerError(
                "DiariZen failed on mix.wav: ValueError: negative dimensions are "
                "not allowed"
            )

    monkeypatch.setattr(s4_diarize, "build_diarizer", lambda _: Exploding())

    with pytest.raises(DiarizerError):
        s4_diarize.run(context)

    state = json.loads((context.work_dir / "stage_state.json").read_text())
    record = next(item for item in state["stages"] if item["stage"] == "s4-diarize")
    assert record["status"] == "failed"
    assert "negative dimensions" in record["error"]


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


# --------------------------------------------------------------------------- #
# the adapter itself
# --------------------------------------------------------------------------- #
#
# Its job is ten lines: call the pipeline, turn what comes back into
# ``SpeakerTurn``s.  What is worth testing is not that, it is what happens when
# the call raises -- because the failure this pipeline actually meets on real
# audio is a crash inside pyannote, and the only thing standing between that and
# a stage which records nothing is the wrapping below.


class _ExplodingPipeline:
    """A pipeline whose call raises, standing in for DiariZen on bad audio."""

    def __init__(self, error: Exception) -> None:
        self.error = error
        self.seen: list[str] = []
        # pyannote's own defaults, which is what they are if the adapter
        # forgets to set them.  Starting these at DEFAULT_BATCH_SIZE instead
        # would make "the adapter sets both" untestable: the fake would already
        # hold the value the assertion is looking for.
        self.embedding_batch_size = 1
        self.segmentation_batch_size = 1

    def __call__(self, audio: str) -> object:
        self.seen.append(audio)
        raise self.error


def _adapter(pipeline: object) -> DiariZenDiarizer:
    """A diarizer with its pipeline replaced, so nothing is imported or fetched.

    ``__new__`` rather than the constructor: ``__init__`` imports DiariZen and
    downloads a checkpoint, neither of which this test is about.
    """

    diarizer = DiariZenDiarizer.__new__(DiariZenDiarizer)
    diarizer._pipeline = pipeline
    diarizer.model = "test"
    diarizer.device = "backend default"
    diarizer.batch_size = DEFAULT_BATCH_SIZE
    return diarizer


def test_a_crash_inside_diarizen_becomes_a_diarizer_error(tmp_path: Path) -> None:
    """The failure that actually happened, on a clip with no speech in it.

    The embeddings come back degenerate and pyannote's VBx clustering dies with
    a bare ``ValueError: negative dimensions are not allowed`` -- from a file
    that names neither the diarizer nor the audio.  Reaching the stage as that,
    it is not recognised as a S4 failure, so no record is written.
    """

    diarizer = _adapter(
        _ExplodingPipeline(ValueError("negative dimensions are not allowed"))
    )

    with pytest.raises(DiarizerError, match="DiariZen failed on clip.wav"):
        diarizer.diarize(tmp_path / "clip.wav")


def test_the_original_error_survives_as_the_cause(tmp_path: Path) -> None:
    """Wrapped, not swallowed -- whoever debugs this still gets the traceback.

    The wrapping exists to give the stage something it can recognise.  A version
    that dropped the cause would trade a confusing error for a missing one.
    """

    original = ValueError("negative dimensions are not allowed")
    diarizer = _adapter(_ExplodingPipeline(original))

    with pytest.raises(DiarizerError) as caught:
        diarizer.diarize(tmp_path / "clip.wav")

    assert caught.value.__cause__ is original


def test_an_empty_result_is_not_an_error(tmp_path: Path) -> None:
    """Guards the guard: wrapping must not turn "no speakers" into a failure.

    A clip where nobody speaks is ordinary, and the stage's own answer to it --
    zero turns, written and summarised -- is what the offscreen branch is built
    on.  Only an exception from the pipeline is an error.
    """

    class Silent:
        def __call__(self, audio: str) -> Any:
            return _AnnotationStub()

    diarizer = _adapter(Silent())
    result = diarizer.diarize(tmp_path / "clip.wav")
    assert result.turns == ()


class _AnnotationStub:
    """The two attributes the adapter reads off a pyannote ``Annotation``."""

    def itertracks(self, yield_label: bool = False) -> tuple[()]:
        return ()

    def __len__(self) -> int:
        return 0


class _MemoryPipeline:
    """Runs out of memory until the batch size is small enough for it."""

    def __init__(self, *, fits_at: int) -> None:
        self.fits_at = fits_at
        # 1, as pyannote leaves them -- see _ExplodingPipeline.
        self.embedding_batch_size = 1
        self.segmentation_batch_size = 1
        self.sizes_seen: list[tuple[int, int]] = []
        self.calls = 0

    def __call__(self, audio: str) -> Any:
        self.calls += 1
        self.sizes_seen.append(
            (self.embedding_batch_size, self.segmentation_batch_size)
        )
        if self.embedding_batch_size > self.fits_at:
            raise MemoryError("batch_size ( 32) is probably too large.")
        return _AnnotationStub()


def test_a_card_that_cannot_hold_the_batch_gets_a_smaller_one(
    tmp_path: Path,
) -> None:
    """The failure this was written for, taken from the first real run.

    ``MemoryError: batch_size ( 32) is probably too large`` is pyannote's own
    wrapper around a CUDA out-of-memory, and it failed a video in S4 for no
    reason but the card that video happened to land on.
    """

    pipeline = _MemoryPipeline(fits_at=8)
    diarizer = _adapter(pipeline)
    diarizer.diarize(tmp_path / "clip.wav")

    assert diarizer.batch_size == 8
    # 32 -> 16 -> 8: two refusals, then the one that worked.
    assert [size for size, _ in pipeline.sizes_seen] == [32, 16, 8]


def test_the_size_that_worked_is_remembered(tmp_path: Path) -> None:
    """A batch pays for the discovery once per worker, not once per video.

    Four workers over a thousand videos would otherwise spend the first attempt
    of every one of them finding out the same thing.
    """

    pipeline = _MemoryPipeline(fits_at=8)
    diarizer = _adapter(pipeline)
    diarizer.diarize(tmp_path / "one.wav")
    settled = pipeline.calls
    diarizer.diarize(tmp_path / "two.wav")

    assert pipeline.calls == settled + 1
    assert pipeline.sizes_seen[-1] == (8, 8)


def test_both_batch_sizes_are_set_and_not_just_one(tmp_path: Path) -> None:
    """pyannote keeps one behind a property and the other as a plain attribute.

    Setting one and forgetting the other drops the forgotten one to its own
    default of 1.  Nothing raises, nothing warns: the pipeline simply runs at a
    fraction of the batch it was asked for.
    """

    pipeline = _MemoryPipeline(fits_at=DEFAULT_BATCH_SIZE)
    _adapter(pipeline).diarize(tmp_path / "clip.wav")

    assert pipeline.sizes_seen == [(DEFAULT_BATCH_SIZE, DEFAULT_BATCH_SIZE)]


def test_running_out_at_the_floor_is_reported_as_itself(tmp_path: Path) -> None:
    """Halving stops at one: past that, the batch size is not the problem."""

    pipeline = _MemoryPipeline(fits_at=0)
    diarizer = _adapter(pipeline)

    with pytest.raises(DiarizerError, match="even at a batch size"):
        diarizer.diarize(tmp_path / "clip.wav")
    assert diarizer.batch_size == 1
