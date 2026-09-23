"""Stage S4: who spoke when, in the mixed audio.

Runs a diarizer over S0's demuxed track and writes speaker turns.  Everything
after this -- matching a speaker to a face, deciding what is off-screen -- reads
these turns, so the stage records the numbers that say whether they can be
trusted: how many speakers, how much speech, and how much of it is contested.

Two orderings here are load-bearing.

**Clamp before merging.**  DiariZen reports against the file it was handed, and
S0 established that a demuxed track runs tens of milliseconds longer than its
video.  Merging first and clamping after could leave a turn whose end is past
the end of the picture.

**Merge before dropping short turns.**  A sliding-window diarizer emits many
brief turns for one continuous utterance.  Dropping the brief ones first would
delete exactly the pieces that merging was about to join into a usable turn.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from avannotate.audio.diarize import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MODEL,
    DiarizerError,
    build_diarizer,
)
from avannotate.audio.types import DiarizationResult, SpeakerTurn
from avannotate.model_cache import model_for
from avannotate.stages import s0_preprocess
from avannotate.stages.base import (
    Artifact,
    StageContext,
    StageRecord,
    StageRun,
    StageState,
    config_float,
    config_int,
    config_optional_str,
    config_str,
    hash_file,
    hash_payload,
    write_json,
)

STAGE = "s4-diarize"
#: DiariZen, on torch.
USES_GPU = True
VERSION = "s4-v1"

TURNS_NAME = "turns.jsonl"
SUMMARY_NAME = "summary.json"


@dataclass(frozen=True)
class S4Config:
    backend: str = "diarizen"
    model: str = DEFAULT_MODEL
    device: str | None = None
    #: One speaker's turns closer than this are one utterance.  A window
    #: boundary is not a turn boundary, and 0.2s is the same threshold S10 uses
    #: for the same reason: a breath, not a change of speaker.
    merge_gap_seconds: float = 0.20
    #: Turns still shorter than this after merging are noise.  Below roughly a
    #: syllable there is nothing to transcribe and nothing to extract.
    min_turn_seconds: float = 0.10
    #: Input batch size for DiariZen.  The checkpoint's own 32 was chosen for
    #: a much larger card and does not fit a 24 GB one; the adapter halves it
    #: and retries when it does not, so this is a starting point rather than
    #: a commitment.  Lower it to skip the discovery, raise it on a big card.
    batch_size: int = DEFAULT_BATCH_SIZE

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> S4Config:
        return cls(
            backend=config_str(mapping, "backend", "diarizen"),
            model=config_str(mapping, "model", DEFAULT_MODEL),
            device=config_optional_str(mapping, "device"),
            merge_gap_seconds=config_float(mapping, "merge_gap_seconds", 0.20),
            min_turn_seconds=config_float(mapping, "min_turn_seconds", 0.10),
            batch_size=config_int(mapping, "batch_size", DEFAULT_BATCH_SIZE),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "model": self.model,
            "device": self.device,
            # The token is not written: it is a credential, and a config hash
            # that included it would put it in the resume record on disk.
            "merge_gap_seconds": self.merge_gap_seconds,
            "min_turn_seconds": self.min_turn_seconds,
            "batch_size": self.batch_size,
        }

    def cache_key(self) -> dict[str, object]:
        """What the resume record hashes.  Same as :meth:`to_dict` here, but
        stated separately so a future field can be added to one and not the
        other deliberately rather than by accident."""

        return self.to_dict()


def postprocess(
    result: DiarizationResult, *, duration: float, config: S4Config
) -> tuple[DiarizationResult, dict[str, int]]:
    """Clamp, merge, then drop -- in that order.  See the module docstring."""

    counts = {
        "raw_turns": len(result.turns),
        "outside_video": len(result.turns),
    }
    clamped = result.clamped(duration)
    counts["outside_video"] -= len(clamped.turns)
    counts["clamped_turns"] = len(clamped.turns)

    merged = clamped.merged(max_gap=config.merge_gap_seconds)
    counts["merged_turns"] = len(merged.turns)

    kept = tuple(
        turn for turn in merged.turns if turn.duration >= config.min_turn_seconds
    )
    counts["dropped_short"] = len(merged.turns) - len(kept)
    counts["turns"] = len(kept)

    return DiarizationResult(turns=kept, metadata=dict(merged.metadata)), counts


def run(context: StageContext, *, force: bool = False) -> StageRun:
    config = S4Config.from_mapping(context.config)
    input_hash = _input_hash(context, config)

    state = StageState(context.work_dir)
    reason = state.reason_to_run(STAGE, code_version=VERSION, input_hash=input_hash)
    if reason is None and not force:
        return StageRun(stage=STAGE, skipped=True, reason="outputs present and unchanged")
    trigger = "forced" if reason is None else reason

    timeline = s0_preprocess.load_timeline(context)
    audio = s0_preprocess.audio_path(context)

    try:
        diarizer = model_for(
            STAGE,
            {
                "backend": config.backend,
                "model": config.model,
                "device": config.device,
                "batch_size": config.batch_size,
            },
            build_diarizer,
        )
        # Running it is inside the guard, not after it.  Building was the
        # failure this expected to meet; running is the one that actually
        # happens, on any clip the clusterer cannot make sense of.  The record
        # is the point either way -- it is what makes the batch driver retry
        # this video rather than skip it as already done.
        raw = diarizer.diarize(audio)
    except DiarizerError as error:
        state.save(
            StageRecord(
                stage=STAGE,
                status="failed",
                code_version=VERSION,
                config_hash=hash_payload(config.cache_key()),
                input_hash=input_hash,
                error=str(error),
            )
        )
        raise

    result, counts = postprocess(raw, duration=timeline.duration, config=config)

    turns_path = context.output(STAGE, TURNS_NAME)
    temporary = turns_path.with_name(turns_path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for turn in result.turns:
            handle.write(json.dumps(turn.to_dict(), ensure_ascii=False) + "\n")
    temporary.replace(turns_path)

    summary_path = write_json(
        context.output(STAGE, SUMMARY_NAME),
        {
            "schema_version": "avannotate-diarization-summary-v1",
            "config": config.to_dict(),
            "backend": dict(result.metadata),
            "counts": counts,
            "timeline": {
                "video_duration": timeline.duration,
                "audio_duration": timeline.audio_duration,
            },
            # The number the whole design turns on: how much of the speech is
            # contested.  A corpus at 0.02 does not need the overlap machinery;
            # one at 0.4 needs all of it.
            "diarization": result.summary(),
        },
    )

    artifacts = tuple(
        Artifact.capture(context.work_dir, path) for path in (turns_path, summary_path)
    )
    state.save(
        StageRecord(
            stage=STAGE,
            status="ok",
            code_version=VERSION,
            config_hash=hash_payload(config.cache_key()),
            input_hash=input_hash,
            artifacts=artifacts,
        )
    )

    return StageRun(
        stage=STAGE,
        skipped=False,
        reason=trigger,
        summary={
            "speakers": len(result.speakers),
            "turns": len(result.turns),
            "overlap_ratio": round(result.overlap_ratio, 4),
            "model": config.model,
            # What actually ran, not what was configured.  The adapter halves
            # this on every CUDA OOM and keeps what fitted, and on a shared card
            # that can be a long way below the number in the config -- which is
            # the difference between a card at 90% and one at nothing, so it is
            # the number the summary has to carry.
            "batch_size_used": getattr(diarizer, "batch_size", config.batch_size),
        },
    )


def _input_hash(context: StageContext, config: S4Config) -> str:
    audio = s0_preprocess.audio_path(context)
    timeline = s0_preprocess.load_timeline(context)
    # The audio's content is the input.  The video duration joins it because it
    # is what the clamp uses, so a different timeline is a different result from
    # the same audio.
    return hash_payload(
        {
            "config": config.cache_key(),
            "audio": hash_file(audio),
            "duration": timeline.duration,
        }
    )


def load_turns(context: StageContext) -> tuple[SpeakerTurn, ...]:
    """The turns, in time order."""

    path = context.work_dir / STAGE / TURNS_NAME
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; run {STAGE} first")
    turns: list[SpeakerTurn] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                turns.append(SpeakerTurn.from_dict(json.loads(stripped)))
    return tuple(turns)


def load_result(context: StageContext) -> DiarizationResult:
    """The turns plus the backend metadata S4 recorded."""

    summary_path = context.work_dir / STAGE / SUMMARY_NAME
    if not summary_path.is_file():
        raise FileNotFoundError(f"{summary_path} is missing; run {STAGE} first")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    metadata = payload.get("backend")
    return DiarizationResult(
        turns=load_turns(context),
        metadata=dict(metadata) if isinstance(metadata, dict) else {},
    )


def turns_path(context: StageContext) -> Path:
    path = context.work_dir / STAGE / TURNS_NAME
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; run {STAGE} first")
    return path
