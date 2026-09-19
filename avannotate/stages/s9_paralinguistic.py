"""Stage S9: how each line was said.

S7 gave each segment its own audio, S8 gave it words.  This gives it a manner --
one tag from a closed vocabulary, rendered before the spoken text as
``<F001> whispering: <S>...</S>``.

Three models, one tag
---------------------

No single model covers what the format asks for.  ``emotion2vec+`` classifies
affect into nine classes and has no notion of *whispering*, which is a way of
producing the words rather than a feeling about them; a voice-tagging model
covers delivery but not the non-speech sounds that replace words entirely; and
a general audio tagger covers those sounds but knows nothing about speech
delivery.  So all three run and
:mod:`avannotate.paralinguistic.reduce` chooses between them by precedence --
an event over a delivery over an emotion -- because the three scores are not on
a comparable scale and letting the largest number win would make the priority an
artefact of calibration.

Always the extraction, never the mix
------------------------------------

The opposite of S8's routing, for the opposite reason.  S8 wants the best
*fidelity*, so it reads the mix whenever one voice is on it.  S9 wants the best
*isolation*, because a general audio tagger listening to a mix reports what is
audible rather than what this person did -- a cough from across the room lands
in the segment of whoever happens to be speaking, and ``cough`` on a line that
was not coughed is a wrong tag that nothing downstream can detect.  S7's
extraction holds one person, so a sound in it is that person's.

What that costs: the extraction has been through a separation model, and a
separated voice is not a clean one.  The taggers are classifiers, not
transcribers, and they are more tolerant of separation artefacts than a
recogniser is -- but the honest statement is that this trades one error for
another, and the QA report keeps both models' full score lists so the trade can
be examined rather than assumed.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from avannotate.audio.wav import read_mono
from avannotate.ffmpeg import TARGET_SAMPLE_RATE
from avannotate.paralinguistic.model import Tagger, TaggerError, build_tagger
from avannotate.paralinguistic.reduce import ReduceConfig, reduce_tags, unmapped_labels
from avannotate.paralinguistic.types import DIMENSIONS, SegmentTags
from avannotate.paralinguistic.vocabulary import ALL_TAGS
from avannotate.segment import SegmentationConfig, SpeechSegment, tag_eligible
from avannotate.stages import s7_tse
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

STAGE = "s9-paralinguistic"
VERSION = "s9-v1"

TAGS_NAME = "tags.json"
SUMMARY_NAME = "summary.json"


@dataclass(frozen=True)
class S9Config:
    backend: str = "three-model"
    dimensions: tuple[str, ...] = DIMENSIONS
    device: str | None = None
    download_root: str | None = None
    #: PANNs' ``.pth``.  Leave unset and the package downloads it with a
    #: shelled-out wget, which does nothing at all where wget is absent.
    checkpoint: str | None = None
    #: How many labels to keep per model per segment.  Kept for the record:
    #: when a tag reads wrong, the first question is what the runner-up was.
    top_k: int = 5
    #: Below this no tag is rendered at all.  Shared with the segmenter so the
    #: rule has one definition.
    min_tag_duration: float = 0.50
    emotion_min_score: float = 0.5
    delivery_min_score: float = 0.5
    event_min_score: float = 0.5
    close_margin: float = 0.10
    sample_rate: int = TARGET_SAMPLE_RATE

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> S9Config:
        return cls(
            backend=config_str(mapping, "backend", "three-model"),
            dimensions=_dimensions(mapping.get("dimensions")),
            device=config_optional_str(mapping, "device"),
            download_root=config_optional_str(mapping, "download_root"),
            checkpoint=config_optional_str(mapping, "checkpoint"),
            top_k=config_int(mapping, "top_k", 5),
            min_tag_duration=config_float(mapping, "min_tag_duration", 0.50),
            emotion_min_score=config_float(mapping, "emotion_min_score", 0.5),
            delivery_min_score=config_float(mapping, "delivery_min_score", 0.5),
            event_min_score=config_float(mapping, "event_min_score", 0.5),
            close_margin=config_float(mapping, "close_margin", 0.10),
            sample_rate=config_int(mapping, "sample_rate", TARGET_SAMPLE_RATE),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "dimensions": list(self.dimensions),
            "device": self.device,
            "download_root": self.download_root,
            "checkpoint": self.checkpoint,
            "top_k": self.top_k,
            "min_tag_duration": self.min_tag_duration,
            "emotion_min_score": self.emotion_min_score,
            "delivery_min_score": self.delivery_min_score,
            "event_min_score": self.event_min_score,
            "close_margin": self.close_margin,
            "sample_rate": self.sample_rate,
        }

    def reducer(self) -> ReduceConfig:
        return ReduceConfig(
            min_score={
                "emotion": self.emotion_min_score,
                "delivery": self.delivery_min_score,
                "event": self.event_min_score,
            },
            close_margin=self.close_margin,
        )

    def segmentation(self) -> SegmentationConfig:
        return SegmentationConfig(min_tag_duration=self.min_tag_duration)


def _dimensions(raw: object) -> tuple[str, ...]:
    """Which taggers to run, checked against the ones that exist.

    Configurable because the three packages install separately: a machine with
    only ``funasr`` can still produce emotion tags, and it is better to say so
    in the config than to fail on an import.
    """

    if not isinstance(raw, list) or not raw:
        return DIMENSIONS
    wanted = tuple(str(item) for item in raw)
    unknown = [item for item in wanted if item not in DIMENSIONS]
    if unknown:
        raise ValueError(
            f"unknown tagger dimension(s) {unknown}; known: {list(DIMENSIONS)}"
        )
    return wanted


def tag_segment(
    context: StageContext,
    tagger: Tagger,
    segment: SpeechSegment,
    config: S9Config,
    *,
    eligible: bool,
) -> SegmentTags:
    """One segment's tags, or a record saying why there are none."""

    if not eligible:
        return SegmentTags(
            identity=segment.identity,
            name=segment.name,
            start=segment.start,
            end=segment.end,
            choice=reduce_tags({}, config=config.reducer()),
            eligible=False,
        )

    path = context.work_dir / segment.audio
    samples = read_mono(
        path,
        start_seconds=0.0,
        duration_seconds=segment.duration,
        sample_rate=config.sample_rate,
    )
    reported = tagger.tag(samples)

    # Trimmed to the model's own top-k here rather than in the adapter: the
    # number is a property of how much of the answer is worth keeping, which is
    # the stage's concern, and the adapter should not need to know it.
    kept = {
        dimension: tuple(reported.get(dimension, ()))[: config.top_k]
        for dimension in config.dimensions
    }
    return SegmentTags(
        identity=segment.identity,
        name=segment.name,
        start=segment.start,
        end=segment.end,
        choice=reduce_tags(kept, config=config.reducer()),
        emotion=kept.get("emotion", ()),
        delivery=kept.get("delivery", ()),
        event=kept.get("event", ()),
        unmapped=unmapped_labels(kept, config=config.reducer()),
        eligible=True,
    )


def run(context: StageContext, *, force: bool = False) -> StageRun:
    config = S9Config.from_mapping(context.config)
    input_hash = _input_hash(context, config)

    state = StageState(context.work_dir)
    reason = state.reason_to_run(STAGE, code_version=VERSION, input_hash=input_hash)
    if reason is None and not force:
        return StageRun(stage=STAGE, skipped=True, reason="outputs present and unchanged")
    trigger = "forced" if reason is None else reason

    segments = [SpeechSegment.from_dict(item) for item in s7_tse.load_segments(context)]

    try:
        tagger = build_tagger(
            {
                "backend": config.backend,
                "dimensions": list(config.dimensions),
                "device": config.device,
                "download_root": config.download_root,
                "checkpoint": config.checkpoint,
            }
        )
    except TaggerError as error:
        state.save(
            StageRecord(
                stage=STAGE,
                status="failed",
                code_version=VERSION,
                config_hash=hash_payload(config.to_dict()),
                input_hash=input_hash,
                error=str(error),
            )
        )
        raise

    tagged = [
        tag_segment(
            context,
            tagger,
            segment,
            config,
            eligible=tag_eligible(segment, config.segmentation()),
        )
        for segment in segments
    ]

    tags_path = write_json(
        context.output(STAGE, TAGS_NAME),
        {
            "schema_version": "avannotate-tags-v1",
            "config": config.to_dict(),
            "backend": {"name": tagger.name, "dimensions": list(config.dimensions)},
            "vocabulary": sorted(ALL_TAGS),
            "segments": [item.to_dict() for item in tagged],
        },
    )

    chosen = [item for item in tagged if item.eligible]
    tagged_tags = [item for item in chosen if item.choice.tag]
    summary_path = write_json(
        context.output(STAGE, SUMMARY_NAME),
        {
            "schema_version": "avannotate-tags-summary-v1",
            "segments": len(tagged),
            "eligible": len(chosen),
            "tagged": len(tagged_tags),
            "untagged": len(chosen) - len(tagged_tags),
            "too_short": len(tagged) - len(chosen),
            "by_tag": _counts([item.choice.tag for item in tagged_tags]),
            "by_dimension": _counts(
                [item.choice.dimension for item in tagged_tags if item.choice.dimension]
            ),
            "unmapped": _counts(
                [label for item in tagged for label in item.unmapped]
            ),
            "close_calls": sum(
                1
                for item in tagged_tags
                if len(item.choice.candidates) > 1
                and item.choice.margin < config.close_margin
            ),
        },
    )

    artifacts = tuple(
        Artifact.capture(context.work_dir, path) for path in (tags_path, summary_path)
    )
    state.save(
        StageRecord(
            stage=STAGE,
            status="ok",
            code_version=VERSION,
            config_hash=hash_payload(config.to_dict()),
            input_hash=input_hash,
            artifacts=artifacts,
        )
    )

    return StageRun(
        stage=STAGE,
        skipped=False,
        reason=trigger,
        summary={
            "eligible": len(chosen),
            "tagged": len(tagged_tags),
            "untagged": len(chosen) - len(tagged_tags),
            "too_short": len(tagged) - len(chosen),
        },
    )


def _counts(values: Sequence[object]) -> dict[str, int]:
    """How often each thing happened, keyed in a stable order."""

    counts: dict[str, int] = {}
    for value in values:
        if value is None:
            continue
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _input_hash(context: StageContext, config: S9Config) -> str:
    stat = context.source.stat()
    return hash_payload(
        {
            "config": config.to_dict(),
            "segments": hash_file(s7_tse.segments_path(context)),
            "video": {"size": stat.st_size, "mtime": stat.st_mtime},
        }
    )


def load_tags(context: StageContext) -> tuple[dict[str, object], ...]:
    """Every segment's scores, tagged or not."""

    path = tags_path(context)
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("segments")
    if not isinstance(raw, list):
        raise ValueError(f"{path} has no segments list; re-run {STAGE}")
    return tuple(item for item in raw if isinstance(item, dict))


def load_segment_tag(context: StageContext, name: str) -> str | None:
    """The tag for one segment, by S7's name for it."""

    for item in load_tags(context):
        if str(item.get("name")) == name:
            tag = item.get("tag")
            return str(tag) if tag else None
    return None


def tags_path(context: StageContext) -> Path:
    path = context.work_dir / STAGE / TAGS_NAME
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; run {STAGE} first")
    return path
