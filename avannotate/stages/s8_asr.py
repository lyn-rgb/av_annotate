"""Stage S8: what each person said.

S7 produced one audio file per speaking segment holding one person's voice.
This turns each of those into words -- with the times they were said, in the
language the video is actually in, and with the recogniser's own confidence
figures kept so a later reviewer can tell a transcript from a guess.

Two decisions do most of the work.

**Which audio to transcribe.**  The mix when one person is talking, S7's
extraction when two are -- because a recogniser handed two overlapping voices
returns one fluent transcript containing both people's words, and there is
nothing in the output that says so.  :mod:`avannotate.asr.plan` makes that
choice from intervals, before any model runs.

**What language to ask for.**  Detection on a two-second clip is not a weak
version of the right answer, it is often a confident wrong one, and a wrong
language makes the recogniser translate instead of transcribe -- which reads
perfectly and is worth nothing.  So the language is decided once for the video
from the segments long enough to be evidence, and short segments are *told* what
language they are in rather than asked.  See :mod:`avannotate.asr.language`.

The cost follows speech, not video: a segment is transcribed once, and the only
pass over anything else is up to 30 seconds of audio when a video has no segment
long enough to establish its language.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from avannotate.asr import language as language_module
from avannotate.asr.audio import AsrAudioError, concat, read_source
from avannotate.asr.model import AsrError, Transcriber, build_transcriber
from avannotate.asr.plan import plan_sources
from avannotate.asr.text import clip_to_span, hallucination_flags, join_words, offset_words
from avannotate.asr.types import (
    SOURCE_MIX,
    SegmentSource,
    SpeechSegment,
    Transcription,
)
from avannotate.ffmpeg import TARGET_SAMPLE_RATE
from avannotate.schema import Language
from avannotate.stages import s0_preprocess, s6_associate, s7_tse
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

STAGE = "s8-asr"
VERSION = "s8-v1"

TRANSCRIPTS_NAME = "transcripts.json"
SUMMARY_NAME = "summary.json"

#: Whisper detects a language from a single 30-second window, so a longer
#: sample buys nothing but costs a longer pass.
DEFAULT_DETECT_SECONDS = 30.0

#: Silence put between the clips of the detection sample.  Without it the
#: recogniser hears two sentences butting together, which is a word boundary
#: nobody spoke.
DEFAULT_DETECT_GAP_SECONDS = 0.25


@dataclass(frozen=True)
class S8Config:
    backend: str = "faster-whisper"
    model: str = "large-v3"
    device: str | None = None
    compute_type: str | None = None
    #: Where the checkpoint is cached.  Worth setting on a server that runs
    #: several jobs at once, or that has no direct route to Hugging Face.
    download_root: str | None = None
    beam_size: int = 5
    #: Voice activity filtering is off by default because S7 has already
    #: trimmed every segment to speech.  There is no silence left for it to
    #: find, and it is another place timestamps can shift.
    vad_filter: bool = False
    condition_on_previous_text: bool = False
    #: Audio read around a segment from the mix.  The extraction carries none:
    #: S7 trimmed it to the segment before writing it.
    context_seconds: float = 0.25
    min_overlap_seconds: float = 0.10
    #: A segment must be at least this long to be evidence about the video's
    #: language.  Shorter ones are transcribed in the decided language instead.
    min_detect_seconds: float = language_module.MIN_VOTE_SECONDS
    detect_seconds: float = DEFAULT_DETECT_SECONDS
    detect_gap_seconds: float = DEFAULT_DETECT_GAP_SECONDS
    #: Hallucination heuristics.  Recorded, never gated: a video of somebody
    #: whispering is a video of low log-probabilities and still correct.
    min_avg_logprob: float = -1.0
    max_no_speech_prob: float = 0.6
    max_compression_ratio: float = 2.4
    sample_rate: int = TARGET_SAMPLE_RATE

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> S8Config:
        return cls(
            backend=config_str(mapping, "backend", "faster-whisper"),
            model=config_str(mapping, "model", "large-v3"),
            device=config_optional_str(mapping, "device"),
            compute_type=config_optional_str(mapping, "compute_type"),
            download_root=config_optional_str(mapping, "download_root"),
            beam_size=config_int(mapping, "beam_size", 5),
            vad_filter=bool(mapping.get("vad_filter", False)),
            condition_on_previous_text=bool(
                mapping.get("condition_on_previous_text", False)
            ),
            context_seconds=config_float(mapping, "context_seconds", 0.25),
            min_overlap_seconds=config_float(mapping, "min_overlap_seconds", 0.10),
            min_detect_seconds=config_float(
                mapping, "min_detect_seconds", language_module.MIN_VOTE_SECONDS
            ),
            detect_seconds=config_float(
                mapping, "detect_seconds", DEFAULT_DETECT_SECONDS
            ),
            detect_gap_seconds=config_float(
                mapping, "detect_gap_seconds", DEFAULT_DETECT_GAP_SECONDS
            ),
            min_avg_logprob=config_float(mapping, "min_avg_logprob", -1.0),
            max_no_speech_prob=config_float(mapping, "max_no_speech_prob", 0.6),
            max_compression_ratio=config_float(mapping, "max_compression_ratio", 2.4),
            sample_rate=config_int(mapping, "sample_rate", TARGET_SAMPLE_RATE),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "model": self.model,
            "device": self.device,
            "compute_type": self.compute_type,
            "download_root": self.download_root,
            "beam_size": self.beam_size,
            "vad_filter": self.vad_filter,
            "condition_on_previous_text": self.condition_on_previous_text,
            "context_seconds": self.context_seconds,
            "min_overlap_seconds": self.min_overlap_seconds,
            "min_detect_seconds": self.min_detect_seconds,
            "detect_seconds": self.detect_seconds,
            "detect_gap_seconds": self.detect_gap_seconds,
            "min_avg_logprob": self.min_avg_logprob,
            "max_no_speech_prob": self.max_no_speech_prob,
            "max_compression_ratio": self.max_compression_ratio,
            "sample_rate": self.sample_rate,
        }


def _transcribe(
    context: StageContext,
    transcriber: Transcriber,
    source: SegmentSource,
    config: S8Config,
    *,
    language: str | None,
) -> Transcription:
    """Read one segment's span and recognise it.

    ``language`` is what makes the two passes different: ``None`` asks the
    recogniser to work it out from this clip, a code tells it.
    """

    samples = read_source(source, root=context.work_dir, sample_rate=config.sample_rate)
    return transcriber.transcribe(samples, language=language)


def _record(
    source: SegmentSource,
    transcription: Transcription,
    config: S8Config,
    *,
    video_language: str,
    silent: bool,
) -> dict[str, object]:
    """One segment's line in ``transcripts.json``.

    Trimming happens before the text is joined, not after: a word inside the
    padding belongs to whoever spoke before this person, and joining the whole
    window and then cutting the string would keep it.
    """

    words = clip_to_span(
        offset_words(transcription.words, origin=source.origin),
        start=source.segment.start,
        end=source.segment.end,
    )
    flags = list(
        hallucination_flags(
            words=len(words),
            avg_logprob=transcription.avg_logprob,
            no_speech_prob=transcription.no_speech_prob,
            compression_ratio=transcription.compression_ratio,
            min_avg_logprob=config.min_avg_logprob,
            max_no_speech_prob=config.max_no_speech_prob,
            max_compression_ratio=config.max_compression_ratio,
        )
    )
    # Only a detected language can disagree: a forced one is what it was forced
    # to, and flagging that would make every short segment look wrong.
    if transcription.detected and transcription.language not in (video_language, ""):
        flags.append("language_mismatch")
    if silent:
        flags.append("silent_extraction")

    return {
        **source.to_dict(),
        "text": join_words(words),
        "words": [word.to_dict() for word in words],
        "language": transcription.language,
        "language_probability": round(transcription.language_probability, 4),
        "detected": transcription.detected,
        "avg_logprob": round(transcription.avg_logprob, 4),
        "no_speech_prob": round(transcription.no_speech_prob, 4),
        "compression_ratio": round(transcription.compression_ratio, 4),
        "flags": flags,
    }


def _detect_from_sample(
    context: StageContext,
    transcriber: Transcriber,
    sources: tuple[SegmentSource, ...],
    config: S8Config,
) -> tuple[str, float] | None:
    """The language, from the longest clips this video has.

    Only reached when no single segment was long enough to be evidence.  The
    signals are read through the planned sources like everything else, so an
    overlapping clip contributes the extraction rather than the mix.

    Asks for detection rather than transcription: the language comes from a
    single 30-second window before any decoding, so this costs a forward pass
    over the mel instead of a decode whose text would be thrown away.
    """

    chosen = sorted(sources, key=lambda item: -item.segment.duration)
    picked: list[SegmentSource] = []
    total = 0.0
    for source in chosen:
        if total >= config.detect_seconds:
            break
        picked.append(source)
        total += source.segment.duration + config.detect_gap_seconds
    if not picked:
        return None

    chunks = [
        read_source(source, root=context.work_dir, sample_rate=config.sample_rate)
        for source in picked
    ]
    samples = concat(
        chunks, gap_seconds=config.detect_gap_seconds, sample_rate=config.sample_rate
    )
    return transcriber.detect(samples)


def run(context: StageContext, *, force: bool = False) -> StageRun:
    config = S8Config.from_mapping(context.config)
    input_hash = _input_hash(context, config)

    state = StageState(context.work_dir)
    reason = state.reason_to_run(STAGE, code_version=VERSION, input_hash=input_hash)
    if reason is None and not force:
        return StageRun(stage=STAGE, skipped=True, reason="outputs present and unchanged")
    trigger = "forced" if reason is None else reason

    timeline = s0_preprocess.load_timeline(context)
    speech = s6_associate.load_identity_speech(context)
    raw = s7_tse.load_segments(context)
    mix = str(s0_preprocess.audio_path(context).relative_to(context.work_dir))

    segments = [SpeechSegment.from_dict(item) for item in raw]
    silent = {str(item.get("name")) for item in raw if item.get("silent")}
    sources = plan_sources(
        segments,
        speech,
        mix_audio=mix,
        duration=timeline.duration,
        context_seconds=config.context_seconds,
        min_overlap_seconds=config.min_overlap_seconds,
    )

    try:
        transcriber = build_transcriber(
            {
                "backend": config.backend,
                "model": config.model,
                "device": config.device,
                "compute_type": config.compute_type,
                "download_root": config.download_root,
                "beam_size": config.beam_size,
                "vad_filter": config.vad_filter,
                "condition_on_previous_text": config.condition_on_previous_text,
            }
        )
    except AsrError as error:
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

    # Pass one: the segments long enough to say what language the video is in.
    # They are transcribed with detection on, so the answer comes from the audio
    # rather than from a guess made before any audio was read.
    recognised: dict[str, Transcription] = {}
    try:
        for source in sources:
            if source.segment.duration < config.min_detect_seconds:
                continue
            recognised[source.segment.name] = _transcribe(
                context, transcriber, source, config, language=None
            )

        votes = language_module.collect_votes(
            [
                (source.segment.duration, recognised[source.segment.name])
                for source in sources
                if source.segment.name in recognised
            ],
            min_seconds=config.min_detect_seconds,
        )

        if votes:
            video = Language(code=votes[0].code, confidence=votes[0].probability, source="segments")
        else:
            sample = _detect_from_sample(context, transcriber, sources, config)
            video = (
                Language(code=sample[0], confidence=sample[1], source="sample")
                if sample is not None and sample[0]
                else Language(code="unknown", source="none")
            )

        # Pass two: the short ones, told what language they are in rather than
        # asked.  A two-second clip has far less evidence than whisper's 30 s
        # detection window, and its guess is a translation as often as a
        # transcription.
        forced = video.code if video.code != "unknown" else None
        for source in sources:
            if source.segment.name in recognised:
                continue
            recognised[source.segment.name] = _transcribe(
                context, transcriber, source, config, language=forced
            )
    except (AsrAudioError, AsrError) as error:
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

    records = [
        _record(
            source,
            recognised[source.segment.name],
            config,
            video_language=video.code,
            silent=source.segment.name in silent,
        )
        for source in sources
        if source.segment.name in recognised
    ]

    disagreements = language_module.disagreements(
        [
            (source.segment.duration, recognised[source.segment.name])
            for source in sources
            if source.segment.name in recognised
        ],
        language=video.code,
    )
    counted = _flag_counts(records)

    transcripts_path = write_json(
        context.output(STAGE, TRANSCRIPTS_NAME),
        {
            "schema_version": "avannotate-transcripts-v1",
            "config": config.to_dict(),
            "backend": {"name": transcriber.name, "model": config.model},
            "language": {
                "code": video.code,
                "confidence": video.confidence,
                "source": video.source,
                "votes": [vote.to_dict() for vote in votes],
                "disagreements": [
                    {"language": code, "seconds": secs} for code, secs in disagreements
                ],
            },
            "transcripts": records,
        },
    )
    summary_path = write_json(
        context.output(STAGE, SUMMARY_NAME),
        {
            "schema_version": "avannotate-transcripts-summary-v1",
            "language": video.code,
            "segments": len(sources),
            "transcribed": len(records),
            "empty": sum(1 for record in records if not record["words"]),
            "flagged": sum(1 for record in records if record["flags"]),
            "seconds": round(sum(item.segment.duration for item in sources), 3),
            "by_source": {
                SOURCE_MIX: sum(1 for item in sources if not item.overlapping),
                "extracted": sum(1 for item in sources if item.overlapping),
            },
            "flags": counted,
            "disagreement_seconds": round(sum(secs for _, secs in disagreements), 3),
            "identities": sorted({item.segment.identity for item in sources}),
        },
    )

    artifacts = tuple(
        Artifact.capture(context.work_dir, path)
        for path in (transcripts_path, summary_path)
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
            "language": video.code,
            "transcribed": len(records),
            "empty": sum(1 for record in records if not record["words"]),
            "flagged": sum(1 for record in records if record["flags"]),
        },
    )


def _flag_counts(records: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        raw = record.get("flags")
        for flag in raw if isinstance(raw, list) else []:
            counts[str(flag)] = counts.get(str(flag), 0) + 1
    return dict(sorted(counts.items()))


def _input_hash(context: StageContext, config: S8Config) -> str:
    stat = context.source.stat()
    return hash_payload(
        {
            "config": config.to_dict(),
            "segments": hash_file(s7_tse.segments_path(context)),
            "assignments": hash_file(s6_associate.assignments_path(context)),
            "video": {"size": stat.st_size, "mtime": stat.st_mtime},
        }
    )


def load_transcripts(context: StageContext) -> tuple[dict[str, object], ...]:
    path = transcripts_path(context)
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("transcripts")
    if not isinstance(raw, list):
        raise ValueError(f"{path} has no transcripts list; re-run {STAGE}")
    return tuple(item for item in raw if isinstance(item, dict))


def load_language(context: StageContext) -> Language:
    """The video's language, as S8 decided it."""

    path = transcripts_path(context)
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("language")
    if not isinstance(raw, dict):
        raise ValueError(f"{path} has no language object; re-run {STAGE}")
    confidence = raw.get("confidence")
    return Language(
        code=str(raw.get("code", "unknown")),
        confidence=float(confidence) if isinstance(confidence, (int, float)) else None,
        source=str(raw.get("source", "unknown")),
    )


def transcripts_path(context: StageContext) -> Path:
    path = context.work_dir / STAGE / TRANSCRIPTS_NAME
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; run {STAGE} first")
    return path
