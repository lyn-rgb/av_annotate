"""Tests for transcription's routing, language decision, and text handling.

The recogniser is not installed here, so the adapter is not exercised.  What is
exercised is everything that decides what it is handed and what its output
means -- and those are the parts that fail silently.  A segment routed to the
mix while someone else was talking produces a fluent transcript with the wrong
words in it; a transcript read from the extraction while its timestamps came
from the mix is a second out.  Neither looks like an error from the outside.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from avannotate.asr.audio import concat, read_source
from avannotate.asr.language import collect_votes, decide, disagreements
from avannotate.asr.plan import is_overlapping, plan_sources, window
from avannotate.asr.text import (
    clip_to_span,
    collapse_whitespace,
    hallucination_flags,
    join_words,
    offset_words,
)
from avannotate.asr.types import (
    SOURCE_EXTRACTED,
    SOURCE_MIX,
    SegmentSource,
    SpeechSegment,
    TranscribedWord,
    Transcription,
)
from avannotate.audio.wav import WavError, write_pcm16
from avannotate.interval import Interval
from avannotate.stages import s0_preprocess, s6_associate, s7_tse, s8_asr
from avannotate.stages.base import StageContext

RATE = 16000


def _segment(
    name: str = "F001_0000",
    *,
    identity: str = "F001",
    start: float = 1.0,
    end: float = 2.0,
    audio: str = "s7-tse/audio/F001/F001_0000.wav",
) -> SpeechSegment:
    return SpeechSegment(identity=identity, name=name, start=start, end=end, audio=audio)


def _word(text: str, start: float, end: float) -> TranscribedWord:
    return TranscribedWord(text=text, start=start, end=end)


def _transcription(
    *,
    language: str = "en",
    probability: float = 0.95,
    detected: bool = True,
    words: tuple[TranscribedWord, ...] = (),
    avg_logprob: float = -0.2,
    no_speech_prob: float = 0.01,
    compression_ratio: float = 1.2,
) -> Transcription:
    return Transcription(
        text=join_words(words),
        words=words,
        language=language,
        language_probability=probability,
        detected=detected,
        avg_logprob=avg_logprob,
        no_speech_prob=no_speech_prob,
        compression_ratio=compression_ratio,
    )


# --------------------------------------------------------------------------- #
# which audio a segment is transcribed from
# --------------------------------------------------------------------------- #


def test_a_segment_nobody_else_is_speaking_reads_the_mix() -> None:
    """The mix is what the microphone heard.  Nothing improves on it."""

    segment = _segment()
    sources = plan_sources(
        [segment],
        {"F001": (Interval(1.0, 2.0),)},
        mix_audio="s0-preprocess/audio.wav",
        duration=10.0,
        context_seconds=0.25,
        min_overlap_seconds=0.1,
    )

    assert len(sources) == 1
    assert sources[0].source == SOURCE_MIX
    assert sources[0].path == "s0-preprocess/audio.wav"
    assert not sources[0].overlapping


def test_a_segment_someone_else_is_speaking_over_reads_the_extraction() -> None:
    """The one case the mix cannot serve: two voices, one transcript."""

    segment = _segment()
    sources = plan_sources(
        [segment],
        {"F001": (Interval(1.0, 2.0),), "F002": (Interval(1.4, 1.6),)},
        mix_audio="mix.wav",
        duration=10.0,
        context_seconds=0.25,
        min_overlap_seconds=0.1,
    )

    assert sources[0].source == SOURCE_EXTRACTED
    assert sources[0].path == segment.audio
    assert sources[0].overlapping


def test_someone_speaking_over_themselves_is_not_overlap() -> None:
    """S6 merges one person's turns, but a person can be their own neighbour.

    Counting their own speech as a competing voice would send every segment
    with a nearby sibling to the extraction, which is the worse source.
    """

    segment = _segment()
    speech = {"F001": (Interval(1.0, 1.5), Interval(1.5, 2.0))}
    assert not is_overlapping(segment, speech, min_seconds=0.1)


def test_a_brief_overlap_is_tolerated() -> None:
    """A few milliseconds of a backchannel is not worth the worse source."""

    segment = _segment()
    speech = {"F001": (Interval(1.0, 2.0),), "F002": (Interval(1.99, 2.05),)}

    assert not is_overlapping(segment, speech, min_seconds=0.1)
    assert is_overlapping(segment, speech, min_seconds=0.01)


def test_the_mix_window_carries_context_and_reads_from_video_time() -> None:
    segment = _segment(start=1.0, end=2.0)
    sources = plan_sources(
        [segment],
        {"F001": (Interval(1.0, 2.0),)},
        mix_audio="mix.wav",
        duration=10.0,
        context_seconds=0.25,
        min_overlap_seconds=0.1,
    )

    assert sources[0].origin == 0.75
    assert sources[0].duration == pytest.approx(1.5)
    # The mix's first sample is the video's first sample.
    assert sources[0].file_origin == 0.0
    assert sources[0].seek == pytest.approx(0.75)


def test_the_extraction_seeks_to_its_own_start_not_the_video_start() -> None:
    """The file S7 wrote begins at the segment, so a seek taken from video
    time would read a minute into the past for a segment a minute in."""

    segment = _segment(start=1.0, end=2.0)
    sources = plan_sources(
        [segment],
        {"F001": (Interval(1.0, 2.0),), "F002": (Interval(1.2, 1.8),)},
        mix_audio="mix.wav",
        duration=10.0,
        context_seconds=0.25,
        min_overlap_seconds=0.1,
    )

    assert sources[0].source == SOURCE_EXTRACTED
    assert sources[0].duration == pytest.approx(1.0)
    assert sources[0].origin == 1.0
    assert sources[0].file_origin == 1.0
    assert sources[0].seek == 0.0


def test_the_window_is_clamped_to_the_video_not_the_audio_file() -> None:
    """AAC pads the last frame, so the track runs past the video.

    Reading into that padding would hand the recogniser samples that are not in
    the video, under timestamps claiming they are.
    """

    segment = _segment(start=0.0, end=2.9)
    start, end = window(segment, context_seconds=0.5, duration=3.0)

    assert start == 0.0
    assert end == 3.0


def test_a_segment_past_the_end_of_the_video_reads_itself() -> None:
    """S6 can emit a turn that runs past the video.  A zero-length window would
    fail inside the recogniser, which is a worse answer than the segment."""

    segment = _segment(start=5.0, end=6.0)
    start, end = window(segment, context_seconds=0.25, duration=3.0)

    assert (start, end) == (5.0, 6.0)


def test_sources_are_ordered_by_time() -> None:
    """Two runs over the same video have to produce the same file."""

    segments = [
        _segment(name="F001_0001", start=5.0, end=6.0),
        _segment(name="F002_0000", identity="F002", start=1.0, end=2.0),
        _segment(name="F001_0000", start=3.0, end=4.0),
    ]
    sources = plan_sources(
        segments, {}, mix_audio="mix.wav", duration=10.0,
        context_seconds=0.0, min_overlap_seconds=0.1,
    )

    assert [item.segment.name for item in sources] == [
        "F002_0000",
        "F001_0000",
        "F001_0001",
    ]


# --------------------------------------------------------------------------- #
# words
# --------------------------------------------------------------------------- #


def test_words_join_with_the_spacing_the_recogniser_used() -> None:
    """English tokens carry their own leading space, so joining them is enough.

    Stripping the words and joining on spaces would seem equivalent here and
    break on the next test.
    """

    words = [_word(" I", 0.0, 0.2), _word(" was", 0.2, 0.4), _word(" late", 0.4, 0.6)]
    assert join_words(words) == "I was late"


def test_chinese_words_join_without_spaces() -> None:
    """The tokens have no spaces because the script has none.

    A rule that joined words with a space would produce 我 今天 迟到 -- which is
    wrong in a way that no amount of downstream processing repairs.
    """

    words = [_word("我", 0.0, 0.2), _word("今天", 0.2, 0.4), _word("迟到", 0.4, 0.6)]
    assert join_words(words) == "我今天迟到"


def test_the_join_survives_a_mixed_script() -> None:
    """Code-switching is ordinary and the spacing rule must not fight it."""

    words = [_word("用", 0.0, 0.1), _word(" Python", 0.1, 0.3), _word(" 写", 0.3, 0.5)]
    assert join_words(words) == "用 Python 写"


def test_collapsing_whitespace_leaves_one_space() -> None:
    assert collapse_whitespace("  a \n b\t\tc  ") == "a b c"


def test_padding_words_are_dropped() -> None:
    """A word in the context belongs to whoever spoke before this person."""

    words = [
        _word(" before", 0.70, 0.90),
        _word(" hello", 1.05, 1.25),
        _word(" after", 2.05, 2.25),
    ]
    kept = clip_to_span(words, start=1.0, end=2.0)

    assert [word.text for word in kept] == [" hello"]
    assert join_words(kept) == "hello"


def test_a_word_straddling_the_boundary_goes_to_the_side_it_mostly_lies_in() -> None:
    """Half a word is still a word: the alternative is a sentence with a hole."""

    words = [_word(" mostly", 0.95, 1.20), _word(" barely", 1.95, 2.15)]
    kept = clip_to_span(words, start=1.0, end=2.0)

    assert [word.text for word in kept] == [" mostly"]


def test_offsets_are_added_once() -> None:
    words = [_word(" a", 0.0, 0.2), _word(" b", 0.2, 0.4)]
    moved = offset_words(words, origin=2.5)

    assert [word.start for word in moved] == [2.5, 2.7]
    assert [word.end for word in moved] == [2.7, 2.9]


def test_a_clean_transcript_has_no_flags() -> None:
    assert (
        hallucination_flags(
            words=5,
            avg_logprob=-0.2,
            no_speech_prob=0.01,
            compression_ratio=1.2,
            min_avg_logprob=-1.0,
            max_no_speech_prob=0.6,
            max_compression_ratio=2.4,
        )
        == ()
    )


def test_each_flag_names_a_different_way_of_being_wrong() -> None:
    """The three published heuristics, and the empty case they do not cover."""

    def flags(**kwargs: float) -> tuple[str, ...]:
        base = {
            "words": 5,
            "avg_logprob": -0.2,
            "no_speech_prob": 0.01,
            "compression_ratio": 1.2,
            "min_avg_logprob": -1.0,
            "max_no_speech_prob": 0.6,
            "max_compression_ratio": 2.4,
        }
        return hallucination_flags(**{**base, **kwargs})  # type: ignore[arg-type]

    assert flags(words=0) == ("empty",)
    assert flags(no_speech_prob=0.9) == ("no_speech",)
    assert flags(avg_logprob=-2.0) == ("low_confidence",)
    assert flags(compression_ratio=9.0) == ("repetition",)
    assert flags(words=0, no_speech_prob=0.9) == ("empty", "no_speech")


# --------------------------------------------------------------------------- #
# what language the video is in
# --------------------------------------------------------------------------- #


def test_the_language_is_decided_by_speech_time_not_by_segment_count() -> None:
    """Twenty two-second English clips do not outvote two minutes of Chinese.

    Counting segments would let a video's fringe language win on volume of
    utterances, which is not what a video's language means.
    """

    votes = collect_votes(
        [
            (2.0, _transcription(language="en")) for _ in range(20)
        ]
        + [(60.0, _transcription(language="zh")), (60.0, _transcription(language="zh"))]
    )

    assert votes[0].code == "zh"
    assert votes[0].seconds == pytest.approx(120.0)
    assert decide(votes) == "zh"


def test_a_segment_too_short_to_be_evidence_does_not_vote() -> None:
    votes = collect_votes([(1.0, _transcription(language="en"))])
    assert votes == ()


def test_an_unsure_segment_does_not_vote() -> None:
    votes = collect_votes([(10.0, _transcription(language="en", probability=0.2))])
    assert votes == ()


def test_a_forced_language_does_not_vote_for_itself() -> None:
    """Otherwise the guess would confirm itself and look like evidence."""

    votes = collect_votes([(10.0, _transcription(language="en", detected=False))])
    assert votes == ()


def test_a_video_with_nothing_long_enough_says_so() -> None:
    """Rather than naming whichever segment happened to be longest."""

    assert decide(()) == "unknown"


def test_a_tie_resolves_the_same_way_every_run() -> None:
    votes = collect_votes(
        [(5.0, _transcription(language="zh")), (5.0, _transcription(language="en"))]
    )
    assert [vote.code for vote in votes] == ["en", "zh"]


def test_disagreements_are_counted_not_corrected() -> None:
    """A code-switch and a detection error look identical from here."""

    counted = disagreements(
        [
            (10.0, _transcription(language="en")),
            (8.0, _transcription(language="en")),
            (4.0, _transcription(language="fr")),
            (2.0, _transcription(language="fr")),
        ],
        language="en",
    )

    assert counted == (("fr", 4.0),)


# --------------------------------------------------------------------------- #
# reading the audio
# --------------------------------------------------------------------------- #


def test_a_file_at_the_wrong_rate_is_an_error_not_a_resample(tmp_path: Path) -> None:
    """8 kHz audio does not make the recogniser fail.

    It transcribes the wrong frequencies against the wrong time base and
    returns timestamps at half scale, so every word lands in the wrong place
    and reads plausibly while doing it.  The check lives in the WAV layer, so
    every stage that reads segment audio gets it.
    """

    path = write_pcm16(tmp_path / "eight.wav", np.zeros(8000, dtype=np.float32), sample_rate=8000)
    source = SegmentSource(
        segment=_segment(audio=str(path)),
        path=str(path),
        origin=0.0,
        file_origin=0.0,
        duration=1.0,
        source=SOURCE_EXTRACTED,
        overlapping=False,
    )

    with pytest.raises(WavError, match="8000 Hz"):
        read_source(source, root=tmp_path, sample_rate=RATE)


def test_a_segment_file_is_read_from_its_own_start(tmp_path: Path) -> None:
    """The extraction's first sample is the segment's first sample, not the
    video's -- so a seek computed from video time would read into the past."""

    samples = np.linspace(-0.5, 0.5, RATE, dtype=np.float32)
    path = write_pcm16(tmp_path / "seg.wav", samples, sample_rate=RATE)
    source = SegmentSource(
        segment=_segment(start=4.0, end=5.0, audio=str(path)),
        path=str(path),
        origin=4.0,
        file_origin=4.0,
        duration=1.0,
        source=SOURCE_EXTRACTED,
        overlapping=False,
    )

    read = read_source(source, root=tmp_path, sample_rate=RATE)

    assert len(read) == RATE
    assert read[0] == pytest.approx(float(samples[0]), abs=1e-4)


def test_the_mix_is_read_at_video_time(tmp_path: Path) -> None:
    samples = np.zeros(3 * RATE, dtype=np.float32)
    samples[2 * RATE :] = 0.5
    path = write_pcm16(tmp_path / "mix.wav", samples, sample_rate=RATE)
    source = SegmentSource(
        segment=_segment(start=2.0, end=2.5),
        path=str(path),
        origin=2.0,
        file_origin=0.0,
        duration=0.5,
        source=SOURCE_MIX,
        overlapping=False,
    )

    read = read_source(source, root=tmp_path, sample_rate=RATE)

    assert len(read) == RATE // 2
    assert float(read.mean()) == pytest.approx(0.5, abs=1e-3)


def test_concat_separates_clips_with_silence() -> None:
    """Butting two utterances together invents a word boundary nobody spoke."""

    joined = concat(
        [np.full(100, 0.5, dtype=np.float32), np.full(100, 0.5, dtype=np.float32)],
        gap_seconds=0.01,
        sample_rate=RATE,
    )

    assert len(joined) == 100 + 160 + 100
    assert float(joined[100:260].max()) == 0.0


# --------------------------------------------------------------------------- #
# the stage
# --------------------------------------------------------------------------- #


class _StubTranscriber:
    """Reports words on a fixed grid, so the test can predict every timestamp."""

    name = "stub"

    def __init__(
        self,
        *,
        language: str = "en",
        probability: float = 0.95,
        seconds_per_word: float = 0.5,
        avg_logprob: float = -0.2,
        no_speech_prob: float = 0.01,
    ) -> None:
        self.language = language
        self.probability = probability
        self.seconds_per_word = seconds_per_word
        self.avg_logprob = avg_logprob
        self.no_speech_prob = no_speech_prob
        #: ``(seconds of audio, language asked for)`` per call.
        self.calls: list[tuple[float, str | None]] = []
        #: Seconds of audio per detection, which is not a transcription.
        self.detects: list[float] = []

    def detect(self, samples: np.ndarray) -> tuple[str, float]:
        self.detects.append(round(len(samples) / RATE, 4))
        return self.language, self.probability

    def transcribe(self, samples: np.ndarray, *, language: str | None) -> Transcription:
        seconds = len(samples) / RATE
        self.calls.append((round(seconds, 4), language))

        words: list[TranscribedWord] = []
        step = self.seconds_per_word
        index = 0
        while (index + 0.5) * step < seconds:
            start = (index + 0.1) * step
            words.append(
                TranscribedWord(
                    text=f" w{index}", start=start, end=start + step * 0.8
                )
            )
            index += 1

        forced = language is not None
        return Transcription(
            text=join_words(words),
            words=tuple(words),
            language=language or self.language,
            language_probability=0.0 if forced else self.probability,
            detected=not forced,
            avg_logprob=self.avg_logprob,
            no_speech_prob=self.no_speech_prob,
            compression_ratio=1.2,
        )


def _context(source: Path, root: Path, **config: object) -> StageContext:
    return StageContext(
        video_id=source.stem,
        source=source,
        work_dir=root / "work" / source.stem,
        config=config,
    )


def _write_segments(context: StageContext, segments: list[SpeechSegment]) -> None:
    """S7's artifact, written by hand: S8 only reads it."""

    context.output(s7_tse.STAGE, s7_tse.SEGMENTS_NAME).write_text(
        json.dumps(
            {
                "schema_version": "x",
                "segments": [
                    {
                        **segment.to_dict(),
                        "audio": _write_segment_audio(context, segment),
                        "sample_rate": RATE,
                        "silent": False,
                    }
                    for segment in segments
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_segment_audio(context: StageContext, segment: SpeechSegment) -> str:
    """A tone as long as the segment, at the path S7 would have written."""

    relative = f"{s7_tse.STAGE}/audio/{segment.identity}/{segment.name}.wav"
    target = context.work_dir / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    count = max(1, int(round(segment.duration * RATE)))
    write_pcm16(
        target, (np.sin(np.linspace(0, 50, count)) * 0.4).astype(np.float32), sample_rate=RATE
    )
    return relative


def _write_speech(context: StageContext, speech: dict[str, list[list[float]]]) -> None:
    """S6's artifact: who was speaking when, which is what routing reads."""

    context.output(s6_associate.STAGE, s6_associate.ASSIGNMENTS_NAME).write_text(
        json.dumps(
            {
                "schema_version": "x",
                "assignments": [],
                "identities": [
                    {"face_id": face_id, "track_ids": [], "speaking_intervals": intervals}
                    for face_id, intervals in sorted(speech.items())
                ],
            }
        ),
        encoding="utf-8",
    )


def _staged(
    source: Path,
    root: Path,
    *,
    segments: list[SpeechSegment],
    speech: dict[str, list[list[float]]],
    **config: object,
) -> StageContext:
    context = _context(source, root, **config)
    s0_preprocess.run(context)
    _write_speech(context, speech)
    _write_segments(context, segments)
    return context


SPEECH = {"F001": [[0.5, 1.5]]}
SEGMENTS = [_segment(start=0.5, end=1.5)]


def test_the_stage_writes_a_transcript_per_segment(
    single_shot_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _staged(single_shot_video, tmp_path, segments=SEGMENTS, speech=SPEECH)
    stub = _StubTranscriber()
    monkeypatch.setattr(s8_asr, "build_transcriber", lambda _: stub)

    result = s8_asr.run(context)

    assert not result.skipped
    assert result.summary["transcribed"] == 1

    written = s8_asr.load_transcripts(context)
    assert len(written) == 1
    assert written[0]["identity"] == "F001"
    assert written[0]["text"]
    assert written[0]["words"]


def test_words_land_on_the_video_timeline_not_the_slice(
    single_shot_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The extraction starts at the segment, so a word at 0.0 in the file is at
    0.5 in the video.  A transcript half a second out looks perfectly fine."""

    context = _staged(
        single_shot_video,
        tmp_path,
        segments=SEGMENTS,
        speech=SPEECH,
        min_detect_seconds=0.4,
        # No padding, so the slice begins exactly at the segment's start and
        # the expected offsets are the segment's own times.
        context_seconds=0.0,
    )
    monkeypatch.setattr(s8_asr, "build_transcriber", lambda _: _StubTranscriber())

    s8_asr.run(context)
    written = s8_asr.load_transcripts(context)[0]

    # The stub puts its first word at 0.1 * 0.5 = 0.05 s into the slice, which
    # for a segment starting at 0.5 s is 0.55 s into the video.
    first = written["words"][0]
    assert first["start"] == pytest.approx(0.5 + 0.05, abs=1e-3)
    assert first["end"] <= 1.5


def test_a_long_segment_is_asked_its_language_and_a_short_one_is_told(
    single_shot_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole reason for two passes: detection on a two-second clip is a
    confident guess, and a wrong language makes the recogniser translate."""

    long_segment = _segment(name="F001_0000", start=0.1, end=1.9)
    short_segment = _segment(name="F001_0001", start=2.0, end=2.4)
    context = _staged(
        single_shot_video,
        tmp_path,
        segments=[long_segment, short_segment],
        speech={"F001": [[0.1, 1.9], [2.0, 2.4]]},
        min_detect_seconds=1.0,
    )
    stub = _StubTranscriber(language="zh")
    monkeypatch.setattr(s8_asr, "build_transcriber", lambda _: stub)

    result = s8_asr.run(context)

    asked = [language for seconds, language in stub.calls if seconds > 1.0]
    told = [language for seconds, language in stub.calls if seconds <= 1.0]
    assert asked == [None]
    assert told == ["zh"]
    assert result.summary["language"] == "zh"


def test_a_video_with_no_long_segment_falls_back_to_a_sample(
    single_shot_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _staged(
        single_shot_video,
        tmp_path,
        segments=[_segment(start=0.5, end=1.5)],
        speech=SPEECH,
        min_detect_seconds=30.0,
    )
    stub = _StubTranscriber(language="ja")
    monkeypatch.setattr(s8_asr, "build_transcriber", lambda _: stub)

    result = s8_asr.run(context)

    # One detection over the assembled sample, then one forced transcription.
    # The sample is read through the planned source, so it carries the same
    # context padding a transcription would: 1.0 s of segment plus 2 x 0.25.
    assert stub.detects == [pytest.approx(1.5, abs=0.02)]
    assert [language for _, language in stub.calls] == ["ja"]
    assert result.summary["language"] == "ja"
    assert s8_asr.load_language(context).source == "sample"


def test_an_overlapping_segment_is_transcribed_from_its_extraction(
    single_shot_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """And says so, because a transcript from the extraction is a different
    kind of evidence from one taken off the mix."""

    context = _staged(
        single_shot_video,
        tmp_path,
        segments=[_segment(start=0.5, end=1.5)],
        speech={"F001": [[0.5, 1.5]], "F002": [[0.4, 1.6]]},
        min_detect_seconds=0.4,
    )
    stub = _StubTranscriber()
    monkeypatch.setattr(s8_asr, "build_transcriber", lambda _: stub)

    s8_asr.run(context)
    written = s8_asr.load_transcripts(context)[0]

    assert written["source"] == SOURCE_EXTRACTED
    assert written["overlapping"] is True
    assert written["audio_source"].endswith("F001_0000.wav")
    # A 1.0 s segment, read from a file that is exactly 1.0 s of audio.
    assert stub.calls[0][0] == pytest.approx(1.0, abs=0.01)


def test_a_non_overlapping_segment_reads_the_mix_with_context(
    single_shot_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _staged(
        single_shot_video,
        tmp_path,
        segments=[_segment(start=0.5, end=1.5)],
        speech=SPEECH,
        min_detect_seconds=0.4,
        context_seconds=0.25,
    )
    stub = _StubTranscriber()
    monkeypatch.setattr(s8_asr, "build_transcriber", lambda _: stub)

    s8_asr.run(context)
    written = s8_asr.load_transcripts(context)[0]

    assert written["source"] == SOURCE_MIX
    assert written["overlapping"] is False
    assert stub.calls[0][0] == pytest.approx(1.5, abs=0.02)


def test_a_mismatched_language_is_flagged_and_left_alone(
    single_shot_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _staged(
        single_shot_video,
        tmp_path,
        segments=[
            _segment(name="F001_0000", start=0.1, end=1.9),
            _segment(name="F002_0000", identity="F002", start=2.0, end=2.9),
        ],
        speech={"F001": [[0.1, 1.9]], "F002": [[2.0, 2.9]]},
        min_detect_seconds=0.5,
    )
    speeches = {"en": 0}

    class _Switching(_StubTranscriber):
        def transcribe(self, samples: np.ndarray, *, language: str | None) -> Transcription:
            speeches["en"] += 1
            language_now = "en" if speeches["en"] == 1 else "fr"
            result = super().transcribe(samples, language=language)
            return Transcription(**{**result.__dict__, "language": language or language_now})

    monkeypatch.setattr(s8_asr, "build_transcriber", lambda _: _Switching())

    s8_asr.run(context)
    written = {item["name"]: item for item in s8_asr.load_transcripts(context)}

    assert written["F001_0000"]["language"] == "en"
    assert written["F002_0000"]["language"] == "fr"
    assert "language_mismatch" in written["F002_0000"]["flags"]
    assert s8_asr.load_language(context).code == "en"


def test_a_silent_extraction_is_carried_through(
    single_shot_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S7 says the extraction came out silent; that is the reason to distrust
    whatever the recogniser made of it, so it must survive into S8's output."""

    context = _staged(
        single_shot_video,
        tmp_path,
        segments=SEGMENTS,
        speech=SPEECH,
        min_detect_seconds=0.4,
    )
    payload = json.loads((context.work_dir / s7_tse.STAGE / s7_tse.SEGMENTS_NAME).read_text())
    payload["segments"][0]["silent"] = True
    (context.work_dir / s7_tse.STAGE / s7_tse.SEGMENTS_NAME).write_text(
        json.dumps(payload), encoding="utf-8"
    )
    monkeypatch.setattr(s8_asr, "build_transcriber", lambda _: _StubTranscriber())

    s8_asr.run(context)

    assert "silent_extraction" in s8_asr.load_transcripts(context)[0]["flags"]


def test_a_video_with_no_segments_writes_empty_outputs(
    single_shot_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _staged(single_shot_video, tmp_path, segments=[], speech={})
    monkeypatch.setattr(s8_asr, "build_transcriber", lambda _: _StubTranscriber())

    result = s8_asr.run(context)

    assert result.summary["transcribed"] == 0
    assert s8_asr.load_transcripts(context) == ()
    assert s8_asr.load_language(context).code == "unknown"


def test_second_run_skips(
    single_shot_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _staged(single_shot_video, tmp_path, segments=SEGMENTS, speech=SPEECH)
    monkeypatch.setattr(s8_asr, "build_transcriber", lambda _: _StubTranscriber())

    assert not s8_asr.run(context).skipped
    assert s8_asr.run(context).skipped


def test_loaders_fail_loudly_when_the_stage_has_not_run(tmp_path: Path) -> None:
    context = StageContext(
        video_id="x", source=tmp_path / "x.mp4", work_dir=tmp_path / "work" / "x"
    )
    with pytest.raises(FileNotFoundError, match="run s8-asr first"):
        s8_asr.load_transcripts(context)
    with pytest.raises(FileNotFoundError, match="run s8-asr first"):
        s8_asr.transcripts_path(context)
