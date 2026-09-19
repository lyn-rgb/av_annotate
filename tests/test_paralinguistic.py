"""Tests for the tag vocabulary and the reduction from three models to one tag.

None of the three taggers is installed here, so the model adapters are not
exercised.  What is exercised is the part that decides what a rendered script
says -- and the failure there is not an exception, it is a tag that reads
plausibly and is wrong, or a tag that breaks the parse of every line it appears
on.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from avannotate.annotation import parse_script, render_script
from avannotate.audio.wav import WavError, write_pcm16
from avannotate.paralinguistic.reduce import ReduceConfig, reduce_tags, unmapped_labels
from avannotate.paralinguistic.types import DIMENSIONS, SegmentTags, TagScore
from avannotate.paralinguistic.vocabulary import (
    ALL_TAGS,
    DELIVERY_TAGS,
    EMOTION_TAGS,
    EVENT_TAGS,
    is_tag,
    slug,
    tag_for,
    unmapped,
)
from avannotate.schema import TAG_PATTERN, Annotation, Language, Utterance, VideoMeta
from avannotate.segment import SpeechSegment
from avannotate.stages import s7_tse, s9_paralinguistic
from avannotate.stages.base import StageContext

RATE = 16000


def _scores(**named: float) -> tuple[TagScore, ...]:
    return tuple(TagScore(label=label, score=score) for label, score in named.items())


# --------------------------------------------------------------------------- #
# the vocabulary
# --------------------------------------------------------------------------- #


def test_every_tag_survives_the_script_format() -> None:
    """The invariant the whole closed vocabulary exists to guarantee.

    A tag is rendered inside the utterance line and recovered by the parser's
    ``[\\w-]+``.  A tag with a space or a colon in it would not come back, and
    it would not fail loudly either -- it would be parsed as part of the spoken
    text, on every line it appeared on.
    """

    import re

    pattern = re.compile(TAG_PATTERN)
    for tag in sorted(ALL_TAGS):
        assert pattern.fullmatch(tag), f"{tag!r} would not survive the round trip"
        assert tag == tag.lower()
        assert is_tag(tag)


def test_the_three_dimensions_do_not_claim_the_same_tag() -> None:
    """A tag has to mean one thing, or precedence decides nothing."""

    assert not EMOTION_TAGS & DELIVERY_TAGS
    assert not EMOTION_TAGS & EVENT_TAGS
    assert not DELIVERY_TAGS & EVENT_TAGS


@pytest.mark.parametrize(
    ("dimension", "label", "expected"),
    [
        # emotion2vec's labels are bilingual with a slash, which is why the
        # table carries the joined form rather than the English half alone.
        ("emotion", "生气/angry", "angry"),
        ("emotion", "开心/happy", "happy"),
        ("emotion", "Sad", "sad"),
        ("emotion", "Neutral", None),
        # AudioSet spells its classes with punctuation and spaces, so a slug is
        # the only thing that can key a table over them.
        ("event", "Laughter", "laughter"),
        ("event", "Crying, sobbing", "crying"),
        ("event", "Throat clearing", "throat-clearing"),
        ("event", "Belly laugh", "laughter"),
        ("event", "Cough", "cough"),
        # The voice tagger's own spellings -- this one is a generator over an
        # open vocabulary, so these are the published strings and nothing else
        # is guaranteed to match.
        ("delivery", "Whispering", "whispering"),
        ("delivery", "  whispered  ", "whispering"),
        ("delivery", "whispery voice", "whispering"),
        ("delivery", "laughing while speaking", "laughing"),
        ("delivery", "giggling delivery", "laughing"),
    ],
)
def test_labels_map_to_tags(dimension: str, label: str, expected: str | None) -> None:
    assert tag_for(label, dimension=dimension) == expected


def test_a_label_the_model_invented_maps_to_nothing() -> None:
    """The voice tagger generates rather than classifies, so its output space
    is open -- its own card shows it emitting strings it never lists."""

    for label in ("natural-Sounding", "natural pop", "natural-Suitable for Work"):
        assert tag_for(label, dimension="delivery") is None


def test_a_label_meaning_nothing_becomes_no_tag() -> None:
    """``neutral`` is a result, not a description.

    The format makes the tag optional so an utterance with nothing to say about
    it renders without one.  A rendered ``neutral:`` is worse than silence --
    it looks like a finding.
    """

    for label in ("neutral", "other", "unknown", "Neutral", "其他/other"):
        assert tag_for(label, dimension="emotion") is None


def test_an_unknown_label_is_dropped_and_reported() -> None:
    """A checkpoint that adds a class is not a reason to fail a batch, but it
    is a reason for somebody to look."""

    assert tag_for("screeching", dimension="delivery") is None
    found = unmapped(
        [
            TagScore(label="whispering", score=0.9),
            TagScore(label="screeching", score=0.9),
            TagScore(label="neutral", score=0.9),
        ],
        dimension="delivery",
        min_score=0.5,
    )
    assert found == ("screeching",)


def test_an_unmapped_label_below_the_threshold_is_not_reported() -> None:
    """The signal is *the model is sure of something we have no word for*.

    Every tagger returns a long tail it barely believes; reporting that from
    day one would bury the case the report exists for.
    """

    found = unmapped(
        [TagScore(label="screeching", score=0.05)], dimension="delivery", min_score=0.5
    )
    assert found == ()


def test_a_known_but_unused_label_is_not_reported_as_unknown() -> None:
    """AudioSet's ``Speech`` fires on every speech segment and ``Whispering``
    belongs to the delivery dimension.

    Counting those as unmapped would make the report a constant rather than a
    signal about the model.
    """

    for label in ("Speech", "Conversation", "Whispering", "Shout", "Music"):
        assert unmapped(
            [TagScore(label=label, score=0.9)], dimension="event", min_score=0.5
        ) == ()
        assert tag_for(label, dimension="event") is None


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("Laughter", "laughter"),
        ("Crying, sobbing", "crying-sobbing"),
        ("Throat clearing", "throat-clearing"),
        ("生气/angry", "生气-angry"),
        ("  Spaced  out  ", "spaced-out"),
        ("hyphen-already", "hyphen-already"),
        ("Belly laugh", "belly-laugh"),
    ],
)
def test_labels_slug_to_something_a_table_can_key_on(label: str, expected: str) -> None:
    """The tables are keyed by slug, so a slug that does not round trip would
    make an entry unreachable without ever failing."""

    assert slug(label) == expected
    assert is_tag(slug(label))


# --------------------------------------------------------------------------- #
# three opinions, one tag
# --------------------------------------------------------------------------- #


def test_an_event_beats_delivery_and_delivery_beats_emotion() -> None:
    """The order is the design: the most specific description wins.

    All three can be true at once -- a line can be whispered, frightened, and
    punctuated by a cough -- and the one a reader could not have guessed from
    the words is the event.
    """

    choice = reduce_tags(
        {
            "emotion": _scores(surprised=0.9),
            "delivery": _scores(whispering=0.9),
            "event": _scores(laughter=0.9),
        }
    )
    assert choice.tag == "laughter"
    assert choice.dimension == "event"

    without_event = reduce_tags(
        {"emotion": _scores(surprised=0.9), "delivery": _scores(whispering=0.9)}
    )
    assert without_event.tag == "whispering"
    assert without_event.dimension == "delivery"


def test_a_smaller_score_in_a_higher_dimension_still_wins() -> None:
    """Precedence is not a comparison of scores.

    The three models are calibrated differently -- one softmax over nine
    classes, two sigmoids over hundreds -- so letting the largest number win
    would make the priority between dimensions an artefact of that.
    """

    choice = reduce_tags(
        {"emotion": _scores(surprised=0.99), "event": _scores(laughter=0.55)}
    )
    assert choice.tag == "laughter"


def test_the_highest_score_within_a_dimension_wins() -> None:
    choice = reduce_tags({"emotion": _scores(angry=0.6, happy=0.8, sad=0.7)})
    assert choice.tag == "happy"


def test_a_label_below_its_threshold_does_not_win() -> None:
    choice = reduce_tags({"emotion": _scores(angry=0.4)})
    assert choice.tag is None
    assert "threshold" in choice.reason


def test_a_label_the_vocabulary_does_not_use_does_not_block_the_one_below_it() -> None:
    """The most confident thing about a quiet conversation is often ``Speech``.

    Taking the top label and then failing to map it would discard a real
    detection sitting just below it.
    """

    choice = reduce_tags(
        {"delivery": _scores(speech=0.97, whispering=0.62, music=0.5)}
    )
    assert choice.tag == "whispering"
    assert choice.label == "whispering"


def test_two_labels_meaning_the_same_tag_keep_the_best_score() -> None:
    choice = reduce_tags({"event": _scores(giggling=0.6, laughter=0.8)})
    assert choice.tag == "laughter"
    assert choice.score == pytest.approx(0.8)


def test_nothing_worth_saying_means_no_tag() -> None:
    choice = reduce_tags(
        {"emotion": _scores(neutral=0.99), "delivery": (), "event": ()}
    )
    assert choice.tag is None
    assert choice.candidates == ()


def test_the_runner_up_is_kept() -> None:
    """When a tag reads wrong the question is what the alternative was."""

    choice = reduce_tags(
        {"emotion": _scores(surprised=0.9), "delivery": _scores(whispering=0.87)}
    )
    assert [candidate.dimension for candidate in choice.candidates] == [
        "delivery",
        "emotion",
    ]
    assert choice.margin == pytest.approx(-0.03)


def test_a_dimension_can_win_on_precedence_while_scoring_lower() -> None:
    """The case worth recording: right by the rule, wrong by the numbers.

    Precedence decides, so a frightened whisper scores higher for fear and is
    still tagged ``whispering``.  The margin is what carries that, and it is
    negative -- which is a finding, not a bug.
    """

    choice = reduce_tags(
        {"emotion": _scores(surprised=0.95), "delivery": _scores(whispering=0.6)}
    )

    assert choice.tag == "whispering"
    assert choice.score == pytest.approx(0.6)
    assert choice.margin == pytest.approx(-0.35)
    assert "higher-scoring" in choice.reason


def test_the_reason_distinguishes_a_clear_win_from_a_narrow_one() -> None:
    """Three outcomes a reader has to be able to tell apart: the winner was
    well clear, the winner was barely ahead, and the winner lost on score."""

    clear = reduce_tags(
        {"delivery": _scores(whispering=0.95), "emotion": _scores(surprised=0.30)}
    )
    narrow = reduce_tags(
        {"delivery": _scores(whispering=0.95), "emotion": _scores(surprised=0.90)}
    )
    on_score = reduce_tags(
        {"delivery": _scores(whispering=0.60), "emotion": _scores(surprised=0.95)}
    )

    assert clear.reason == "delivery takes precedence"
    assert narrow.reason == "delivery takes precedence over a close emotion"
    assert on_score.reason == "delivery takes precedence over a higher-scoring emotion"


def test_thresholds_are_per_dimension() -> None:
    """One score is a softmax over nine classes and the others are sigmoids
    over hundreds; the same number does not mean the same thing."""

    config = ReduceConfig(min_score={"emotion": 0.2, "delivery": 0.9, "event": 0.9})
    choice = reduce_tags(
        {"emotion": _scores(angry=0.4), "delivery": _scores(whispering=0.5)},
        config=config,
    )

    assert choice.tag == "angry"


def test_unmapped_labels_are_collected_across_dimensions() -> None:
    """``music`` is a label the event dimension knows and deliberately ignores,
    so it is not evidence that anything has moved."""

    found = unmapped_labels(
        {"emotion": _scores(scornful=0.9), "event": _scores(music=0.9, yodelling=0.8)}
    )
    assert set(found) == {"scornful", "yodelling"}


# --------------------------------------------------------------------------- #
# the stage
# --------------------------------------------------------------------------- #


class _StubTagger:
    """Reports whatever the test says, so the reduction is what is under test."""

    name = "stub"

    def __init__(self, **reported: tuple[TagScore, ...]) -> None:
        self.reported = reported
        self.dimensions = DIMENSIONS
        self.calls: list[float] = []

    def tag(self, samples: np.ndarray) -> dict[str, tuple[TagScore, ...]]:
        self.calls.append(round(len(samples) / RATE, 4))
        return {dimension: self.reported.get(dimension, ()) for dimension in DIMENSIONS}


def _context(source: Path, root: Path, **config: object) -> StageContext:
    return StageContext(
        video_id=source.stem,
        source=source,
        work_dir=root / "work" / source.stem,
        config=config,
    )


def _write_segments(context: StageContext, segments: list[SpeechSegment]) -> None:
    """S7's artifact, written by hand: S9 only reads it."""

    written = []
    for segment in segments:
        relative = f"{s7_tse.STAGE}/audio/{segment.identity}/{segment.name}.wav"
        target = context.work_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        count = max(1, int(round(segment.duration * RATE)))
        write_pcm16(target, np.zeros(count, dtype=np.float32), sample_rate=RATE)
        written.append({**segment.to_dict(), "audio": relative, "sample_rate": RATE})
    context.output(s7_tse.STAGE, s7_tse.SEGMENTS_NAME).write_text(
        json.dumps({"schema_version": "x", "segments": written}), encoding="utf-8"
    )


def _segment(name: str = "F001_0000", *, start: float = 0.0, end: float = 1.0) -> SpeechSegment:
    return SpeechSegment(
        identity="F001",
        name=name,
        start=start,
        end=end,
        audio=f"{s7_tse.STAGE}/audio/F001/{name}.wav",
    )


def test_the_stage_tags_each_segment(
    single_shot_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(single_shot_video, tmp_path)
    _write_segments(context, [_segment()])
    monkeypatch.setattr(
        s9_paralinguistic,
        "build_tagger",
        lambda _: _StubTagger(
            delivery=_scores(whispering=0.9), emotion=_scores(neutral=0.9)
        ),
    )

    result = s9_paralinguistic.run(context)

    assert not result.skipped
    assert result.summary["tagged"] == 1
    assert s9_paralinguistic.load_segment_tag(context, "F001_0000") == "whispering"


def test_the_stage_reads_the_extraction_not_the_mix(
    single_shot_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Opposite of S8's routing, for the opposite reason: a general audio
    tagger on a mix reports what is audible rather than what this person did."""

    context = _context(single_shot_video, tmp_path)
    _write_segments(context, [_segment(end=1.5)])
    stub = _StubTagger(event=_scores(cough=0.8))
    monkeypatch.setattr(s9_paralinguistic, "build_tagger", lambda _: stub)

    s9_paralinguistic.run(context)

    assert stub.calls == [pytest.approx(1.5, abs=0.01)]
    written = s9_paralinguistic.load_tags(context)[0]
    assert written["tag"] == "cough"


def test_a_segment_too_short_to_tag_says_so_rather_than_guessing(
    single_shot_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Classifiers are unreliable on sub-second clips, and a wrong tag is worse
    than none -- the format makes it optional exactly so it can be withheld."""

    context = _context(single_shot_video, tmp_path)
    _write_segments(context, [_segment(end=0.4)])
    stub = _StubTagger(event=_scores(cough=0.99))
    monkeypatch.setattr(s9_paralinguistic, "build_tagger", lambda _: stub)

    result = s9_paralinguistic.run(context)

    assert stub.calls == []
    assert result.summary["tagged"] == 0
    assert result.summary["too_short"] == 1

    written = s9_paralinguistic.load_tags(context)[0]
    assert written["eligible"] is False
    assert written["tag"] is None


def test_a_segment_with_no_strong_signal_gets_no_tag(
    single_shot_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(single_shot_video, tmp_path)
    _write_segments(context, [_segment()])
    monkeypatch.setattr(
        s9_paralinguistic,
        "build_tagger",
        lambda _: _StubTagger(emotion=_scores(neutral=0.99), delivery=_scores(breathy=0.3)),
    )

    result = s9_paralinguistic.run(context)

    assert result.summary["tagged"] == 0
    assert result.summary["untagged"] == 1


def test_every_model_is_asked_and_every_answer_is_kept(
    single_shot_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-running to find out which model was wrong would mean re-running all
    three, so all three answers are kept."""

    context = _context(single_shot_video, tmp_path)
    _write_segments(context, [_segment()])
    monkeypatch.setattr(
        s9_paralinguistic,
        "build_tagger",
        lambda _: _StubTagger(
            event=_scores(laughter=0.8),
            delivery=_scores(whispering=0.7),
            emotion=_scores(happy=0.6),
        ),
    )

    s9_paralinguistic.run(context)
    written = s9_paralinguistic.load_tags(context)[0]

    assert written["tag"] == "laughter"
    assert written["choice"]["dimension"] == "event"
    assert [item["label"] for item in written["scores"]["delivery"]] == ["whispering"]
    assert [item["label"] for item in written["scores"]["emotion"]] == ["happy"]


def test_the_summary_counts_where_the_tags_came_from(
    single_shot_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(single_shot_video, tmp_path)
    _write_segments(
        context,
        [
            _segment("F001_0000", start=0.0, end=1.0),
            _segment("F001_0001", start=1.2, end=2.2),
        ],
    )
    monkeypatch.setattr(
        s9_paralinguistic,
        "build_tagger",
        lambda _: _StubTagger(delivery=_scores(whispering=0.9)),
    )

    s9_paralinguistic.run(context)
    summary = json.loads(
        (context.work_dir / s9_paralinguistic.STAGE / s9_paralinguistic.SUMMARY_NAME).read_text()
    )

    assert summary["by_tag"] == {"whispering": 2}
    assert summary["by_dimension"] == {"delivery": 2}


def test_a_file_at_the_wrong_rate_is_refused(
    single_shot_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard lives in the WAV layer, so every stage that reads segment
    audio gets it rather than each one remembering to check."""

    context = _context(single_shot_video, tmp_path)
    segment = _segment()
    _write_segments(context, [segment])
    path = context.work_dir / segment.audio
    write_pcm16(path, np.zeros(8000, dtype=np.float32), sample_rate=8000)
    monkeypatch.setattr(s9_paralinguistic, "build_tagger", lambda _: _StubTagger())

    with pytest.raises(WavError, match="8000 Hz"):
        s9_paralinguistic.run(context)


def test_second_run_skips(
    single_shot_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(single_shot_video, tmp_path)
    _write_segments(context, [_segment()])
    monkeypatch.setattr(s9_paralinguistic, "build_tagger", lambda _: _StubTagger())

    assert not s9_paralinguistic.run(context).skipped
    assert s9_paralinguistic.run(context).skipped


def test_loaders_fail_loudly_when_the_stage_has_not_run(tmp_path: Path) -> None:
    context = StageContext(
        video_id="x", source=tmp_path / "x.mp4", work_dir=tmp_path / "work" / "x"
    )
    with pytest.raises(FileNotFoundError, match="run s9-paralinguistic first"):
        s9_paralinguistic.load_tags(context)
    with pytest.raises(FileNotFoundError, match="run s9-paralinguistic first"):
        s9_paralinguistic.tags_path(context)


# --------------------------------------------------------------------------- #
# the claim, end to end
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("tag", sorted(ALL_TAGS))
def test_every_tag_renders_and_parses_back(tag: str) -> None:
    """The closed vocabulary is only worth having if it holds.

    Checked by rendering an annotation with each tag and reading it back with
    the real parser -- the same one the deliverable's consumers use.
    """

    annotation = Annotation(
        video=VideoMeta(
            video_id="v", path="v.mp4", duration=10.0, fps=25.0, width=320, height=240
        ),
        utterances=(
            Utterance(
                face_id="F001",
                start=0.5,
                end=2.0,
                text="I was late for work today",
                tag=tag,
            ),
        ),
        language=Language(code="en"),
    )

    text = render_script(annotation)
    parsed = parse_script(text)

    assert len(parsed.utterances) == 1
    assert parsed.utterances[0].tag == tag
    assert parsed.utterances[0].text == "I was late for work today"


def test_a_segment_record_round_trips_through_json() -> None:
    """The stage's own record, checked for the fields a reader needs."""

    tags = SegmentTags(
        identity="F001",
        name="F001_0000",
        start=1.0,
        end=2.0,
        choice=reduce_tags({"event": _scores(laughter=0.8)}),
        event=_scores(laughter=0.8),
    )
    payload = json.loads(json.dumps(tags.to_dict()))

    assert payload["tag"] == "laughter"
    assert payload["duration"] == 1.0
    assert payload["choice"]["candidates"][0]["dimension"] == "event"
