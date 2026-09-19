"""Tests for the script grammar.

The round-trip test is the important one: it is the property that keeps the
renderer and the parser honest, and it is the same check `qa.py` runs on every
video before the annotation is allowed out.
"""

from __future__ import annotations

import pytest

from avannotate.annotation import (
    AnnotationFormatError,
    annotation_from_dict,
    parse_script,
    render_flat,
    render_script,
)
from avannotate.schema import (
    Annotation,
    Language,
    Shot,
    Utterance,
    VideoMeta,
    new_face_id,
    parse_face_id,
)

VIDEO = VideoMeta(
    video_id="sample",
    path="inbox/sample.mp4",
    duration=31.0,
    fps=25.0,
    width=1920,
    height=1080,
)


def _example() -> Annotation:
    return Annotation(
        video=VIDEO,
        global_caption="Two people are discussing in a spacious living room.",
        language=Language(code="en", confidence=0.98, source="whisper_lid"),
        shots=(
            Shot(
                index=1,
                start=0.0,
                end=12.4,
                caption="A bright living room with a sofa and a coffee table.",
            ),
            Shot(index=2, start=12.4, end=31.0, caption="The camera moves closer to the window."),
        ),
        utterances=(
            Utterance(face_id="F001", start=1.24, end=3.02, text="I was late for work today",
                      tag="whispering"),
            Utterance(face_id="F002", start=3.4, end=4.1, text="What's going on?", tag="surprised"),
            Utterance(face_id="F001", start=20.0, end=21.0, text="I forgot my phone"),
        ),
    )


def test_renders_the_specified_example_verbatim() -> None:
    """The format was specified by example; this pins it byte for byte."""

    expected = (
        "[GLOBAL]\n"
        "Two people are discussing in a spacious living room.\n"
        "\n"
        "[SHOT 1 0.0s-12.4s]\n"
        "A bright living room with a sofa and a coffee table.\n"
        "<F001> whispering: <S>I was late for work today</S>\n"
        "<F002> surprised: <S>What's going on?</S>\n"
        "\n"
        "[SHOT 2 12.4s-31.0s]\n"
        "The camera moves closer to the window.\n"
        "<F001> <S>I forgot my phone</S>\n"
    )
    assert render_script(_example()) == expected


def test_round_trip_preserves_the_utterance_stream() -> None:
    """render -> parse must reproduce (face_id, tag, text) in order."""

    annotation = _example()
    parsed = parse_script(render_script(annotation))

    ordered = sorted(annotation.utterances, key=lambda u: u.start)
    original = [(u.face_id, u.tag, u.text) for u in ordered]
    recovered = [(u.face_id, u.tag, u.text) for u in parsed.utterances]
    assert recovered == original

    assert parsed.global_caption == annotation.global_caption
    assert [shot.index for shot in parsed.shots] == [1, 2]
    assert [shot.caption for shot in parsed.shots] == [
        "A bright living room with a sofa and a coffee table.",
        "The camera moves closer to the window.",
    ]


def test_utterance_lands_in_the_shot_containing_its_start() -> None:
    """An utterance is never split across a cut."""

    annotation = _example()
    parsed = parse_script(render_script(annotation))
    first, second = parsed.shots
    assert [u.text for u in first.utterances] == [
        "I was late for work today",
        "What's going on?",
    ]
    assert [u.text for u in second.utterances] == ["I forgot my phone"]


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        ("whispering", "<F001> whispering: <S>hi</S>"),
        (None, "<F001> <S>hi</S>"),
        ("WHISPERING", "<F001> whispering: <S>hi</S>"),
        ("  shouting  ", "<F001> shouting: <S>hi</S>"),
    ],
)
def test_tag_is_optional_and_normalized(tag: str | None, expected: str) -> None:
    annotation = Annotation(
        video=VIDEO,
        utterances=(Utterance(face_id="F001", start=0.0, end=1.0, text="hi", tag=tag),),
    )
    assert expected in render_script(annotation)


def test_no_shots_still_renders_and_parses() -> None:
    """A video that was never shot-detected must not lose its utterances."""

    annotation = Annotation(
        video=VIDEO,
        global_caption="A person talks to camera.",
        utterances=(Utterance(face_id="F001", start=0.0, end=1.0, text="hello"),),
    )
    script = render_script(annotation)
    assert "[SHOT" not in script
    parsed = parse_script(script)
    assert [u.text for u in parsed.utterances] == ["hello"]
    assert parsed.global_caption == "A person talks to camera."


def test_no_utterances_is_valid_output() -> None:
    """A music-only or silent video produces a script, not a crash or hallucination."""

    annotation = Annotation(
        video=VIDEO,
        global_caption="An empty street at dusk.",
        shots=(Shot(index=1, start=0.0, end=5.0, caption="A street."),),
    )
    script = render_script(annotation)
    parsed = parse_script(script)
    assert parsed.utterances == ()
    assert parsed.global_caption == "An empty street at dusk."
    assert [shot.caption for shot in parsed.shots] == ["A street."]


def test_offscreen_speaker_uses_f000_and_round_trips() -> None:
    """Off-screen speech needs an id, because the grammar requires a face tag."""

    annotation = Annotation(
        video=VIDEO,
        global_caption="A narrator over a cityscape.",
        utterances=(Utterance(face_id="F000", start=0.0, end=2.0, text="Once, in this city",
                              tag="whispering"),),
    )
    script = render_script(annotation)
    assert "<F000> whispering: <S>Once, in this city</S>" in script
    parsed = parse_script(script)
    assert parsed.utterances[0].face_id == "F000"
    assert parsed.utterances[0].is_offscreen


@pytest.mark.parametrize(
    "text",
    [
        "a < b",
        "a > b",
        "line\nbreak",
        "carriage\rreturn",
        "",
        "   ",
    ],
)
def test_unrenderable_text_is_rejected_not_mangled(text: str) -> None:
    annotation = Annotation(
        video=VIDEO,
        utterances=(Utterance(face_id="F001", start=0.0, end=1.0, text=text),),
    )
    with pytest.raises(AnnotationFormatError):
        render_script(annotation)


@pytest.mark.parametrize("tag", ["whis pering", "whispering!", "very-happy", "a.b"])
def test_tag_charset_is_enforced(tag: str) -> None:
    annotation = Annotation(
        video=VIDEO,
        utterances=(Utterance(face_id="F001", start=0.0, end=1.0, text="hi", tag=tag),),
    )
    if tag == "very-happy":
        assert "very-happy" in render_script(annotation)
    else:
        with pytest.raises(AnnotationFormatError):
            render_script(annotation)


def test_flat_rendering_keeps_markers_so_it_still_parses() -> None:
    flat = render_flat(_example())
    parsed = parse_script(flat)
    assert [u.text for u in parsed.utterances] == [
        "I was late for work today",
        "What's going on?",
        "I forgot my phone",
    ]


def test_multi_line_caption_is_joined_not_lost() -> None:
    script = (
        "[GLOBAL]\nA wide shot of a room.\n\n"
        "[SHOT 1 0.0s-4.0s]\nA sofa.\nNear a window.\n"
        "<F001> <S>hello</S>\n"
    )
    parsed = parse_script(script)
    assert parsed.shots[0].caption == "A sofa. Near a window."


def test_face_ids_are_one_based_and_never_collide_with_offscreen() -> None:
    assert new_face_id(1) == "F001"
    assert new_face_id(12) == "F012"
    assert parse_face_id("F012") == 12
    with pytest.raises(ValueError):
        new_face_id(0)
    with pytest.raises(ValueError):
        parse_face_id("SPEAKER_00")


def test_dict_round_trip() -> None:
    """The JSON deliverable must rebuild into the same renderable object."""

    annotation = _example()
    payload = {
        "schema_version": "av-annotation-v1",
        "video": {
            "video_id": "sample",
            "path": "inbox/sample.mp4",
            "duration": 31.0,
            "fps": 25.0,
            "width": 1920,
            "height": 1080,
        },
        "global_caption": annotation.global_caption,
        "language": {"code": "en", "confidence": 0.98, "source": "whisper_lid"},
        "shots": [
            {"index": shot.index, "start": shot.start, "end": shot.end, "caption": shot.caption}
            for shot in annotation.shots
        ],
        "utterances": [
            {
                "face_id": u.face_id,
                "start": u.start,
                "end": u.end,
                "text": u.text,
                "tag": u.tag,
                "words": [{"text": "hi", "start": 0.0, "end": 0.1}],
            }
            for u in annotation.utterances
        ],
        "face_tracks": [
            {
                "face_id": "F001",
                "first_seen": 0.0,
                "last_seen": 30.0,
                "speaks": True,
                "total_speech": 3.0,
            }
        ],
    }
    rebuilt = annotation_from_dict(payload)
    assert render_script(rebuilt) == render_script(annotation)
    assert rebuilt.face_track("F001") is not None
    assert rebuilt.utterances[0].words[0].text == "hi"


def test_dict_rejects_wrong_schema_version() -> None:
    with pytest.raises(AnnotationFormatError):
        annotation_from_dict({"schema_version": "av-annotation-v0"})
