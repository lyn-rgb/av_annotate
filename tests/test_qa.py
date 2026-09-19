"""Tests for the gates and metrics.

The gates are what stop a bad annotation leaving the pipeline, so each one needs
a case that actually trips it -- a gate that cannot fail is worse than no gate,
because it reads as coverage.
"""

from __future__ import annotations

import pytest

from avannotate.qa import (
    build_report,
    compute_metrics,
    gate_format_hygiene,
    gate_id_integrity,
    gate_render_round_trip,
    gate_speech_accounting,
    gate_timeline_sanity,
    run_gates,
)
from avannotate.schema import Annotation, FaceTrack, Shot, Utterance, VideoMeta

VIDEO = VideoMeta(
    video_id="sample",
    path="inbox/sample.mp4",
    duration=30.0,
    fps=25.0,
    width=1920,
    height=1080,
)


def _annotation(
    utterances: tuple[Utterance, ...] = (),
    tracks: tuple[FaceTrack, ...] = (),
    shots: tuple[Shot, ...] = (),
) -> Annotation:
    return Annotation(
        video=VIDEO,
        global_caption="A room.",
        utterances=utterances,
        face_tracks=tracks,
        shots=shots,
    )


def _track(face_id: str, speaks: bool = True) -> FaceTrack:
    return FaceTrack(
        face_id=face_id, first_seen=0.0, last_seen=30.0, speaks=speaks, total_speech=1.0
    )


# --------------------------------------------------------------------------- #
# gates
# --------------------------------------------------------------------------- #


def test_round_trip_passes_on_a_good_annotation() -> None:
    annotation = _annotation(
        utterances=(
            Utterance(face_id="F001", start=1.0, end=2.0, text="hello there", tag="whispering"),
            Utterance(face_id="F002", start=3.0, end=4.0, text="and you"),
        ),
        tracks=(_track("F001"), _track("F002")),
    )
    result = gate_render_round_trip(annotation)
    assert result.passed
    assert "2 utterances" in result.detail


def test_round_trip_treats_tag_case_as_equivalent() -> None:
    """The renderer lowercases tags; the gate must compare in rendered form."""

    annotation = _annotation(
        utterances=(Utterance(face_id="F001", start=1.0, end=2.0, text="hi", tag="WHISPERING"),),
        tracks=(_track("F001"),),
    )
    assert gate_render_round_trip(annotation).passed


def test_round_trip_fails_when_the_text_cannot_be_rendered() -> None:
    annotation = _annotation(
        utterances=(Utterance(face_id="F001", start=1.0, end=2.0, text="a < b"),),
        tracks=(_track("F001"),),
    )
    result = gate_render_round_trip(annotation)
    assert not result.passed
    assert "rendering failed" in result.detail


def test_round_trip_preserves_shot_membership_for_out_of_range_utterances() -> None:
    """An utterance past the last shot's end must not drift into it on re-parse."""

    shots = (Shot(index=1, start=0.0, end=5.0, caption="One."),)
    annotation = _annotation(
        utterances=(
            Utterance(face_id="F001", start=1.0, end=2.0, text="early"),
            Utterance(face_id="F001", start=20.0, end=21.0, text="late"),
        ),
        tracks=(_track("F001"),),
        shots=shots,
    )
    from avannotate.annotation import parse_script, render_script

    parsed = parse_script(render_script(annotation))
    assert [item.text for item in parsed.utterances] == ["early", "late"]
    assert [item.text for item in parsed.shots[0].utterances] == ["early", "late"]


def test_id_integrity_catches_duplicate_ids() -> None:
    annotation = _annotation(tracks=(_track("F001"), _track("F001")))
    result = gate_id_integrity(annotation)
    assert not result.passed
    assert "twice" in result.detail


def test_id_integrity_forbids_f000_as_a_track() -> None:
    annotation = _annotation(tracks=(_track("F000"),))
    result = gate_id_integrity(annotation)
    assert not result.passed
    assert "reserved" in result.detail


def test_id_integrity_catches_dangling_reference() -> None:
    annotation = _annotation(
        utterances=(Utterance(face_id="F009", start=1.0, end=2.0, text="who?"),),
        tracks=(_track("F001"),),
    )
    result = gate_id_integrity(annotation)
    assert not result.passed
    assert "F009" in result.detail


def test_id_integrity_allows_offscreen_utterances() -> None:
    annotation = _annotation(
        utterances=(Utterance(face_id="F000", start=1.0, end=2.0, text="narration"),),
        tracks=(_track("F001"),),
    )
    assert gate_id_integrity(annotation).passed


def test_timeline_sanity_catches_overlap_within_one_face() -> None:
    annotation = _annotation(
        utterances=(
            Utterance(face_id="F001", start=1.0, end=3.0, text="one"),
            Utterance(face_id="F001", start=2.0, end=4.0, text="two"),
        ),
        tracks=(_track("F001"),),
    )
    result = gate_timeline_sanity(annotation)
    assert not result.passed
    assert "overlapping" in result.detail


def test_timeline_sanity_allows_two_faces_at_once() -> None:
    """Simultaneous speech is the point, not a defect."""

    annotation = _annotation(
        utterances=(
            Utterance(face_id="F001", start=1.0, end=3.0, text="one"),
            Utterance(face_id="F002", start=1.0, end=3.0, text="two"),
        ),
        tracks=(_track("F001"), _track("F002")),
    )
    assert gate_timeline_sanity(annotation).passed


def test_timeline_sanity_catches_out_of_range_span() -> None:
    annotation = _annotation(
        utterances=(Utterance(face_id="F001", start=29.0, end=31.0, text="past the end"),),
        tracks=(_track("F001"),),
    )
    result = gate_timeline_sanity(annotation)
    assert not result.passed
    assert "outside" in result.detail


def test_speech_accounting_within_tolerance() -> None:
    annotation = _annotation(
        utterances=(Utterance(face_id="F001", start=0.0, end=3.0, text="hi"),),
        tracks=(_track("F001"),),
    )
    assert gate_speech_accounting(annotation, 3.2).passed


def test_speech_accounting_catches_lost_speech() -> None:
    annotation = _annotation(
        utterances=(Utterance(face_id="F001", start=0.0, end=3.0, text="hi"),),
        tracks=(_track("F001"),),
    )
    result = gate_speech_accounting(annotation, 10.0)
    assert not result.passed
    assert "off by" in result.detail


def test_speech_accounting_skips_when_no_total_supplied() -> None:
    result = gate_speech_accounting(_annotation(), None)
    assert result.passed
    assert "not checked" in result.detail


def test_format_hygiene_catches_empty_text() -> None:
    annotation = _annotation(
        utterances=(Utterance(face_id="F001", start=0.0, end=1.0, text="   "),),
        tracks=(_track("F001"),),
    )
    result = gate_format_hygiene(annotation)
    assert not result.passed
    assert "empty" in result.detail


@pytest.mark.parametrize("text", ["a <S>b</S>", "line\nbreak", "a > b"])
def test_format_hygiene_catches_unrenderable_text(text: str) -> None:
    annotation = _annotation(
        utterances=(Utterance(face_id="F001", start=0.0, end=1.0, text=text),),
        tracks=(_track("F001"),),
    )
    assert not gate_format_hygiene(annotation).passed


def test_run_gates_returns_every_gate_in_a_stable_order() -> None:
    names = [gate.name for gate in run_gates(_annotation(), vad_speech_seconds=0.0)]
    assert names == [
        "format_hygiene",
        "id_integrity",
        "timeline_sanity",
        "speech_accounting",
        "render_round_trip",
    ]


def test_report_passes_only_when_every_gate_passes() -> None:
    good = build_report(
        _annotation(
            utterances=(Utterance(face_id="F001", start=0.0, end=2.0, text="hi"),),
            tracks=(_track("F001"),),
        ),
        vad_speech_seconds=2.0,
    )
    assert good.passed
    assert good.failing == ()

    bad = build_report(
        _annotation(
            utterances=(Utterance(face_id="F009", start=0.0, end=2.0, text="hi"),),
            tracks=(_track("F001"),),
        )
    )
    assert not bad.passed
    assert [gate.name for gate in bad.failing] == ["id_integrity"]


def test_report_serializes() -> None:
    report = build_report(_annotation(), vad_speech_seconds=None)
    payload = report.to_dict()
    assert payload["video_id"] == "sample"
    assert payload["passed"] is True
    assert isinstance(payload["gates"], list)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #


def test_overlap_counts_two_faces_talking_at_once() -> None:
    annotation = _annotation(
        utterances=(
            Utterance(face_id="F001", start=0.0, end=4.0, text="one"),
            Utterance(face_id="F002", start=2.0, end=6.0, text="two"),
        ),
        tracks=(_track("F001"), _track("F002")),
    )
    metrics = compute_metrics(annotation)
    assert metrics["overlap_seconds"] == pytest.approx(2.0)
    assert metrics["utterance_union_seconds"] == pytest.approx(6.0)


def test_adjacent_utterances_of_one_face_are_not_an_overlap() -> None:
    annotation = _annotation(
        utterances=(
            Utterance(face_id="F001", start=0.0, end=2.0, text="one"),
            Utterance(face_id="F001", start=2.0, end=4.0, text="two"),
        ),
        tracks=(_track("F001"),),
    )
    assert compute_metrics(annotation)["overlap_seconds"] == pytest.approx(0.0)


def test_assigned_ratio_separates_offscreen_speech() -> None:
    annotation = _annotation(
        utterances=(
            Utterance(face_id="F001", start=0.0, end=3.0, text="on screen"),
            Utterance(face_id="F000", start=4.0, end=5.0, text="narration"),
        ),
        tracks=(_track("F001"),),
    )
    metrics = compute_metrics(annotation)
    assert metrics["assigned_ratio"] == pytest.approx(0.75)
    assert metrics["offscreen_speech_seconds"] == pytest.approx(1.0)


def test_assigned_ratio_is_zero_when_nothing_was_said() -> None:
    metrics = compute_metrics(_annotation(tracks=(_track("F001"),)))
    assert metrics["assigned_ratio"] == 0.0
    assert metrics["overlap_ratio"] == 0.0


def test_metrics_count_tags_and_flags() -> None:
    annotation = _annotation(
        utterances=(
            Utterance(face_id="F001", start=0.0, end=3.0, text="one", tag="whispering"),
            Utterance(face_id="F001", start=5.0, end=5.4, text="two", flags=("short",)),
        ),
        tracks=(_track("F001"),),
    )
    metrics = compute_metrics(annotation)
    assert metrics["tagged_utterance_count"] == 1.0
    assert metrics["short_utterance_count"] == 1.0
    assert metrics["utterance_count"] == 2.0
