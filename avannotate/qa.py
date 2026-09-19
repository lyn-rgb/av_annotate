"""Automatic gates and reported metrics.

The gates decide whether a video's annotation may leave the pipeline; the
metrics are what a human looks at when a batch's numbers move.  They are
separate because a metric crossing a threshold is a judgement call, while a
failed gate is a defect: an annotation that does not parse, or whose speech does
not add up, is wrong regardless of how good it looks.

The strongest gate is the render round trip.  It exercises the actual deliverable
rather than a proxy for it, and one failure mode it catches -- a transcript
containing ``<S>`` or a newline -- would otherwise survive every other check and
be discovered by a downstream parser.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from avannotate.annotation import (
    AnnotationFormatError,
    normalize_tag,
    parse_script,
    render_script,
)
from avannotate.interval import Interval, total_duration
from avannotate.schema import OFFSCREEN_FACE_ID, Annotation, Utterance

#: Speech accounting compares two independently produced numbers (the diarizer's
#: total and the pipeline's attributed total), so exact equality is not the bar;
#: this is the slack allowed before the books are declared unbalanced.
DEFAULT_ACCOUNTING_TOLERANCE = 0.5


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class QaReport:
    video_id: str
    gates: tuple[GateResult, ...]
    metrics: dict[str, float] = field(default_factory=dict)
    notes: dict[str, object] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(gate.passed for gate in self.gates)

    @property
    def failing(self) -> tuple[GateResult, ...]:
        return tuple(gate for gate in self.gates if not gate.passed)

    def to_dict(self) -> dict[str, object]:
        return {
            "video_id": self.video_id,
            "passed": self.passed,
            "gates": [
                {"name": gate.name, "passed": gate.passed, "detail": gate.detail}
                for gate in self.gates
            ],
            "metrics": self.metrics,
            "notes": self.notes,
        }


# --------------------------------------------------------------------------- #
# gates
# --------------------------------------------------------------------------- #


def gate_render_round_trip(annotation: Annotation) -> GateResult:
    """Render, parse back, and compare the utterance stream."""

    name = "render_round_trip"
    try:
        script = render_script(annotation)
    except AnnotationFormatError as error:
        return GateResult(name, False, f"rendering failed: {error}")

    try:
        parsed = parse_script(script)
    except AnnotationFormatError as error:  # pragma: no cover - parser is total
        return GateResult(name, False, f"parsing the rendered script failed: {error}")

    ordered = sorted(annotation.utterances, key=lambda item: (item.start, item.end))
    # Tags are compared in their rendered form: the renderer lowercases them, so
    # comparing raw would report a mismatch for a script that is correct.
    expected = [(item.face_id, normalize_tag(item.tag), item.text) for item in ordered]
    recovered = [(item.face_id, item.tag, item.text) for item in parsed.utterances]

    if expected != recovered:
        for index, (want, got) in enumerate(zip(expected, recovered, strict=False)):
            if want != got:
                return GateResult(
                    name,
                    False,
                    f"utterance {index} changed on round trip: {want!r} became {got!r}",
                )
        return GateResult(
            name,
            False,
            f"round trip produced {len(recovered)} utterances, expected {len(expected)}",
        )

    return GateResult(name, True, f"{len(expected)} utterances round-tripped exactly")


def gate_speech_accounting(
    annotation: Annotation,
    vad_speech_seconds: float | None,
    tolerance: float = DEFAULT_ACCOUNTING_TOLERANCE,
) -> GateResult:
    """Attributed speech plus off-screen speech must account for the VAD total.

    A mismatch means speech was lost or double-counted somewhere between
    diarization and the utterance list -- the failure that makes a dataset quietly
    smaller than it should be.
    """

    name = "speech_accounting"
    if vad_speech_seconds is None:
        return GateResult(name, True, "no VAD total supplied; accounting not checked")

    attributed = sum(item.duration for item in annotation.utterances)
    difference = attributed - vad_speech_seconds
    if abs(difference) > tolerance:
        return GateResult(
            name,
            False,
            f"attributed {attributed:.2f}s against VAD {vad_speech_seconds:.2f}s "
            f"(off by {difference:+.2f}s, tolerance {tolerance:.2f}s)",
        )
    return GateResult(
        name,
        True,
        f"attributed {attributed:.2f}s against VAD {vad_speech_seconds:.2f}s",
    )


def gate_timeline_sanity(annotation: Annotation) -> GateResult:
    """Spans must be ordered, in range, and never overlap within one face."""

    name = "timeline_sanity"
    duration = annotation.video.duration

    for item in annotation.utterances:
        if item.end < item.start:
            return GateResult(
                name, False, f"{item.face_id} utterance ends before it starts: {item.start}"
            )
        if item.start < 0.0 or item.end > duration + 1e-6:
            return GateResult(
                name,
                False,
                f"{item.face_id} utterance [{item.start:.2f}, {item.end:.2f}] "
                f"falls outside the video's {duration:.2f}s",
            )

    for face_id in {item.face_id for item in annotation.utterances}:
        spans = sorted(
            (item.start, item.end) for item in annotation.utterances if item.face_id == face_id
        )
        for (_, previous_end), (start, _) in zip(spans, spans[1:], strict=False):
            if start < previous_end - 1e-6:
                return GateResult(
                    name,
                    False,
                    f"{face_id} has overlapping utterances at {start:.2f}s "
                    f"(previous ended {previous_end:.2f}s)",
                )

    return GateResult(name, True, f"{len(annotation.utterances)} spans are ordered and in range")


def gate_id_integrity(annotation: Annotation) -> GateResult:
    """Ids must be unique, resolvable, and must not put F000 in the face table."""

    name = "id_integrity"

    seen: set[str] = set()
    for track in annotation.face_tracks:
        if track.face_id in seen:
            return GateResult(name, False, f"face id {track.face_id} appears twice")
        seen.add(track.face_id)

    if OFFSCREEN_FACE_ID in seen:
        return GateResult(
            name,
            False,
            f"{OFFSCREEN_FACE_ID} is reserved for off-screen speech and cannot be a face track",
        )

    for item in annotation.utterances:
        if item.face_id == OFFSCREEN_FACE_ID:
            continue
        if seen and item.face_id not in seen:
            return GateResult(
                name, False, f"utterance references {item.face_id}, which has no face track"
            )

    return GateResult(name, True, f"{len(seen)} face ids, all referenced ids resolve")


def gate_format_hygiene(annotation: Annotation) -> GateResult:
    """Text must be non-empty and free of the characters the grammar reserves.

    ``render_script`` enforces this by raising; checking it here as well means a
    violation is reported as a gate with the offending segment named, instead of
    as an exception from inside the renderer.
    """

    name = "format_hygiene"
    for item in annotation.utterances:
        if not item.text.strip():
            return GateResult(
                name, False, f"{item.face_id} utterance at {item.start:.2f}s has empty text"
            )
        for unsafe in ("<", ">", "\n", "\r"):
            if unsafe in item.text:
                return GateResult(
                    name,
                    False,
                    f"{item.face_id} utterance at {item.start:.2f}s contains {unsafe!r}",
                )
    return GateResult(name, True, f"{len(annotation.utterances)} utterances are renderable")


def run_gates(
    annotation: Annotation,
    *,
    vad_speech_seconds: float | None = None,
    accounting_tolerance: float = DEFAULT_ACCOUNTING_TOLERANCE,
) -> tuple[GateResult, ...]:
    """Every gate, in a fixed order so reports diff cleanly across runs."""

    return (
        gate_format_hygiene(annotation),
        gate_id_integrity(annotation),
        gate_timeline_sanity(annotation),
        gate_speech_accounting(annotation, vad_speech_seconds, accounting_tolerance),
        gate_render_round_trip(annotation),
    )


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #


def _overlap_seconds(utterances: Sequence[Utterance]) -> float:
    """Time during which two or more distinct faces are speaking.

    A sweep over boundaries.  Distinct faces rather than distinct utterances:
    one person's two adjacent segments are not an overlap, two people talking at
    once are -- and that is the case the whole design exists to handle.
    """

    events: list[tuple[float, int, str]] = []
    for item in utterances:
        events.append((item.start, 1, item.face_id))
        events.append((item.end, -1, item.face_id))
    if not events:
        return 0.0
    # Ends before starts at the same timestamp, so a span that begins exactly
    # where another ends is adjacency, not overlap.
    events.sort(key=lambda event: (event[0], event[1]))

    active: dict[str, int] = {}
    total = 0.0
    index = 0
    previous_time = events[0][0]

    while index < len(events):
        time = events[index][0]
        if time > previous_time and len(active) >= 2:
            total += time - previous_time
        while index < len(events) and events[index][0] == time:
            _, delta, face_id = events[index]
            active[face_id] = active.get(face_id, 0) + delta
            if active[face_id] <= 0:
                del active[face_id]
            index += 1
        previous_time = time

    return total


def compute_metrics(annotation: Annotation) -> dict[str, float]:
    """Reported numbers.  Nothing here gates; a human reads them."""

    utterances = annotation.utterances
    face_speech = sum(item.duration for item in utterances if not item.is_offscreen)
    offscreen_speech = sum(item.duration for item in utterances if item.is_offscreen)
    total = face_speech + offscreen_speech

    duration = annotation.video.duration
    durations = [item.duration for item in utterances]
    spans = [Interval(item.start, item.end) for item in utterances]
    union = total_duration(spans)

    overlap_seconds = _overlap_seconds(utterances)

    metrics: dict[str, float] = {
        "utterance_count": float(len(utterances)),
        "face_count": float(len(annotation.face_tracks)),
        "speaking_face_count": float(sum(1 for track in annotation.face_tracks if track.speaks)),
        "face_speech_seconds": face_speech,
        "offscreen_speech_seconds": offscreen_speech,
        "speech_seconds": total,
        "speech_ratio": total / duration if duration > 0 else 0.0,
        "assigned_ratio": face_speech / total if total > 0 else 0.0,
        "utterance_union_seconds": union,
        "overlap_seconds": overlap_seconds,
        "overlap_ratio": overlap_seconds / union if union > 0 else 0.0,
        "mean_utterance_seconds": sum(durations) / len(durations) if durations else 0.0,
        "max_utterance_seconds": max(durations) if durations else 0.0,
        "short_utterance_count": float(sum(1 for item in utterances if "short" in item.flags)),
        "tagged_utterance_count": float(sum(1 for item in utterances if item.tag)),
        "empty_text_count": float(sum(1 for item in utterances if not item.text.strip())),
    }
    return metrics


def build_report(
    annotation: Annotation,
    *,
    vad_speech_seconds: float | None = None,
    accounting_tolerance: float = DEFAULT_ACCOUNTING_TOLERANCE,
    notes: dict[str, object] | None = None,
) -> QaReport:
    """Every gate, every metric, and whatever the caller wants to add.

    ``accounting_tolerance`` is a parameter here rather than baked into the gate
    because it is the one threshold that depends on the corpus rather than on
    this code: a set with a lot of off-screen speech has a real gap between
    attributed and heard speech, and a caller measuring that gap should be able
    to see it without also reading a failure.
    """

    return QaReport(
        video_id=annotation.video.video_id,
        gates=run_gates(
            annotation,
            vad_speech_seconds=vad_speech_seconds,
            accounting_tolerance=accounting_tolerance,
        ),
        metrics=compute_metrics(annotation),
        notes=notes or {},
    )
