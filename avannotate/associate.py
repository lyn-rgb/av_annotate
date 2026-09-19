"""Stage S8: decide which diarization speaker is which face.

This is the step the whole pipeline exists for, and the one most able to fail
quietly: a wrong assignment still produces a fluent transcript, a plausible
audio clip and a valid script.  So the score, the margin over the runner-up and
the coverage all travel with the result, and the caller is expected to gate on
them rather than trust the label.

The score is a **contrast**, not a raw probability.  A face that the detector
scores as talking the entire video would otherwise win every speaker by
accident; subtracting its talking rate outside the speaker's turns is what makes
the score about *this* speaker rather than about the face's overall talkativeness.

Assignment uses the Hungarian algorithm rather than greedy selection.  Greedy is
visibly worse on the ordinary case where two faces both match one speaker well
and the second-best pairing is the correct one, and it would report a margin
computed against a pairing that should never have been chosen.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from avannotate.interval import Interval, overlap_duration, total_duration
from avannotate.schema import OFFSCREEN_FACE_ID

#: Below this score a speaker is treated as having no visible source -- narration,
#: a phone call, someone off camera -- rather than being pinned to the best-looking
#: face, which for an off-screen speaker is noise.
DEFAULT_MIN_SCORE = 0.05

#: Below this margin the assignment is reported but flagged: two faces explain the
#: speaker comparably well, which is what a track split or a look-alike looks like.
DEFAULT_MIN_MARGIN = 0.05

#: A face must be on screen for at least this fraction of the speaker's speech to
#: be considered at all.  Without it, a face glimpsed for two frames while someone
#: else talks for a minute would compete on equal terms.
DEFAULT_MIN_COVERAGE = 0.25

#: How hard a constantly-talking face is penalized.  See the module docstring.
DEFAULT_CONTRAST_PENALTY = 0.5


@dataclass(frozen=True)
class SpeechSample:
    """One ASD output: how likely this face is speaking at ``time``."""

    time: float
    probability: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(f"probability out of range: {self.probability}")


@dataclass(frozen=True)
class FaceObservation:
    """What S1-S7 know about one face: when it is visible, and its ASD trace."""

    face_id: str
    visible: tuple[Interval, ...]
    speech: tuple[SpeechSample, ...] = ()

    @property
    def first_seen(self) -> float:
        return min((interval.start for interval in self.visible), default=0.0)

    @property
    def last_seen(self) -> float:
        return max((interval.end for interval in self.visible), default=0.0)


@dataclass(frozen=True)
class SpeakerTurn:
    """One diarization segment."""

    speaker: str
    start: float
    end: float

    @property
    def interval(self) -> Interval:
        return Interval(self.start, self.end)


@dataclass(frozen=True)
class SpeakerAssignment:
    """The verdict for one diarization speaker.

    ``ambiguous`` and ``merged`` are the flags QA gates on.  The assignment is
    still reported when either is set -- a caller reviewing a bad video wants to
    see the best guess and its competition, not a blank.
    """

    speaker: str
    face_id: str
    score: float
    margin: float
    coverage: float
    candidate: str | None
    ambiguous: bool
    reason: str
    #: True when this speaker shares its face with a higher-scoring one, which is
    #: what diarization over-segmentation looks like: one person split into two
    #: clusters.  Not an error, but worth counting.
    merged: bool = False

    @property
    def is_offscreen(self) -> bool:
        return self.face_id == OFFSCREEN_FACE_ID


def _mean_probability(
    samples: Sequence[SpeechSample], inside: Sequence[Interval]
) -> float | None:
    """Mean ASD probability of samples falling in ``inside``, or ``None`` if none do."""

    selected = [
        sample.probability
        for sample in samples
        if any(interval.start <= sample.time < interval.end for interval in inside)
    ]
    if not selected:
        return None
    return sum(selected) / len(selected)


def _complement(samples: Sequence[SpeechSample], inside: Sequence[Interval]) -> list[SpeechSample]:
    return [
        sample
        for sample in samples
        if not any(interval.start <= sample.time < interval.end for interval in inside)
    ]


@dataclass(frozen=True)
class AssociationConfig:
    min_score: float = DEFAULT_MIN_SCORE
    min_margin: float = DEFAULT_MIN_MARGIN
    min_coverage: float = DEFAULT_MIN_COVERAGE
    contrast_penalty: float = DEFAULT_CONTRAST_PENALTY


def _score_pair(
    face: FaceObservation, turns: Sequence[Interval], config: AssociationConfig
) -> tuple[float, float] | None:
    """Return ``(score, coverage)``, or ``None`` when the face is not eligible."""

    speech_duration = total_duration(turns)
    if speech_duration <= 0.0:
        return None

    coverage = overlap_duration(face.visible, turns) / speech_duration
    if coverage < config.min_coverage:
        return None

    inside = _mean_probability(face.speech, turns)
    if inside is None:
        return None

    outside_samples = _complement(face.speech, turns)
    outside = (
        sum(sample.probability for sample in outside_samples) / len(outside_samples)
        if outside_samples
        else 0.0
    )

    return inside - config.contrast_penalty * outside, coverage


def _hungarian(cost: list[list[float]]) -> list[int]:
    """Minimum-cost assignment of rows to columns, one each.

    Straight implementation of the shortest-augmenting-path form, O(n^2 m).
    Requires ``len(cost) <= len(cost[0])``; callers transpose when they do not.
    Returns the column chosen for each row, or ``-1`` if a row went unassigned.
    """

    rows = len(cost)
    if rows == 0:
        return []
    columns = len(cost[0])
    if rows > columns:
        raise ValueError("cost matrix must have at least as many columns as rows")

    infinity = float("inf")
    row_potential = [0.0] * (rows + 1)
    column_potential = [0.0] * (columns + 1)
    # column_match[j] is the 1-based row assigned to column j; 0 means free.
    column_match = [0] * (columns + 1)
    previous = [0] * (columns + 1)

    for row in range(1, rows + 1):
        column_match[0] = row
        column = 0
        min_reduced = [infinity] * (columns + 1)
        used = [False] * (columns + 1)

        while True:
            used[column] = True
            current_row = column_match[column]
            delta = infinity
            next_column = 0
            for candidate in range(1, columns + 1):
                if used[candidate]:
                    continue
                reduced = (
                    cost[current_row - 1][candidate - 1]
                    - row_potential[current_row]
                    - column_potential[candidate]
                )
                if reduced < min_reduced[candidate]:
                    min_reduced[candidate] = reduced
                    previous[candidate] = column
                if min_reduced[candidate] < delta:
                    delta = min_reduced[candidate]
                    next_column = candidate
            for candidate in range(columns + 1):
                if used[candidate]:
                    row_potential[column_match[candidate]] += delta
                    column_potential[candidate] -= delta
                else:
                    min_reduced[candidate] -= delta
            column = next_column
            if column_match[column] == 0:
                break

        while column:
            previous_column = previous[column]
            column_match[column] = column_match[previous_column]
            column = previous_column

    assignment = [-1] * rows
    for column in range(1, columns + 1):
        if column_match[column] > 0:
            assignment[column_match[column] - 1] = column - 1
    return assignment


def assign_speakers(
    faces: Sequence[FaceObservation],
    turns: Sequence[SpeakerTurn],
    config: AssociationConfig | None = None,
) -> tuple[SpeakerAssignment, ...]:
    """Match every diarization speaker to a face, or to ``F000``.

    Deterministic: speakers and faces are ordered internally, so the same input
    always yields the same output, which is what makes resume and cross-run
    diffing possible.
    """

    active = config or AssociationConfig()

    by_speaker: dict[str, list[Interval]] = {}
    for turn in turns:
        by_speaker.setdefault(turn.speaker, []).append(turn.interval)

    speaker_names = sorted(by_speaker)
    if not speaker_names:
        return ()

    ordered_faces = sorted(faces, key=lambda face: face.face_id)
    if not ordered_faces:
        return tuple(
            SpeakerAssignment(
                speaker=name,
                face_id=OFFSCREEN_FACE_ID,
                score=0.0,
                margin=0.0,
                coverage=0.0,
                candidate=None,
                ambiguous=False,
                reason="no faces were tracked in this video",
            )
            for name in speaker_names
        )

    # score[i][j] for face i, speaker j; None marks an ineligible pair.
    scores: list[list[float | None]] = []
    coverages: list[list[float]] = []
    for face in ordered_faces:
        row: list[float | None] = []
        coverage_row: list[float] = []
        for name in speaker_names:
            outcome = _score_pair(face, by_speaker[name], active)
            if outcome is None:
                row.append(None)
                coverage_row.append(0.0)
            else:
                pair_score, pair_coverage = outcome
                row.append(pair_score)
                coverage_row.append(pair_coverage)
        scores.append(row)
        coverages.append(coverage_row)

    # The solver wants rows <= columns and a finite cost for every entry, so an
    # ineligible pair gets a large finite cost rather than infinity.  Maximising
    # the score is minimising its negation.
    ineligible = 1000.0

    def to_cost(values: Sequence[float | None]) -> list[float]:
        return [-value if value is not None else ineligible for value in values]

    transpose = len(ordered_faces) > len(speaker_names)
    if transpose:
        matrix = [
            to_cost([scores[i][column] for i in range(len(ordered_faces))])
            for column in range(len(speaker_names))
        ]
    else:
        matrix = [to_cost(row) for row in scores]

    mapping = _hungarian(matrix)
    pairs: dict[str, int] = {}
    for row_index, column_index in enumerate(mapping):
        if column_index < 0:
            continue
        if transpose:
            pairs[speaker_names[row_index]] = column_index
        else:
            pairs[speaker_names[column_index]] = row_index

    def best_face(column: int) -> tuple[int, float] | None:
        best: tuple[int, float] | None = None
        for index in range(len(ordered_faces)):
            value = scores[index][column]
            if value is None:
                continue
            if best is None or value > best[1]:
                best = (index, value)
        return best

    def runner_up_face(column: int, exclude: int) -> tuple[float, str] | None:
        options: list[tuple[float, str]] = []
        for index in range(len(ordered_faces)):
            if index == exclude:
                continue
            value = scores[index][column]
            if value is not None:
                options.append((value, ordered_faces[index].face_id))
        return max(options) if options else None

    results: list[SpeakerAssignment] = []
    for column, name in enumerate(speaker_names):
        seated = pairs.get(name)
        face_index = (
            seated if seated is not None and scores[seated][column] is not None else None
        )
        merged = False

        if face_index is None:
            # The one-to-one pass could not seat this speaker.  Before calling it
            # off-screen, check whether it belongs to a face another speaker
            # already claimed: diarization routinely splits one person into two
            # clusters, and sending the spare cluster to F000 would relabel real
            # speech as narration -- a corruption nothing downstream can see.
            alternative = best_face(column)
            if alternative is not None and alternative[1] >= active.min_score:
                face_index, merged = alternative[0], True

        if face_index is None:
            alternative = best_face(column)
            if alternative is None:
                results.append(
                    SpeakerAssignment(
                        speaker=name,
                        face_id=OFFSCREEN_FACE_ID,
                        score=0.0,
                        margin=0.0,
                        coverage=0.0,
                        candidate=None,
                        ambiguous=False,
                        reason="no face was on screen for enough of this speaker's speech",
                    )
                )
            else:
                index, value = alternative
                results.append(
                    SpeakerAssignment(
                        speaker=name,
                        face_id=OFFSCREEN_FACE_ID,
                        score=value,
                        margin=0.0,
                        coverage=coverages[index][column],
                        candidate=ordered_faces[index].face_id,
                        ambiguous=False,
                        reason=(
                            f"best face {ordered_faces[index].face_id} scored {value:.3f}, "
                            f"below the {active.min_score:.3f} floor; treated as off-screen"
                        ),
                    )
                )
            continue

        score = scores[face_index][column]
        assert score is not None  # face_index was chosen only from non-None entries
        coverage = coverages[face_index][column]
        face_id = ordered_faces[face_index].face_id
        runner_up = runner_up_face(column, face_index)
        margin = score - runner_up[0] if runner_up is not None else score

        if score < active.min_score:
            results.append(
                SpeakerAssignment(
                    speaker=name,
                    face_id=OFFSCREEN_FACE_ID,
                    score=score,
                    margin=margin,
                    coverage=coverage,
                    candidate=face_id,
                    ambiguous=False,
                    reason=(
                        f"best face {face_id} scored {score:.3f}, below the "
                        f"{active.min_score:.3f} floor; treated as off-screen"
                    ),
                )
            )
            continue

        ambiguous = margin < active.min_margin
        if merged:
            reason = (
                f"{face_id} scored {score:.3f}; shared with a higher-scoring speaker, "
                "so this cluster is treated as a split of the same person"
            )
        elif ambiguous and runner_up is not None:
            reason = (
                f"{face_id} scored {score:.3f} against {runner_up[1]} at {runner_up[0]:.3f}; "
                f"margin below {active.min_margin:.3f}"
            )
        else:
            reason = f"{face_id} scored {score:.3f} over coverage {coverage:.2f}"

        results.append(
            SpeakerAssignment(
                speaker=name,
                face_id=face_id,
                score=score,
                margin=margin,
                coverage=coverage,
                candidate=face_id,
                ambiguous=ambiguous,
                reason=reason,
                merged=merged,
            )
        )

    return tuple(results)
