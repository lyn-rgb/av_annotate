"""What the taggers report, and what the pipeline keeps.

Three models answer three different questions about the same audio.  This is
the shape of each answer and the shape of the one tag that survives them.

Kept apart from the reduction so the reduction can be tested on labels nobody
had to run a model to produce -- which matters more here than elsewhere, because
the interesting cases are disagreements between dimensions, and those are rare
in real audio and trivial to write down by hand.
"""

from __future__ import annotations

from dataclasses import dataclass

from avannotate.coercion import coerce_number

#: The three questions, in the order they take precedence when more than one has
#: something to say.  See :func:`avannotate.paralinguistic.reduce.reduce_tags`.
DIMENSIONS: tuple[str, ...] = ("event", "delivery", "emotion")


@dataclass(frozen=True)
class TagScore:
    """One label and the model's score for it."""

    label: str
    score: float

    def to_dict(self) -> dict[str, object]:
        return {"label": self.label, "score": round(self.score, 4)}

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> TagScore:
        return cls(
            label=str(payload.get("label", "")),
            score=coerce_number(payload.get("score", 0.0), "score"),
        )


@dataclass(frozen=True)
class TagCandidate:
    """One dimension's best answer, after mapping and thresholding."""

    dimension: str
    tag: str
    label: str
    score: float

    def to_dict(self) -> dict[str, object]:
        return {
            "dimension": self.dimension,
            "tag": self.tag,
            "label": self.label,
            "score": round(self.score, 4),
        }


@dataclass(frozen=True)
class TagChoice:
    """The one tag an utterance gets, and what it beat.

    ``candidates`` holds every dimension's best answer in precedence order, so
    the winner is ``candidates[0]`` and the margin over the next one is
    recoverable.  A reviewer reading a rendered ``whispering:`` can then see
    whether it was a whisper or a coin toss between a whisper and a shout.
    """

    tag: str | None = None
    dimension: str | None = None
    label: str | None = None
    score: float = 0.0
    candidates: tuple[TagCandidate, ...] = ()
    reason: str = ""

    @property
    def margin(self) -> float:
        """How far clear the winner is of the best-scoring *other* dimension.

        Against the highest score among the losers, not against the next one in
        precedence order -- those are different candidates, and only the first
        answers "was this a close call".

        **Negative is a normal value and the one worth looking at.**  Precedence
        picks the winner, so a dimension can lose while scoring higher: a
        frightened whisper scores 0.9 for fear and 0.6 for whispering, and the
        whisper still wins.  A negative margin says the tag is right by the rule
        and wrong by the numbers, which is exactly what a reviewer needs to see.
        """

        others = [candidate.score for candidate in self.candidates[1:]]
        if not others:
            return self.score
        return self.score - max(others)

    def to_dict(self) -> dict[str, object]:
        return {
            "tag": self.tag,
            "dimension": self.dimension,
            "label": self.label,
            "score": round(self.score, 4),
            "margin": round(self.margin, 4),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "reason": self.reason,
        }


@dataclass(frozen=True)
class SegmentTags:
    """Every model's answer for one segment, plus the tag that came out.

    All three models' outputs are kept, not just the winner: when a tag reads
    wrong, the question is always which model was wrong, and re-running to find
    out would mean re-running all of them.
    """

    identity: str
    name: str
    start: float
    end: float
    choice: TagChoice
    emotion: tuple[TagScore, ...] = ()
    delivery: tuple[TagScore, ...] = ()
    event: tuple[TagScore, ...] = ()
    #: Labels no table had an entry for, per dimension.
    unmapped: tuple[str, ...] = ()
    #: False when the segment was too short for any tagger to be trusted, in
    #: which case nothing was run and the empty results mean "not asked".
    eligible: bool = True

    @property
    def duration(self) -> float:
        return self.end - self.start

    def scores_for(self, dimension: str) -> tuple[TagScore, ...]:
        return {
            "emotion": self.emotion,
            "delivery": self.delivery,
            "event": self.event,
        }.get(dimension, ())

    def to_dict(self) -> dict[str, object]:
        return {
            "identity": self.identity,
            "name": self.name,
            "start": round(self.start, 4),
            "end": round(self.end, 4),
            "duration": round(self.duration, 4),
            "eligible": self.eligible,
            "tag": self.choice.tag,
            "choice": self.choice.to_dict(),
            "scores": {
                dimension: [score.to_dict() for score in self.scores_for(dimension)]
                for dimension in DIMENSIONS
            },
            "unmapped": list(self.unmapped),
        }
