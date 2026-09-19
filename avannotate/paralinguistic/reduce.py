"""Three models' opinions, reduced to one tag.

The deliverable has room for one tag per utterance.  Three models each have an
opinion about the same audio, and they can all be right at once -- a line can be
delivered in a whisper, in a frightened voice, over a cough.  Something has to
choose, and the choice is the design.

The order
---------

``event`` beats ``delivery`` beats ``emotion``.  The rule is which description
tells a reader something they could not already tell from the words:

* An **event** is not a way of speaking at all -- there may be no words in the
  segment.  ``laughter`` on a line is a fact about the audio that the transcript
  cannot contain, and it is the most specific thing anyone has to say.
* **Delivery** is how the words were produced.  ``whispering`` is checkable by
  listening and is what an editor would need to reproduce the line.
* **Emotion** is an inference about the speaker's state from the delivery.  It
  is the most valuable thing to know and the least reliable thing to act on,
  which is why it is the fallback rather than the headline.

The alternative -- letting the highest score win across dimensions -- was
rejected because the three scores are not comparable.  One is a softmax over
nine classes, the others are sigmoids over hundreds of independent labels, and
comparing them would make the priority between dimensions an artefact of how
each model happens to be calibrated.

Within a dimension the highest score wins, but only among labels that map to a
tag.  A model's top label is often one the vocabulary does not use -- the most
confident thing about a quiet conversation is frequently ``Speech`` -- and
letting an unmapped label win would then discard a real detection sitting just
below it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from avannotate.paralinguistic.types import (
    DIMENSIONS,
    TagCandidate,
    TagChoice,
    TagScore,
)
from avannotate.paralinguistic.vocabulary import tag_for, unmapped


@dataclass(frozen=True)
class ReduceConfig:
    """How much each model has to believe something before it is believed."""

    #: One threshold per dimension, because the scales are not comparable: a
    #: nine-way softmax and a several-hundred-way sigmoid put the same
    #: confidence at different numbers.  These are starting points to be
    #: calibrated against a sample, not measurements.
    min_score: Mapping[str, float]

    #: Ties between two dimensions are broken by precedence, so this is only
    #: used to record that a choice was close.  Not a gate.
    close_margin: float = 0.10

    @classmethod
    def defaults(cls) -> ReduceConfig:
        return cls(min_score={dimension: 0.5 for dimension in DIMENSIONS})


def _best(
    scores: Sequence[TagScore], *, dimension: str, min_score: float
) -> TagCandidate | None:
    """The highest-scoring label that maps to a tag and clears the threshold.

    Two labels can map to the same tag -- ``giggling`` and ``laughing`` both
    mean ``laughing`` -- so this keeps the best score per tag and then takes the
    best of those, rather than taking the best label and hoping it maps.
    """

    best: dict[str, TagCandidate] = {}
    for item in scores:
        if item.score < min_score:
            continue
        tag = tag_for(item.label, dimension=dimension)
        if tag is None:
            continue
        current = best.get(tag)
        if current is None or item.score > current.score:
            best[tag] = TagCandidate(
                dimension=dimension, tag=tag, label=item.label, score=item.score
            )
    if not best:
        return None
    return max(best.values(), key=lambda candidate: candidate.score)


def reduce_tags(
    scores: Mapping[str, Sequence[TagScore]], *, config: ReduceConfig | None = None
) -> TagChoice:
    """One tag from every dimension's answer, or none.

    ``scores`` is dimension name to that model's labels, highest first or not --
    order is not read, only the scores are.
    """

    active = config or ReduceConfig.defaults()

    candidates: list[TagCandidate] = []
    for dimension in DIMENSIONS:
        found = _best(
            scores.get(dimension, ()),
            dimension=dimension,
            min_score=active.min_score.get(dimension, 0.5),
        )
        if found is not None:
            candidates.append(found)

    if not candidates:
        return TagChoice(
            reason="no dimension scored a known label above its threshold",
            candidates=(),
        )

    winner = candidates[0]
    choice = TagChoice(
        tag=winner.tag,
        dimension=winner.dimension,
        label=winner.label,
        score=winner.score,
        candidates=tuple(candidates),
        reason=f"{winner.dimension} takes precedence",
    )
    # Compared against the best-scoring rival rather than the next one in
    # precedence order, so a precedence win over a higher score is reported as
    # the close call it is -- see TagChoice.margin, where that shows up negative.
    if len(candidates) > 1 and choice.margin < active.close_margin:
        rival = max(candidates[1:], key=lambda candidate: candidate.score)
        if choice.margin < 0.0:
            reason = (
                f"{winner.dimension} takes precedence over a higher-scoring "
                f"{rival.dimension}"
            )
        else:
            reason = f"{winner.dimension} takes precedence over a close {rival.dimension}"
        return replace(choice, reason=reason)
    return choice


def unmapped_labels(
    scores: Mapping[str, Sequence[TagScore]], *, config: ReduceConfig | None = None
) -> tuple[str, ...]:
    """Confident labels no table had an entry for, in dimension order."""

    active = config or ReduceConfig.defaults()
    found: list[str] = []
    for dimension in DIMENSIONS:
        found.extend(
            unmapped(
                scores.get(dimension, ()),
                dimension=dimension,
                min_score=active.min_score.get(dimension, 0.5),
            )
        )
    return tuple(found)
