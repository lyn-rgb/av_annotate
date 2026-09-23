"""The pipeline's stages, in dependency order.

Each stage is a module exposing ``STAGE``, ``VERSION`` and ``run(context,
*, force=False) -> StageRun``.  ``VERSION`` is part of the resume record, so
bumping it invalidates that stage's cached output and nothing else -- which is
what makes a logic change safe to ship mid-batch.
"""

from __future__ import annotations

from types import ModuleType
from typing import Protocol

from avannotate.stages import (
    s0_preprocess,
    s1_faces,
    s2_tracks,
    s3_cluster,
    s4_diarize,
    s5_asd,
    s6_associate,
    s7_tse,
    s8_asr,
    s9_paralinguistic,
    s10_caption,
    s11_compose,
)
from avannotate.stages.base import StageContext, StageRun


class Stage(Protocol):
    """The shape every stage module satisfies."""

    STAGE: str
    VERSION: str
    #: Whether this stage's work happens on a graphics card.
    #:
    #: Not a dependency -- every stage here falls back to the CPU, badly.  It
    #: is what the batch sizes its pool by: a stage that uses a card gets one
    #: worker per card, because two workers on one card swap weights for every
    #: video; a stage that does not gets a pool sized by cores, because the
    #: cards have nothing to do with how fast it goes.  Getting it wrong is not
    #: a crash either way -- it is a corpus that takes longer than it needed
    #: to, quietly, which is why it is declared rather than guessed.
    USES_GPU: bool

    def run(self, context: StageContext, *, force: bool = False) -> StageRun: ...


#: Ordered by dependency.  Everything up to S7 is implemented; the rest are
#: named here so the CLI can report them as known-but-absent, not unknown.
STAGE_ORDER: tuple[str, ...] = (
    "s0-preprocess",
    "s1-faces",
    "s2-tracks",
    "s3-cluster",
    "s4-diarize",
    "s5-asd",
    "s6-associate",
    "s7-tse",
    "s8-asr",
    "s9-paralinguistic",
    "s10-caption",
    "s11-compose",
)

_MODULES: dict[str, ModuleType] = {
    s0_preprocess.STAGE: s0_preprocess,
    s1_faces.STAGE: s1_faces,
    s2_tracks.STAGE: s2_tracks,
    s3_cluster.STAGE: s3_cluster,
    s4_diarize.STAGE: s4_diarize,
    s5_asd.STAGE: s5_asd,
    s6_associate.STAGE: s6_associate,
    s7_tse.STAGE: s7_tse,
    s8_asr.STAGE: s8_asr,
    s9_paralinguistic.STAGE: s9_paralinguistic,
    s10_caption.STAGE: s10_caption,
    s11_compose.STAGE: s11_compose,
}


def available_stages() -> tuple[str, ...]:
    return tuple(name for name in STAGE_ORDER if name in _MODULES)


def uses_gpu(name: str) -> bool:
    """Whether this stage's work happens on a card.  Unknown stages say yes.

    The conservative answer for a name that is not in the pipeline: a mistyped
    stage is a bug to be reported elsewhere, and claiming it does not want a
    card would size a pool for work nobody has described.
    """

    module = _MODULES.get(name)
    return True if module is None else bool(getattr(module, "USES_GPU", True))


def get_stage(name: str) -> ModuleType:
    """Look up a stage module, failing with the list of what does exist."""

    module = _MODULES.get(name)
    if module is None:
        known = ", ".join(STAGE_ORDER)
        raise KeyError(f"unknown or unimplemented stage {name!r}; known stages: {known}")
    return module


__all__ = [
    "STAGE_ORDER",
    "Stage",
    "available_stages",
    "get_stage",
    "s0_preprocess",
    "uses_gpu",
]
