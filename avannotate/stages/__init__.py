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
)
from avannotate.stages.base import StageContext, StageRun


class Stage(Protocol):
    """The shape every stage module satisfies."""

    STAGE: str
    VERSION: str

    def run(self, context: StageContext, *, force: bool = False) -> StageRun: ...


#: Ordered by dependency.  S0 is the only one implemented; the rest are named
#: here so the CLI can report them as known-but-absent rather than unknown.
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
}


def available_stages() -> tuple[str, ...]:
    return tuple(name for name in STAGE_ORDER if name in _MODULES)


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
]
