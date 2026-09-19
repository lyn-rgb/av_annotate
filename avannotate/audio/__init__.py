"""Speaker diarization: who spoke when, in the mixed audio.

The turn geometry lives here and is testable on its own; the model that
produces the turns is behind :mod:`avannotate.audio.diarize`.
"""

from avannotate.audio.diarize import DiariZenDiarizer, Diarizer, DiarizerError, build_diarizer
from avannotate.audio.types import DiarizationResult, SpeakerTurn, overlap_intervals

__all__ = [
    "DiariZenDiarizer",
    "DiarizationResult",
    "Diarizer",
    "DiarizerError",
    "SpeakerTurn",
    "build_diarizer",
    "overlap_intervals",
]
