"""Active speaker detection: which face is talking, frame by frame.

The windowing and crop geometry are plain arithmetic and are tested on their
own; the network is behind :mod:`avannotate.asd.model`.
"""

from avannotate.asd.crop import CROP_SIZE, CropBox, clamp_box, crop_box
from avannotate.asd.model import (
    AsdError,
    AsdModel,
    LoCoNetAsd,
    build_asd_model,
    features_for_window,
)
from avannotate.asd.stitch import stitch_predictions
from avannotate.asd.types import AsdResult, SpeakingSample, TrackSpeaking, Window
from avannotate.asd.window import context_speakers, plan_windows, tracks_in_window

__all__ = [
    "CROP_SIZE",
    "AsdError",
    "AsdModel",
    "AsdResult",
    "CropBox",
    "LoCoNetAsd",
    "SpeakingSample",
    "TrackSpeaking",
    "Window",
    "build_asd_model",
    "clamp_box",
    "context_speakers",
    "crop_box",
    "features_for_window",
    "plan_windows",
    "stitch_predictions",
    "tracks_in_window",
]
