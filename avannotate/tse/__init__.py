"""Target speaker extraction: one person's voice, out of the mix.

The planning is plain arithmetic and is tested on its own; the network and the
crop-video writing are behind :mod:`avannotate.tse.model` and
:mod:`avannotate.tse.crop_video`.
"""

from avannotate.tse.plan import (
    ExtractionSegment,
    identity_boxes,
    plan_extractions,
    segment_name,
)

__all__ = [
    "ExtractionSegment",
    "identity_boxes",
    "plan_extractions",
    "segment_name",
]
