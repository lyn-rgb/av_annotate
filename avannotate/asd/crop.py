"""Turning a detected face box into the crop the network expects.

A face detector returns a tight box around the visible face.  Every ASD model is
trained on something looser -- a square-ish region including forehead, chin and
some background -- and the difference matters more than it looks: the mouth
region is where the signal is, and a crop that cuts the chin off removes it.

**The margin convention is unverified.**  LoCoNet's preprocessing follows
TalkNet's, which exposes a ``cropScale`` of 0.40, but the exact arithmetic --
whether the margin is a fraction of the width on each side or of the larger
side, and whether the result is squared before resizing -- could not be checked
against the repository from this machine.  The interpretation used here is the
one TalkNet's argument name suggests, and :func:`crop_box` is the only place it
appears.  Verify against the checkpoint's own preprocessing on the first server
run; a mismatch is a domain shift applied to every frame.
"""

from __future__ import annotations

from dataclasses import dataclass

from avannotate.coercion import coerce_number

#: **Zero, because LoCoNet has no crop scale.**  Reading ``cropScale = 0.40``
#: off TalkNet and assuming it was inherited was wrong: a case-insensitive
#: search of the LoCoNet repository for ``cropScale`` or ``crop_scale`` returns
#: nothing, and its AVA preprocessing crops the detected box directly --
#: ``frame[y1:y2, x1:x2]`` -- with no expansion and no squaring, leaving the
#: aspect ratio to be flattened later by ``cv2.resize(face, (112, 112))``.
#:
#: So the box is used as the detector reported it.  Expanding it here would
#: frame every face more loosely than the checkpoint was trained on, which is a
#: domain shift applied to every frame of every video -- and one that degrades
#: the scores quietly rather than raising.
DEFAULT_MARGIN = 0.0

#: What every ASD checkpoint in this family resizes its crops to.  Square and
#: hard: the aspect ratio is not preserved, which is the repository's own
#: behaviour and therefore the behaviour this has to reproduce.
CROP_SIZE = 112


@dataclass(frozen=True)
class CropBox:
    """A rectangle in source-frame pixels, integral and inside the frame."""

    x: int
    y: int
    width: int
    height: int

    @property
    def size(self) -> tuple[int, int]:
        return (self.width, self.height)

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    def to_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> CropBox:
        return cls(
            x=int(coerce_number(payload["x"], "x")),
            y=int(coerce_number(payload["y"], "y")),
            width=int(coerce_number(payload["width"], "width")),
            height=int(coerce_number(payload["height"], "height")),
        )


def clamp_box(box: CropBox, *, frame_width: int, frame_height: int) -> CropBox:
    """Trim a box to the frame, returning a zero-area box when it falls outside.

    Clamping rather than sliding: a face at the edge is genuinely cut off by the
    frame, and pretending otherwise by sliding the box inward would put the
    face off-centre in a crop the model expects to be centred.
    """

    if frame_width <= 0 or frame_height <= 0:
        raise ValueError(f"invalid frame size {frame_width}x{frame_height}")

    left = max(0, min(box.x, frame_width))
    top = max(0, min(box.y, frame_height))
    right = max(left, min(box.x + box.width, frame_width))
    bottom = max(top, min(box.y + box.height, frame_height))
    return CropBox(x=left, y=top, width=right - left, height=bottom - top)


def crop_box(
    box: tuple[float, float, float, float],
    *,
    frame_width: int,
    frame_height: int,
    margin: float = DEFAULT_MARGIN,
) -> CropBox:
    """Expand a face box by ``margin`` and fit it to the frame.

    ``box`` is ``(x, y, width, height)`` -- the same convention as
    :class:`avannotate.faces.types.Detection`, which is what S2's tracklets
    carry.
    """

    if margin < 0.0:
        raise ValueError(f"margin cannot be negative, got {margin}")

    x, y, width, height = (float(value) for value in box)
    if width <= 0.0 or height <= 0.0:
        return CropBox(x=int(x), y=int(y), width=0, height=0)

    pad_x = width * margin
    pad_y = height * margin
    expanded = CropBox(
        x=int(round(x - pad_x)),
        y=int(round(y - pad_y)),
        width=int(round(width + 2.0 * pad_x)),
        height=int(round(height + 2.0 * pad_y)),
    )
    return clamp_box(expanded, frame_width=frame_width, frame_height=frame_height)
