"""Building the tensor one forward pass consumes.

``[S, T, 112, 112]`` crops and ``[4T, 128]`` features, for one target and its
context speakers over one window.  The arithmetic that decides *which* box goes
in which cell is separated from the pixel work, because that is where a silent
error lives: a track's crops shifted by one frame still look like faces, and the
model will still return a confident answer about the wrong instant.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from avannotate import threads
from avannotate.asd.crop import CROP_SIZE, crop_box
from avannotate.asd.types import Window
from avannotate.faces.track import Tracklet
from avannotate.faces.types import Frame
from avannotate.matching import Box

#: Frames of a track that fall outside its own detections take the nearest
#: known box.  A detector misses frames; the face does not vanish in them.
_Boxes = tuple[Box | None, ...]


def boxes_for_window(
    tracklet: Tracklet, window: Window, *, fill: bool = True
) -> _Boxes:
    """One box per frame of the window, or ``None`` where the track has none.

    A detection is a sighting, not an existence: between two of them the face
    was there and the detector simply did not report it.  With ``fill`` those
    gaps take the most recent known box, and a leading gap takes the first box
    seen later in the window -- otherwise the model would be shown a blank tile
    and asked whether that face is talking.  Without ``fill`` they stay ``None``,
    which is what a caller measuring coverage wants.
    """

    by_frame = {detection.frame_index: detection.box for detection in tracklet.detections}
    boxes: list[Box | None] = []
    held: Box | None = None
    for frame_index in range(window.start_frame, window.end_frame):
        found = by_frame.get(frame_index)
        if found is not None:
            held = found
        # Without ``fill`` the gap stays a gap: a caller measuring how much of
        # the window this track actually covers wants the absences visible.
        boxes.append(found if found is not None else (held if fill else None))

    if not fill:
        return tuple(boxes)

    first = next((box for box in boxes if box is not None), None)
    return tuple(first if box is None else box for box in boxes)


def read_faces(
    frames: Sequence[Frame],
    boxes: Sequence[Box | None],
    *,
    size: int = CROP_SIZE,
    margin: float,
) -> NDArray[np.uint8]:
    """Cut and resize one face per frame, as ``[T, size, size]`` greyscale.

    A frame with no box contributes a black tile rather than being skipped: the
    model's input is a fixed-length sequence, and dropping a frame would shift
    every later one against the audio it is supposed to line up with.
    """

    if len(frames) != len(boxes):
        raise ValueError(
            f"{len(frames)} frames but {len(boxes)} boxes; they would not align"
        )

    import cv2
    threads.cap_opencv()

    tiles = np.zeros((len(frames), size, size), dtype=np.uint8)
    for index, (frame, box) in enumerate(zip(frames, boxes, strict=True)):
        if box is None:
            continue
        height, width = frame.shape[:2]
        cut = crop_box(box, frame_width=width, frame_height=height, margin=margin)
        if cut.area == 0:
            continue
        patch = frame[cut.y : cut.y + cut.height, cut.x : cut.x + cut.width]
        if patch.size == 0:
            continue
        tiles[index] = cv2.resize(patch, (size, size), interpolation=cv2.INTER_LINEAR)
    return tiles


def stack_speakers(tiles: Sequence[NDArray[np.uint8]]) -> NDArray[np.float32]:
    """``[S, T, size, size]`` uint8 tiles to the ``[S, T, size, size]`` float input.

    **Left in the 0..255 range on purpose.**  LoCoNet normalises inside its own
    visual frontend -- ``(x / 255 - 0.4161) / 0.1688`` -- so dividing here as
    well would apply the shift twice and hand the network a batch sitting around
    -2.5 instead of around 0.5.  Nothing would fail; every activation in the
    first layer would simply be wrong, which is the kind of mistake that shows
    up as an unimpressive mAP rather than as a bug report.

    The uint8 is widened to float32 because that is what a torch tensor of
    images has to be, not because the values change.
    """

    if not tiles:
        raise ValueError("no speaker tiles to stack")
    lengths = {tile.shape[0] for tile in tiles}
    if len(lengths) != 1:
        raise ValueError(f"speakers have differing frame counts: {sorted(lengths)}")
    stacked = np.stack(list(tiles), axis=0).astype(np.float32)
    return np.ascontiguousarray(stacked)
