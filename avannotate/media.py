"""What counts as a video file.

One predicate, in its own module, because three places need to agree on it and
two of them cannot import from each other.  The CLI's directory scan, the
batch's list reader and the test fixtures all decide "is this a video", and the
answer is not the suffix -- see :func:`is_media_file`.

It grew in ``cli.py``, which is the entry point: a library module importing the
entry point to reach a predicate is backwards, and under a batch run it is a
cycle, because the entry point imports the batch.
"""

from __future__ import annotations

from pathlib import Path

VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"})


def is_media_file(path: Path) -> bool:
    """Whether a path is a video, as opposed to something wearing a video's name.

    The suffix alone is not enough, and the exception is not exotic.  macOS
    writes an AppleDouble sidecar beside every file it copies onto a filesystem
    that cannot hold extended attributes -- a network mount, an external drive,
    an HDD formatted by something else -- naming it ``._`` plus the original.
    It holds that file's attributes and none of its bytes, and it ends in
    ``.mp4`` exactly like the video it describes.

    A scan that trusts the suffix hands it to ffprobe, which answers ``moov atom
    not found`` and marks the video failed.  Worse than it sounds: the failure
    reads as the video's fault, and it is attached to a video that is fine.  It
    is not hypothetical here -- the corpus this was written against arrived with
    one beside it.

    Anything starting with a dot is skipped, not only ``._``.  A hidden file is
    hidden deliberately, every other tool that lists a directory agrees, and a
    rule that enumerates the artefacts is a rule that has to be extended the
    next time one appears.  Naming a file explicitly is still honoured: this
    governs what a scan picks up on its own, not what it was told to open.
    """

    return not path.name.startswith(".") and path.suffix.lower() in VIDEO_SUFFIXES
