"""How many threads one worker may have, and why it is not "all of them".

The parallelism in this pipeline is across videos: N worker processes, each
running one video at a time.  Every numeric library in the stack adds a second,
nested layer of it -- OpenCV keeps a pool, OpenMP keeps one for numpy's BLAS,
torch keeps one, onnxruntime keeps one -- and each of those defaults to a
thread per core.

So a sixteen-worker run on a hundred-and-twenty-eight-core machine asks for two
thousand threads to do sixteen videos' worth of work.  It does not fail; it
just spends the machine on itself.  The observed shape is a worker burning
seven cores while the card it is meant to be feeding sits at three percent, and
the fix is to divide the threads rather than multiply them.

**Set before the libraries are imported, because they read it once.**  That is
the whole reason this runs in the worker initialiser: by the time a stage has
imported numpy, the pool exists and an environment variable changed afterwards
does nothing.

The share is cores divided by workers, so the two layers together come to about
one machine rather than the square of it.  An operator who has set one of these
variables themselves is left alone -- they know something this does not.
"""

from __future__ import annotations

import os

#: Variables the numeric stack reads at import.  All of them, because which one
#: a given library honours depends on how it was built: numpy from PyPI may be
#: OpenBLAS or MKL, and torch has its own pool that respects OMP by default.
VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

#: What the last :func:`cap` decided, for :func:`cap_opencv` to reuse.
_share = 1


def share_for(workers: int) -> int:
    """How many threads each of ``workers`` processes should get."""

    return max(1, (os.cpu_count() or 1) // max(1, workers))


def cap(workers: int) -> int:
    """Divide the machine's threads among the workers.  Returns the share.

    Called once per worker, before anything heavy is imported.
    """

    global _share
    _share = share_for(workers)
    for name in VARIABLES:
        os.environ.setdefault(name, str(_share))
    return _share


def cap_opencv() -> None:
    """Give OpenCV the same share.

    It is the one library here with no environment variable for its thread
    count -- ``setNumThreads`` is the only way in -- so every module that
    imports it has to ask.  Without this, ``cv2.resize`` on a frame opens a pool
    sized by the core count, in every worker, and its threads wait by spinning.
    """

    try:
        import cv2
    except ModuleNotFoundError:  # pragma: no cover - the stages without OpenCV
        return
    cv2.setNumThreads(_share)
