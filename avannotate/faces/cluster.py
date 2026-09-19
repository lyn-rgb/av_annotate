"""Agglomerative clustering of identity vectors.

This is what rejoins the tracklets a camera pan splits.  Measured on the sample
corpus: two fragments of one person, 211 px apart after a pan, score 0.78 cosine
against each other and 0.08-0.12 against everyone else.  Position cannot link
them; appearance can, with a margin wide enough that the threshold is not
delicate.

Average linkage rather than single or complete.  Single linkage chains: one
borderline pair joins two clusters and they stay joined.  Complete linkage
splits a person whose appearance drifts across a long clip -- a face turning,
lighting changing -- because it insists every pair be close.  Average linkage
tolerates drift within a person while still refusing to bridge two people, which
is the trade this corpus needs.

Merges use Lance-Williams for average linkage, updating the distance matrix in
place rather than recomputing cluster pairs from scratch.  O(n^3), where ``n`` is
the tracklet count per video -- tens, not thousands.  A video with thousands of
tracklets has a tracking problem that clustering cannot fix.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

#: Cosine distance below which two vectors are the same person.
#:
#: The corpus separates at 0.22 (same person: 1 - 0.78) against 0.88 (different
#: people), so anything between them works.  This sits nearer the same-person
#: side because splitting one person across two ids corrupts the output quietly,
#: while merging two people is a mistake the QA margin can still surface.
DEFAULT_MAX_DISTANCE = 0.45

#: Slack when comparing distances, so a tie is decided by index rather than by
#: the last bit of a float.
_TIE_EPSILON = 1e-9


def _normalize(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # A zero vector would divide by zero and poison the matrix; nothing should
    # produce one, but a single NaN makes every distance NaN and every cluster a
    # singleton, with no error anywhere.
    safe = np.where(norms > 0.0, norms, 1.0)
    return np.asarray(vectors / safe, dtype=np.float32)


def cluster_vectors(
    vectors: NDArray[np.float32], *, max_distance: float = DEFAULT_MAX_DISTANCE
) -> tuple[tuple[int, ...], ...]:
    """Group rows of ``vectors`` by cosine similarity.

    Returns clusters of original row indices, each sorted, and the clusters
    ordered by their smallest member.  Deterministic: ties break to the lowest
    indices, so the same input always yields the same grouping -- which is what
    lets a rerun be diffed against the previous one.
    """

    count = len(vectors)
    if count == 0:
        return ()
    if count == 1:
        return ((0,),)

    normalized = _normalize(np.asarray(vectors, dtype=np.float32))
    distances = (1.0 - normalized @ normalized.T).astype(np.float64)
    np.fill_diagonal(distances, np.inf)

    clusters: list[list[int]] = [[index] for index in range(count)]
    sizes = [1] * count
    # Slots stay in ascending order as they are retired, so scanning them in
    # order makes the first strictly-closest pair the lowest-indexed one.
    slots = list(range(count))

    while len(slots) > 1:
        best = _closest_pair(distances, slots, max_distance)
        if best is None:
            break
        left, right = best

        left_size, right_size = sizes[left], sizes[right]
        merged_size = left_size + right_size
        for other in slots:
            if other in (left, right):
                continue
            combined = (
                left_size * distances[left][other] + right_size * distances[right][other]
            ) / merged_size
            distances[left][other] = combined
            distances[other][left] = combined

        clusters[left].extend(clusters[right])
        sizes[left] = merged_size
        slots.remove(right)

    return tuple(
        tuple(sorted(clusters[slot])) for slot in sorted(slots, key=lambda s: min(clusters[s]))
    )


def _closest_pair(
    distances: NDArray[np.float64], slots: list[int], max_distance: float
) -> tuple[int, int] | None:
    """The nearest pair of live slots within the threshold, or ``None``."""

    best: tuple[int, int] | None = None
    best_distance = float("inf")
    for position, left in enumerate(slots):
        for right in slots[position + 1 :]:
            value = float(distances[left][right])
            if value <= max_distance and value < best_distance - _TIE_EPSILON:
                best_distance = value
                best = (left, right)
    return best
