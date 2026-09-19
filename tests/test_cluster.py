"""Tests for identity clustering.

The numbers mirror the measured corpus: two fragments of one person score 0.78
cosine (0.22 distance) and different people 0.09 (0.91 distance).
"""

from __future__ import annotations

import numpy as np
import pytest

from avannotate.faces.cluster import cluster_vectors

DIM = 8


def _unit(channel: int = 0) -> np.ndarray:
    """A unit vector along one axis."""

    vector = np.zeros(DIM, dtype=np.float32)
    vector[channel] = 1.0
    return vector


def _blend(first: np.ndarray, second: np.ndarray, weight: float) -> np.ndarray:
    """A unit vector ``weight`` of the way from ``first`` toward ``second``.

    Explicit endpoints rather than a channel index: a helper that nudges toward
    a default direction silently returns its input when that direction is the
    one it started from, and the test then asserts nothing.
    """

    blended = first * (1.0 - weight) + second * weight
    return (blended / np.linalg.norm(blended)).astype(np.float32)


def _stack(*vectors: np.ndarray) -> np.ndarray:
    return np.asarray(vectors, dtype=np.float32)


def test_blend_helper_is_a_real_perturbation() -> None:
    """Guards the tests below: a no-op blend would make several of them vacuous."""

    a, b = _unit(0), _unit(1)
    assert float(a @ _blend(a, b, 0.3)) < 0.99


def test_empty_input_yields_nothing() -> None:
    assert cluster_vectors(np.zeros((0, DIM), dtype=np.float32)) == ()


def test_a_single_vector_is_its_own_cluster() -> None:
    assert cluster_vectors(_stack(_unit(0))) == ((0,),)


def test_identical_vectors_merge() -> None:
    vector = _unit(0)
    assert cluster_vectors(_stack(vector, vector.copy())) == ((0, 1),)


def test_orthogonal_vectors_stay_apart() -> None:
    assert cluster_vectors(_stack(_unit(0), _unit(1))) == ((0,), (1,))


def test_two_similar_plus_one_different() -> None:
    a, b = _unit(0), _unit(1)
    clusters = cluster_vectors(_stack(a, _blend(a, b, 0.2), b))
    assert clusters == ((0, 1), (2,))


def test_the_threshold_decides_the_merge() -> None:
    a, b = _unit(0), _unit(1)
    vectors = _stack(a, _blend(a, b, 0.6))

    assert cluster_vectors(vectors, max_distance=0.9) == ((0, 1),)
    assert cluster_vectors(vectors, max_distance=0.01) == ((0,), (1,))


def test_average_linkage_refuses_to_chain() -> None:
    """A-B and B-C close, A-C far: average linkage must not put all three together.

    Single linkage would: it merges on B's closeness to each in turn, and one
    person's identity absorbs their scene partner's.
    """

    a, b_axis = _unit(0), _unit(1)
    # b close to a, c close to b but far from a.
    vector_b = _blend(a, b_axis, 0.3)
    vector_c = _blend(a, b_axis, 0.7)

    distance_ab = 1.0 - float(a @ vector_b)
    distance_bc = 1.0 - float(vector_b @ vector_c)
    distance_ac = 1.0 - float(a @ vector_c)
    assert distance_ab < distance_bc < distance_ac  # the premise

    # Once a and b have merged, the distance average linkage sees to c is the
    # mean of their two distances -- and that mean is what must exceed the
    # threshold, while the smaller of the two (what single linkage would use)
    # does not.
    average_to_c = (distance_bc + distance_ac) / 2
    threshold = average_to_c * 0.9
    assert distance_ab <= threshold
    assert distance_bc <= threshold, "single linkage would chain here"
    assert average_to_c > threshold, "average linkage must stop here"

    clusters = cluster_vectors(_stack(a, vector_b, vector_c), max_distance=threshold)
    assert sorted(len(cluster) for cluster in clusters) == [1, 2]


def test_transitive_grouping_of_three_alike_vectors() -> None:
    a, b_axis = _unit(0), _unit(1)
    clusters = cluster_vectors(
        _stack(a, _blend(a, b_axis, 0.1), _blend(a, b_axis, 0.15)),
    )
    assert clusters == ((0, 1, 2),)


def test_a_zero_vector_does_not_poison_the_result() -> None:
    """A NaN anywhere turns every distance NaN and every cluster a singleton."""

    clusters = cluster_vectors(_stack(_unit(0), np.zeros(DIM, dtype=np.float32), _unit(0)))
    assert (0, 2) in clusters


def test_clusters_are_ordered_and_sorted_within() -> None:
    a, b = _unit(0), _unit(1)
    clusters = cluster_vectors(_stack(_blend(a, b, 0.2), a, b, _blend(a, b, 0.1)))
    assert [cluster[0] for cluster in clusters] == sorted(cluster[0] for cluster in clusters)
    for cluster in clusters:
        assert list(cluster) == sorted(cluster)


def test_result_is_deterministic_across_runs() -> None:
    a, b = _unit(0), _unit(1)
    vectors = _stack(a, _blend(a, b, 0.15), b, _blend(a, b, 0.25))
    assert cluster_vectors(vectors) == cluster_vectors(vectors.copy())


def test_unnormalized_input_is_handled() -> None:
    """Real embeddings arrive normalized, but a raw average of them may not be."""

    clusters = cluster_vectors(
        _stack(np.full(DIM, 3.0, dtype=np.float32), np.full(DIM, 0.5, dtype=np.float32)),
    )
    assert clusters == ((0, 1),)


def test_every_index_appears_exactly_once() -> None:
    rng = np.random.default_rng(7)
    clusters = cluster_vectors(rng.normal(size=(30, DIM)).astype(np.float32))
    flattened = [index for cluster in clusters for index in cluster]
    assert sorted(flattened) == list(range(30))


@pytest.mark.parametrize("count", [2, 5, 20])
def test_random_input_does_not_crash(count: int) -> None:
    rng = np.random.default_rng(count)
    clusters = cluster_vectors(rng.normal(size=(count, DIM)).astype(np.float32))
    assert sum(len(cluster) for cluster in clusters) == count
