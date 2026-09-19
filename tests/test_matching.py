"""Tests for the assignment and overlap primitives.

The solver is checked against brute force: a subtly wrong assignment is the kind
of failure that produces plausible output and cannot be detected downstream.
"""

from __future__ import annotations

import itertools
import random

import pytest

from avannotate.matching import hungarian, iou, iou_distance, linear_assignment


def _brute_force(cost: list[list[float]]) -> float:
    """Optimal total cost, by trying every injective row -> column mapping."""

    rows, columns = len(cost), len(cost[0])
    return min(
        sum(cost[i][choice[i]] for i in range(rows))
        for choice in itertools.permutations(range(columns), rows)
    )


# --------------------------------------------------------------------------- #
# the solver
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", range(12))
def test_hungarian_matches_brute_force_square(seed: int) -> None:
    rng = random.Random(seed)
    size = rng.randint(1, 5)
    cost = [[rng.uniform(-2, 2) for _ in range(size)] for _ in range(size)]
    assignment = hungarian(cost)
    assert sorted(assignment) == list(range(size))
    assert sum(cost[i][assignment[i]] for i in range(size)) == pytest.approx(_brute_force(cost))


@pytest.mark.parametrize("seed", range(12))
def test_hungarian_matches_brute_force_rectangular(seed: int) -> None:
    rng = random.Random(seed + 100)
    rows = rng.randint(1, 4)
    columns = rows + rng.randint(0, 3)
    cost = [[rng.uniform(-2, 2) for _ in range(columns)] for _ in range(rows)]
    assignment = hungarian(cost)
    assert len(set(assignment)) == rows
    assert all(0 <= column < columns for column in assignment)
    assert sum(cost[i][assignment[i]] for i in range(rows)) == pytest.approx(_brute_force(cost))


def test_hungarian_rejects_more_rows_than_columns() -> None:
    with pytest.raises(ValueError):
        hungarian([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])


def test_hungarian_prefers_globally_better_over_greedy() -> None:
    """The case greedy gets wrong: the top-scoring pair blocks a better total."""

    scores = [[0.90, 0.80], [0.85, 0.10]]
    assignment = hungarian([[-value for value in row] for row in scores])
    assert assignment == [1, 0]
    assert scores[0][1] + scores[1][0] == pytest.approx(1.65)


def test_hungarian_on_empty_input() -> None:
    assert hungarian([]) == []


# --------------------------------------------------------------------------- #
# thresholded assignment
# --------------------------------------------------------------------------- #


def test_linear_assignment_drops_pairs_over_the_threshold() -> None:
    cost = [[0.1, 0.9], [0.9, 0.2]]
    matches, unmatched_rows, unmatched_columns = linear_assignment(cost, threshold=0.5)
    assert sorted(matches) == [(0, 0), (1, 1)]
    assert unmatched_rows == []
    assert unmatched_columns == []


def test_linear_assignment_reports_both_sides_unmatched() -> None:
    cost = [[0.9, 0.9]]
    matches, unmatched_rows, unmatched_columns = linear_assignment(cost, threshold=0.5)
    assert matches == []
    assert unmatched_rows == [0]
    assert unmatched_columns == [0, 1]


def test_linear_assignment_handles_more_rows_than_columns() -> None:
    """The transposed case: more detections than tracks."""

    cost = [[0.1], [0.2], [0.9]]
    matches, unmatched_rows, unmatched_columns = linear_assignment(cost, threshold=0.5)
    assert matches == [(0, 0)]
    assert unmatched_rows == [1, 2]
    assert unmatched_columns == []


def test_linear_assignment_on_empty_input() -> None:
    assert linear_assignment([], threshold=0.5) == ([], [], [])
    assert linear_assignment([[]], threshold=0.5) == ([], [0], [])


def test_linear_assignment_with_no_rows_reports_every_column_unmatched() -> None:
    """The tracker's first frame: no tracks exist, so every detection starts one.

    A rowless matrix carries no width, so the caller has to supply it.  Getting
    this wrong does not raise -- it returns no unmatched detections, and the
    tracker silently creates nothing.
    """

    matches, unmatched_rows, unmatched_columns = linear_assignment(
        [], threshold=0.8, column_count=3
    )
    assert matches == []
    assert unmatched_rows == []
    assert unmatched_columns == [0, 1, 2]


def test_linear_assignment_without_a_column_count_cannot_guess_the_width() -> None:
    assert linear_assignment([], threshold=0.8, column_count=None) == ([], [], [])


def test_linear_assignment_prefers_a_poor_pair_over_an_unmatched_row() -> None:
    """A match just over the threshold is dropped, but a better one is kept even
    when a competing row could also take it."""

    cost = [[0.4, 0.6], [0.45, 0.6]]
    matches, unmatched_rows, _ = linear_assignment(cost, threshold=0.5)
    assert matches == [(0, 0)]
    assert unmatched_rows == [1]


# --------------------------------------------------------------------------- #
# overlap
# --------------------------------------------------------------------------- #


def test_iou_of_identical_boxes_is_one() -> None:
    assert iou((0.0, 0.0, 10.0, 10.0), (0.0, 0.0, 10.0, 10.0)) == pytest.approx(1.0)


def test_iou_of_disjoint_boxes_is_zero() -> None:
    assert iou((0.0, 0.0, 10.0, 10.0), (20.0, 20.0, 10.0, 10.0)) == 0.0


def test_iou_of_touching_boxes_is_zero() -> None:
    """Half-open rectangles: sharing an edge is not overlapping."""

    assert iou((0.0, 0.0, 10.0, 10.0), (10.0, 0.0, 10.0, 10.0)) == 0.0


def test_iou_of_half_overlap() -> None:
    # Intersection 10x10 = 100; union 100 + 100 - 100 = 100... boxes are 20x20.
    value = iou((0.0, 0.0, 20.0, 20.0), (10.0, 0.0, 20.0, 20.0))
    assert value == pytest.approx(100.0 / 300.0)


def test_iou_with_a_degenerate_box() -> None:
    assert iou((0.0, 0.0, 0.0, 10.0), (0.0, 0.0, 10.0, 10.0)) == 0.0


def test_iou_distance_matrix_shape_and_values() -> None:
    rows = [(0.0, 0.0, 10.0, 10.0)]
    columns = [(0.0, 0.0, 10.0, 10.0), (100.0, 100.0, 10.0, 10.0)]
    distance = iou_distance(rows, columns)
    assert len(distance) == 1
    assert distance[0][0] == pytest.approx(0.0)
    assert distance[0][1] == pytest.approx(1.0)
