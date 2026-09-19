"""Assignment and box-overlap primitives.

Two stages need these: S2 matches detections to tracks frame by frame, and S8
matches speakers to faces once per video.  They were written for S8 and moved
here when the tracker arrived, because a second copy of an assignment algorithm
is a second place for it to be subtly wrong.

Pure Python, no numpy: S8 lives in the deterministic core, which is testable
without any of the array or model dependencies.
"""

from __future__ import annotations

from collections.abc import Sequence

#: A box as ``(x, y, width, height)`` in pixels.
Box = tuple[float, float, float, float]


def hungarian(cost: list[list[float]]) -> list[int]:
    """Minimum-cost assignment of rows to columns, one each.

    Straight implementation of the shortest-augmenting-path form, O(n^2 m).
    Requires ``len(cost) <= len(cost[0])``; callers transpose when they do not.
    Returns the column chosen for each row, or ``-1`` if a row went unassigned.
    """

    rows = len(cost)
    if rows == 0:
        return []
    columns = len(cost[0])
    if rows > columns:
        raise ValueError("cost matrix must have at least as many columns as rows")

    infinity = float("inf")
    row_potential = [0.0] * (rows + 1)
    column_potential = [0.0] * (columns + 1)
    # column_match[j] is the 1-based row assigned to column j; 0 means free.
    column_match = [0] * (columns + 1)
    previous = [0] * (columns + 1)

    for row in range(1, rows + 1):
        column_match[0] = row
        column = 0
        min_reduced = [infinity] * (columns + 1)
        used = [False] * (columns + 1)

        while True:
            used[column] = True
            current_row = column_match[column]
            delta = infinity
            next_column = 0
            for candidate in range(1, columns + 1):
                if used[candidate]:
                    continue
                reduced = (
                    cost[current_row - 1][candidate - 1]
                    - row_potential[current_row]
                    - column_potential[candidate]
                )
                if reduced < min_reduced[candidate]:
                    min_reduced[candidate] = reduced
                    previous[candidate] = column
                if min_reduced[candidate] < delta:
                    delta = min_reduced[candidate]
                    next_column = candidate
            for candidate in range(columns + 1):
                if used[candidate]:
                    row_potential[column_match[candidate]] += delta
                    column_potential[candidate] -= delta
                else:
                    min_reduced[candidate] -= delta
            column = next_column
            if column_match[column] == 0:
                break

        while column:
            previous_column = previous[column]
            column_match[column] = column_match[previous_column]
            column = previous_column

    assignment = [-1] * rows
    for column in range(1, columns + 1):
        if column_match[column] > 0:
            assignment[column_match[column] - 1] = column - 1
    return assignment


def linear_assignment(
    cost: list[list[float]], *, threshold: float, column_count: int | None = None
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Minimum-cost matching, discarding pairs that cost more than ``threshold``.

    Returns ``(matches, unmatched_rows, unmatched_columns)``.  The threshold is
    applied *after* the solve, not by forbidding pairs beforehand: a pair that
    is merely poor is still worth taking when the alternative is leaving both
    sides unassigned, and that is the difference between a tracker that survives
    a missed detection and one that fragments.

    ``column_count`` supplies the width when ``cost`` has no rows.  That case is
    not an edge case: a tracker with no tracks yet has a 0 x N cost matrix, and
    every detection is unmatched.  Inferring the width from an empty list
    returns none of them, so no track is ever created -- a failure that looks
    like "the tracker found nothing" rather than like a bug.
    """

    rows = len(cost)
    if rows == 0:
        return [], [], list(range(column_count or 0))
    columns = len(cost[0])
    if columns == 0:
        return [], list(range(rows)), []

    transpose = rows > columns
    matrix = (
        [[cost[r][c] for r in range(rows)] for c in range(columns)] if transpose else cost
    )

    assignment = hungarian(matrix)
    matches: list[tuple[int, int]] = []
    for index, chosen in enumerate(assignment):
        if chosen < 0:
            continue
        row, column = (chosen, index) if transpose else (index, chosen)
        if cost[row][column] <= threshold:
            matches.append((row, column))

    matched_rows = {row for row, _ in matches}
    matched_columns = {column for _, column in matches}
    return (
        matches,
        [row for row in range(rows) if row not in matched_rows],
        [column for column in range(columns) if column not in matched_columns],
    )


def iou(first: Box, second: Box) -> float:
    """Intersection over union of two ``(x, y, w, h)`` boxes."""

    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    if aw <= 0.0 or ah <= 0.0 or bw <= 0.0 or bh <= 0.0:
        return 0.0

    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)
    if right <= left or bottom <= top:
        return 0.0

    intersection = (right - left) * (bottom - top)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0.0 else 0.0


def iou_distance(rows: Sequence[Box], columns: Sequence[Box]) -> list[list[float]]:
    """``1 - IoU`` for every pair -- a cost matrix for :func:`linear_assignment`."""

    return [[1.0 - iou(row, column) for column in columns] for row in rows]
