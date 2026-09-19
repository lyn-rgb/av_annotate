"""A constant-velocity Kalman filter over ``(x, y, aspect, height)``.

This is ByteTrack's motion model, kept stateless: the mean and covariance live
on the track, and every method here takes them in and hands them back.  A filter
holding its own state would be one more thing to reset at the right moment, and
its bugs would look like tracking bugs.

Why a filter at all, when faces barely move between sampled frames: the point is
what happens when a detection is *missing*.  With no prediction a track freezes
at its last box, so after half a second of occlusion it no longer overlaps the
face it belongs to and the association fails.  With a prediction it keeps
drifting along its velocity and re-associates.  Aspect ratio rather than width
is the fourth state because a face's aspect ratio is far more stable than its
apparent width as it turns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

#: State is ``x, y, aspect, height`` plus their velocities.
STATE_DIM = 8

#: Measurement is the four observed quantities, without velocities.
MEASUREMENT_DIM = 4

Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]


def _diagonal(values: Vector) -> Matrix:
    return np.diag(values)


def _matmul(left: Matrix, right: Matrix) -> Matrix:
    """``left @ right``, via ``np.dot``, pinned back to float64.

    Not style: on numpy 2.2.6 over Apple's Accelerate BLAS, the ``matmul``
    ufunc emits "divide by zero encountered in matmul" on operands that are
    finite and produce finite results -- three warnings per call, all spurious.
    ``np.dot`` takes a different path and is silent.  Wrapping the calls in
    ``errstate`` would have hidden the noise too, and a real overflow with it.

    The ``asarray`` is because numpy types ``dot``'s result loosely, and an
    untyped array here would propagate into every subsequent arithmetic
    expression in the filter.
    """

    result: Matrix = np.asarray(np.dot(left, right), dtype=np.float64)
    return result


@dataclass(frozen=True)
class KalmanFilter:
    """Standard prediction and correction for a linear motion model."""

    std_weight_position: float = 1.0 / 20.0
    std_weight_velocity: float = 1.0 / 160.0

    def initiate(self, measurement: Vector) -> tuple[Vector, Matrix]:
        """A fresh track: observed position, zero velocity, wide uncertainty.

        Position uncertainty scales with the box height because a large face has
        proportionally larger absolute jitter, and a fixed pixel sigma would be
        far too tight for a close-up and far too loose for a distant face.
        """

        height = float(measurement[3])
        position = np.array(
            [
                2 * self.std_weight_position * height,
                2 * self.std_weight_position * height,
                1e-2,
                2 * self.std_weight_position * height,
            ]
        )
        velocity = np.array(
            [
                10 * self.std_weight_velocity * height,
                10 * self.std_weight_velocity * height,
                1e-5,
                10 * self.std_weight_velocity * height,
            ]
        )
        mean = np.concatenate([measurement, np.zeros(MEASUREMENT_DIM)])
        covariance = _diagonal(np.concatenate([position, velocity]) ** 2)
        return mean, covariance

    def _transition(self) -> Matrix:
        matrix = np.eye(STATE_DIM)
        for index in range(MEASUREMENT_DIM):
            matrix[index, MEASUREMENT_DIM + index] = 1.0
        return matrix

    def predict(self, mean: Vector, covariance: Matrix) -> tuple[Vector, Matrix]:
        height = float(mean[3])
        position = np.array(
            [
                self.std_weight_position * height,
                self.std_weight_position * height,
                1e-2,
                self.std_weight_position * height,
            ]
        )
        velocity = np.array(
            [
                self.std_weight_velocity * height,
                self.std_weight_velocity * height,
                1e-5,
                self.std_weight_velocity * height,
            ]
        )
        motion_covariance = _diagonal(np.concatenate([position, velocity]) ** 2)

        transition = self._transition()
        predicted_mean = _matmul(transition, mean)
        predicted_covariance = (
            _matmul(_matmul(transition, covariance), transition.T) + motion_covariance
        )
        return predicted_mean, predicted_covariance

    def project(self, mean: Vector, covariance: Matrix) -> tuple[Vector, Matrix]:
        """The measurement distribution implied by the current state."""

        height = float(mean[3])
        std = np.array(
            [
                self.std_weight_position * height,
                self.std_weight_position * height,
                1e-2,
                self.std_weight_position * height,
            ]
        )
        innovation_covariance = _diagonal(std**2)
        observation = np.eye(MEASUREMENT_DIM, STATE_DIM)
        projected_mean = _matmul(observation, mean)
        projected_covariance = (
            _matmul(_matmul(observation, covariance), observation.T) + innovation_covariance
        )
        return projected_mean, projected_covariance

    def update(
        self, mean: Vector, covariance: Matrix, measurement: Vector
    ) -> tuple[Vector, Matrix]:
        projected_mean, projected_covariance = self.project(mean, covariance)
        observation = np.eye(MEASUREMENT_DIM, STATE_DIM)

        # solve rather than inv: the innovation covariance is symmetric positive
        # definite, and forming its inverse is both slower and less stable.  The
        # asarray pins the result: numpy types solve loosely, and an untyped
        # gain would propagate into every expression that uses it.
        gain: Matrix = np.asarray(
            np.linalg.solve(projected_covariance.T, _matmul(covariance, observation.T).T).T,
            dtype=np.float64,
        )
        corrected_mean = mean + _matmul(gain, measurement - projected_mean)
        corrected_covariance = covariance - _matmul(
            _matmul(gain, projected_covariance), gain.T
        )
        return corrected_mean, corrected_covariance
