"""Gradient descent on quadratic objectives J(theta) = 1/2 theta^T A theta.

Pure numpy — no manim imports. Every function here is exercised by
tests/test_gradient_descent.py before any scene or web page consumes it.

The whole demo hangs on one identity: with A = Q diag(lambda) Q^T and
c_t = Q^T theta_t, the GD iterate theta_{t+1} = (I - alpha A) theta_t
decouples into d independent scalar recurrences

    c_i(t) = (1 - alpha * lambda_i)^t * c_i(0),

so GD converges iff |1 - alpha*lambda_i| < 1 for all i, i.e. alpha < 2/lambda_max.
"""

from __future__ import annotations

import numpy as np


def rotation(angle_rad: float) -> np.ndarray:
    """2x2 rotation matrix."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s], [s, c]])


def make_quadratic(lambdas, angle_rad: float = 0.0) -> np.ndarray:
    """SPD matrix A = Q diag(lambdas) Q^T, Q = rotation by angle (2-D only for angle != 0)."""
    lambdas = np.asarray(lambdas, dtype=float)
    if angle_rad == 0.0:
        return np.diag(lambdas)
    if lambdas.shape != (2,):
        raise ValueError("rotation angle only supported for 2-D quadratics")
    q = rotation(angle_rad)
    return q @ np.diag(lambdas) @ q.T


def loss(A: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """J(theta) = 1/2 theta^T A theta; theta may be a single point or a path (T, d)."""
    theta = np.atleast_2d(theta)
    return 0.5 * np.einsum("ti,ij,tj->t", theta, A, theta)


def gd_path(A: np.ndarray, theta0, alpha: float, steps: int) -> np.ndarray:
    """Iterates [theta_0, ..., theta_steps], shape (steps+1, d)."""
    theta = np.asarray(theta0, dtype=float)
    path = np.empty((steps + 1, theta.size))
    path[0] = theta
    step_matrix = np.eye(theta.size) - alpha * A
    for t in range(steps):
        theta = step_matrix @ theta
        path[t + 1] = theta
    return path


def eigen_frame(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(lambdas ascending, Q with matching eigenvector columns)."""
    lambdas, Q = np.linalg.eigh(A)
    return lambdas, Q


def to_eigen(A: np.ndarray, path: np.ndarray) -> np.ndarray:
    """Path expressed in eigen-coordinates c = Q^T theta."""
    _, Q = eigen_frame(A)
    return path @ Q


def mode_factors(A: np.ndarray, alpha: float) -> np.ndarray:
    """Per-mode contraction factors 1 - alpha*lambda_i (ascending lambda order)."""
    lambdas, _ = eigen_frame(A)
    return 1.0 - alpha * lambdas


def closed_form_path(A: np.ndarray, theta0, alpha: float, steps: int) -> np.ndarray:
    """theta_t = Q (I - alpha Lambda)^t Q^T theta_0 — the decoupled solution."""
    lambdas, Q = eigen_frame(A)
    c0 = Q.T @ np.asarray(theta0, dtype=float)
    t = np.arange(steps + 1)[:, None]
    c = (1.0 - alpha * lambdas) ** t * c0
    return c @ Q.T


def convergence_threshold(A: np.ndarray) -> float:
    """Largest stable learning rate: alpha_crit = 2 / lambda_max."""
    lambdas, _ = eigen_frame(A)
    return 2.0 / lambdas[-1]


def optimal_alpha(A: np.ndarray) -> float:
    """Fixed step minimizing the worst-case rate: 2 / (lambda_min + lambda_max)."""
    lambdas, _ = eigen_frame(A)
    return 2.0 / (lambdas[0] + lambdas[-1])


def worst_rate(A: np.ndarray, alpha: float) -> float:
    """max_i |1 - alpha*lambda_i| — the geometric error rate; < 1 means convergence."""
    return float(np.max(np.abs(mode_factors(A, alpha))))


def diverges(A: np.ndarray, theta0, alpha: float, steps: int = 2000) -> bool:
    """True if GD blows up from a generic start.

    Renormalizes the iterate every step and averages the log growth over the
    second half of the run (once the dominant mode has taken over), so the
    verdict is overflow-proof and sharp arbitrarily close to the boundary.
    """
    theta = np.asarray(theta0, dtype=float)
    theta = theta / np.linalg.norm(theta)
    step_matrix = np.eye(theta.size) - alpha * A
    burn = steps // 2
    acc = 0.0
    for t in range(steps):
        theta = step_matrix @ theta
        growth = np.linalg.norm(theta)
        theta /= growth
        if t >= burn:
            acc += np.log(growth)
    return acc > 0.0


def contour_ellipse(A: np.ndarray, level: float, n: int = 200) -> np.ndarray:
    """Points on {theta : J(theta) = level} for 2-D A, shape (n, 2)."""
    lambdas, Q = eigen_frame(A)
    phi = np.linspace(0.0, 2.0 * np.pi, n)
    radii = np.sqrt(2.0 * level / lambdas)
    circle = np.stack([radii[0] * np.cos(phi), radii[1] * np.sin(phi)], axis=1)
    return circle @ Q.T
