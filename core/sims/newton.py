"""Newton vs gradient descent (CS205L HW7-9), pure numpy.

Newton's step -H^{-1} grad solves a quadratic exactly in ONE step from any
start (affine invariance: it optimizes in the metric of the Hessian, where
kappa = 1). GD needs ~ kappa log(1/eps) steps. Quasi-Newton (BFGS) learns the
curvature from gradient differences via the secant condition B s = y.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------- objectives
def quadratic(A: np.ndarray):
    f = lambda x: 0.5 * x @ A @ x
    grad = lambda x: A @ x
    hess = lambda x: A
    return f, grad, hess


def rosenbrock(a: float = 1.0, b: float = 100.0):
    def f(x):
        return (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2

    def grad(x):
        return np.array([
            -2 * (a - x[0]) - 4 * b * x[0] * (x[1] - x[0] ** 2),
            2 * b * (x[1] - x[0] ** 2),
        ])

    def hess(x):
        return np.array([
            [2 - 4 * b * (x[1] - 3 * x[0] ** 2), -4 * b * x[0]],
            [-4 * b * x[0], 2 * b],
        ])

    return f, grad, hess


# ---------------------------------------------------------------- optimizers
def gd_run(grad, x0, alpha: float, steps: int = 10000, tol: float = 1e-8):
    """Fixed-step GD; returns (path, first index with |grad| < tol or None)."""
    x = np.asarray(x0, dtype=float)
    path = [x.copy()]
    hit = None
    for t in range(steps):
        g = grad(x)
        if np.linalg.norm(g) < tol and hit is None:
            hit = t
            break
        x = x - alpha * g
        path.append(x.copy())
    return np.array(path), hit


def newton_run(grad, hess, x0, steps: int = 50, tol: float = 1e-10,
               damped: bool = False, f=None):
    """Full or backtracking-damped Newton."""
    x = np.asarray(x0, dtype=float)
    path = [x.copy()]
    for _ in range(steps):
        g = grad(x)
        if np.linalg.norm(g) < tol:
            break
        d = np.linalg.solve(hess(x), -g)
        t = 1.0
        if damped and f is not None:
            while f(x + t * d) > f(x) + 1e-4 * t * (g @ d) and t > 1e-8:
                t *= 0.5
        x = x + t * d
        path.append(x.copy())
    return np.array(path)


def bfgs_run(f, grad, x0, steps: int = 200, tol: float = 1e-8):
    """BFGS on the inverse Hessian H; returns (path, secant_residuals)."""
    x = np.asarray(x0, dtype=float)
    n = x.size
    H = np.eye(n)                      # inverse-Hessian approximation
    path = [x.copy()]
    secant_residuals = []
    g = grad(x)
    for _ in range(steps):
        if np.linalg.norm(g) < tol:
            break
        d = -H @ g
        # backtracking line search on the Armijo condition
        t = 1.0
        while f(x + t * d) > f(x) + 1e-4 * t * (g @ d) and t > 1e-12:
            t *= 0.5
        s = t * d
        x_new = x + s
        g_new = grad(x_new)
        y = g_new - g
        sy = s @ y
        if sy > 1e-12:
            rho = 1.0 / sy
            I = np.eye(n)
            H = (I - rho * np.outer(s, y)) @ H @ (I - rho * np.outer(y, s)) \
                + rho * np.outer(s, s)
            # secant condition on the inverse: H y = s
            secant_residuals.append(float(np.linalg.norm(H @ y - s)))
        x, g = x_new, g_new
        path.append(x.copy())
    return np.array(path), secant_residuals


def whitening_map(A: np.ndarray) -> np.ndarray:
    """A^{-1/2}: the change of variables in which the quadratic has kappa=1."""
    lam, Q = np.linalg.eigh(A)
    return Q @ np.diag(lam ** -0.5) @ Q.T


def gd_steps_to_tol(A: np.ndarray, x0, tol: float = 1e-6) -> int:
    """Steps of optimally-stepped GD until |grad| < tol on the quadratic."""
    lam = np.linalg.eigvalsh(A)
    alpha = 2.0 / (lam[0] + lam[-1])
    _, grad, _ = quadratic(A)
    path, hit = gd_run(grad, x0, alpha, steps=200000, tol=tol)
    return hit if hit is not None else len(path)
