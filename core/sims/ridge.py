"""Ridge vs OLS bias-variance (CS229 PS2.2), pure numpy.

w_hat(lam) = (X^T X + lam I)^{-1} X^T y with y = X w* + noise. In the
eigenbasis of X^T X (eigenvalues gamma_i, eigenvectors v_i):

    bias^2(lam) = lam^2 * sum_i (v_i . w*)^2 / (gamma_i + lam)^2
    var(lam)    = sigma^2 * sum_i gamma_i / (gamma_i + lam)^2

OLS (lam = 0) is unbiased but pays sigma^2 sum 1/gamma_i; the MSE curve is
U-shaped with strictly negative slope at 0 - some ridge always helps.
"""

from __future__ import annotations

import numpy as np


def make_design(n: int = 40, gammas=(0.4, 8.0), seed: int = 9) -> np.ndarray:
    """Fixed design X whose Gram matrix X^T X has the given eigenvalues."""
    rng = np.random.default_rng(seed)
    d = len(gammas)
    raw = rng.normal(size=(n, d))
    q, _ = np.linalg.qr(raw)                 # n x d with orthonormal columns
    return q @ np.diag(np.sqrt(np.asarray(gammas, float)))


def ridge_estimator(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)


def estimate_cloud(X: np.ndarray, w_star: np.ndarray, sigma: float,
                   lam: float, trials: int = 400, seed: int = 0) -> np.ndarray:
    """Monte Carlo draws of w_hat(lam) over fresh noise, shape (trials, d)."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    mean = X @ w_star
    out = np.empty((trials, d))
    for t in range(trials):
        y = mean + sigma * rng.normal(size=n)
        out[t] = ridge_estimator(X, y, lam)
    return out


def closed_form(X: np.ndarray, w_star: np.ndarray, sigma: float,
                lam: float) -> dict:
    """bias^2, variance, and mse of w_hat(lam), from the PS2.2 formulas."""
    gam, V = np.linalg.eigh(X.T @ X)
    proj = V.T @ w_star
    bias2 = float(np.sum((lam * proj / (gam + lam)) ** 2))
    var = float(sigma**2 * np.sum(gam / (gam + lam) ** 2))
    return {"bias2": bias2, "var": var, "mse": bias2 + var}


def curves(X: np.ndarray, w_star: np.ndarray, sigma: float,
           lams: np.ndarray) -> dict:
    vals = [closed_form(X, w_star, sigma, l) for l in lams]
    return {
        "lams": lams,
        "bias2": np.array([v["bias2"] for v in vals]),
        "var": np.array([v["var"] for v in vals]),
        "mse": np.array([v["mse"] for v in vals]),
    }


def mse_slope_at_zero(X: np.ndarray, w_star: np.ndarray, sigma: float,
                      eps: float = 1e-6) -> float:
    a = closed_form(X, w_star, sigma, 0.0)["mse"]
    b = closed_form(X, w_star, sigma, eps)["mse"]
    return (b - a) / eps


def shrinkage_factors(X: np.ndarray, lam: float) -> np.ndarray:
    """Per-eigenmode multiplier gamma_i/(gamma_i + lam), ascending gamma."""
    gam = np.linalg.eigvalsh(X.T @ X)
    return gam / (gam + lam)
