"""SVD vs PCA (CS229 PS4.1, CS246 HW2.1, CS205L HW3), pure numpy.

Two stories, one decomposition:
  PCA:  the unit u maximizing sum (u.x)^2 — equivalently minimizing squared
        reconstruction error, by the per-point Pythagorean identity
            ||x||^2 = ||f_u(x)||^2 + ||x - f_u(x)||^2.
  SVD:  M = U S V^T — rotate, stretch along axes, rotate.
Bridge: M^T M = V S^2 V^T, so PCA directions ARE the right singular vectors
and the covariance eigenvalues are the squared singular values.
"""

from __future__ import annotations

import numpy as np


def rotation(t: float) -> np.ndarray:
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s], [s, c]])


def make_cloud(n: int = 60, angle: float = 0.6, sig1: float = 2.0,
               sig2: float = 0.6, seed: int = 4) -> np.ndarray:
    """Centered 2-D cloud whose principal direction is `angle`."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 2)) * np.array([sig1, sig2])
    pts = z @ rotation(angle).T
    return pts - pts.mean(axis=0)


def unit(theta: float) -> np.ndarray:
    return np.array([np.cos(theta), np.sin(theta)])


def project(points: np.ndarray, u: np.ndarray) -> dict:
    """Projections onto span(u): kept component, residual, and their energies."""
    coeffs = points @ u
    proj = np.outer(coeffs, u)
    resid = points - proj
    return {
        "coeffs": coeffs,
        "proj": proj,
        "resid": resid,
        "keep": float(np.sum(coeffs**2)),
        "lose": float(np.sum(resid**2)),
        "total": float(np.sum(points**2)),
    }


def sweep_variance(points: np.ndarray, n_angles: int = 181) -> dict:
    thetas = np.linspace(0.0, np.pi, n_angles)
    keep = np.array([project(points, unit(t))["keep"] for t in thetas])
    total = float(np.sum(points**2))
    return {"thetas": thetas, "keep": keep, "lose": total - keep, "total": total}


def principal_dir(points: np.ndarray) -> np.ndarray:
    """Top right-singular vector of the data matrix (canonical sign: x >= 0)."""
    _, _, vt = np.linalg.svd(points, full_matrices=False)
    v = vt[0]
    return v if v[0] >= 0 else -v


def svd_stages(M: np.ndarray) -> dict:
    """U, S, Vt plus the rotation angles of U and V (2x2, det-corrected so
    both factors are pure rotations)."""
    U, S, Vt = np.linalg.svd(M)
    # fold reflections into the sign of the second singular axis
    if np.linalg.det(Vt) < 0:
        Vt = np.diag([1, -1]) @ Vt
        U = U @ np.diag([1, -1])
    if np.linalg.det(U) < 0:
        U = U @ np.diag([1, -1])
        S = S * np.array([1, -1])
    return {
        "U": U, "S": S, "Vt": Vt,
        "angle_u": float(np.arctan2(U[1, 0], U[0, 0])),
        "angle_v": float(np.arctan2(Vt[0, 1], Vt[0, 0])),
    }


def mtm_eigen(M: np.ndarray) -> dict:
    lam, Q = np.linalg.eigh(M.T @ M)
    return {"eigvals": lam[::-1], "eigvecs": Q[:, ::-1]}  # descending
