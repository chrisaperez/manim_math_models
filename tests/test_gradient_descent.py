"""The math behind Demo 1, proven before it is drawn."""

import numpy as np
import pytest

from core.sims import gradient_descent as gd

RNG = np.random.default_rng(229)


def random_spd(d: int) -> np.ndarray:
    m = RNG.normal(size=(d, d))
    q, _ = np.linalg.qr(m)
    lambdas = RNG.uniform(0.5, 10.0, size=d)
    return q @ np.diag(lambdas) @ q.T


@pytest.mark.parametrize("d", [2, 3, 8])
def test_iterates_match_closed_form(d):
    A = random_spd(d)
    theta0 = RNG.normal(size=d)
    alpha = 0.9 * gd.convergence_threshold(A)
    iterated = gd.gd_path(A, theta0, alpha, steps=60)
    closed = gd.closed_form_path(A, theta0, alpha, steps=60)
    np.testing.assert_allclose(iterated, closed, atol=1e-9)


def test_modes_decouple_exactly():
    A = gd.make_quadratic([1.0, 6.0], angle_rad=np.deg2rad(30))
    theta0 = np.array([1.5, -0.7])
    alpha = 0.25
    c = gd.to_eigen(A, gd.gd_path(A, theta0, alpha, steps=40))
    lambdas, Q = gd.eigen_frame(A)
    t = np.arange(41)[:, None]
    expected = (1 - alpha * lambdas) ** t * (Q.T @ theta0)
    np.testing.assert_allclose(c, expected, atol=1e-10)


@pytest.mark.parametrize("d", [2, 5])
def test_divergence_boundary_is_two_over_lambda_max(d):
    """Bisect the alpha where GD flips from converging to diverging; must be 2/lambda_max."""
    A = random_spd(d)
    theta0 = RNG.normal(size=d)  # generic: has weight on the top eigenvector
    crit = gd.convergence_threshold(A)
    lo, hi = 0.5 * crit, 1.5 * crit
    assert not gd.diverges(A, theta0, lo, steps=2000)
    assert gd.diverges(A, theta0, hi, steps=2000)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if gd.diverges(A, theta0, mid, steps=2000):
            hi = mid
        else:
            lo = mid
    assert abs(0.5 * (lo + hi) - crit) < 1e-6 * crit


def test_loss_monotone_below_one_over_lambda_max():
    A = random_spd(4)
    theta0 = RNG.normal(size=4)
    lambdas, _ = gd.eigen_frame(A)
    path = gd.gd_path(A, theta0, 0.99 / lambdas[-1], steps=100)
    j = gd.loss(A, path)
    assert np.all(np.diff(j) <= 1e-12)


def test_optimal_alpha_rate_is_kappa_ratio():
    A = gd.make_quadratic([2.0, 18.0], angle_rad=0.4)
    kappa = 18.0 / 2.0
    rate = gd.worst_rate(A, gd.optimal_alpha(A))
    assert np.isclose(rate, (kappa - 1) / (kappa + 1), atol=1e-12)
    # and no other alpha does better
    for alpha in np.linspace(0.01, gd.convergence_threshold(A) - 1e-3, 200):
        assert gd.worst_rate(A, alpha) >= rate - 1e-12


def test_contour_ellipse_lies_on_level_set():
    A = gd.make_quadratic([1.0, 5.0], angle_rad=1.1)
    pts = gd.contour_ellipse(A, level=3.0)
    np.testing.assert_allclose(gd.loss(A, pts), 3.0, atol=1e-10)
