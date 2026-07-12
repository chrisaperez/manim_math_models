"""Demo 11's math: the PS2.2 formulas vs Monte Carlo, and 'ridge always helps'."""

import numpy as np

from core.sims import ridge as rg

X = rg.make_design(n=40, gammas=(0.4, 8.0), seed=9)
W_STAR = np.array([1.1, 0.8])
SIGMA = 0.8


def test_design_has_requested_gram_eigenvalues():
    gam = np.linalg.eigvalsh(X.T @ X)
    np.testing.assert_allclose(np.sort(gam), [0.4, 8.0], atol=1e-10)


def test_closed_form_matches_monte_carlo():
    for lam in (0.0, 0.5, 3.0):
        cloud = rg.estimate_cloud(X, W_STAR, SIGMA, lam, trials=6000, seed=1)
        emp_bias2 = float(np.sum((cloud.mean(axis=0) - W_STAR) ** 2))
        emp_var = float(cloud.var(axis=0).sum())
        cf = rg.closed_form(X, W_STAR, SIGMA, lam)
        assert abs(emp_var - cf["var"]) < 0.08 * max(cf["var"], 0.05)
        assert abs(emp_bias2 - cf["bias2"]) < 0.05


def test_ols_is_unbiased_and_pays_in_variance():
    cf = rg.closed_form(X, W_STAR, SIGMA, 0.0)
    assert cf["bias2"] < 1e-20
    expected_var = SIGMA**2 * (1 / 0.4 + 1 / 8.0)
    assert abs(cf["var"] - expected_var) < 1e-10


def test_bias_rises_and_variance_falls_in_lambda():
    lams = np.linspace(0, 20, 200)
    c = rg.curves(X, W_STAR, SIGMA, lams)
    assert np.all(np.diff(c["bias2"]) >= -1e-12)
    assert np.all(np.diff(c["var"]) <= 1e-12)


def test_some_ridge_always_helps():
    """PS2.2(d): MSE'(0) < 0, and the U-curve bottoms at an interior lambda."""
    assert rg.mse_slope_at_zero(X, W_STAR, SIGMA) < 0
    lams = np.linspace(0, 15, 3000)
    c = rg.curves(X, W_STAR, SIGMA, lams)
    star = np.argmin(c["mse"])
    assert 0 < star < len(lams) - 1
    assert c["mse"][star] < c["mse"][0]      # strictly better than OLS


def test_shrinkage_hits_weak_directions_hardest():
    f = rg.shrinkage_factors(X, lam=2.0)      # ascending gamma order
    assert f[0] < f[1]                        # small-gamma mode shrunk more
    assert np.all((f > 0) & (f < 1))
