"""Demo 10's math: Jensen, asymmetry, the chain rule, and the MLE bridge."""

import numpy as np

from core.sims import kl_mle as km

RNG = np.random.default_rng(43)


def rand_dist(k: int) -> np.ndarray:
    v = RNG.uniform(0.05, 1.0, size=k)
    return v / v.sum()


def test_nonnegative_and_zero_iff_equal():
    for _ in range(20):
        p, q = rand_dist(6), rand_dist(6)
        assert km.kl(p, q) >= 0
        assert km.kl(p, p) < 1e-14
    p = rand_dist(6)
    q = rand_dist(6)
    if not np.allclose(p, q):
        assert km.kl(p, q) > 1e-6


def test_asymmetry():
    p = np.array([0.9, 0.05, 0.05])
    q = np.array([1 / 3, 1 / 3, 1 / 3])
    assert abs(km.kl(p, q) - km.kl(q, p)) > 0.2


def test_chain_rule_exact():
    for _ in range(10):
        jp = RNG.uniform(0.02, 1.0, size=(3, 4)); jp /= jp.sum()
        jq = RNG.uniform(0.02, 1.0, size=(3, 4)); jq /= jq.sum()
        parts = km.chain_rule_parts(jp, jq)
        assert abs(parts["total"] - parts["marginal"] - parts["conditional"]) < 1e-12
        assert parts["marginal"] >= -1e-15          # each piece is itself a KL
        assert parts["conditional"] >= -1e-15


def test_min_kl_is_max_likelihood_is_sample_mean():
    counts = km.sample_counts(theta_true=0.62, n_samples=400, seed=5)
    p_hat = km.empirical_dist(counts)
    thetas = np.linspace(0.01, 0.99, 981)
    kl_c = km.kl_curve(p_hat, thetas)
    ll_c = km.loglik_curve(p_hat, thetas)
    # same optimizer, from both objectives, equal to the closed form
    assert thetas[np.argmin(kl_c)] == thetas[np.argmax(ll_c)]
    assert abs(thetas[np.argmin(kl_c)] - km.mle_theta(counts)) < 2e-3


def test_kl_and_loglik_differ_by_the_entropy_constant():
    counts = km.sample_counts(theta_true=0.4, n_samples=300, seed=7)
    p_hat = km.empirical_dist(counts)
    thetas = np.linspace(0.05, 0.95, 91)
    kl_c = km.kl_curve(p_hat, thetas)
    ll_c = km.loglik_curve(p_hat, thetas)
    h = km.entropy(p_hat)
    np.testing.assert_allclose(kl_c, -h - ll_c, atol=1e-12)
