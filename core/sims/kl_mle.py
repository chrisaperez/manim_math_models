"""KL divergence & maximum likelihood (CS229 PS4.3), pure numpy.

KL(P||Q) = sum P log(P/Q) >= 0 by Jensen, zero iff P = Q; it chains over
joints; and minimizing KL(P_hat || P_theta) over a model family is EXACTLY
maximizing the average log-likelihood, because they differ by the constant
entropy of the data:

    KL(P_hat || P_theta) = -H(P_hat) - (1/n) sum_i log P_theta(x_i).
"""

from __future__ import annotations

import math

import numpy as np


def kl(p: np.ndarray, q: np.ndarray) -> float:
    """KL(P||Q) for discrete distributions (0 log 0 = 0; requires q>0 where p>0)."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    mask = p > 0
    return float(-np.sum(p[mask] * np.log(p[mask])))


def chain_rule_parts(joint_p: np.ndarray, joint_q: np.ndarray) -> dict:
    """KL(P(X,Y)||Q(X,Y)) = KL(P(X)||Q(X)) + KL(P(Y|X)||Q(Y|X)).

    Rows index X, columns Y. Returns each term for exact verification.
    """
    px, qx = joint_p.sum(axis=1), joint_q.sum(axis=1)
    total = kl(joint_p.ravel(), joint_q.ravel())
    marginal = kl(px, qx)
    conditional = 0.0
    for i in range(joint_p.shape[0]):
        if px[i] > 0:
            conditional += px[i] * kl(joint_p[i] / px[i], joint_q[i] / qx[i])
    return {"total": total, "marginal": marginal, "conditional": conditional}


# ---------------------------------------------------------------- MLE bridge
def binom_pmf(theta: float, n: int = 5) -> np.ndarray:
    ks = np.arange(n + 1)
    return np.array([math.comb(n, int(k)) * theta**k * (1 - theta) ** (n - k)
                     for k in ks])


def sample_counts(theta_true: float, n_samples: int, n: int = 5,
                  seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    draws = rng.binomial(n, theta_true, size=n_samples)
    return np.bincount(draws, minlength=n + 1)


def empirical_dist(counts: np.ndarray) -> np.ndarray:
    return counts / counts.sum()


def kl_curve(p_hat: np.ndarray, thetas: np.ndarray, n: int = 5) -> np.ndarray:
    return np.array([kl(p_hat, binom_pmf(t, n)) for t in thetas])


def loglik_curve(p_hat: np.ndarray, thetas: np.ndarray, n: int = 5) -> np.ndarray:
    """Average log-likelihood (1/n_samples) sum log P_theta(x_i) as an
    expectation under the empirical distribution."""
    out = []
    for t in thetas:
        pmf = binom_pmf(t, n)
        mask = p_hat > 0
        out.append(float(np.sum(p_hat[mask] * np.log(pmf[mask]))))
    return np.array(out)


def mle_theta(counts: np.ndarray, n: int = 5) -> float:
    """Closed form for Binomial(n, theta): sample mean / n."""
    ks = np.arange(len(counts))
    return float((counts * ks).sum() / (counts.sum() * n))
