"""AdaBoost with decision stumps (CS229 PS3.3-3.4), pure numpy.

Round t: pick the stump minimizing weighted error eps_t, weight it by
alpha_t = 0.5*ln((1-eps_t)/eps_t), and reweight the data so that the stump
just used becomes worthless (weighted error exactly 1/2 under the new
weights). Training error obeys the PS3.4 chain:

    (1/n) sum 1[H(x_i) != y_i]  <=  prod_t Z_t
                                 =   prod_t 2*sqrt(eps_t(1-eps_t))
                                 <=  exp(-2 sum_t gamma_t^2),  gamma_t = 1/2 - eps_t.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Stump:
    """h(x) = sign * (+1 if x[axis] > thresh else -1)."""
    axis: int
    thresh: float
    sign: int

    def predict(self, X: np.ndarray) -> np.ndarray:
        raw = np.where(X[:, self.axis] > self.thresh, 1, -1)
        return self.sign * raw


@dataclass
class Round:
    stump: Stump
    eps: float
    alpha: float
    z: float
    weights_before: np.ndarray
    weights_after: np.ndarray


def best_stump(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[Stump, float]:
    """Exhaustive search over axes, midpoints, and polarities."""
    best, best_eps = None, np.inf
    for axis in range(X.shape[1]):
        values = np.unique(X[:, axis])
        cuts = np.concatenate([[values[0] - 1.0],
                               (values[:-1] + values[1:]) / 2,
                               [values[-1] + 1.0]])
        for thresh in cuts:
            raw = np.where(X[:, axis] > thresh, 1, -1)
            for sign in (1, -1):
                eps = float(w[(sign * raw) != y].sum())
                if eps < best_eps:
                    best, best_eps = Stump(axis, float(thresh), sign), eps
    return best, best_eps


def boost_round(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> Round:
    stump, eps = best_stump(X, y, w)
    eps = min(max(eps, 1e-12), 1 - 1e-12)
    alpha = 0.5 * np.log((1 - eps) / eps)
    margin = y * stump.predict(X)
    w_new = w * np.exp(-alpha * margin)
    z = float(w_new.sum())                    # = 2 sqrt(eps (1-eps))
    w_new = w_new / z
    return Round(stump, eps, float(alpha), z, w.copy(), w_new)


def run(X: np.ndarray, y: np.ndarray, T: int) -> list[Round]:
    w = np.full(len(y), 1.0 / len(y))
    rounds = []
    for _ in range(T):
        rnd = boost_round(X, y, w)
        rounds.append(rnd)
        w = rnd.weights_after
    return rounds


def committee_scores(X: np.ndarray, rounds: list[Round]) -> np.ndarray:
    """f(x) = sum_t alpha_t h_t(x)."""
    f = np.zeros(len(X))
    for rnd in rounds:
        f += rnd.alpha * rnd.stump.predict(X)
    return f


def train_error(X: np.ndarray, y: np.ndarray, rounds: list[Round]) -> float:
    return float(np.mean(np.sign(committee_scores(X, rounds)) != y))


def bound_history(X: np.ndarray, y: np.ndarray, rounds: list[Round]) -> dict:
    """Per-round: train error, prod Z_t, exp(-2 sum gamma^2)."""
    errs, prod_z, exp_bound = [], [], []
    pz, sg2 = 1.0, 0.0
    for t in range(1, len(rounds) + 1):
        errs.append(train_error(X, y, rounds[:t]))
        pz *= rounds[t - 1].z
        sg2 += (0.5 - rounds[t - 1].eps) ** 2
        prod_z.append(pz)
        exp_bound.append(float(np.exp(-2 * sg2)))
    return {"train_error": errs, "prod_z": prod_z, "exp_bound": exp_bound}


def demo_data(seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """A 2-D +/- layout no single stump can solve (diagonal-ish classes)."""
    rng = np.random.default_rng(seed)
    pos = np.concatenate([
        rng.normal([-1.6, 1.4], 0.45, size=(6, 2)),
        rng.normal([1.7, -1.3], 0.45, size=(6, 2)),
    ])
    neg = np.concatenate([
        rng.normal([1.6, 1.5], 0.45, size=(6, 2)),
        rng.normal([-1.7, -1.5], 0.45, size=(6, 2)),
    ])
    X = np.vstack([pos, neg])
    y = np.array([1] * len(pos) + [-1] * len(neg))
    return X, y
