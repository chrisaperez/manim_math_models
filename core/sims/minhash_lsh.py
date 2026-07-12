"""MinHash & LSH (CS246 HW1.3-1.4), pure numpy.

The miracle: for a uniform random permutation pi,
    P[min pi(A) = min pi(B)] = |A ∩ B| / |A ∪ B| = J(A, B),
because the first row of A ∪ B under pi is uniform over the union, and the
signatures agree iff that row lies in the intersection.

Banding: b bands of r rows -> P[candidate] = 1 - (1 - s^r)^b, an S-curve with
threshold ~ (1/b)^{1/r}. And HW1 3(c)'s warning: cyclic shifts are NOT enough.
"""

from __future__ import annotations

import numpy as np


def jaccard(A: set, B: set) -> float:
    return len(A & B) / len(A | B) if A | B else 0.0


def minhash_agreement(A: set, B: set, k: int, seed: int = 0) -> float:
    """Fraction of k independent random permutations whose minhashes agree."""
    rng = np.random.default_rng(seed)
    universe = sorted(A | B)
    idx = {x: i for i, x in enumerate(universe)}
    a_mask = np.array([x in A for x in universe])
    b_mask = np.array([x in B for x in universe])
    agree = 0
    for _ in range(k):
        perm = rng.permutation(len(universe))
        agree += perm[a_mask].min() == perm[b_mask].min()
    return agree / k


def cyclic_agreement(A: set, B: set, n: int) -> float:
    """Collision rate over the n cyclic shifts pi_s(x) = (x + s) mod n.
    HW1 3(c): this does NOT generally equal the Jaccard similarity."""
    agree = 0
    for s in range(n):
        ha = min((x + s) % n for x in A)
        hb = min((x + s) % n for x in B)
        agree += ha == hb
    return agree / n


def dont_know_prob(n: int, m: int, k: int) -> tuple[float, float]:
    """(exact-ish sampled-rows probability ((n-m)/n)^k, bound e^{-km/n}).
    HW1 3(a-b): probability that k sampled rows are all 0 for a column
    with m ones out of n rows."""
    exact = ((n - m) / n) ** k
    bound = float(np.exp(-k * m / n))
    return exact, bound


def s_curve(s, r: int, b: int):
    """P[candidate pair] = 1 - (1 - s^r)^b."""
    s = np.asarray(s, dtype=float)
    return 1.0 - (1.0 - s**r) ** b


def s_curve_threshold(r: int, b: int) -> float:
    return (1.0 / b) ** (1.0 / r)


def band_collision_empirical(s: float, r: int, b: int, trials: int = 20000,
                             seed: int = 0) -> float:
    """Monte Carlo: each signature row agrees independently w.p. s; a pair is
    a candidate iff some band has all r rows agreeing."""
    rng = np.random.default_rng(seed)
    rows = rng.random((trials, b, r)) < s
    return float(rows.all(axis=2).any(axis=1).mean())
