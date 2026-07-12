"""Demo 7's math: agreement = Jaccard, the cyclic pitfall, and the S-curve."""

import math

import numpy as np

from core.sims import minhash_lsh as mh


A = {"a", "b", "c", "d"}
B = {"c", "d", "e", "f"}          # J = 2/6 = 1/3


def test_agreement_converges_to_jaccard():
    j = mh.jaccard(A, B)
    est = mh.minhash_agreement(A, B, k=20000, seed=1)
    se = math.sqrt(j * (1 - j) / 20000)
    assert abs(est - j) < 5 * se


def test_cyclic_permutations_are_not_enough():
    """HW1 3(c): A={0,3}, B={1,3} in Z_4 gives cyclic collision rate 1/2 but
    true Jaccard 1/3 — dependence breaks uniformity over the union."""
    A4, B4 = {0, 3}, {1, 3}
    assert abs(mh.jaccard(A4, B4) - 1 / 3) < 1e-12
    assert abs(mh.cyclic_agreement(A4, B4, n=4) - 1 / 2) < 1e-12


def test_dont_know_bound_holds_and_decays():
    n, m = 1000, 40
    prev = 1.0
    for k in (5, 25, 125, 625):
        exact, bound = mh.dont_know_prob(n, m, k)
        assert exact <= bound + 1e-12
        assert exact < prev
        prev = exact
    # HW1 3(b): k = 10n/m drives it below e^-10
    exact, _ = mh.dont_know_prob(n, m, k=10 * n // m)
    assert exact <= math.e**-10 + 1e-12


def test_s_curve_matches_simulation():
    for s, r, b in [(0.3, 4, 8), (0.6, 4, 8), (0.8, 6, 16)]:
        theory = float(mh.s_curve(s, r, b))
        emp = mh.band_collision_empirical(s, r, b, trials=40000, seed=2)
        se = math.sqrt(max(theory * (1 - theory), 1e-6) / 40000)
        assert abs(emp - theory) < 5 * se + 0.005


def test_threshold_sits_on_the_steep_part():
    for r, b in [(3, 4), (5, 20), (8, 32)]:
        t = mh.s_curve_threshold(r, b)
        p = float(mh.s_curve(t, r, b))
        assert 0.25 < p < 0.85          # the inflection region
        grid = np.linspace(0.01, 0.99, 99)
        vals = mh.s_curve(grid, r, b)
        assert np.all(np.diff(vals) >= 0)      # monotone (flat near 0 at fp precision)
        assert vals[-1] - vals[0] > 0.95       # and genuinely S-shaped
