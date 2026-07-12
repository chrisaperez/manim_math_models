"""Demo 5's math: no false negatives, calibrated false positives, honest CMS."""

import math

import numpy as np

from core.sims import sketches as sk


def test_bloom_never_lies_with_a_zero():
    """No false negatives: every inserted item queries MAYBE."""
    bf = sk.BloomFilter(m=64, k=3)
    items = [f"w{i}" for i in range(20)]
    for w in items:
        bf.insert(w)
    assert all(bf.query(w) for w in items)


def test_bloom_fp_matches_theory():
    m, k, n = 256, 3, 60
    emp = sk.fp_empirical(m, k, n, probes=8000, seed=1)
    theo = sk.fp_theoretical(m, k, n)
    se = math.sqrt(theo * (1 - theo) / 8000)
    assert abs(emp - theo) < 5 * se + 0.01


def test_optimal_k_minimizes_theory():
    m, n = 240, 30  # m/n = 8 -> k* = 8 ln2 = 5.55
    kstar = sk.optimal_k(m, n)
    best_int = min(range(1, 13), key=lambda k: sk.fp_theoretical(m, k, n))
    assert abs(best_int - kstar) <= 1.0


def test_cms_never_underestimates_and_meets_eps_delta():
    eps, delta = 0.05, 0.1
    w, d = sk.cms_params(eps, delta)
    cms = sk.CountMinSketch(w, d)
    stream = sk.zipf_stream(n_items=300, length=5000, seed=3)
    truth = {}
    for it in stream:
        cms.update(it)
        truth[it] = truth.get(it, 0) + 1
    t = len(stream)
    errs = []
    for it, c in truth.items():
        est = cms.query(it)
        assert est >= c                      # one-sided by construction
        errs.append(est - c)
    frac_bad = np.mean([e > eps * t for e in errs])
    assert frac_bad <= delta


def test_cms_row_estimates_all_upper_bounds():
    cms = sk.CountMinSketch(50, 4)
    for i in range(200):
        cms.update(f"x{i % 17}")
    for i in range(17):
        rows = cms.row_estimates(f"x{i}")
        assert (rows >= cms.query(f"x{i}")).all()
        assert cms.query(f"x{i}") >= (200 // 17)  # every item appeared >= 11x
