"""Demo 6's math: mass conservation, leaks, traps, and the eigenvector."""

import numpy as np

from core.sims import pagerank as pr


M_CORE = pr.link_matrix(pr.DEMO_EDGES, 6)


def test_columns_sum_to_one_without_dead_ends():
    np.testing.assert_allclose(M_CORE.sum(axis=0), np.ones(6), atol=1e-12)


def test_pure_walk_conserves_mass():
    r = np.full(6, 1 / 6)
    for _ in range(50):
        r = pr.step(M_CORE, r)
    assert abs(pr.total_mass(r) - 1.0) < 1e-12


def test_dead_end_leaks_mass_monotonically():
    edges, n = pr.with_dead_end()
    M = pr.link_matrix(edges, n)
    r = np.full(n, 1 / n)
    masses = [pr.total_mass(r)]
    for _ in range(20):
        r = pr.step(M, r)
        masses.append(pr.total_mass(r))
    assert all(m2 < m1 + 1e-15 for m1, m2 in zip(masses, masses[1:]))
    assert masses[-1] < 0.5  # substantial leak after 20 steps


def test_trap_hoards_mass_without_teleport():
    edges, n = pr.with_trap()
    M = pr.link_matrix(edges, n)
    r = np.full(n, 1 / n)
    for _ in range(300):
        r = pr.step(M, r)
    assert abs(pr.total_mass(r) - 1.0) < 1e-9      # conserved...
    assert r[6] + r[7] > 0.999                      # ...but all in the trap


def test_teleport_rescues_the_trap_and_conserves():
    edges, n = pr.with_trap()
    M = pr.link_matrix(edges, n)
    r, _, iters = pr.power_iterate(M, beta=0.85)
    assert abs(pr.total_mass(r) - 1.0) < 1e-9
    assert r[6] + r[7] < 0.5                        # trap no longer absorbs
    assert iters < 200


def test_power_iteration_matches_eigenvector():
    r, _, _ = pr.power_iterate(M_CORE, beta=0.85)
    v = pr.eig_rank(M_CORE, beta=0.85)
    np.testing.assert_allclose(r, v, atol=1e-8)


def test_convergence_rate_is_about_beta():
    beta = 0.8
    _, hist, _ = pr.power_iterate(M_CORE, beta=beta, tol=1e-14, max_iter=80)
    deltas = [np.abs(b - a).sum() for a, b in zip(hist, hist[1:])]
    ratios = [d2 / d1 for d1, d2 in zip(deltas[5:20], deltas[6:21]) if d1 > 1e-13]
    assert all(rho <= beta + 0.05 for rho in ratios)
