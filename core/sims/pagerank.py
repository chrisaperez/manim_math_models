"""PageRank (CS246 HW3.1-3.2), pure numpy.

r' = beta * M r + (1-beta) * 1/n, with M column-stochastic (M[i,j] = 1/deg(j)
if j -> i). Chris's HW3 1(a): column sums 1 => total rank conserved. Dead ends
leak mass (zero column); spider traps hoard it; teleportation fixes both the
trap and (in the google_step form used here) redistributes what the walk keeps.
"""

from __future__ import annotations

import numpy as np

# 6-node demo web: a small strongly-connected core
DEMO_EDGES = [
    (0, 1), (0, 2), (1, 2), (2, 0), (2, 3), (3, 4), (4, 0), (4, 5),
    (5, 0), (1, 3), (3, 1),
]


def with_dead_end(edges=DEMO_EDGES, n: int = 6):
    """Node n becomes a sink: edges in, none out."""
    return list(edges) + [(2, n), (4, n)], n + 1


def with_trap(edges=DEMO_EDGES, n: int = 6):
    """Nodes n, n+1 form a 2-cycle trap reachable from the core."""
    return list(edges) + [(2, n), (n, n + 1), (n + 1, n)], n + 2


def link_matrix(edges, n: int) -> np.ndarray:
    """Column-stochastic M; a dead end contributes a zero column."""
    M = np.zeros((n, n))
    out_deg = np.zeros(n)
    for j, _ in edges:
        out_deg[j] += 1
    for j, i in edges:
        M[i, j] = 1.0 / out_deg[j]
    return M


def step(M: np.ndarray, r: np.ndarray) -> np.ndarray:
    return M @ r


def google_step(M: np.ndarray, r: np.ndarray, beta: float) -> np.ndarray:
    n = len(r)
    return beta * (M @ r) + (1.0 - beta) / n


def power_iterate(M: np.ndarray, beta: float = 0.85, tol: float = 1e-12,
                  max_iter: int = 500):
    n = M.shape[0]
    r = np.full(n, 1.0 / n)
    history = [r]
    for it in range(max_iter):
        r_new = google_step(M, r, beta)
        history.append(r_new)
        delta = np.abs(r_new - r).sum()
        r = r_new
        if delta < tol:
            return r, history, it + 1
    return r, history, max_iter


def eig_rank(M: np.ndarray, beta: float = 0.85) -> np.ndarray:
    """Principal eigenvector of the Google matrix, L1-normalized."""
    n = M.shape[0]
    G = beta * M + (1.0 - beta) / n * np.ones((n, n))
    vals, vecs = np.linalg.eig(G)
    v = np.real(vecs[:, np.argmax(np.real(vals))])
    v = np.abs(v)
    return v / v.sum()


def total_mass(r: np.ndarray) -> float:
    return float(r.sum())
