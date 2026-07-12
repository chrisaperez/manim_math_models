"""Bloom filters & count-min sketch (CS246 HW4.4 + streaming lectures).

Bloom: m bits, k salted hashes. P[false positive] ~ (1 - e^{-kn/m})^k, minimized
at k = (m/n) ln 2. Zeros never lie: no false negatives, ever.

Count-min: d rows x w counters, estimate = min over rows. Never underestimates;
with w = ceil(e/eps), d = ceil(ln(1/delta)):  P[err > eps*t] <= delta.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np


def _hash(item: str, salt: int, mod: int) -> int:
    h = hashlib.sha256(f"{salt}:{item}".encode()).digest()
    return int.from_bytes(h[:8], "big") % mod


class BloomFilter:
    def __init__(self, m: int, k: int):
        self.m, self.k = m, k
        self.bits = np.zeros(m, dtype=bool)

    def positions(self, item: str) -> list[int]:
        return [_hash(item, i, self.m) for i in range(self.k)]

    def insert(self, item: str) -> None:
        self.bits[self.positions(item)] = True

    def query(self, item: str) -> bool:
        return bool(self.bits[self.positions(item)].all())


def fp_theoretical(m: int, k: int, n: int) -> float:
    return (1.0 - math.exp(-k * n / m)) ** k


def optimal_k(m: int, n: int) -> float:
    return (m / n) * math.log(2)


def fp_empirical(m: int, k: int, n: int, probes: int = 4000,
                 seed: int = 0) -> float:
    bf = BloomFilter(m, k)
    for i in range(n):
        bf.insert(f"member-{seed}-{i}")
    hits = sum(bf.query(f"probe-{seed}-{j}") for j in range(probes))
    return hits / probes


def find_false_positive(bf: BloomFilter, candidates: list[str]) -> str | None:
    """First candidate that was never inserted but still reads MAYBE."""
    for w in candidates:
        if bf.query(w):
            return w
    return None


class CountMinSketch:
    def __init__(self, w: int, d: int):
        self.w, self.d = w, d
        self.table = np.zeros((d, w), dtype=np.int64)

    def positions(self, item: str) -> list[int]:
        return [_hash(item, 100 + j, self.w) for j in range(self.d)]

    def update(self, item: str, count: int = 1) -> None:
        for j, pos in enumerate(self.positions(item)):
            self.table[j, pos] += count

    def row_estimates(self, item: str) -> np.ndarray:
        return np.array([self.table[j, pos]
                         for j, pos in enumerate(self.positions(item))])

    def query(self, item: str) -> int:
        return int(self.row_estimates(item).min())


def cms_params(eps: float, delta: float) -> tuple[int, int]:
    """(w, d) from the HW4 analysis: w = ceil(e/eps), d = ceil(ln(1/delta))."""
    return math.ceil(math.e / eps), math.ceil(math.log(1.0 / delta))


def zipf_stream(n_items: int, length: int, seed: int = 0) -> list[str]:
    rng = np.random.default_rng(seed)
    ranks = rng.zipf(1.3, size=length)
    return [f"item-{min(int(r), n_items)}" for r in ranks]
