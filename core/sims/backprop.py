"""Batched gradients through one dense layer (CS229 PS2.4), pure numpy.

Layer:  Z = X W + b,   A = sigma(Z),   L = (1/2n) ||A - Y||^2

The three identities the demo animates, with shapes:

    dL/dZ = (dL/dA) ⊙ sigma'(Z)          (n,k) — Hadamard mask
    dL/dX = (dL/dZ) W^T                  (n,k)(k,d) -> (n,d)
    dL/dW = X^T (dL/dZ)                  (d,n)(n,k) -> (d,k)
    dL/db = 1^T (dL/dZ)                  sum over the batch

Everything is verified against central finite differences in tests.
"""

from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z: np.ndarray) -> np.ndarray:
    s = sigmoid(z)
    return s * (1.0 - s)


def forward(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> dict:
    Z = X @ W + b
    A = sigmoid(Z)
    return {"Z": Z, "A": A}


def loss(A: np.ndarray, Y: np.ndarray) -> float:
    n = A.shape[0]
    return float(np.sum((A - Y) ** 2) / (2 * n))


def backward(X: np.ndarray, W: np.ndarray, b: np.ndarray, Y: np.ndarray) -> dict:
    """All gradients of L = (1/2n)||sigma(XW+b) - Y||^2, plus intermediates."""
    n = X.shape[0]
    f = forward(X, W, b)
    Z, A = f["Z"], f["A"]
    G = (A - Y) / n                    # dL/dA — one error vector per example
    dZ = G * sigmoid_prime(Z)          # Hadamard gate
    return {
        "Z": Z, "A": A, "G": G, "dZ": dZ,
        "dX": dZ @ W.T,
        "dW": X.T @ dZ,
        "db": dZ.sum(axis=0),
        "loss": loss(A, Y),
    }


def outer_product_decomp(X: np.ndarray, dZ: np.ndarray) -> np.ndarray:
    """Per-example rank-1 votes: stack of x_i^T delta_i, shape (n, d, k).

    Summing over the first axis reproduces dL/dW exactly — each data point
    stamps its own outer product onto the weight update.
    """
    return np.einsum("ni,nj->nij", X, dZ)


def fd_grad(param_name: str, X, W, b, Y, eps: float = 1e-6) -> np.ndarray:
    """Central finite-difference gradient of the loss w.r.t. one parameter."""
    params = {"X": X.copy(), "W": W.copy(), "b": b.copy()}
    target = params[param_name]
    grad = np.zeros_like(target)
    it = np.nditer(target, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        orig = target[idx]
        target[idx] = orig + eps
        lp = loss(forward(params["X"], params["W"], params["b"])["A"], Y)
        target[idx] = orig - eps
        lm = loss(forward(params["X"], params["W"], params["b"])["A"], Y)
        target[idx] = orig
        grad[idx] = (lp - lm) / (2 * eps)
    return grad
