"""Demo 2's math: every analytic gradient must match finite differences."""

import numpy as np
import pytest

from core.sims import backprop as bp

RNG = np.random.default_rng(2024)


@pytest.fixture
def problem():
    n, d, k = 7, 3, 4
    return (
        RNG.normal(size=(n, d)),
        RNG.normal(size=(d, k)) * 0.8,
        RNG.normal(size=k) * 0.3,
        RNG.uniform(0.1, 0.9, size=(n, k)),
    )


@pytest.mark.parametrize("param", ["X", "W", "b"])
def test_gradients_match_finite_differences(problem, param):
    X, W, b, Y = problem
    grads = bp.backward(X, W, b, Y)
    fd = bp.fd_grad(param, X, W, b, Y)
    np.testing.assert_allclose(grads["d" + param], fd, atol=1e-6)


def test_shapes_are_the_bookkeeping_story(problem):
    X, W, b, Y = problem
    g = bp.backward(X, W, b, Y)
    assert g["dZ"].shape == (7, 4)          # n x k
    assert g["dX"].shape == X.shape          # (n,k)(k,d) -> n x d
    assert g["dW"].shape == W.shape          # (d,n)(n,k) -> d x k
    assert g["db"].shape == b.shape


def test_outer_products_sum_to_dW(problem):
    X, W, b, Y = problem
    g = bp.backward(X, W, b, Y)
    votes = bp.outer_product_decomp(X, g["dZ"])
    assert votes.shape == (7, 3, 4)
    np.testing.assert_allclose(votes.sum(axis=0), g["dW"], atol=1e-12)


def test_saturated_units_get_no_gradient():
    """The Hadamard gate: where |Z| is huge, sigma'(Z) ~ 0 kills the signal."""
    X = np.array([[50.0, 0.1]])
    W = np.eye(2)
    b = np.zeros(2)
    Y = np.zeros((1, 2))
    g = bp.backward(X, W, b, Y)
    assert abs(g["dZ"][0, 0]) < 1e-15      # saturated unit: dead gradient
    assert abs(g["dZ"][0, 1]) > 1e-4       # live unit: signal flows
