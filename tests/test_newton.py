"""Demo 8's math: one-step Newton, kappa-scaling GD, honest BFGS."""

import numpy as np
import pytest

from core.sims import newton as nt
from core.sims import gradient_descent as gd

RNG = np.random.default_rng(205)


def test_newton_solves_any_quadratic_in_one_step():
    for _ in range(5):
        A = gd.make_quadratic(RNG.uniform(0.5, 20, size=2), RNG.uniform(0, np.pi))
        _, grad, hess = nt.quadratic(A)
        x0 = RNG.normal(size=2) * 3
        path = nt.newton_run(grad, hess, x0)
        assert len(path) == 2                       # exactly one step taken
        assert np.linalg.norm(path[-1]) < 1e-10     # and it lands on the min


def test_gd_iteration_count_scales_with_kappa():
    x0 = np.array([1.0, 1.0])
    n10 = nt.gd_steps_to_tol(np.diag([1.0, 10.0]), x0)
    n100 = nt.gd_steps_to_tol(np.diag([1.0, 100.0]), x0)
    ratio = n100 / n10
    assert 5 < ratio < 20                           # ~linear in kappa


def test_whitening_makes_kappa_one():
    A = gd.make_quadratic([0.5, 12.0], 0.7)
    W = nt.whitening_map(A)
    Aw = W @ A @ W
    lam = np.linalg.eigvalsh(Aw)
    assert abs(lam[-1] / lam[0] - 1.0) < 1e-10


def test_bfgs_secant_condition_every_update():
    f, grad, _ = nt.rosenbrock()
    _, residuals = nt.bfgs_run(f, grad, np.array([-1.2, 1.0]))
    assert len(residuals) > 5
    assert max(residuals) < 1e-8


def test_bfgs_beats_gd_on_rosenbrock():
    f, grad, _ = nt.rosenbrock()
    x0 = np.array([-1.2, 1.0])
    path_bfgs, _ = nt.bfgs_run(f, grad, x0, steps=400, tol=1e-6)
    assert np.linalg.norm(path_bfgs[-1] - np.array([1.0, 1.0])) < 1e-4
    path_gd, hit = nt.gd_run(grad, x0, alpha=1e-3, steps=len(path_bfgs) * 50,
                             tol=1e-6)
    # GD with a stable step is nowhere near done in 50x the budget
    assert hit is None
    assert np.linalg.norm(path_gd[-1] - np.array([1.0, 1.0])) > 1e-2


def test_damped_newton_converges_on_rosenbrock():
    f, grad, hess = nt.rosenbrock()
    path = nt.newton_run(grad, hess, np.array([-1.2, 1.0]), steps=100,
                         damped=True, f=f)
    assert np.linalg.norm(path[-1] - np.array([1.0, 1.0])) < 1e-8
