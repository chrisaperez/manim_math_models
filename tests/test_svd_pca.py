"""Demo 3's math: Pythagoras per point, variance-max = error-min, SVD = PCA."""

import numpy as np
import pytest

from core.sims import svd_pca as sp

CLOUD = sp.make_cloud(n=80, angle=0.6, seed=11)
M = sp.rotation(0.9) @ np.diag([2.2, 0.8]) @ sp.rotation(-0.35)


def test_pythagoras_per_point():
    u = sp.unit(1.1)
    p = sp.project(CLOUD, u)
    lhs = np.sum(CLOUD**2, axis=1)
    rhs = np.sum(p["proj"]**2, axis=1) + np.sum(p["resid"]**2, axis=1)
    np.testing.assert_allclose(lhs, rhs, atol=1e-12)


def test_keep_plus_lose_is_constant_over_the_sweep():
    sw = sp.sweep_variance(CLOUD)
    np.testing.assert_allclose(sw["keep"] + sw["lose"], sw["total"], atol=1e-9)


def test_variance_max_is_error_min_is_principal_dir():
    sw = sp.sweep_variance(CLOUD, n_angles=3601)
    v = sp.principal_dir(CLOUD)
    theta_star = sw["thetas"][np.argmax(sw["keep"])]
    # same angle minimizes the loss (they are mirror objectives)
    assert np.argmax(sw["keep"]) == np.argmin(sw["lose"])
    dot = abs(sp.unit(theta_star) @ v)
    assert dot > 0.9999


def test_svd_reconstructs_and_factors_are_rotations():
    st = sp.svd_stages(M)
    recon = st["U"] @ np.diag(st["S"]) @ st["Vt"]
    np.testing.assert_allclose(recon, M, atol=1e-12)
    for R in (st["U"], st["Vt"]):
        np.testing.assert_allclose(R @ R.T, np.eye(2), atol=1e-12)
        assert np.linalg.det(R) > 0  # pure rotation after sign-folding


def test_mtm_eigenvectors_are_right_singular_vectors():
    st = sp.svd_stages(M)
    eig = sp.mtm_eigen(M)
    np.testing.assert_allclose(eig["eigvals"], np.sort(st["S"]**2)[::-1], atol=1e-10)
    for i in range(2):
        v_svd = st["Vt"][i]
        v_eig = eig["eigvecs"][:, i]
        assert abs(abs(v_svd @ v_eig) - 1) < 1e-10  # equal up to sign


def test_principal_dir_of_cloud_matches_mtm_eigenvector():
    v = sp.principal_dir(CLOUD)
    top = sp.mtm_eigen(CLOUD)["eigvecs"][:, 0]
    assert abs(abs(v @ top) - 1) < 1e-10
