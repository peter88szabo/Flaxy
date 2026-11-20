import numpy as np

from conftest import fd_gradient, random_geometries
from flaxy.baseline import Baseline
from flaxy.system import ATOM_PERMS


def _fitted_baseline(seed=1):
    X = random_geometries(400, np.linspace(6, 12, 400), seed=seed)
    base = Baseline.build(X, nh3_degree=3, ch_degree=4)
    rng = np.random.default_rng(seed)
    # Random but fixed coefficients exercise every column.
    base.coef = rng.normal(0, 1e-3, base.design(X[:1]).shape[1])
    base.e_ref = -94.9
    return base


def test_gradient_matches_finite_differences(geoms):
    base = _fitted_baseline()
    _, g = base.energy_grad(geoms)
    g_fd = fd_gradient(base.energy, geoms)
    assert np.max(np.abs(g - g_fd)) < 1e-8


def test_permutation_invariance_and_force_covariance(geoms):
    base = _fitted_baseline()
    E, g = base.energy_grad(geoms)
    for p in ATOM_PERMS:
        Ep, gp = base.energy_grad(geoms[:, p])
        assert np.max(np.abs(Ep - E)) < 1e-12
        assert np.max(np.abs(gp - g[:, p])) < 1e-12


def test_rigid_motion_and_zero_net_force(geoms):
    base = _fitted_baseline()
    E, g = base.energy_grad(geoms)
    q, _ = np.linalg.qr(np.random.default_rng(3).normal(size=(3, 3)))
    Er, gr = base.energy_grad(geoms @ q + 1.7)
    assert np.max(np.abs(Er - E)) < 1e-12
    assert np.max(np.abs(gr - g @ q)) < 1e-12
    assert np.max(np.abs(g.sum(1))) < 1e-12
    torque = np.cross(geoms, g).sum(1)
    assert np.max(np.abs(torque)) < 1e-12


def test_fit_recovers_synthetic_coefficients():
    base = _fitted_baseline()
    X = random_geometries(600, np.linspace(8, 12, 600), seed=7)
    tb = base.blocks["tail"]
    # Physically sized tail: every column contributes ~1e-5 Hartree on this set.
    col = np.sqrt(np.mean(base.tail.design(X) ** 2, axis=0))
    base.coef[tb] = 1e-5 / col * np.sign(base.coef[tb])
    truth = base.coef.copy()
    E = base.energy(X)
    info = base.fit(X, E, ridge=0.0)
    assert np.allclose(base.coef[tb], truth[tb], rtol=1e-6, atol=0)
    assert info["rmse"] < 1e-10
