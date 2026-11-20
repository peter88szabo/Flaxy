import numpy as np

from conftest import ISOPRENE_O3, fd_hessian, fd_richardson, jitter, random_geometries
from src.baseline import Baseline
from src.features import PairFeatures
from src.system import System, nh3_ch

SYS = nh3_ch()


def _random_baseline(system, X, degrees=(3, 4), seed=1):
    base = Baseline.build(system, X, PairFeatures(system), degrees=degrees)
    rng = np.random.default_rng(seed)
    base.coef = rng.normal(0, 1e-3, base.design(X[:1]).shape[1])
    base.e_ref = -94.9
    return base


def _nh3_ch():
    return _random_baseline(SYS, random_geometries(400, np.linspace(6, 12, 400), seed=1))


def test_gradient_and_hessian_match_finite_differences(geoms):
    base = _nh3_ch()
    E, g, H = base.evaluate(geoms, order=2)
    assert np.max(np.abs(g - fd_richardson(base.energy, geoms))) < 1e-8
    assert np.max(np.abs(H - fd_hessian(lambda X: base.evaluate(X)[1], geoms))) < 1e-7
    assert np.max(np.abs(H - H.transpose(0, 2, 1))) < 1e-12


def test_permutation_invariance_and_covariance(geoms):
    base = _nh3_ch()
    E, g, H = base.evaluate(geoms, order=2)
    for p in SYS.atom_perms:
        Ep, gp, Hp = base.evaluate(geoms[:, p], order=2)
        idx = (3 * p[:, None] + np.arange(3)).ravel()
        assert np.max(np.abs(Ep - E)) < 1e-12
        assert np.max(np.abs(gp - g[:, p])) < 1e-12
        assert np.max(np.abs(Hp - H[:, idx][:, :, idx])) < 1e-10


def test_rigid_motion_and_zero_net_force(geoms):
    base = _nh3_ch()
    E, g = base.evaluate(geoms)
    q, _ = np.linalg.qr(np.random.default_rng(3).normal(size=(3, 3)))
    Er, gr = base.evaluate(geoms @ q + 1.7)
    assert np.max(np.abs(Er - E)) < 1e-12
    assert np.max(np.abs(gr - g @ q)) < 1e-12
    assert np.max(np.abs(g.sum(1))) < 1e-12
    assert np.max(np.abs(np.cross(geoms, g).sum(1))) < 1e-12


def test_fit_recovers_synthetic_tail():
    base = _nh3_ch()
    X = random_geometries(600, np.linspace(8, 12, 600), seed=7)
    tb = base.blocks["tail"]
    col = np.sqrt(np.mean(base.tail.design(X) ** 2, axis=0))
    base.coef[tb] = 1e-5 / col * np.sign(base.coef[tb])  # each column ~1e-5 Hartree
    truth = base.coef.copy()
    info = base.fit(X, base.energy(X), ridge=0.0)
    assert np.allclose(base.coef[tb], truth[tb], rtol=1e-6, atol=0)
    assert info["rmse"] < 1e-10


def test_isoprene_ozone_tail_invariant_under_48_permutations():
    system = System(ISOPRENE_O3["symbols"], (range(13), range(13, 16)), bonds=ISOPRENE_O3["bonds"])
    X = jitter(ISOPRENE_O3["X"], 3, sd=0.04, seed=2)
    base = _random_baseline(system, X, degrees=(1, 2))
    E, g = base.evaluate(X)
    assert np.max(np.abs(g - fd_richardson(base.energy, X))) < 1e-8
    for p in system.atom_perms:
        Ep, gp = base.evaluate(X[:, p])
        assert np.max(np.abs(Ep - E)) < 1e-12 and np.max(np.abs(gp - g[:, p])) < 1e-12
    _, _, H = base.evaluate(X[:1], order=2)
    assert np.max(np.abs(H - fd_hessian(lambda Z: base.evaluate(Z)[1], X[:1]))) < 1e-7
