import numpy as np
import pytest

from conftest import ISOPRENE_O3, fd_hessian, fd_richardson, jitter, random_geometries
from src.analysis import harmonic_frequencies
from src.baseline import Baseline
from src.features import PairFeatures
from src.geometry import pair_vectors
from src.invariants import InvariantFeatures
from src.krr import KRR
from src.surface import Surface
from src.system import System, nh3_ch
from src.tt import GaussianTT
from src.units import ANG_BOHR

SYS = nh3_ch()


def _baseline(system, X, degrees=(3, 4)):
    base = Baseline.build(system, X, PairFeatures(system), degrees=degrees)
    rng = np.random.default_rng(5)
    base.coef = rng.normal(0, 1e-3, base.design(X[:1]).shape[1])
    base.e_ref = -94.9
    return base


def _pips(system, X, degree=2):
    _, r = pair_vectors(X, system)
    return InvariantFeatures.build(system, PairFeatures(system), degree).standardize(r)


def _inner(kind, pips, system, X, seed=2):
    Z = pips(pair_vectors(X, system)[1])
    if kind == "krr":
        target = np.random.default_rng(seed).normal(0, 1e-2, len(Z))
        ones = np.ones(len(Z))
        return KRR(length=3.0, ridge=1e-6).fit(Z, target, ones, ones)
    # Near-constant random cores: a product over ~45 PIP sites must stay O(1).
    return GaussianTT.initialize(Z, n_gauss=3, rank=3, noise=0.03, seed=4)


def _surface(kind, system, X, r_in=None, r_out=None, degrees=(3, 4)):
    pips = _pips(system, X)
    return Surface(system, _baseline(system, X, degrees), PairFeatures(system),
                   _inner(kind, pips, system, X), r_in, r_out, pips)


@pytest.fixture(scope="module")
def train():
    return random_geometries(60, np.linspace(1.8, 9.5, 60), seed=11)


@pytest.fixture(scope="module", params=["krr", "tt"])
def surface(request, train):
    return _surface(request.param, SYS, train, 5.0 * ANG_BOHR, 7.0 * ANG_BOHR)


def _fd_relative_error(surface, X):
    """Max |analytic - Richardson FD| relative to max(1, max |F|)."""
    _, F = surface.energy_forces(X)
    return np.max(np.abs(-F - fd_richardson(surface.energy, X))) / max(1.0, np.abs(F).max())


def test_pips_are_exactly_invariant_with_exact_derivatives(train):
    pips = _pips(SYS, train, degree=3)
    _, r = pair_vectors(train[:4], SYS)
    Z, dZ, d2Z = pips(r, order=2)
    for P in SYS.pair_perms:
        assert np.max(np.abs(pips(r[:, P]) - Z)) < 1e-12
    ap, h = SYS.active_pairs, 1e-5
    for e in range(len(ap)):
        rp, rm = r.copy(), r.copy()
        rp[:, ap[e]] += h
        rm[:, ap[e]] -= h
        Zp, dZp = pips(rp, 1)
        Zm, dZm = pips(rm, 1)
        assert np.max(np.abs((Zp - Zm) / (2 * h) - dZ[:, :, e])) < 1e-6 * max(1, np.abs(dZ).max())
        assert np.max(np.abs((dZp - dZm) / (2 * h) - d2Z[:, :, :, e])) < 1e-6 * max(1, np.abs(d2Z).max())


def test_forces_match_finite_differences(surface, geoms):
    assert _fd_relative_error(surface, geoms) < 1e-7


def test_switch_region_forces(surface):
    X = random_geometries(6, np.linspace(5.1, 6.9, 6), seed=21)
    assert _fd_relative_error(surface, X) < 1e-7


def test_hessian_matches_gradient_differences(surface):
    X = random_geometries(4, np.array([2.5, 4.0, 5.5, 6.5]), seed=31)
    E, F, H = surface.energy_forces_hessian(X)
    E1, F1 = surface.energy_forces(X)
    assert np.allclose(E, E1, rtol=0, atol=1e-13) and np.allclose(F, F1, rtol=0, atol=1e-12)
    H_fd = fd_hessian(lambda Z: -surface.energy_forces(Z)[1], X)
    assert np.max(np.abs(H - H_fd)) / max(1.0, np.abs(H).max()) < 1e-6
    assert np.max(np.abs(H - H.transpose(0, 2, 1))) < 1e-9
    t = np.tile(np.eye(3), (SYS.n_atoms, 1))   # translations are exact null vectors
    assert np.max(np.abs(H @ t)) / max(1.0, np.abs(H).max()) < 1e-10


def test_permutation_and_rigid_invariance(surface, geoms):
    E, F, H = surface.energy_forces_hessian(geoms)
    for p in SYS.atom_perms:
        Ep, Fp, Hp = surface.energy_forces_hessian(geoms[:, p])
        idx = (3 * p[:, None] + np.arange(3)).ravel()
        assert np.max(np.abs(Ep - E)) < 1e-11
        assert np.max(np.abs(Fp - F[:, p])) < 1e-10
        assert np.max(np.abs(Hp - H[:, idx][:, :, idx])) < 1e-8
    q, _ = np.linalg.qr(np.random.default_rng(8).normal(size=(3, 3)))
    Er, Fr = surface.energy_forces(geoms @ q - 0.4)
    assert np.max(np.abs(Er - E)) < 1e-11
    assert np.max(np.abs(Fr - F @ q)) < 1e-10
    assert np.max(np.abs(F.sum(1))) < 1e-11


def test_outside_r_out_is_baseline(surface):
    X = random_geometries(4, np.array([7.2, 10.0, 15.0, 40.0]), seed=3)
    E, F, H = surface.energy_forces_hessian(X)
    Eb, gb, Hb = surface.baseline.evaluate(X, order=2)
    assert np.array_equal(E, Eb) and np.array_equal(F, -gb) and np.array_equal(H, Hb)


def test_save_load_roundtrip(surface, geoms, tmp_path):
    path = tmp_path / "model.npz"
    surface.save(path, meta={"test": True})
    again = Surface.load(path)
    E, F = surface.energy_forces(geoms)
    E2, F2 = again.energy_forces(geoms)
    assert np.array_equal(E, E2) and np.array_equal(F, F2)


def test_unimolecular_surface_and_frequencies():
    """One fragment (NH3 alone): V = U + f(Z), no switch; 6 external modes projected out."""
    nh3 = System(["N", "H", "H", "H"], bonds=[(0, 1), (0, 2), (0, 3)])
    X0 = [[0.0, 0.0, 0.0], [0.94, 0.0, -0.38], [-0.47, 0.81, -0.38], [-0.47, -0.81, -0.38]]
    X = jitter(X0, 30, sd=0.05, seed=4)
    for kind in ("krr", "tt"):
        pes = _surface(kind, nh3, X, degrees=(2,))
        assert _fd_relative_error(pes, X[:3]) < 1e-7
        E, F, H = pes.energy_forces_hessian(X[:2])
        for p in nh3.atom_perms:
            assert np.max(np.abs(pes.energy(X[:2, p]) - E)) < 1e-11
        assert len(harmonic_frequencies(H[0], X[0], nh3.masses)) == 6


def test_isoprene_ozone_surface_invariant_under_48_permutations():
    system = System(ISOPRENE_O3["symbols"], (range(13), range(13, 16)), bonds=ISOPRENE_O3["bonds"],
                    active=(0, 1, 2, 3, 13, 14, 15))
    X = jitter(ISOPRENE_O3["X"], 6, sd=0.04, seed=2)
    pes = _surface("krr", system, X, 5.0 * ANG_BOHR, 7.0 * ANG_BOHR, degrees=(1, 2))
    E, F = pes.energy_forces(X[:2])
    for p in system.atom_perms:
        Ep, Fp = pes.energy_forces(X[:2, p])
        assert np.max(np.abs(Ep - E)) < 1e-11 and np.max(np.abs(Fp - F[:, p])) < 1e-10
    assert _fd_relative_error(pes, X[:1]) < 1e-7


def _teacher_problem(n=240):
    X = random_geometries(n, np.linspace(1.8, 5.0, n), seed=13)
    Z = _pips(SYS, X)(pair_vectors(X, SYS)[1])
    teacher = GaussianTT.initialize(Z, n_gauss=3, rank=2, noise=0.05, seed=1)
    rng = np.random.default_rng(9)
    student = GaussianTT(None, None, teacher.centers, teacher.widths,
                         [G + 0.01 * rng.normal(size=G.shape) for G in teacher.cores], teacher.transforms)
    return Z, teacher.predict(Z), student


def test_tt_two_site_fit_reaches_exact_target():
    """Two-site ALS on PIP inputs fits an exact TT target when the rank may grow."""
    Z, target, student = _teacher_problem()
    ones = np.ones(len(Z))
    hist = student.fit(Z, target, ones, ones, Z, target, ones, ones, sweeps=1, rank_cap=40,
                       svd_tol=0.0, ridge=1e-12, prox=1e-10, log=lambda *_: None)
    assert min(h[1]["train"] for h in hist) / np.std(target) < 1e-4


def test_tt_one_site_sweeps_are_monotone():
    Z, target, student = _teacher_problem()
    ones = np.ones(len(Z))
    hist = student.fit(Z, target, ones, ones, sweeps=3, rank_cap=3, two_site_sweeps=0,
                       ridge=1e-12, prox=1e-10, log=lambda *_: None)
    train = [h[1]["train"] for h in hist]
    assert all(b <= a * (1 + 1e-12) for a, b in zip(train, train[1:]))
    assert train[-1] < 0.6 * train[0]
