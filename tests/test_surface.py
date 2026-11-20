import numpy as np
import pytest

from conftest import fd_hessian, fd_richardson, jitter, random_geometries
from src.analysis import harmonic_frequencies
from src.baseline import Baseline
from src.features import PairFeatures
from src.geometry import pair_vectors
from src.krr import SymKRR
from src.surface import Surface
from src.system import System, nh3_ch
from src.tt import GaussianTT
from src.units import ANG_BOHR

SYS = nh3_ch()


def _baseline(system, X):
    base = Baseline.build(system, X, PairFeatures(system), degrees=(3, 4))
    rng = np.random.default_rng(5)
    base.coef = rng.normal(0, 1e-3, base.design(X[:1]).shape[1])
    base.e_ref = -94.9
    return base


def _active_features(system, X):
    _, r = pair_vectors(X, system)
    return PairFeatures(system)(r)[0][:, system.active_pairs]


def _inner(kind, system, X, seed=2):
    Y = _active_features(system, X)
    if kind == "krr":
        target = np.random.default_rng(seed).normal(0, 1e-2, len(Y))
        ones = np.ones(len(Y))
        return SymKRR(system.active_pair_perms, length=0.4, ridge=1e-6).fit(Y, target, ones, ones)
    return GaussianTT.initialize(Y, system.active_pair_perms, system.active_pair_type,
                                 n_gauss=3, rank=3, noise=0.3, seed=4)


@pytest.fixture(scope="module")
def train():
    return random_geometries(60, np.linspace(1.8, 9.5, 60), seed=11)


@pytest.fixture(scope="module", params=["krr", "tt"])
def surface(request, train):
    return Surface(SYS, _baseline(SYS, train), PairFeatures(SYS), _inner(request.param, SYS, train),
                   5.0 * ANG_BOHR, 7.0 * ANG_BOHR)


def _fd_relative_error(surface, X):
    """Max |analytic - Richardson FD| relative to max(1, max |F|)."""
    _, F = surface.energy_forces(X)
    return np.max(np.abs(-F - fd_richardson(surface.energy, X))) / max(1.0, np.abs(F).max())


def test_forces_match_finite_differences(surface, geoms):
    assert _fd_relative_error(surface, geoms) < 1e-7


def test_switch_region_forces(surface):
    X = random_geometries(6, np.linspace(5.1, 6.9, 6), seed=21)
    assert _fd_relative_error(surface, X) < 1e-7


def test_hessian_matches_gradient_differences(surface):
    X = random_geometries(4, np.array([2.5, 4.0, 5.5, 6.5]), seed=31)
    E, F, H = surface.energy_forces_hessian(X)
    E1, F1 = surface.energy_forces(X)
    assert np.array_equal(E, E1) and np.allclose(F, F1, rtol=0, atol=1e-13)
    H_fd = fd_hessian(lambda Z: -surface.energy_forces(Z)[1], X)
    assert np.max(np.abs(H - H_fd)) / max(1.0, np.abs(H).max()) < 1e-6
    assert np.max(np.abs(H - H.transpose(0, 2, 1))) < 1e-9
    # Translations are exact null vectors of the Hessian.
    t = np.tile(np.eye(3), (SYS.n_atoms, 1))
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
    """One fragment (NH3 alone): V = U + f_G, no switch; 6 external modes projected out."""
    nh3 = System(["N", "H", "H", "H"], bonds=[(0, 1), (0, 2), (0, 3)])
    X0 = [[0.0, 0.0, 0.0], [0.94, 0.0, -0.38], [-0.47, 0.81, -0.38], [-0.47, -0.81, -0.38]]
    X = jitter(X0, 30, sd=0.05, seed=4)
    base = Baseline.build(nh3, X, PairFeatures(nh3), degrees=(2,))
    base.coef = np.random.default_rng(1).normal(0, 1e-3, base.design(X[:1]).shape[1])
    for kind in ("krr", "tt"):
        pes = Surface(nh3, base, PairFeatures(nh3), _inner(kind, nh3, X))
        assert _fd_relative_error(pes, X[:3]) < 1e-7
        E, F, H = pes.energy_forces_hessian(X[:2])
        for p in nh3.atom_perms:
            assert np.max(np.abs(pes.energy(X[:2, p]) - E)) < 1e-11
        assert len(harmonic_frequencies(H[0], X[0], nh3.masses)) == 6


def test_tt_fit_recovers_teacher():
    """Projected two-site ALS reaches an exact projected-TT target of the same dictionary.

    The ridge selects the symmetrized representative, whose rank can be up to |G|
    times the teacher rank, so the cap must allow growth (here 16 for rank 2)."""
    X = random_geometries(240, np.linspace(1.8, 5.0, 240), seed=13)
    Y = _active_features(SYS, X)
    teacher = GaussianTT.initialize(Y, SYS.active_pair_perms, SYS.active_pair_type,
                                    n_gauss=3, rank=2, noise=0.5, seed=1)
    target = teacher.predict(Y)
    rng = np.random.default_rng(9)
    student = GaussianTT(teacher.pair_perms, teacher.pair_types, teacher.centers, teacher.widths,
                         [G + 0.05 * rng.normal(size=G.shape) for G in teacher.cores], teacher.transforms)
    ones = np.ones(len(Y))
    hist = student.fit(Y, target, ones, ones, Y, target, ones, ones, sweeps=1,
                       rank_cap=16, ridge=1e-12, prox=1e-10, log=lambda *_: None)
    assert min(h[1]["train"] for h in hist) / np.std(target) < 1e-5
