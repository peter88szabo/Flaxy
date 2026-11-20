import numpy as np
import pytest

from conftest import fd_richardson, random_geometries
from flaxy.baseline import Baseline
from flaxy.features import PairFeatures
from flaxy.geometry import pair_vectors
from flaxy.krr import SymKRR
from flaxy.surface import Surface
from flaxy.system import ATOM_PERMS
from flaxy.tt import GaussianTT
from flaxy.units import ANG_BOHR


def _baseline(X):
    base = Baseline.build(X, nh3_degree=3, ch_degree=4)
    rng = np.random.default_rng(5)
    base.coef = rng.normal(0, 1e-3, base.design(X[:1]).shape[1])
    base.e_ref = -94.9
    return base


def _features(X):
    _, r = pair_vectors(X)
    return PairFeatures()(r)[0]


@pytest.fixture(scope="module")
def train():
    return random_geometries(60, np.linspace(1.8, 9.5, 60), seed=11)


@pytest.fixture(scope="module", params=["krr", "tt"])
def surface(request, train):
    feats = PairFeatures()
    Y = _features(train)
    rng = np.random.default_rng(2)
    target = rng.normal(0, 1e-2, len(Y))
    scale = np.ones(len(Y))
    if request.param == "krr":
        inner = SymKRR(length=0.4, ridge=1e-6).fit(Y, target, np.ones(len(Y)), scale)
    else:
        inner = GaussianTT.initialize(Y, n_gauss=3, rank=3, noise=0.3, seed=4)
    return Surface(_baseline(train), feats, inner, 6.0 * ANG_BOHR, 9.0 * ANG_BOHR)


def _fd_relative_error(surface, X):
    """Max |analytic - Richardson FD| relative to max(1, max |F|)."""
    _, F = surface.energy_forces(X)
    return np.max(np.abs(-F - fd_richardson(surface.energy, X))) / max(1.0, np.abs(F).max())


def test_forces_match_finite_differences(surface, geoms):
    assert _fd_relative_error(surface, geoms) < 1e-7


def test_switch_region_forces(surface):
    X = random_geometries(6, np.linspace(6.1, 8.9, 6), seed=21)
    assert _fd_relative_error(surface, X) < 1e-7


def test_permutation_and_rigid_invariance(surface, geoms):
    E, F = surface.energy_forces(geoms)
    for p in ATOM_PERMS:
        Ep, Fp = surface.energy_forces(geoms[:, p])
        assert np.max(np.abs(Ep - E)) < 1e-11
        assert np.max(np.abs(Fp - F[:, p])) < 1e-10
    q, _ = np.linalg.qr(np.random.default_rng(8).normal(size=(3, 3)))
    Er, Fr = surface.energy_forces(geoms @ q - 0.4)
    assert np.max(np.abs(Er - E)) < 1e-11
    assert np.max(np.abs(Fr - F @ q)) < 1e-10
    assert np.max(np.abs(F.sum(1))) < 1e-11


def test_outside_r_out_is_baseline(surface):
    X = random_geometries(4, np.array([9.2, 10.0, 15.0, 40.0]), seed=3)
    E, F = surface.energy_forces(X)
    Eb, gb = surface.baseline.energy_grad(X)
    assert np.array_equal(E, Eb) and np.array_equal(F, -gb)


def test_save_load_roundtrip(surface, geoms, tmp_path):
    path = tmp_path / "model.npz"
    surface.save(path, meta={"test": True})
    again = Surface.load(path)
    E, F = surface.energy_forces(geoms)
    E2, F2 = again.energy_forces(geoms)
    assert np.array_equal(E, E2) and np.array_equal(F, F2)


def test_tt_fit_recovers_teacher(train):
    """Projected two-site ALS reaches an exact projected-TT target of the same dictionary.

    The ridge selects the symmetrized representative, whose rank can be up to |G|
    times the teacher rank, so the cap must allow growth (here 16 for rank 2)."""
    X = random_geometries(240, np.linspace(1.8, 5.0, 240), seed=13)
    Y = _features(X)
    teacher = GaussianTT.initialize(Y, n_gauss=3, rank=2, noise=0.5, seed=1)
    target = teacher.predict(Y)
    student = GaussianTT(teacher.centers, teacher.widths,
                         [G + 0.05 * np.random.default_rng(9).normal(size=G.shape)
                          for G in teacher.cores], teacher.transforms)
    ones = np.ones(len(Y))
    hist = student.fit(Y, target, ones, ones, Y, target, ones, ones, sweeps=1,
                       rank_cap=16, ridge=1e-12, prox=1e-10, log=lambda *_: None)
    rel = min(h[1]["train"] for h in hist) / np.std(target)
    assert rel < 1e-5
