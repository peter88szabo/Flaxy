import numpy as np
import pytest

from src.units import ANG_BOHR


def random_geometries(n, R_ang, seed=0):
    """NH3 + CH geometries near equilibrium, random orientations, COM distance R (bohr out)."""
    rng = np.random.default_rng(seed)
    nh3 = np.array([[0.0, 0.0, 0.0], [0.94, 0.0, -0.38], [-0.47, 0.81, -0.38],
                    [-0.47, -0.81, -0.38]])
    ch = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.12]])
    out = []
    for k in range(n):
        a = (nh3 + rng.normal(0, 0.04, nh3.shape)) @ _rotation(rng)
        b = (ch + rng.normal(0, 0.03, ch.shape)) @ _rotation(rng)
        R = R_ang if np.isscalar(R_ang) else R_ang[k]
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        out.append(np.vstack([a - a.mean(0), b - b.mean(0) + R * direction]))
    return np.array(out) * ANG_BOHR


def _rotation(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q


def jitter(X0_ang, n, sd=0.05, seed=0):
    """Random rigid rotations of a reference geometry plus Cartesian noise (bohr out)."""
    rng = np.random.default_rng(seed)
    X0 = np.asarray(X0_ang) - np.mean(X0_ang, axis=0)
    return np.array([(X0 + rng.normal(0, sd, X0.shape)) @ _rotation(rng) for _ in range(n)]) * ANG_BOHR


@pytest.fixture
def geoms():
    return random_geometries(5, np.array([2.0, 4.0, 6.5, 8.0, 11.0]))


def fd_gradient(f, X, h=1e-5):
    """Central differences of a batched scalar function f(X)->(B,)."""
    g = np.zeros_like(X)
    for i in range(X.shape[1]):
        for k in range(3):
            Xp, Xm = X.copy(), X.copy()
            Xp[:, i, k] += h
            Xm[:, i, k] -= h
            g[:, i, k] = (f(Xp) - f(Xm)) / (2 * h)
    return g


def fd_richardson(f, X, h=2e-4):
    """Richardson-extrapolated central differences: O(h^4) truncation, less roundoff."""
    return (4.0 * fd_gradient(f, X, h / 2) - fd_gradient(f, X, h)) / 3.0


def fd_hessian(grad, X, h=2e-4):
    """Richardson-extrapolated Hessian [b, 3N, 3N] from an analytic gradient grad(X)->[b, N, 3]."""
    def one(step):
        n3 = X.shape[1] * 3
        H = np.zeros((len(X), n3, n3))
        for i in range(X.shape[1]):
            for k in range(3):
                Xp, Xm = X.copy(), X.copy()
                Xp[:, i, k] += step
                Xm[:, i, k] -= step
                H[:, 3 * i + k] = ((grad(Xp) - grad(Xm)) / (2 * step)).reshape(len(X), -1)
        return H
    return (4.0 * one(h / 2) - one(h)) / 3.0


ISOPRENE_O3 = {
    "symbols": ["C"] * 5 + ["H"] * 8 + ["O"] * 3,
    "bonds": [(0, 1), (1, 2), (2, 3), (1, 4), (0, 5), (0, 6), (2, 7), (3, 8), (3, 9), (4, 10),
              (4, 11), (4, 12), (13, 14), (14, 15)],
    # Approximate isoprene (C1=C2(C5)-C3=C4) and ozone about 4 Angstrom apart (Angstrom).
    "X": [[0.00, 0.00, 0.00], [1.34, 0.00, 0.00], [2.10, 1.25, 0.00], [3.44, 1.25, 0.00],
          [2.08, -1.30, 0.00], [-0.55, 0.93, 0.00], [-0.55, -0.93, 0.00], [1.55, 2.19, 0.00],
          [3.99, 2.18, 0.00], [3.99, 0.32, 0.00], [1.45, -2.19, 0.00], [2.72, -1.35, 0.89],
          [2.72, -1.35, -0.89], [1.20, 0.30, 4.00], [2.27, 0.30, 4.65], [3.34, 0.30, 4.00]],
}
