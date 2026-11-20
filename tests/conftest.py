import numpy as np
import pytest

from flaxy.units import ANG_BOHR


def random_geometries(n, R_ang, seed=0):
    """NH3 + CH geometries near equilibrium, random orientations, COM distance R."""
    rng = np.random.default_rng(seed)
    nh3 = np.array([[0.0, 0.0, 0.0], [0.94, 0.0, -0.38], [-0.47, 0.81, -0.38],
                    [-0.47, -0.81, -0.38]])
    ch = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.12]])
    out = []
    for k in range(n):
        def rot():
            q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            return q
        a = (nh3 + rng.normal(0, 0.04, nh3.shape)) @ rot()
        b = (ch + rng.normal(0, 0.03, ch.shape)) @ rot()
        R = R_ang if np.isscalar(R_ang) else R_ang[k]
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        out.append(np.vstack([a - a.mean(0), b - b.mean(0) + R * direction]))
    return np.array(out) * ANG_BOHR


@pytest.fixture
def geoms():
    return random_geometries(5, np.array([2.0, 4.0, 6.5, 8.0, 11.0]))


def fd_richardson(f, X, h=2e-4):
    """Richardson-extrapolated central differences: O(h^4) truncation, less roundoff."""
    return (4.0 * fd_gradient(f, X, h / 2) - fd_gradient(f, X, h)) / 3.0


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
