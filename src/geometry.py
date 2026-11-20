"""Pair distances and their Cartesian chain rule to second order; fragment origins (bohr)."""
import numpy as np


def pair_vectors(X, system):
    """Return d[b,e] = X_i - X_j and r[b,e] for all pairs in the system's pair order."""
    d = X[:, system.pair_i] - X[:, system.pair_j]
    return d, np.linalg.norm(d, axis=2)


def pair_grad_to_cart(dEdr, d, r, system):
    """Chain dE/dr[b,e] to dE/dX[b,i,k] using dr/dX_i = d/r = -dr/dX_j."""
    g = dEdr[..., None] * d / r[..., None]
    out = np.zeros((len(dEdr), system.n_atoms, 3))
    np.add.at(out, (slice(None), system.pair_i), g)
    np.add.at(out, (slice(None), system.pair_j), -g)
    return out


def pair_jacobian(d, r, system):
    """J[b, e, 3N] = dr_e/dx."""
    u = d / r[..., None]
    J = np.zeros((len(d), system.n_pairs, system.n_atoms, 3))
    e = np.arange(system.n_pairs)
    J[:, e, system.pair_i] = u
    J[:, e, system.pair_j] = -u
    return J.reshape(len(d), system.n_pairs, -1)


def pair_hess_to_cart(dEdr, d2Edr2, d, r, system):
    """Cartesian Hessian H[b, 3N, 3N] = J^T (d2E/dr2) J + sum_e (dE/dr_e) d2r_e/dx2."""
    J = pair_jacobian(d, r, system)
    H = np.einsum("bpi,bpq,bqj->bij", J, d2Edr2, J)
    u = d / r[..., None]
    T = (np.eye(3)[None, None] - u[..., :, None] * u[..., None, :]) / r[..., None, None]
    T *= dEdr[..., None, None]
    H4 = H.reshape(len(d), system.n_atoms, 3, system.n_atoms, 3)
    for e, (i, j) in enumerate(zip(system.pair_i, system.pair_j)):
        H4[:, i, :, i, :] += T[:, e]
        H4[:, j, :, j, :] += T[:, e]
        H4[:, i, :, j, :] -= T[:, e]
        H4[:, j, :, i, :] -= T[:, e]
    return H


def origin_weights(system, which):
    """Atom indices of fragment `which` and their centre-of-mass weights."""
    idx = list(system.fragment(which))
    m = system.masses[idx]
    return idx, m / m.sum()


def _origin_coefficients(system):
    """c[i] with R_vec = o_B - o_A = sum_i c_i X_i (two-fragment systems only)."""
    if system.n_fragments != 2:
        raise ValueError("centre-of-mass separation needs a two-fragment system")
    c = np.zeros(system.n_atoms)
    for which, sign in ((0, -1.0), (1, 1.0)):
        idx = list(system.fragment(which))
        m = system.masses[idx]
        c[idx] = sign * m / m.sum()
    return c


def com_vector(X, system):
    """R = o_B - o_A between fragment centres of mass."""
    return np.einsum("i,bik->bk", _origin_coefficients(system), X)


def com_distance_grad(X, system, order=1):
    """R, dR/dX[b,i,k] and, for order 2, d2R/dX2[b,3N,3N]."""
    c = _origin_coefficients(system)
    Rv = np.einsum("i,bik->bk", c, X)
    R = np.linalg.norm(Rv, axis=1)
    n = Rv / R[:, None]
    g = c[None, :, None] * n[:, None, :]
    if order < 2:
        return R, g
    P = (np.eye(3)[None] - n[:, :, None] * n[:, None, :]) / R[:, None, None]
    H = np.einsum("i,j,bkl->bikjl", c, c, P).reshape(len(X), 3 * system.n_atoms, -1)
    return R, g, H
