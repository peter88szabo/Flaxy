"""Pair distances, their Cartesian chain rule, and fragment origins (bohr)."""
import numpy as np

from .system import FRAG_A, FRAG_B, MASSES, N_ATOMS, PAIR_I, PAIR_J


def pair_vectors(X):
    """Return d[b,e] = X_i - X_j and r[b,e] for all pairs in site order."""
    d = X[:, PAIR_I] - X[:, PAIR_J]
    return d, np.linalg.norm(d, axis=2)


def pair_grad_to_cart(dEdr, d, r):
    """Chain dE/dr[b,e] to dE/dX[b,i,k] using dr/dX_i = d/r = -dr/dX_j."""
    g = dEdr[..., None] * d / r[..., None]
    out = np.zeros((len(dEdr), N_ATOMS, 3))
    for e, (i, j) in enumerate(zip(PAIR_I, PAIR_J)):
        out[:, i] += g[:, e]
        out[:, j] -= g[:, e]
    return out


def _weights(idx):
    m = MASSES[list(idx)]
    return m / m.sum()


W_A, W_B = _weights(FRAG_A), _weights(FRAG_B)


def com_vector(X):
    """R = o_B - o_A between fragment centres of mass."""
    return (np.einsum("i,bik->bk", W_B, X[:, list(FRAG_B)])
            - np.einsum("i,bik->bk", W_A, X[:, list(FRAG_A)]))


def com_distance_grad(X):
    """Return R and dR/dX[b,i,k]."""
    Rv = com_vector(X)
    R = np.linalg.norm(Rv, axis=1)
    n = Rv / R[:, None]
    g = np.zeros((len(X), N_ATOMS, 3))
    g[:, list(FRAG_A)] = -W_A[None, :, None] * n[:, None]
    g[:, list(FRAG_B)] = W_B[None, :, None] * n[:, None]
    return R, g
