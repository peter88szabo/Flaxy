"""Harmonic analysis from the analytic Hessian."""
import numpy as np

from .units import AMU_ME, HARTREE_CM


def external_modes(X, masses):
    """Orthonormal mass-weighted translations and rotations (columns) at X (bohr)."""
    m = np.asarray(masses, dtype=float) * AMU_ME
    sq = np.sqrt(m)
    Xc = X - (m[:, None] * X).sum(0) / m.sum()
    vecs = []
    for k in range(3):
        t = np.zeros_like(X)
        t[:, k] = 1.0
        vecs.append((t * sq[:, None]).ravel())
        axis = np.eye(3)[k]
        vecs.append((np.cross(axis, Xc) * sq[:, None]).ravel())
    U, S, _ = np.linalg.svd(np.array(vecs).T, full_matrices=False)
    return U[:, S > 1e-6 * S.max()]


def harmonic_frequencies(H, X, masses):
    """Signed harmonic wavenumbers (cm^-1, negative = imaginary) of the internal modes.

    H: Cartesian Hessian [3N, 3N] in Hartree/bohr^2 at X [N, 3] (bohr); masses in Da.
    Translations and rotations (5 for linear, 6 otherwise) are projected out.
    """
    m = np.repeat(np.asarray(masses, dtype=float) * AMU_ME, 3)
    Hm = H / np.sqrt(m)[:, None] / np.sqrt(m)[None, :]
    E = external_modes(X, masses)
    P = np.eye(len(m)) - E @ E.T
    lam, vec = np.linalg.eigh(P @ Hm @ P)
    # Drop the eigenvectors that live in the external space.
    internal = np.linalg.norm(E.T @ vec, axis=0) < 0.5
    lam = lam[internal]
    return np.sign(lam) * np.sqrt(np.abs(lam)) * HARTREE_CM
