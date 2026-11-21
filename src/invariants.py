"""Permutation-invariant polynomial (PIP) input features for the short-range model.

Z_k(x) = sum_{m in orbit k} prod_e y_e^{m_e},   y_e = exp(-r_e / a_e),

orbit sums of Morse monomials of the active pairs up to a total degree, generated
from the system's permutation group (same construction as the fragment polynomials
in monomer.py). Every Z_k is invariant by construction, so any model of Z is
invariant and is evaluated once per geometry. Features are standardized with the
training mean and standard deviation.

A monomial is exp(-sum_e c_e r_e) with c_e = m_e / a_e, so derivatives in the
active pair distances are exact and cheap:
  dZ_k/dr_e = -sum_{m in k} c_me M_m,   d2Z_k/dr_e dr_f = sum_{m in k} c_me c_mf M_m.
"""
import numpy as np

from .monomer import orbit_basis


class InvariantFeatures:
    def __init__(self, pairs, a, exponents, orbit, mean=None, std=None):
        self.pairs = np.asarray(pairs, dtype=int)       # global indices of the active pairs
        self.a = np.asarray(a, dtype=float)
        self.exponents = np.asarray(exponents)
        self.orbit = np.asarray(orbit, dtype=int)
        self.n = int(self.orbit.max()) + 1
        self._rates = self.exponents / self.a[None, :]
        self._S = np.zeros((len(self.orbit), self.n))   # monomial -> invariant
        self._S[np.arange(len(self.orbit)), self.orbit] = 1.0
        self.mean = np.zeros(self.n) if mean is None else np.asarray(mean, dtype=float)
        self.std = np.ones(self.n) if std is None else np.asarray(std, dtype=float)

    @classmethod
    def build(cls, system, features, degree=3, max_monomials=200000):
        """PIPs of system.active_pairs, with Morse ranges from `features`."""
        pairs = system.active_pairs
        exps, orbit = orbit_basis(len(pairs), degree, system.active_pair_perms, max_monomials)
        return cls(pairs, features.a[pairs], exps, orbit)

    def standardize(self, r):
        """Set mean and std from training pair distances r[b, n_pairs]; returns self."""
        Z = self._monomials(r) @ self._S
        self.mean, self.std = Z.mean(0), Z.std(0)
        self.std[self.std == 0] = 1.0
        return self

    def _monomials(self, r):
        return np.exp(-r[:, self.pairs] @ self._rates.T)

    def __call__(self, r, order=0):
        """Z[b, n]; order 1 adds dZ/dr[b, n, n_active]; order 2 adds d2Z/dr2[b, n, n_active, n_active].

        Derivatives are with respect to the active pair distances (system.active_pairs order).
        """
        M = self._monomials(r)
        Z = (M @ self._S - self.mean) / self.std
        if order == 0:
            return Z
        St = (self._S / self.std[None, :]).T                       # (n, n_mono)
        MR = M[:, :, None] * self._rates[None]                       # (B, n_mono, n_active)
        dZ = -np.matmul(St[None], MR)                                # (B, n, n_active)
        if order == 1:
            return Z, dZ
        d2Z = np.einsum("km,bme,mf->bkef", St, MR, self._rates, optimize=True)
        return Z, dZ, d2Z

    def state(self, prefix="pip_"):
        return {f"{prefix}pairs": self.pairs, f"{prefix}a": self.a, f"{prefix}exponents": self.exponents,
                f"{prefix}orbit": self.orbit, f"{prefix}mean": self.mean, f"{prefix}std": self.std}

    @classmethod
    def from_state(cls, s, prefix="pip_"):
        return cls(s[f"{prefix}pairs"], s[f"{prefix}a"], s[f"{prefix}exponents"], s[f"{prefix}orbit"],
                   s[f"{prefix}mean"], s[f"{prefix}std"])
