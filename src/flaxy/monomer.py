"""Permutation-invariant polynomials in Morse variables for the isolated fragments.

With y_e = exp(-r_e / a_e), a monomial prod_e y_e^{m_e} is exp(-sum_e m_e r_e / a_e),
so every monomial and its r-gradient follow from one matrix product and one exp.
Invariant basis functions are orbit sums of monomials under the fragment group.
"""
import itertools

import numpy as np

from .system import CH_PAIR, NH3_PAIRS, PAIR_PERMS


class ExpPoly:
    def __init__(self, pairs, a, exponents, orbit):
        self.pairs = np.asarray(pairs)          # global pair indices used
        self.a = np.asarray(a, dtype=float)     # Morse range per used pair (bohr)
        self.exponents = np.asarray(exponents)  # (n_mono, n_var)
        self.orbit = np.asarray(orbit)          # (n_mono,) basis index of each monomial
        self.n_basis = int(self.orbit.max()) + 1
        self._rates = self.exponents / self.a[None, :]

    def monomials(self, r):
        return np.exp(-r[:, self.pairs] @ self._rates.T)

    def design(self, r):
        """(B, n_basis) invariant basis values."""
        M = self.monomials(r)
        out = np.zeros((len(r), self.n_basis))
        np.add.at(out.T, self.orbit, M.T)
        return out

    def value_grad_r(self, r, coef):
        """Energy and dE/dr[b, 15] (zero outside the fragment pairs)."""
        w = coef[self.orbit]
        Mw = self.monomials(r) * w[None, :]
        dEdr = np.zeros_like(r)
        dEdr[:, self.pairs] = -Mw @ self._rates
        return Mw.sum(1), dEdr

    def state(self, prefix):
        return {f"{prefix}pairs": self.pairs, f"{prefix}a": self.a,
                f"{prefix}exponents": self.exponents, f"{prefix}orbit": self.orbit}

    @classmethod
    def from_state(cls, s, prefix):
        return cls(s[f"{prefix}pairs"], s[f"{prefix}a"], s[f"{prefix}exponents"], s[f"{prefix}orbit"])


def _orbit_basis(n_var, degree, var_perms):
    """Monomials of total degree 1..degree grouped into orbits under var_perms."""
    exps, orbit, seen = [], [], {}
    for total in range(1, degree + 1):
        for combo in itertools.combinations_with_replacement(range(n_var), total):
            e = np.bincount(combo, minlength=n_var)
            images = {tuple(np.bincount(np.asarray(p)[list(combo)], minlength=n_var))
                      for p in var_perms}
            key = min(images)
            if key not in seen:
                seen[key] = len(seen)
            exps.append(e)
            orbit.append(seen[key])
    return np.array(exps), np.array(orbit)


def nh3_basis(degree=6, a_nh=1.5, a_hh=2.0):
    """S3-invariant polynomial for NH3 in its 6 internal distances (sites 0..5)."""
    pairs = NH3_PAIRS
    local = {g: k for k, g in enumerate(pairs)}
    # Pair permutation restricted to NH3 pairs: pair e maps to PAIR_PERMS[P][e].
    var_perms = [[local[PAIR_PERMS[P][e]] for e in pairs] for P in range(len(PAIR_PERMS))]
    exps, orbit = _orbit_basis(len(pairs), degree, var_perms)
    a = np.array([a_hh] * 3 + [a_nh] * 3)
    return ExpPoly(pairs, a, exps, orbit)


def ch_basis(degree=8, a_ch=1.5):
    exps = np.arange(1, degree + 1)[:, None]
    return ExpPoly([CH_PAIR], [a_ch], exps, np.arange(degree))
