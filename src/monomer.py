"""Permutation-invariant polynomials in Morse variables (fragment or whole-molecule energies).

With y_e = exp(-r_e / a_e), a monomial prod_e y_e^{m_e} is exp(-sum_e m_e r_e / a_e),
so all monomials and their r-derivatives follow from one matrix product and one
exp. Invariant basis functions are orbit sums of monomials under the group.
"""
import itertools
import math

import numpy as np


class ExpPoly:
    def __init__(self, pairs, a, exponents, orbit):
        self.pairs = np.asarray(pairs, dtype=int)        # global pair indices used
        self.a = np.asarray(a, dtype=float)              # Morse range per used pair (bohr)
        self.exponents = np.asarray(exponents)           # (n_mono, n_var)
        self.orbit = np.asarray(orbit, dtype=int)        # (n_mono,) basis index of each monomial
        self.n_basis = int(self.orbit.max()) + 1
        self._rates = self.exponents / self.a[None, :]

    def monomials(self, r):
        return np.exp(-r[:, self.pairs] @ self._rates.T)

    def design(self, r):
        """(B, n_basis) invariant basis values."""
        out = np.zeros((len(r), self.n_basis))
        np.add.at(out.T, self.orbit, self.monomials(r).T)
        return out

    def evaluate(self, r, coef, order=1):
        """Energy, dE/dr[b, P] and, for order 2, d2E/dr2[b, P, P] (zero outside self.pairs)."""
        Mw = self.monomials(r) * coef[self.orbit][None, :]
        E = Mw.sum(1)
        g = np.zeros_like(r)
        g[:, self.pairs] = -Mw @ self._rates
        if order < 2:
            return E, g
        H = np.zeros((len(r), r.shape[1], r.shape[1]))
        H[np.ix_(np.arange(len(r)), self.pairs, self.pairs)] = \
            np.einsum("bm,mi,mj->bij", Mw, self._rates, self._rates)
        return E, g, H

    def state(self, prefix):
        return {f"{prefix}pairs": self.pairs, f"{prefix}a": self.a,
                f"{prefix}exponents": self.exponents, f"{prefix}orbit": self.orbit}

    @classmethod
    def from_state(cls, s, prefix):
        return cls(s[f"{prefix}pairs"], s[f"{prefix}a"], s[f"{prefix}exponents"], s[f"{prefix}orbit"])


def orbit_basis(n_var, degree, var_perms, max_monomials=200000):
    """Monomials of total degree 1..degree and their orbit index under var_perms.

    var_perms[g][k] = l means the group element maps variable k onto variable l.
    """
    count = math.comb(n_var + degree, degree) - 1
    if count > max_monomials:
        raise ValueError(f"{count:,} monomials (n_var={n_var}, degree={degree}) > max_monomials; "
                         "lower the degree or use a fragment kernel model")
    exps = np.array([np.bincount(c, minlength=n_var)
                     for total in range(1, degree + 1)
                     for c in itertools.combinations_with_replacement(range(n_var), total)], dtype=np.int16)
    index = {e.tobytes(): k for k, e in enumerate(exps)}
    orbit = np.arange(len(exps))
    for p in var_perms:
        image = np.empty_like(exps)
        image[:, np.asarray(p)] = exps                   # exponent of y_k moves to y_{p[k]}
        orbit = np.minimum(orbit, [index[row.tobytes()] for row in image])
    _, orbit = np.unique(orbit, return_inverse=True)
    return exps, orbit


def fragment_poly(system, features, which, degree, max_monomials=200000):
    """Invariant polynomial of fragment `which` (0 or 1; 0 is the whole molecule when
    the system has one fragment). Returns None if the fragment has no internal pairs."""
    pairs = system.fragment_pairs(which)
    if len(pairs) == 0 or degree < 1:
        return None
    local = {int(g): k for k, g in enumerate(pairs)}
    var_perms = [[local[int(P[e])] for e in pairs] for P in system.pair_perms]
    exps, orbit = orbit_basis(len(pairs), degree, var_perms, max_monomials)
    return ExpPoly(pairs, features.a[pairs], exps, orbit)
