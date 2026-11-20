"""Separable asymptotic baseline U = d + U_NH3(q_A) + U_CH(r) + L_damp(R, Omega, q).

All parameters enter linearly, so (d, U_NH3, U_CH, L) are fitted together by one
weighted least-squares solve on the long-range geometries. The monomer columns
depend only on internal coordinates while the tail columns depend on R and
orientation, so the blocks are nearly orthogonal on randomly oriented data.
"""
import numpy as np
import scipy.linalg as sl

from .geometry import pair_grad_to_cart, pair_vectors
from .monomer import ExpPoly, ch_basis, nh3_basis
from .tail import TERMS, Tail


class Baseline:
    def __init__(self, nh3: ExpPoly, ch: ExpPoly, tail: Tail):
        self.nh3, self.ch, self.tail = nh3, ch, tail
        self.e_ref = 0.0
        self.coef = None

    @classmethod
    def build(cls, X_ref, nh3_degree=6, ch_degree=8, a_nh=1.5, a_hh=2.0, a_ch=1.5,
              beta=1.0, terms=None):
        kw = {"beta": beta} if terms is None else {"beta": beta, "terms": terms}
        return cls(nh3_basis(nh3_degree, a_nh, a_hh), ch_basis(ch_degree, a_ch),
                   Tail.from_geometries(X_ref, **kw))

    @property
    def blocks(self):
        nA, nB, nT = self.nh3.n_basis, self.ch.n_basis, len(self.tail.terms)
        return {"const": slice(0, 1), "nh3": slice(1, 1 + nA),
                "ch": slice(1 + nA, 1 + nA + nB), "tail": slice(1 + nA + nB, 1 + nA + nB + nT)}

    def design(self, X):
        _, r = pair_vectors(X)
        return np.hstack([np.ones((len(X), 1)), self.nh3.design(r), self.ch.design(r),
                          self.tail.design(X)])

    def fit(self, X, E, weights=None, ridge=1e-10):
        """Weighted ridge least squares of all columns at once.

        Returns diagnostics including tail coefficient standard errors.
        """
        w = np.ones(len(E)) if weights is None else np.asarray(weights, dtype=float)
        self.e_ref = float(np.median(E))
        A = self.design(X)
        coef, stderr, cond, resid = _wls(A, E - self.e_ref, w, ridge)
        self.coef = coef
        tb = self.blocks["tail"]
        return {"rmse": float(np.sqrt(np.mean(resid**2))),
                "tail": {t: (float(c), float(s)) for t, c, s in
                         zip(self.tail.terms, coef[tb], stderr[tb])},
                "condition": cond, "n": int(len(E)), "n_params": int(A.shape[1])}

    def fit_two_stage(self, X, E, weights, tail_rows, mono_rows, ridge=1e-10):
        """Stage 1: all columns on tail_rows (clean, long range) -> keep the tail.
        Stage 2: constant and monomer columns on mono_rows with the tail frozen.
        """
        stage1 = self.fit(X[tail_rows], E[tail_rows], weights[tail_rows], ridge)
        b, tail = self.blocks, self.coef[self.blocks["tail"]].copy()
        Xm, Em, wm = X[mono_rows], E[mono_rows], weights[mono_rows]
        A = self.design(Xm)
        keep = np.arange(b["ch"].stop)  # const, nh3 and ch columns precede the tail
        y = Em - self.e_ref - A[:, b["tail"]] @ tail
        coef, _, cond, resid = _wls(A[:, keep], y, wm, ridge)
        self.coef = np.concatenate([coef, tail])
        stage1.update({"mono_rmse": float(np.sqrt(np.mean(resid**2))),
                       "mono_condition": cond, "mono_n": int(len(Em))})
        return stage1

    def parts(self, X):
        """Energy components (Hartree): monomers and interaction, without e_ref."""
        _, r = pair_vectors(X)
        b = self.blocks
        return {"nh3": self.nh3.design(r) @ self.coef[b["nh3"]],
                "ch": self.ch.design(r) @ self.coef[b["ch"]],
                "tail": self.tail.design(X) @ self.coef[b["tail"]]}

    def energy(self, X):
        return self.e_ref + self.design(X) @ self.coef

    def energy_grad(self, X):
        """Energy (Hartree) and gradient dE/dX (Hartree/bohr), analytic."""
        d, r = pair_vectors(X)
        b = self.blocks
        eA, gA = self.nh3.value_grad_r(r, self.coef[b["nh3"]])
        eB, gB = self.ch.value_grad_r(r, self.coef[b["ch"]])
        eT, gT = self.tail.value_grad(X, self.coef[b["tail"]])
        E = self.e_ref + self.coef[0] + eA + eB + eT
        return E, pair_grad_to_cart(gA + gB, d, r) + gT

    def state(self):
        s = {"base_coef": self.coef, "base_e_ref": np.array(self.e_ref)}
        s.update(self.nh3.state("nh3_"))
        s.update(self.ch.state("ch_"))
        s.update(self.tail.state())
        return s

    @classmethod
    def from_state(cls, s):
        obj = cls(ExpPoly.from_state(s, "nh3_"), ExpPoly.from_state(s, "ch_"), Tail.from_state(s))
        obj.coef = np.asarray(s["base_coef"])
        obj.e_ref = float(s["base_e_ref"])
        return obj


def tail_powers(terms):
    return [TERMS[t][0] for t in terms]


def _wls(A, y, w, ridge):
    """Weighted ridge solve via SVD of the column-scaled design (no squared condition).

    Returns coefficients, their standard errors, the condition number and residuals.
    """
    scale = np.sqrt(np.mean(A**2, axis=0))
    scale[scale == 0] = 1.0
    sw = np.sqrt(w)
    U, S, Vt = sl.svd(A / scale * sw[:, None], full_matrices=False, lapack_driver="gesvd")
    filt = S / (S**2 + ridge * np.mean(S**2))
    cs = Vt.T @ (filt * (U.T @ (y * sw)))
    coef = cs / scale
    resid = y - A @ coef
    sigma2 = np.sum(w * resid**2) / max(len(y) - A.shape[1], 1)
    stderr = np.sqrt(np.einsum("ij,j->i", Vt.T**2, filt**2) * sigma2) / scale
    return coef, stderr, float(S[0] / S[-1]), resid
