"""Multipole-shaped long-range basis for NH3 + CH, Tang-Toennies damped.

Shapes come from Cartesian interaction tensors T^(m) = grad^m (1/R) contracted
with covariant fragment tensors built from bond vectors (no body frames, no
singularity for planar NH3):

  p_A = sum_i (X_Hi - X_N) / l_A         NH3 dipole direction (C3 axis)
  Q_A = sum_i (s s^T - |s|^2 I/3),  s = (X_Hi - X_N)/l_S     NH3 quadrupole shape
  p_B = (X_H - X_C) / l_B                CH dipole direction
  Q_B = p_B p_B^T - |p_B|^2 I / 3        CH quadrupole shape

R = o_B - o_A between centres of mass, n = R/|R|. Each column is one shape
times f_n(beta R)/R^n; the fitted coefficient carries the physical magnitude.
For the dipole-dipole column the coefficient is -mu_A mu_B when |p| = 1 and both
dipoles point along +p (N->H for NH3, C->H for CH, i.e. C-H+ polarity).
"""
import numpy as np

from . import dual as D
from .system import FRAG_A, FRAG_B, MASSES

TERMS = {
    # name: (inverse power, description)
    "dd": (3, "dipole-dipole"),
    "dq": (4, "dipole(NH3)-quadrupole(CH)"),
    "qd": (4, "quadrupole(NH3)-dipole(CH)"),
    "qq": (5, "quadrupole-quadrupole"),
    "c6": (6, "isotropic R^-6 (dispersion + induction)"),
    "c6a": (6, "R^-6 anisotropy along NH3 axis"),
    "c6b": (6, "R^-6 anisotropy along CH axis"),
    "ddA": (3, "dipole-dipole x NH3 dipole stretch |p_A|^2-1"),
    "ddB": (3, "dipole-dipole x CH dipole stretch |p_B|^2-1"),
    "c8": (8, "isotropic R^-8"),
}
DEFAULT_TERMS = ("dd", "dq", "qd", "qq", "c6", "c6a", "c6b", "ddA", "ddB")


class Tail:
    def __init__(self, scales, beta=1.0, terms=DEFAULT_TERMS):
        self.lA, self.lS, self.lB = (float(scales[k]) for k in ("lA", "lS", "lB"))
        self.beta = float(beta)
        self.terms = tuple(terms)
        for t in self.terms:
            if t not in TERMS:
                raise ValueError(f"unknown tail term {t}")

    @classmethod
    def from_geometries(cls, X, **kw):
        """Set l_A, l_S, l_B to medians over X so that |p| ~ 1 on typical geometries."""
        s = X[:, 1:4] - X[:, [0]]
        scales = {"lA": np.median(np.linalg.norm(s.sum(1), axis=1)),
                  "lS": np.median(np.linalg.norm(s, axis=2)),
                  "lB": np.median(np.linalg.norm(X[:, 5] - X[:, 4], axis=1))}
        return cls(scales, **kw)

    def columns(self, X):
        """Dict of Duals, one per term, differentiable w.r.t. all 18 coordinates."""
        x = D.seed(X)
        oA = _origin(x, FRAG_A)
        oB = _origin(x, FRAG_B)
        Rv = D.vsub(oB, oA)
        R = D.dot(Rv, Rv).sqrt()
        n = D.vscale(Rv, R.reciprocal())

        s = [D.vsub(x[i], x[0]) for i in (1, 2, 3)]
        pA = D.vscale(D.vadd(D.vadd(s[0], s[1]), s[2]), 1.0 / self.lA)
        QA = D.outer_traceless(s, 1.0 / self.lS**2)
        pB = D.vscale(D.vsub(x[5], x[4]), 1.0 / self.lB)
        QB = D.outer_traceless([pB])

        u, v = D.dot(pA, n), D.dot(pB, n)
        QAn, QBn = D.matvec(QA, n), D.matvec(QB, n)
        nQAn, nQBn = D.dot(n, QAn), D.dot(n, QBn)
        inv = {k: D.damped_inverse_power(R, k, self.beta) for k in {TERMS[t][0] for t in self.terms}}

        dd = 3.0 * u * v - D.dot(pA, pB)
        shape = {
            "dd": lambda: dd,
            "dq": lambda: 15.0 * u * nQBn - 6.0 * D.dot(pA, QBn),
            "qd": lambda: 15.0 * v * nQAn - 6.0 * D.dot(pB, QAn),
            "qq": lambda: 105.0 * nQAn * nQBn - 60.0 * D.dot(QAn, QBn) + 6.0 * D.contract(QA, QB),
            "c6": lambda: None,
            "c6a": lambda: nQAn,
            "c6b": lambda: nQBn,
            "ddA": lambda: dd * (D.dot(pA, pA) - 1.0),
            "ddB": lambda: dd * (D.dot(pB, pB) - 1.0),
            "c8": lambda: None,
        }
        out = {}
        for t in self.terms:
            a = shape[t]()
            radial = inv[TERMS[t][0]]
            out[t] = radial if a is None else a * radial
        return out

    def design(self, X):
        cols = self.columns(X)
        return np.stack([cols[t].v for t in self.terms], axis=1)

    def value_grad(self, X, coef):
        cols = self.columns(X)
        total = None
        for t, c in zip(self.terms, coef):
            term = cols[t] * float(c)
            total = term if total is None else total + term
        return total.v, total.g.reshape(len(X), -1, 3)

    def state(self):
        return {"tail_scales": np.array([self.lA, self.lS, self.lB]),
                "tail_beta": np.array(self.beta),
                "tail_terms": np.array(self.terms)}

    @classmethod
    def from_state(cls, s):
        lA, lS, lB = s["tail_scales"]
        return cls({"lA": lA, "lS": lS, "lB": lB}, float(s["tail_beta"]),
                   tuple(str(t) for t in s["tail_terms"]))


def _origin(x, idx):
    m = MASSES[list(idx)]
    w = m / m.sum()
    return [sum(x[i][k] * w[j] for j, i in enumerate(idx)) for k in range(3)]
