"""Reference energy U, linear in all its parameters.

Two fragments:  U = d + U_A(q_A) + U_B(q_B) + L_damp(R, Omega, q)   (asymptotic channel)
One fragment:   U = d (+ optional whole-molecule invariant polynomial)

The fragment polynomials depend only on internal coordinates and the tail on R and
orientation, so on randomly oriented data the blocks are nearly orthogonal and can
be fitted together by one weighted least-squares solve.
"""
import json

import numpy as np
import scipy.linalg as sl

from .geometry import pair_grad_to_cart, pair_hess_to_cart, pair_vectors
from .monomer import ExpPoly, fragment_poly
from .tail import DEFAULT_KINDS, Tail


class Baseline:
    def __init__(self, system, polys, tail):
        self.system, self.polys, self.tail = system, list(polys), tail
        self.e_ref = 0.0
        self.coef = None

    @classmethod
    def build(cls, system, X_ref, features, degrees=(8, 8), beta=1.0, kinds=DEFAULT_KINDS,
              quadrupoles="total"):
        """X_ref (bohr) sets the tail's orbit vectors and scales (two fragments only)."""
        polys = [fragment_poly(system, features, f, degrees[f] if f < len(degrees) else 0)
                 for f in range(system.n_fragments)]
        tail = (Tail.from_geometries(system, X_ref, beta, kinds, quadrupoles)
                if system.n_fragments == 2 else None)
        return cls(system, polys, tail)

    @property
    def blocks(self):
        sizes = [("const", 1)] + [(f"poly{f}", p.n_basis if p else 0) for f, p in enumerate(self.polys)]
        sizes.append(("tail", len(self.tail.terms) if self.tail else 0))
        out, start = {}, 0
        for name, size in sizes:
            out[name] = slice(start, start + size)
            start += size
        return out

    def design(self, X):
        _, r = pair_vectors(X, self.system)
        cols = [np.ones((len(X), 1))] + [p.design(r) for p in self.polys if p]
        if self.tail:
            cols.append(self.tail.design(X))
        return np.hstack(cols)

    def fit(self, X, E, weights=None, ridge=1e-10):
        """Weighted ridge least squares of all columns at once."""
        w = np.ones(len(E)) if weights is None else np.asarray(weights, dtype=float)
        self.e_ref = float(np.median(E))
        A = self.design(X)
        coef, stderr, cond, resid = _wls(A, E - self.e_ref, w, ridge)
        self.coef = coef
        info = {"rmse": float(np.sqrt(np.mean(resid**2))), "condition": cond,
                "n": int(len(E)), "n_params": int(A.shape[1])}
        if self.tail:
            tb = self.blocks["tail"]
            info["tail"] = {n: (float(c), float(s)) for n, c, s in
                            zip(self.tail.names, coef[tb], stderr[tb])}
        return info

    def fit_two_stage(self, X, E, weights, tail_rows, mono_rows, ridge=1e-10):
        """Stage 1: all columns on tail_rows (clean, long range) -> keep the tail.
        Stage 2: constant and fragment columns on mono_rows with the tail frozen."""
        if not self.tail:
            raise ValueError("two-stage fit needs a two-fragment system")
        stage1 = self.fit(X[tail_rows], E[tail_rows], weights[tail_rows], ridge)
        tb = self.blocks["tail"]
        tail = self.coef[tb].copy()
        A = self.design(X[mono_rows])
        y = E[mono_rows] - self.e_ref - A[:, tb] @ tail
        coef, _, cond, resid = _wls(A[:, :tb.start], y, weights[mono_rows], ridge)
        self.coef = np.concatenate([coef, tail])
        stage1.update({"mono_rmse": float(np.sqrt(np.mean(resid**2))),
                       "mono_condition": cond, "mono_n": int(mono_rows.sum())})
        return stage1

    def parts(self, X):
        """Energy components (Hartree) without e_ref and the constant."""
        _, r = pair_vectors(X, self.system)
        b, out = self.blocks, {}
        for f, p in enumerate(self.polys):
            if p:
                out[f"poly{f}"] = p.design(r) @ self.coef[b[f"poly{f}"]]
        if self.tail:
            out["tail"] = self.tail.design(X) @ self.coef[b["tail"]]
        return out

    def energy(self, X):
        return self.e_ref + self.design(X) @ self.coef

    def evaluate(self, X, order=1):
        """Energy (Hartree), gradient dE/dX [b, N, 3] (Hartree/bohr) and, for order 2,
        Hessian [b, 3N, 3N] (Hartree/bohr^2). All analytic."""
        d, r = pair_vectors(X, self.system)
        b = self.blocks
        E = np.full(len(X), self.e_ref + self.coef[0])
        dEdr = np.zeros_like(r)
        d2E = np.zeros((len(X), r.shape[1], r.shape[1])) if order == 2 else None
        for f, p in enumerate(self.polys):
            if p:
                res = p.evaluate(r, self.coef[b[f"poly{f}"]], order)
                E += res[0]
                dEdr += res[1]
                if order == 2:
                    d2E += res[2]
        grad = pair_grad_to_cart(dEdr, d, r, self.system)
        hess = pair_hess_to_cart(dEdr, d2E, d, r, self.system) if order == 2 else None
        if self.tail:
            res = self.tail.evaluate(X, self.coef[b["tail"]], order)
            E += res[0]
            grad += res[1]
            if order == 2:
                hess += res[2]
        return (E, grad) if order < 2 else (E, grad, hess)

    def energy_grad(self, X):
        return self.evaluate(X, order=1)

    def state(self):
        s = {"base_coef": self.coef, "base_e_ref": np.array(self.e_ref),
             "base_layout": np.array(json.dumps({"polys": [p is not None for p in self.polys],
                                                 "tail": self.tail is not None}))}
        for f, p in enumerate(self.polys):
            if p:
                s.update(p.state(f"poly{f}_"))
        if self.tail:
            s.update(self.tail.state())
        return s

    @classmethod
    def from_state(cls, s, system):
        layout = json.loads(str(s["base_layout"]))
        polys = [ExpPoly.from_state(s, f"poly{f}_") if has else None for f, has in enumerate(layout["polys"])]
        tail = Tail.from_state(s, system) if layout["tail"] else None
        obj = cls(system, polys, tail)
        obj.coef = np.asarray(s["base_coef"])
        obj.e_ref = float(s["base_e_ref"])
        return obj


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
