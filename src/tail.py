"""Multipole-shaped long-range basis between two fragments, Tang-Toennies damped.

Shapes come from Cartesian interaction tensors T^(m) = grad^m (1/R) contracted
with covariant fragment tensors built from atom positions relative to the
fragment centre of mass o_F. Equivalent atoms (one orbit under the permutation
group) share one effective charge, so the fragment dipole lies in the span of

  v_{F,c} = sum_{i in orbit c} (X_i - o_F) / l_{F,c},

one vector per orbit, minus one (sum_c M_c v_c = 0 at the centre of mass). An
atom or a fragment whose orbits are all centred (e.g. homonuclear diatomic) gets
no dipole vector. Quadrupole shapes are Q_F = sum_i (s s^T - |s|^2 I/3) / l_F^2
over the whole fragment ("total") or per orbit ("orbits"). Every column is one
shape times f_n(beta R)/R^n; the fitted coefficient carries the magnitude. Terms:

  dd  R^-3  dipole-dipole               dq  R^-4  dipole(A)-quadrupole(B)
  qd  R^-4  quadrupole(A)-dipole(B)     qq  R^-5  quadrupole-quadrupole
  c6  R^-6  isotropic                   c6a, c6b  R^-6 anisotropy n.Q.n
  flex      dd x (|v|^2 - 1), dipole change with fragment distortion
"""
import json

import numpy as np

from . import dual as D
from .geometry import origin_weights

POWER = {"dd": 3, "dq": 4, "qd": 4, "qq": 5, "c6": 6, "c6a": 6, "c6b": 6, "flexA": 3, "flexB": 3}
DEFAULT_KINDS = ("dd", "dq", "qd", "qq", "c6", "c6a", "c6b", "flex")


class Tail:
    def __init__(self, system, vec_orbits, vec_scales, quad_sets, quad_scales, beta=1.0,
                 kinds=DEFAULT_KINDS):
        if system.n_fragments != 2:
            raise ValueError("a long-range tail needs a two-fragment system")
        self.system = system
        self.vec_orbits = [[tuple(o) for o in f] for f in vec_orbits]      # per fragment
        self.vec_scales = [list(map(float, s)) for s in vec_scales]
        self.quad_sets = [[tuple(o) for o in f] for f in quad_sets]
        self.quad_scales = [float(s) for s in quad_scales]
        self.beta = float(beta)
        self.kinds = tuple(kinds)
        self.terms = self._term_list()

    @classmethod
    def from_geometries(cls, system, X, beta=1.0, kinds=DEFAULT_KINDS, quadrupoles="total", tol=1e-6):
        """Choose orbits and scales from reference geometries X (bohr)."""
        vec_orbits, vec_scales, quad_sets, quad_scales = [], [], [], []
        for f in (0, 1):
            idx, w = origin_weights(system, f)
            o = np.einsum("i,bik->bk", w, X[:, idx])
            orbits = system.atom_orbits(f)
            vs, ss = [], []
            for orb in orbits[1:]:
                norm = np.median(np.linalg.norm((X[:, list(orb)] - o[:, None]).sum(1), axis=1))
                if norm > tol:
                    vs.append(orb)
                    ss.append(norm)
            rms = np.median(np.sqrt(np.mean(np.sum((X[:, idx] - o[:, None]) ** 2, axis=2), axis=1)))
            sets = [tuple(idx)] if quadrupoles == "total" else [o_ for o_ in orbits]
            if len(idx) == 1 or rms < tol:
                sets = []
            vec_orbits.append(vs)
            vec_scales.append(ss)
            quad_sets.append(sets)
            quad_scales.append(max(rms, tol))
        return cls(system, vec_orbits, vec_scales, quad_sets, quad_scales, beta, kinds)

    def _term_list(self):
        va, vb = range(len(self.vec_orbits[0])), range(len(self.vec_orbits[1]))
        qa, qb = range(len(self.quad_sets[0])), range(len(self.quad_sets[1]))
        k = set(self.kinds)
        t = []
        t += [("dd", a, b) for a in va for b in vb] if "dd" in k else []
        t += [("dq", a, b) for a in va for b in qb] if "dq" in k else []
        t += [("qd", a, b) for a in qa for b in vb] if "qd" in k else []
        t += [("qq", a, b) for a in qa for b in qb] if "qq" in k else []
        t += [("c6", 0, 0)] if "c6" in k else []
        t += [("c6a", a, 0) for a in qa] if "c6a" in k else []
        t += [("c6b", 0, b) for b in qb] if "c6b" in k else []
        if "flex" in k:
            t += [("flexA", a, b) for a in va for b in vb] + [("flexB", a, b) for a in va for b in vb]
        return t

    @property
    def names(self):
        return [f"{k}:{a}:{b}" for k, a, b in self.terms]

    def columns(self, X, order=1):
        """List of Duals, one per term, differentiable w.r.t. all Cartesian coordinates."""
        x = D.seed(X, order)
        sysm = self.system
        o, vec, quad = [], [], []
        for f in (0, 1):
            idx, w = origin_weights(sysm, f)
            of = [sum(x[i][k] * float(wi) for i, wi in zip(idx, w)) for k in range(3)]
            o.append(of)
            vec.append([D.vscale(_vsum([D.vsub(x[i], of) for i in orb]), 1.0 / s)
                        for orb, s in zip(self.vec_orbits[f], self.vec_scales[f])])
            quad.append([D.outer_traceless([D.vsub(x[i], of) for i in qs], 1.0 / self.quad_scales[f] ** 2)
                         for qs in self.quad_sets[f]])
        Rv = D.vsub(o[1], o[0])
        R = D.dot(Rv, Rv).sqrt()
        n = D.vscale(Rv, R.reciprocal())
        inv = {p: D.damped_inverse_power(R, p, self.beta) for p in {POWER[k] for k, _, _ in self.terms}}
        Qn = [[D.matvec(Q, n) for Q in quad[f]] for f in (0, 1)]
        nQn = [[D.dot(n, q) for q in Qn[f]] for f in (0, 1)]
        vn = [[D.dot(v, n) for v in vec[f]] for f in (0, 1)]

        def dd(a, b):
            return 3.0 * vn[0][a] * vn[1][b] - D.dot(vec[0][a], vec[1][b])

        cols = []
        for kind, a, b in self.terms:
            if kind == "dd":
                s = dd(a, b)
            elif kind == "dq":
                s = 15.0 * vn[0][a] * nQn[1][b] - 6.0 * D.dot(vec[0][a], Qn[1][b])
            elif kind == "qd":
                s = 15.0 * vn[1][b] * nQn[0][a] - 6.0 * D.dot(vec[1][b], Qn[0][a])
            elif kind == "qq":
                s = (105.0 * nQn[0][a] * nQn[1][b] - 60.0 * D.dot(Qn[0][a], Qn[1][b])
                     + 6.0 * D.contract(quad[0][a], quad[1][b]))
            elif kind == "c6":
                s = None
            elif kind == "c6a":
                s = nQn[0][a]
            elif kind == "c6b":
                s = nQn[1][b]
            elif kind == "flexA":
                s = dd(a, b) * (D.dot(vec[0][a], vec[0][a]) - 1.0)
            else:  # flexB
                s = dd(a, b) * (D.dot(vec[1][b], vec[1][b]) - 1.0)
            radial = inv[POWER[kind]]
            cols.append(radial if s is None else s * radial)
        return cols

    def design(self, X):
        return np.stack([c.v for c in self.columns(X)], axis=1)

    def evaluate(self, X, coef, order=1):
        """Energy, gradient [b, N, 3] and, for order 2, Hessian [b, 3N, 3N]."""
        total = None
        for c, w in zip(self.columns(X, order), coef):
            t = c * float(w)
            total = t if total is None else total + t
        g = total.g.reshape(len(X), -1, 3)
        return (total.v, g) if order < 2 else (total.v, g, total.h)

    def state(self, prefix="tail_"):
        spec = {"vec_orbits": self.vec_orbits, "vec_scales": self.vec_scales,
                "quad_sets": self.quad_sets, "quad_scales": self.quad_scales,
                "beta": self.beta, "kinds": self.kinds}
        return {f"{prefix}spec": np.array(json.dumps(spec))}

    @classmethod
    def from_state(cls, s, system, prefix="tail_"):
        spec = json.loads(str(s[f"{prefix}spec"]))
        return cls(system, spec["vec_orbits"], spec["vec_scales"], spec["quad_sets"],
                   spec["quad_scales"], spec["beta"], spec["kinds"])


def _vsum(vectors):
    out = vectors[0]
    for v in vectors[1:]:
        out = D.vadd(out, v)
    return out
