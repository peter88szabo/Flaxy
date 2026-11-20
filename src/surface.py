"""The assembled potential energy surface with analytic forces and Hessians.

V(x) = U(x) + s(x) f_G(y_active(x))

U        reference energy (baseline.py): constant (+ whole-molecule polynomial) for
         one fragment; fragment energies + long-range tail for two fragments
s(x)     1 for one fragment; 1 - g(R) for two fragments, g a C2 step in the centre-
         of-mass separation between r_in and r_out, so V = U exactly beyond r_out
f_G      group-projected short-range model (SymKRR or GaussianTT) of the active pair
         features y = exp(-r/a)

Derivatives: grad V = grad U + s grad f + f grad s,
             H_V = H_U + s H_f + grad s grad f^T + grad f grad s^T + f H_s.
"""
import json

import numpy as np

from .baseline import Baseline
from .features import PairFeatures
from .geometry import com_distance_grad, pair_grad_to_cart, pair_hess_to_cart, pair_vectors
from .krr import SymKRR
from .switch import rising
from .system import System
from .tt import GaussianTT
from .units import ANG_BOHR

INNER = {"krr": SymKRR, "tt": GaussianTT}
FORMAT = "flaxy-surface-v2"


class Surface:
    def __init__(self, system, baseline, features, inner, r_in=None, r_out=None):
        """r_in, r_out in bohr; required for two-fragment systems, ignored otherwise."""
        self.system, self.baseline, self.features, self.inner = system, baseline, features, inner
        self.switched = system.n_fragments == 2
        if self.switched and (r_in is None or r_out is None):
            raise ValueError("two-fragment surfaces need r_in and r_out (bohr)")
        self.r_in = None if r_in is None else float(r_in)
        self.r_out = None if r_out is None else float(r_out)

    # ---------------------------------------------------------------- pieces
    def scale(self, X, order=0):
        """s(x) = 1 - g(R) (two fragments) or 1, with Cartesian gradient / Hessian."""
        n3 = 3 * self.system.n_atoms
        if not self.switched:
            out = (np.ones(len(X)), np.zeros((len(X), self.system.n_atoms, 3)),
                   np.zeros((len(X), n3, n3)))
            return out[:order + 1] if order else out[0]
        res = com_distance_grad(X, self.system, order=max(order, 1))
        g, dg, d2g = rising(res[0], self.r_in, self.r_out)
        s = 1.0 - g
        if order == 0:
            return s
        ds = -dg[:, None, None] * res[1]
        if order == 1:
            return s, ds
        gR = res[1].reshape(len(X), -1)
        d2s = -(d2g[:, None, None] * gR[:, :, None] * gR[:, None, :] + dg[:, None, None] * res[2])
        return s, ds, d2s

    def inner_features(self, X, order=0):
        d, r = pair_vectors(X, self.system)
        y = self.features(r, order=max(order, 1))
        return d, r, y

    def inner_residual_target(self, X, E):
        """Training target E - U for the short-range model and the scale s(x)."""
        return E - self.baseline.energy(X), self.scale(X)

    # ---------------------------------------------------------------- energies
    def energy(self, X):
        E = self.baseline.energy(X)
        s = self.scale(X)
        act = np.flatnonzero(s > 0)
        if len(act) and self.inner is not None:
            _, _, (y, _) = self.inner_features(X[act])
            E[act] += s[act] * self.inner.predict(y[:, self.system.active_pairs])
        return E

    def energy_forces(self, X):
        """Energy (Hartree) and forces -dV/dX (Hartree/bohr) for X[b, N, 3] in bohr."""
        E, F = self._evaluate(X, order=1)
        return E, F

    def energy_forces_hessian(self, X):
        """Energy, forces and the Cartesian Hessian [b, 3N, 3N] (Hartree/bohr^2)."""
        return self._evaluate(X, order=2)

    def energy_forces_angstrom(self, X_ang):
        """Convenience: X in Angstrom; returns Hartree and Hartree/bohr."""
        return self.energy_forces(np.asarray(X_ang, dtype=float) * ANG_BOHR)

    def _evaluate(self, X, order):
        res = self.baseline.evaluate(X, order)
        E, grad = res[0], res[1]
        hess = res[2] if order == 2 else None
        sc = self.scale(X, order)
        s = sc[0]
        act = np.flatnonzero(s > 0)
        if len(act) and self.inner is not None:
            sysm, ap = self.system, self.system.active_pairs
            Xa = X[act]
            d, r, feats = self.inner_features(Xa, order)
            y, dy = feats[0], feats[1]
            out = self.inner.predict(y[:, ap], order=order)
            f, dfdy = out[0], out[1]
            dEdr = np.zeros_like(r)
            dEdr[:, ap] = dfdy * dy[:, ap]
            gf = pair_grad_to_cart(dEdr, d, r, sysm)
            sa, dsa = s[act], sc[1][act]
            E[act] += sa * f
            grad[act] += sa[:, None, None] * gf + f[:, None, None] * dsa
            if order == 2:
                d2y = feats[2]
                d2Edr2 = np.zeros((len(act), sysm.n_pairs, sysm.n_pairs))
                d2Edr2[np.ix_(np.arange(len(act)), ap, ap)] = \
                    out[2] * dy[:, ap, None] * dy[:, None, ap]
                d2Edr2[:, ap, ap] += dfdy * d2y[:, ap]
                Hf = pair_hess_to_cart(dEdr, d2Edr2, d, r, sysm)
                gfv, dsv = gf.reshape(len(act), -1), dsa.reshape(len(act), -1)
                hess[act] += (sa[:, None, None] * Hf + f[:, None, None] * sc[2][act]
                              + gfv[:, :, None] * dsv[:, None, :] + dsv[:, :, None] * gfv[:, None, :])
        F = -grad
        return (E, F) if order < 2 else (E, F, hess)

    # --------------------------------------------------------------------- I/O
    def save(self, path, meta=None):
        state = {"format": np.array(FORMAT), "inner_kind": np.array(_inner_kind(self.inner)),
                 "r_switch": np.array([np.nan if v is None else v for v in (self.r_in, self.r_out)]),
                 "meta": np.array(json.dumps(meta or {}))}
        state.update(self.system.state())
        state.update(self.baseline.state())
        state.update(self.features.state())
        if self.inner is not None:
            state.update(self.inner.state())
        np.savez_compressed(path, **state)

    @classmethod
    def load(cls, path):
        s = np.load(path, allow_pickle=False)
        if str(s["format"]) != FORMAT:
            raise ValueError(f"{path}: unknown format {s['format']}")
        system = System.from_state(s)
        kind = str(s["inner_kind"])
        inner = None if kind == "none" else INNER[kind].from_state(s)
        r_in, r_out = (None if np.isnan(v) else float(v) for v in s["r_switch"])
        return cls(system, Baseline.from_state(s, system), PairFeatures.from_state(s, system),
                   inner, r_in, r_out)


def _inner_kind(inner):
    for k, v in INNER.items():
        if isinstance(inner, v):
            return k
    return "none"
