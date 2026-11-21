"""The assembled potential energy surface with analytic forces and Hessians.

V(x) = U(x) + s(x) f(Z(x))

U        reference energy (baseline.py): constant (+ whole-molecule polynomial) for
         one fragment; fragment energies + long-range tail for two fragments
s(x)     1 for one fragment; 1 - g(R) for two fragments, g a C2 step in the centre-
         of-mass separation between r_in and r_out, so V = U exactly beyond r_out
Z(x)     permutation-invariant polynomial (PIP) features of the active pairs
         (invariants.py): the symmetry is built into the input
f        short-range model of Z (KRR or GaussianTT), evaluated once per geometry

Derivatives: grad V = grad U + s grad f + f grad s,
             H_V = H_U + s H_f + grad s grad f^T + grad f grad s^T + f H_s,
with grad f = (df/dZ)(dZ/dr)(dr/dx) and the full second-order chain rule for H_f.
"""
import json

import numpy as np

from .baseline import Baseline
from .features import PairFeatures
from .geometry import com_distance_grad, pair_grad_to_cart, pair_hess_to_cart, pair_vectors
from .invariants import InvariantFeatures
from .krr import KRR
from .switch import rising
from .system import System
from .tt import GaussianTT
from .units import ANG_BOHR

INNER = {"krr": KRR, "tt": GaussianTT}
FORMAT = "flaxy-surface-v3"


class Surface:
    def __init__(self, system, baseline, features, inner=None, r_in=None, r_out=None, pips=None):
        """r_in, r_out in bohr (two-fragment systems only); pips: InvariantFeatures input of inner."""
        self.system, self.baseline, self.features = system, baseline, features
        self.inner, self.pips = inner, pips
        if inner is not None and pips is None:
            raise ValueError("the short-range model needs PIP inputs (InvariantFeatures)")
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

    def model_inputs(self, X):
        """PIP features Z[b, n] seen by the short-range model."""
        return self.pips(pair_vectors(X, self.system)[1])

    def inner_residual_target(self, X, E):
        """Training target E - U for the short-range model and the scale s(x)."""
        return E - self.baseline.energy(X), self.scale(X)

    # ---------------------------------------------------------------- energies
    def energy(self, X):
        E = self.baseline.energy(X)
        if self.inner is None:
            return E
        s = self.scale(X)
        act = np.flatnonzero(s > 0)
        if len(act):
            E[act] += s[act] * self.inner.predict(self.model_inputs(X[act]))
        return E

    def energy_forces(self, X):
        """Energy (Hartree) and forces -dV/dX (Hartree/bohr) for X[b, N, 3] in bohr."""
        return self._evaluate(X, order=1)

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
        if self.inner is not None:
            sc = self.scale(X, order)
            act = np.flatnonzero(sc[0] > 0)
            if len(act):
                self._add_inner(X[act], act, order, sc, E, grad, hess)
        F = -grad
        return (E, F) if order < 2 else (E, F, hess)

    def _add_inner(self, Xa, act, order, sc, E, grad, hess):
        sysm, ap, B = self.system, self.system.active_pairs, len(act)
        d, r = pair_vectors(Xa, sysm)
        zres = self.pips(r, order)
        Z, dZ = zres[0], zres[1]
        out = self.inner.predict(Z, order=order)
        f, dfdZ = out[0], out[1]
        dEdr = np.zeros_like(r)
        dEdr[:, ap] = np.einsum("bk,bke->be", dfdZ, dZ)
        gf = pair_grad_to_cart(dEdr, d, r, sysm)
        s, ds = sc[0][act], sc[1][act]
        E[act] += s * f
        grad[act] += s[:, None, None] * gf + f[:, None, None] * ds
        if order == 2:
            d2Edr2 = np.zeros((B, sysm.n_pairs, sysm.n_pairs))
            d2Edr2[np.ix_(np.arange(B), ap, ap)] = (
                np.einsum("bke,bkl,blf->bef", dZ, out[2], dZ, optimize=True)
                + np.einsum("bk,bkef->bef", dfdZ, zres[2], optimize=True))
            Hf = pair_hess_to_cart(dEdr, d2Edr2, d, r, sysm)
            gfv, dsv = gf.reshape(B, -1), ds.reshape(B, -1)
            hess[act] += (s[:, None, None] * Hf + f[:, None, None] * sc[2][act]
                          + gfv[:, :, None] * dsv[:, None, :] + dsv[:, :, None] * gfv[:, None, :])

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
            state.update(self.pips.state())
        np.savez_compressed(path, **state)

    @classmethod
    def load(cls, path):
        s = np.load(path, allow_pickle=False)
        if str(s["format"]) != FORMAT:
            raise ValueError(f"{path}: format {s['format']} is not {FORMAT}; refit the surface")
        system = System.from_state(s)
        kind = str(s["inner_kind"])
        inner = None if kind == "none" else INNER[kind].from_state(s)
        pips = None if inner is None else InvariantFeatures.from_state(s)
        r_in, r_out = (None if np.isnan(v) else float(v) for v in s["r_switch"])
        return cls(system, Baseline.from_state(s, system), PairFeatures.from_state(s, system),
                   inner, r_in, r_out, pips)


def _inner_kind(inner):
    for k, v in INNER.items():
        if isinstance(inner, v):
            return k
    return "none"
