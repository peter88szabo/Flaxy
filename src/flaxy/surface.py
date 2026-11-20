"""The assembled capture surface and its analytic forces.

V(x) = U_base(x) + (1 - g(R)) f_G(y(x)),

U_base = d + U_NH3 + U_CH + L_damp  (baseline.py), g = rising quintic step in
the COM distance between r_in and r_out, f_G an S3-projected short-range model
(SymKRR or GaussianTT) of the pair features. Beyond r_out, V = U_base exactly.
"""
import json

import numpy as np

from .baseline import Baseline
from .features import PairFeatures
from .geometry import com_distance_grad, pair_grad_to_cart, pair_vectors
from .krr import SymKRR
from .switch import rising
from .tt import GaussianTT
from .units import ANG_BOHR

INNER = {"krr": SymKRR, "tt": GaussianTT}
FORMAT = "flaxy-capture-v1"


class Surface:
    def __init__(self, baseline: Baseline, features: PairFeatures, inner, r_in, r_out):
        """r_in, r_out in bohr."""
        self.baseline, self.features, self.inner = baseline, features, inner
        self.r_in, self.r_out = float(r_in), float(r_out)

    def gate(self, X):
        R, dR = com_distance_grad(X)
        g, dg = rising(R, self.r_in, self.r_out)
        return g, dg[:, None, None] * dR

    def inner_residual_target(self, X, E):
        """Training target for f and the operator scale 1 - g."""
        g, _ = self.gate(X)
        return E - self.baseline.energy(X), 1.0 - g

    def energy(self, X):
        E = self.baseline.energy(X)
        g, _ = self.gate(X)
        active = g < 1.0
        if active.any() and self.inner is not None:
            _, r = pair_vectors(X[active])
            y, _ = self.features(r)
            E[active] += (1.0 - g[active]) * self.inner.predict(y)
        return E

    def energy_forces(self, X):
        """Energy (Hartree) and forces -dV/dX (Hartree/bohr) for X[b, 6, 3] in bohr."""
        E, grad = self.baseline.energy_grad(X)
        g, dg = self.gate(X)
        active = np.flatnonzero(g < 1.0)
        if len(active) and self.inner is not None:
            Xa = X[active]
            d, r = pair_vectors(Xa)
            y, dydr = self.features(r)
            f, dfdy = self.inner.predict(y, grad=True)
            s = 1.0 - g[active]
            E[active] += s * f
            grad[active] += s[:, None, None] * pair_grad_to_cart(dfdy * dydr, d, r) \
                - f[:, None, None] * dg[active]
        return E, -grad

    def energy_forces_angstrom(self, X_ang):
        """Convenience: X in Angstrom, returns Hartree and Hartree/bohr."""
        return self.energy_forces(np.asarray(X_ang, dtype=float) * ANG_BOHR)

    def save(self, path, meta=None):
        state = {"format": np.array(FORMAT), "r_switch": np.array([self.r_in, self.r_out]),
                 "inner_kind": np.array(_inner_kind(self.inner)),
                 "meta": np.array(json.dumps(meta or {}))}
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
        kind = str(s["inner_kind"])
        inner = None if kind == "none" else INNER[kind].from_state(s)
        r_in, r_out = s["r_switch"]
        return cls(Baseline.from_state(s), PairFeatures.from_state(s), inner, r_in, r_out)


def _inner_kind(inner):
    for k, v in INNER.items():
        if isinstance(inner, v):
            return k
    return "none"
