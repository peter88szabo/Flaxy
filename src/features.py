"""Pair features y_e = exp(-r_e / a_e) with ranges tied by pair type (bohr).

Default ranges by pair type (system.pair_type = "<elements>:<location>:<kind>"):
inter-fragment pairs `inter`, bonded pairs `bond`, mobile-atom pairs `mobile`,
all other intra-fragment pairs `nonbond`. Override any type with `ranges`.
"""
import numpy as np


class PairFeatures:
    def __init__(self, system, ranges=None, bond=1.5, nonbond=2.0, inter=4.0, mobile=2.0):
        self.system = system
        self.ranges = {}
        for t in sorted(set(system.pair_type)):
            _, loc, kind = t.split(":")
            default = inter if loc == "AB" else {"bond": bond, "mobile": mobile}.get(kind, nonbond)
            self.ranges[t] = float((ranges or {}).get(t, default))
        self.a = np.array([self.ranges[t] for t in system.pair_type])

    def __call__(self, r, order=1):
        """y[b, e], dy/dr and, for order 2, d2y/dr2."""
        y = np.exp(-r / self.a)
        return (y, -y / self.a) if order < 2 else (y, -y / self.a, y / self.a**2)

    def state(self, prefix="feat_"):
        keys = sorted(self.ranges)
        return {f"{prefix}types": np.array(keys), f"{prefix}ranges": np.array([self.ranges[k] for k in keys])}

    @classmethod
    def from_state(cls, s, system, prefix="feat_"):
        return cls(system, {str(k): float(v) for k, v in zip(s[f"{prefix}types"], s[f"{prefix}ranges"])})
