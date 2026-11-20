"""Pair features y_e = exp(-r_e / a_e) with type-tied ranges (bohr)."""
import numpy as np

from .system import PAIR_TYPE, type_array

DEFAULT_RANGES = {"HH_A": 2.0, "NH_A": 1.5, "CH_B": 1.5,
                  "HC": 3.0, "NC": 3.0, "NH5": 3.0, "HH5": 3.0}


class PairFeatures:
    def __init__(self, ranges=None):
        self.ranges = dict(DEFAULT_RANGES if ranges is None else ranges)
        missing = set(PAIR_TYPE) - set(self.ranges)
        if missing:
            raise ValueError(f"missing feature ranges for {sorted(missing)}")
        self.a = type_array(self.ranges)

    def __call__(self, r):
        """Return y[b, e] and dy/dr[b, e]."""
        y = np.exp(-r / self.a)
        return y, -y / self.a

    def state(self, prefix="feat_"):
        return {f"{prefix}types": np.array(sorted(self.ranges)),
                f"{prefix}ranges": np.array([self.ranges[k] for k in sorted(self.ranges)])}

    @classmethod
    def from_state(cls, s, prefix="feat_"):
        return cls({str(k): float(v) for k, v in zip(s[f"{prefix}types"], s[f"{prefix}ranges"])})
