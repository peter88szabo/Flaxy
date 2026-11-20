"""S3-projected kernel ridge regression: the uncompressed Gaussian-process reference.

f(x) = sum_t beta_t k_G(y(x), y(x_t)),  k_G(y, y') = |G|^-1 sum_P k(y, y'[gather_P]).

The model enters the surface as (1 - g(x)) f(x), so the ridge system uses the
operator S = diag(1 - g):  (S K S + lambda W^-1) alpha = r,  beta = S alpha.
Energies only. Kernels: Matern-5/2 (default) or Gaussian, both C2 at d = 0.
"""
import numpy as np
import scipy.linalg as sl

from .system import PAIR_PERMS

SQRT5 = np.sqrt(5.0)


def _sqdist(A, B):
    d2 = (A**2).sum(1)[:, None] + (B**2).sum(1)[None, :] - 2.0 * A @ B.T
    return np.maximum(d2, 0.0)


def _kernel(d2, length, kind):
    """Kernel value and q = -(dk/dy)/(y - y'), i.e. dk/dy = -q (y - y')."""
    if kind == "gauss":
        k = np.exp(-0.5 * d2 / length**2)
        return k, k / length**2
    if kind == "matern52":
        s = SQRT5 * np.sqrt(d2) / length
        e = np.exp(-s)
        return (1.0 + s + s * s / 3.0) * e, (5.0 / (3.0 * length**2)) * (1.0 + s) * e
    raise ValueError(f"unknown kernel {kind}")


class SymKRR:
    def __init__(self, length=0.3, ridge=1e-6, kind="matern52"):
        self.length, self.ridge, self.kind = float(length), float(ridge), kind
        self.centers = None  # (n, 15) features of training geometries
        self.beta = None
        self.y_scale = 1.0

    def gram(self, Y1, Y2, chunk=4096):
        out = np.zeros((len(Y1), len(Y2)))
        for P in PAIR_PERMS:
            Y2p = Y2[:, P]
            for s in range(0, len(Y1), chunk):
                k, _ = _kernel(_sqdist(Y1[s:s + chunk], Y2p), self.length, self.kind)
                out[s:s + chunk] += k
        return out / len(PAIR_PERMS)

    def fit(self, Y, target, weight, scale):
        """Y features, target residual (Hartree), row weights, scale = 1 - g > 0."""
        keep = scale > 0
        Y, target, weight, scale = Y[keep], target[keep], weight[keep], scale[keep]
        self.y_scale = float(np.std(target)) or 1.0
        K = self.gram(Y, Y)
        K *= scale[:, None] * scale[None, :]
        K[np.diag_indices_from(K)] += self.ridge / weight
        alpha = sl.cho_solve(sl.cho_factor(K, overwrite_a=True), target / self.y_scale)
        self.centers = Y
        self.beta = alpha * scale * self.y_scale
        return self

    def predict(self, Y, grad=False, chunk=2048):
        """f(Y) and, if grad, df/dy[b, e]."""
        f = np.zeros(len(Y))
        g = np.zeros_like(Y) if grad else None
        for P in PAIR_PERMS:
            C = self.centers[:, P]
            for s in range(0, len(Y), chunk):
                y = Y[s:s + chunk]
                k, q = _kernel(_sqdist(y, C), self.length, self.kind)
                f[s:s + chunk] += k @ self.beta
                if grad:
                    qb = q * self.beta[None, :]
                    g[s:s + chunk] -= qb.sum(1)[:, None] * y - qb @ C
        n = len(PAIR_PERMS)
        return (f / n, g / n) if grad else f / n

    def state(self, prefix="krr_"):
        return {f"{prefix}centers": self.centers, f"{prefix}beta": self.beta,
                f"{prefix}hyper": np.array([self.length, self.ridge]),
                f"{prefix}kind": np.array(self.kind)}

    @classmethod
    def from_state(cls, s, prefix="krr_"):
        length, ridge = s[f"{prefix}hyper"]
        obj = cls(length, ridge, str(s[f"{prefix}kind"]))
        obj.centers, obj.beta = s[f"{prefix}centers"], s[f"{prefix}beta"]
        return obj
