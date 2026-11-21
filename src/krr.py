"""Kernel ridge regression on permutation-invariant (PIP) input features.

f(Z) = sum_t beta_t k(Z, Z_t). The inputs are already invariant, so the model is
evaluated once per geometry. The model enters the surface as s(x) f(Z(x)) with
s = 1 - g (two fragments) or 1, so the ridge system uses S = diag(s):
(S K S + lambda W^-1) alpha = r, beta = S alpha.
Kernels: Matern-5/2 (default) or Gaussian; analytic gradient and Hessian in Z.
"""
import numpy as np
import scipy.linalg as sl

SQRT5 = np.sqrt(5.0)


def _sqdist(A, B):
    return np.maximum((A**2).sum(1)[:, None] + (B**2).sum(1)[None, :] - 2.0 * A @ B.T, 0.0)


def _kernel(d2, length, kind):
    """k, q and c with dk/dz = -q (z - z') and d2k/dz2 = -q I + c (z - z')(z - z')^T."""
    if kind == "gauss":
        k = np.exp(-0.5 * d2 / length**2)
        return k, k / length**2, k / length**4
    if kind == "matern52":
        s = SQRT5 * np.sqrt(d2) / length
        e = np.exp(-s)
        return (1.0 + s + s * s / 3.0) * e, (5.0 / (3.0 * length**2)) * (1.0 + s) * e, \
            (25.0 / (3.0 * length**4)) * e
    raise ValueError(f"unknown kernel {kind}")


class KRR:
    def __init__(self, length=3.0, ridge=1e-6, kind="matern52"):
        self.length, self.ridge, self.kind = float(length), float(ridge), kind
        self.centers = None
        self.beta = None

    def fit(self, Z, target, weight, scale, chunk=4096):
        """Z inputs, target residual (Hartree), row weights, scale = s(x) > 0."""
        keep = scale > 0
        Z, target, weight, scale = Z[keep], target[keep], weight[keep], scale[keep]
        y_scale = float(np.std(target)) or 1.0
        K = np.empty((len(Z), len(Z)))
        for s in range(0, len(Z), chunk):
            K[s:s + chunk] = _kernel(_sqdist(Z[s:s + chunk], Z), self.length, self.kind)[0]
        K *= scale[:, None] * scale[None, :]
        K[np.diag_indices_from(K)] += self.ridge / weight
        alpha = sl.cho_solve(sl.cho_factor(K, overwrite_a=True), target / y_scale)
        self.centers = Z
        self.beta = alpha * scale * y_scale
        return self

    def predict(self, Z, order=0, chunk=1024):
        """f(Z); order 1 adds df/dZ[b, n]; order 2 adds d2f/dZ2[b, n, n]."""
        f = np.zeros(len(Z))
        g = np.zeros_like(Z) if order >= 1 else None
        H = np.zeros((len(Z), Z.shape[1], Z.shape[1])) if order == 2 else None
        C = self.centers
        step = chunk if order < 2 else 16
        for s in range(0, len(Z), step):
            z = Z[s:s + step]
            k, q, c = _kernel(_sqdist(z, C), self.length, self.kind)
            f[s:s + step] = k @ self.beta
            if order >= 1:
                qb = q * self.beta[None, :]
                g[s:s + step] = qb @ C - qb.sum(1)[:, None] * z
            if order == 2:
                diff = z[:, None, :] - C[None, :, :]
                H[s:s + step] = (np.einsum("bt,bti,btj->bij", c * self.beta[None, :], diff, diff, optimize=True)
                                 - qb.sum(1)[:, None, None] * np.eye(Z.shape[1])[None])
        return f if order == 0 else ((f, g) if order == 1 else (f, g, H))

    def state(self, prefix="krr_"):
        return {f"{prefix}centers": self.centers, f"{prefix}beta": self.beta,
                f"{prefix}hyper": np.array([self.length, self.ridge]), f"{prefix}kind": np.array(self.kind)}

    @classmethod
    def from_state(cls, s, prefix="krr_"):
        length, ridge = s[f"{prefix}hyper"]
        obj = cls(length, ridge, str(s[f"{prefix}kind"]))
        obj.centers, obj.beta = s[f"{prefix}centers"], s[f"{prefix}beta"]
        return obj
