"""Group-projected kernel ridge regression: the uncompressed reference model.

f(y) = sum_t beta_t k_G(y, y_t),  k_G(y, y') = |G|^-1 sum_P k(y, y'[gather_P]).

`pair_perms` (|G| x n_features) are the feature gathers of the permutation group.
The model enters the surface as s(x) f(y(x)) with s = 1 - g (bimolecular) or 1,
so the ridge system uses S = diag(s):  (S K S + lambda W^-1) alpha = r, beta = S alpha.
Kernels: Matern-5/2 (default) or Gaussian; analytic gradient and Hessian in y.
"""
import numpy as np
import scipy.linalg as sl

SQRT5 = np.sqrt(5.0)


def _sqdist(A, B):
    return np.maximum((A**2).sum(1)[:, None] + (B**2).sum(1)[None, :] - 2.0 * A @ B.T, 0.0)


def _kernel(d2, length, kind):
    """k, q and c with dk/dy = -q (y - y') and d2k/dy2 = -q I + c (y - y')(y - y')^T."""
    if kind == "gauss":
        k = np.exp(-0.5 * d2 / length**2)
        return k, k / length**2, k / length**4
    if kind == "matern52":
        s = SQRT5 * np.sqrt(d2) / length
        e = np.exp(-s)
        return (1.0 + s + s * s / 3.0) * e, (5.0 / (3.0 * length**2)) * (1.0 + s) * e, \
            (25.0 / (3.0 * length**4)) * e
    raise ValueError(f"unknown kernel {kind}")


class SymKRR:
    def __init__(self, pair_perms, length=0.8, ridge=1e-6, kind="matern52"):
        self.pair_perms = np.asarray(pair_perms)
        self.length, self.ridge, self.kind = float(length), float(ridge), kind
        self.centers = None
        self.beta = None

    def gram(self, Y1, Y2, chunk=4096):
        out = np.zeros((len(Y1), len(Y2)))
        for P in self.pair_perms:
            Y2p = Y2[:, P]
            for s in range(0, len(Y1), chunk):
                out[s:s + chunk] += _kernel(_sqdist(Y1[s:s + chunk], Y2p), self.length, self.kind)[0]
        return out / len(self.pair_perms)

    def fit(self, Y, target, weight, scale):
        """Y features, target residual (Hartree), row weights, scale = s(x) > 0."""
        keep = scale > 0
        Y, target, weight, scale = Y[keep], target[keep], weight[keep], scale[keep]
        y_scale = float(np.std(target)) or 1.0
        K = self.gram(Y, Y)
        K *= scale[:, None] * scale[None, :]
        K[np.diag_indices_from(K)] += self.ridge / weight
        alpha = sl.cho_solve(sl.cho_factor(K, overwrite_a=True), target / y_scale)
        self.centers = Y
        self.beta = alpha * scale * y_scale
        return self

    def predict(self, Y, order=0, chunk=1024):
        """f(Y); order 1 adds df/dy[b, d]; order 2 adds d2f/dy2[b, d, d]."""
        n = len(self.pair_perms)
        f = np.zeros(len(Y))
        g = np.zeros_like(Y) if order >= 1 else None
        H = np.zeros((len(Y), Y.shape[1], Y.shape[1])) if order == 2 else None
        step = chunk if order < 2 else 32
        for P in self.pair_perms:
            C = self.centers[:, P]
            for s in range(0, len(Y), step):
                y = Y[s:s + step]
                k, q, c = _kernel(_sqdist(y, C), self.length, self.kind)
                f[s:s + step] += k @ self.beta
                if order >= 1:
                    qb = q * self.beta[None, :]
                    g[s:s + step] -= qb.sum(1)[:, None] * y - qb @ C
                if order == 2:
                    diff = y[:, None, :] - C[None, :, :]
                    H[s:s + step] += (np.einsum("bt,bti,btj->bij", c * self.beta[None, :], diff, diff)
                                      - qb.sum(1)[:, None, None] * np.eye(Y.shape[1])[None])
        out = (f / n,)
        if order >= 1:
            out += (g / n,)
        if order == 2:
            out += (H / n,)
        return out[0] if order == 0 else out

    def state(self, prefix="krr_"):
        return {f"{prefix}centers": self.centers, f"{prefix}beta": self.beta,
                f"{prefix}hyper": np.array([self.length, self.ridge]),
                f"{prefix}kind": np.array(self.kind), f"{prefix}perms": self.pair_perms}

    @classmethod
    def from_state(cls, s, prefix="krr_"):
        length, ridge = s[f"{prefix}hyper"]
        obj = cls(s[f"{prefix}perms"], length, ridge, str(s[f"{prefix}kind"]))
        obj.centers, obj.beta = s[f"{prefix}centers"], s[f"{prefix}beta"]
        return obj
