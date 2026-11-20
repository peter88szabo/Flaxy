"""Gaussian tensor train over the 15 pair features, S3-projected, fitted by ALS.

f(y) = A_1(y_1) ... A_d(y_d),   A_j(y) = sum_k G_j[:, k, :] phi_jk(y),
phi_j0 = 1 (constant feature, so low-order couplings cost little rank) and
phi_jk = exp(-(y - c_jk)^2 / (2 w_jk^2)) for k >= 1.

The surface uses the projection f_G(y) = |G|^-1 sum_P f(y[gather_P]) times the
gate operator (1 - g). Fitting uses the two-site alternating scheme: merge
cores j, j+1, solve the linear least-squares problem for the merged core with
ridge (full coefficient-tensor norm in the mixed-canonical gauge) and a proximal
term, then split by truncated SVD, which adapts the bond rank.
"""
import numpy as np
import scipy.linalg as sl

from .system import N_PAIRS, PAIR_PERMS, PAIR_TYPE

NP = len(PAIR_PERMS)


class GaussianTT:
    def __init__(self, centers, widths, cores, transforms=None, y_scale=1.0):
        self.centers = [np.asarray(c, dtype=float) for c in centers]
        self.widths = [np.asarray(w, dtype=float) for w in widths]
        self.cores = [np.asarray(G, dtype=float) for G in cores]
        # Per-site whitening W_j: features are [1, gaussians] @ W_j (see whiten()).
        self.transforms = ([np.eye(len(c) + 1) for c in self.centers] if transforms is None
                           else [np.asarray(W, dtype=float) for W in transforms])
        self.y_scale = float(y_scale)

    # ----------------------------------------------------------------- features
    def raw_phi(self, y, j, deriv=False):
        """(B, b+1) raw dictionary [1, gaussians] of site j and optional y-derivatives."""
        u = (y[:, None] - self.centers[j][None, :]) / self.widths[j][None, :]
        gauss = np.exp(-0.5 * u * u)
        val = np.hstack([np.ones((len(y), 1)), gauss])
        if not deriv:
            return val
        der = np.hstack([np.zeros((len(y), 1)), -u * gauss / self.widths[j][None, :]])
        return val, der

    def phi(self, y, j, deriv=False):
        """Whitened features of site j (B, m_j) and, optionally, their y-derivatives."""
        W = self.transforms[j]
        if not deriv:
            return self.raw_phi(y, j) @ W
        val, der = self.raw_phi(y, j, deriv=True)
        return val @ W, der @ W

    def whiten(self, Y, eig_tol=1e-10):
        """Set W_j = V L^-1/2 from the empirical Gram of each site's raw dictionary.

        Pools equivalent pairs, since the projected model sees every pair of a type
        at every site of that type. Near-dependent directions are dropped.
        """
        types = np.array(PAIR_TYPE)
        for j in range(N_PAIRS):
            pooled = Y[:, types == PAIR_TYPE[j]].ravel()
            Phi = self.raw_phi(pooled, j)
            lam, V = np.linalg.eigh(Phi.T @ Phi / len(pooled))
            keep = lam > eig_tol * lam.max()
            self.transforms[j] = V[:, keep] / np.sqrt(lam[keep])[None, :]

    def constant_coefficients(self, j):
        """Coefficients a with phi_j(y) @ a = 1 for all y."""
        e0 = np.zeros(len(self.centers[j]) + 1)
        e0[0] = 1.0
        return np.linalg.lstsq(self.transforms[j], e0, rcond=None)[0]

    @property
    def ranks(self):
        return [G.shape[2] for G in self.cores[:-1]]

    @property
    def n_params(self):
        return int(sum(G.size for G in self.cores))

    # --------------------------------------------------------------- evaluation
    def predict(self, Y, grad=False, chunk=4096):
        f = np.zeros(len(Y))
        g = np.zeros_like(Y) if grad else None
        for s in range(0, len(Y), chunk):
            fs, gs = self._predict_chunk(Y[s:s + chunk], grad)
            f[s:s + chunk] = fs
            if grad:
                g[s:s + chunk] = gs
        f *= self.y_scale
        if grad:
            g *= self.y_scale
        return (f, g) if grad else f

    def _predict_chunk(self, Y, grad):
        B = len(Y)
        f = np.zeros(B)
        g = np.zeros_like(Y) if grad else None
        for P in PAIR_PERMS:
            Yp = Y[:, P]
            left = [np.ones((B, 1))]
            feats = []
            for j, G in enumerate(self.cores):
                ph = self.phi(Yp[:, j], j, deriv=grad)
                feats.append(ph)
                v = ph[0] if grad else ph
                left.append(_step(left[-1], v, G))
            f += left[-1][:, 0]
            if grad:
                right = np.ones((B, 1))
                for j in range(N_PAIRS - 1, -1, -1):
                    G = self.cores[j]
                    v, dv = feats[j]
                    dmid = _step(left[j], dv, G)          # (B, r_j)
                    g[:, P[j]] += np.einsum("bc,bc->b", dmid, right)
                    right = _step_right(v, G, right)
        return f / NP, (g / NP if grad else None)

    # ------------------------------------------------------------------ fitting
    @classmethod
    def initialize(cls, Y, n_gauss=6, width_factor=1.0, rank=2, noise=1e-2, seed=0):
        """Quantile centres per pair type (pooled over equivalent pairs)."""
        rng = np.random.default_rng(seed)
        centers, widths = [], []
        types = np.array(PAIR_TYPE)
        for j in range(N_PAIRS):
            pooled = Y[:, types == PAIR_TYPE[j]].ravel()
            q = np.quantile(pooled, (np.arange(n_gauss) + 0.5) / n_gauss)
            q = np.maximum.accumulate(q + 1e-9 * np.arange(n_gauss))
            gaps = np.diff(q)
            spacing = np.r_[gaps[0], 0.5 * (gaps[:-1] + gaps[1:]), gaps[-1]]
            centers.append(q)
            widths.append(width_factor * np.maximum(spacing, 1e-4))
        tt = cls(centers, widths, [None] * N_PAIRS)
        tt.whiten(Y)
        # Start near the constant function: pass the constant through every bond.
        for j in range(N_PAIRS):
            rl = 1 if j == 0 else rank
            rr = 1 if j == N_PAIRS - 1 else rank
            const = tt.constant_coefficients(j)
            G = noise * rng.normal(size=(rl, len(const), rr))
            for a in range(min(rl, rr)):
                G[a, :, a] += const
            tt.cores[j] = G
        return tt

    def fit(self, Y, target, weight, scale, Y_val=None, target_val=None, weight_val=None,
            scale_val=None, sweeps=6, rank_cap=8, svd_tol=1e-8, ridge=1e-8, prox=1e-6,
            chunk=1024, accept_tol=1e-3, log=print):
        """Two-site ALS. Rows with scale = 1 - g = 0 carry no information and are dropped.

        An update is kept only if the weighted training loss does not rise by more
        than accept_tol (relative); truncation at the rank cap can otherwise undo
        progress. Returns a history of (half-sweep, scores, ranks).
        """
        keep = scale > 0
        Y, r, w, s = Y[keep], target[keep], weight[keep], scale[keep]
        self.y_scale = float(np.std(r)) or 1.0
        r = r / self.y_scale
        w = w / w.mean()
        phis = [[self.phi(Y[:, P[j]], j) for j in range(N_PAIRS)] for P in PAIR_PERMS]
        self._right_canonical()
        R_env = self._right_envs(phis)
        L_env = [np.ones((NP, len(Y), 1))] + [None] * (N_PAIRS - 1)

        def score():
            out = {"train": _rmse(self._fitted(phis, s), r, w) * self.y_scale}
            if Y_val is not None:
                out["valid"] = _rmse(scale_val * self.predict(Y_val), target_val, weight_val)
            return out

        def loss():
            return float(np.sum(w * (self._fitted(phis, s) - r) ** 2) / np.sum(w))

        best = (np.inf, [G.copy() for G in self.cores])
        history = []
        current = loss()
        for sweep in range(sweeps):
            for forward in (True, False):
                sites = range(N_PAIRS - 1) if forward else range(N_PAIRS - 2, -1, -1)
                rejected = 0
                for j in sites:
                    old = (self.cores[j].copy(), self.cores[j + 1].copy())
                    rl, b1 = L_env[j].shape[2], phis[0][j].shape[1]
                    b2, rr = phis[0][j + 1].shape[1], R_env[j + 1].shape[2]
                    left = _with_site_left(L_env[j], phis, j)            # (NP, n, rl*b1)
                    right = _with_site_right(phis, j + 1, R_env[j + 1])  # (NP, n, b2*rr)
                    K_old = np.tensordot(self.cores[j], self.cores[j + 1], axes=1)
                    H, rhs, ML, MR = _normal(left, right, r, w, s, chunk, gram=True)
                    K = _ridge_solve(H, rhs, K_old.ravel(), ridge, prox)
                    A, S, Bt = _metric_svd(K.reshape(rl * b1, b2 * rr), ML, MR, rank_cap, svd_tol)
                    k = len(S)
                    if forward:
                        # Keep the data-orthonormal left factor, then re-fit core j+1.
                        self.cores[j] = A.reshape(rl, b1, k)
                        L_env[j + 1] = _env_left(L_env[j], phis, j, self.cores[j])
                        H1, rhs1, _, _ = _normal(L_env[j + 1], right, r, w, s, chunk)
                        guess = (S[:, None] * Bt).ravel()
                        self.cores[j + 1] = _ridge_solve(H1, rhs1, guess, ridge, prox).reshape(k, b2, rr)
                    else:
                        self.cores[j + 1] = Bt.reshape(k, b2, rr)
                        R_env[j] = _env_right(R_env[j + 1], phis, j + 1, self.cores[j + 1])
                        H0, rhs0, _, _ = _normal(left, R_env[j], r, w, s, chunk)
                        guess = (A * S[None, :]).ravel()
                        self.cores[j] = _ridge_solve(H0, rhs0, guess, ridge, prox).reshape(rl, b1, k)
                    new = loss()
                    if new > current * (1.0 + accept_tol):
                        rejected += 1
                        self.cores[j], self.cores[j + 1] = old
                        if forward:
                            L_env[j + 1] = _env_left(L_env[j], phis, j, self.cores[j])
                        else:
                            R_env[j] = _env_right(R_env[j + 1], phis, j + 1, self.cores[j + 1])
                    else:
                        current = new
                sc = score()
                sc["rejected"] = rejected
                history.append((sweep + 0.5 * (not forward) + 0.5, sc, list(self.ranks)))
                key = sc.get("valid", sc["train"])
                if key < best[0]:
                    best = (key, [G.copy() for G in self.cores])
                log(f"  TT sweep {sweep + 1}{'>' if forward else '<'} train {sc['train']:.3e}"
                    f" valid {sc.get('valid', float('nan')):.3e} Eh  ranks {self.ranks}"
                    f"  rejected {rejected}")
        self.cores = best[1]
        return history

    def _fitted(self, phis, s):
        out = np.zeros(len(s))
        for p in range(NP):
            v = np.ones((len(s), 1))
            for j in range(N_PAIRS):
                v = _step(v, phis[p][j], self.cores[j])
            out += v[:, 0]
        return s * out / NP

    def _right_canonical(self):
        for j in range(N_PAIRS - 1, 0, -1):
            rl, b, rr = self.cores[j].shape
            q, t = np.linalg.qr(self.cores[j].reshape(rl, b * rr).T)
            self.cores[j] = q.T.reshape(-1, b, rr)
            self.cores[j - 1] = np.tensordot(self.cores[j - 1], t.T, axes=1)

    def _right_envs(self, phis):
        n = len(phis[0][0])
        R = [None] * N_PAIRS
        R[N_PAIRS - 1] = np.ones((NP, n, 1))
        for j in range(N_PAIRS - 2, -1, -1):
            R[j] = _env_right(R[j + 1], phis, j + 1, self.cores[j + 1])
        return R

    # -------------------------------------------------------------------- state
    def state(self, prefix="tt_"):
        s = {f"{prefix}n": np.array(N_PAIRS), f"{prefix}y_scale": np.array(self.y_scale)}
        for j in range(N_PAIRS):
            s[f"{prefix}core{j}"] = self.cores[j]
            s[f"{prefix}centers{j}"] = self.centers[j]
            s[f"{prefix}widths{j}"] = self.widths[j]
            s[f"{prefix}whiten{j}"] = self.transforms[j]
        return s

    @classmethod
    def from_state(cls, s, prefix="tt_"):
        n = int(s[f"{prefix}n"])
        return cls([s[f"{prefix}centers{j}"] for j in range(n)],
                   [s[f"{prefix}widths{j}"] for j in range(n)],
                   [s[f"{prefix}core{j}"] for j in range(n)],
                   [s[f"{prefix}whiten{j}"] for j in range(n)], float(s[f"{prefix}y_scale"]))


def _step(left, phi, G):
    """left (B, rl), phi (B, b), G (rl, b, rr) -> (B, rr)."""
    rl, b, rr = G.shape
    return (left[:, :, None] * phi[:, None, :]).reshape(len(left), rl * b) @ G.reshape(rl * b, rr)


def _step_right(phi, G, right):
    """phi (B, b), G (rl, b, rr), right (B, rr) -> (B, rl)."""
    rl, b, rr = G.shape
    t = right @ G.transpose(2, 1, 0).reshape(rr, b * rl)   # (B, b*rl)
    return np.einsum("bk,bka->ba", phi, t.reshape(len(phi), b, rl))


def _env_left(L, phis, j, G):
    return np.stack([_step(L[p], phis[p][j], G) for p in range(NP)])


def _env_right(R, phis, j, G):
    return np.stack([_step_right(phis[p][j], G, R[p]) for p in range(NP)])


def _with_site_left(L, phis, j):
    """(NP, n, rl) x site-j features -> (NP, n, rl*b) row-wise Kronecker products."""
    return np.stack([(L[p][:, :, None] * phis[p][j][:, None, :]).reshape(len(L[p]), -1)
                     for p in range(NP)])


def _with_site_right(phis, j, R):
    """site-j features x (NP, n, rr) -> (NP, n, b*rr)."""
    return np.stack([(phis[p][j][:, :, None] * R[p][:, None, :]).reshape(len(R[p]), -1)
                     for p in range(NP)])


def _normal(left, right, r, w, s, chunk, gram=False):
    """Normal equations for the projected design D = s/|G| sum_P left_P (x) right_P.

    With gram=True also return the weighted Gram matrices of the left and right
    environment functions (metric for the truncation)."""
    n, pd, qd = left.shape[1], left.shape[2], right.shape[2]
    H = np.zeros((pd * qd, pd * qd))
    rhs = np.zeros(pd * qd)
    ML, MR = np.zeros((pd, pd)), np.zeros((qd, qd))
    for a in range(0, n, chunk):
        c = slice(a, a + chunk)
        D = np.zeros((len(r[c]), pd * qd))
        for p in range(NP):
            D += (left[p, c][:, :, None] * right[p, c][:, None, :]).reshape(len(D), -1)
            if gram:
                ws = (w[c] * s[c] ** 2)[:, None]
                ML += left[p, c].T @ (ws * left[p, c])
                MR += right[p, c].T @ (ws * right[p, c])
        D *= (s[c] / NP)[:, None]
        Dw = D * w[c, None]
        H += D.T @ Dw
        rhs += Dw.T @ r[c]
    norm = NP * np.sum(w * s**2)
    return H, rhs, ML / norm, MR / norm


def _ridge_solve(H, rhs, guess, ridge, prox):
    """(H + (ridge + prox) m I) k = rhs + prox m guess, m = mean diag(H)."""
    m = np.mean(np.diag(H))
    H = H.copy()
    H[np.diag_indices_from(H)] += (ridge + prox) * m
    return sl.cho_solve(sl.cho_factor(H, overwrite_a=True), rhs + prox * m * guess)


def _psd_sqrt(M, eps=1e-12):
    lam, V = np.linalg.eigh(M)
    keep = lam > eps * lam.max()
    V, lam = V[:, keep], lam[keep]
    return (V * np.sqrt(lam)) @ V.T, (V / np.sqrt(lam)) @ V.T


def _metric_svd(K, ML, MR, rank_cap, tol):
    """Truncated SVD of K in the data metric: K ~ A diag(S) Bt with A, Bt^T
    orthonormal w.r.t. ML, MR. Discards the smallest directions whose combined
    norm is below tol of the total (in function norm), capped at rank_cap."""
    sL, iL = _psd_sqrt(ML)
    sR, iR = _psd_sqrt(MR)
    U, S, Vt = sl.svd(sL @ K @ sR, full_matrices=False)
    tail = np.sqrt(np.cumsum((S**2)[::-1])[::-1])  # tail[k] = norm of S[k:]
    cut = tol * np.linalg.norm(S)
    rank = next((k for k in range(1, len(S)) if tail[k] <= cut), len(S))
    rank = max(1, min(rank, rank_cap))
    return iL @ U[:, :rank], S[:rank], Vt[:rank] @ iR


def _rmse(pred, target, weight):
    return float(np.sqrt(np.sum(weight * (pred - target) ** 2) / np.sum(weight)))
