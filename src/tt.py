"""Gaussian tensor train over pair features, projected onto a permutation group, fitted by ALS.

f(y) = A_1(y_1) ... A_d(y_d),   A_j(y) = sum_k G_j[:, k, :] phi_jk(y),
phi_j0 = 1 (constant feature, so low-order couplings cost little rank) and
phi_jk = exp(-(y - c_jk)^2 / (2 w_jk^2)) for k >= 1, whitened per site.

The surface uses the projection f_G(y) = |G|^-1 sum_P f(y[gather_P]) times the
switch factor s(x). Fitting uses the two-site alternating scheme: merge cores
j, j+1, solve the linear least-squares problem for the merged core with ridge and a
proximal term, split it by an SVD in the data metric (adapts the bond rank), and
re-fit the neighbouring core. Analytic gradient and Hessian in y.
"""
import numpy as np
import scipy.linalg as sl


class GaussianTT:
    def __init__(self, pair_perms, pair_types, centers, widths, cores, transforms=None, y_scale=1.0):
        self.pair_perms = np.asarray(pair_perms)
        self.pair_types = tuple(str(t) for t in pair_types)
        self.centers = [np.asarray(c, dtype=float) for c in centers]
        self.widths = [np.asarray(w, dtype=float) for w in widths]
        self.cores = [None if G is None else np.asarray(G, dtype=float) for G in cores]
        # Per-site whitening W_j: features are [1, gaussians] @ W_j (see whiten()).
        self.transforms = ([np.eye(len(c) + 1) for c in self.centers] if transforms is None
                           else [np.asarray(W, dtype=float) for W in transforms])
        self.y_scale = float(y_scale)

    @property
    def n_sites(self):
        return len(self.centers)

    # ----------------------------------------------------------------- features
    def raw_phi(self, y, j, order=0):
        """Raw dictionary [1, gaussians] of site j (B, b+1) and its first/second y-derivatives."""
        u = (y[:, None] - self.centers[j][None, :]) / self.widths[j][None, :]
        gauss = np.exp(-0.5 * u * u)
        one, zero = np.ones((len(y), 1)), np.zeros((len(y), 1))
        out = [np.hstack([one, gauss])]
        if order >= 1:
            out.append(np.hstack([zero, -u * gauss / self.widths[j][None, :]]))
        if order >= 2:
            out.append(np.hstack([zero, (u * u - 1.0) * gauss / self.widths[j][None, :] ** 2]))
        return out if order else out[0]

    def phi(self, y, j, order=0):
        """Whitened features of site j (B, m_j) and, for order >= 1, their y-derivatives."""
        W = self.transforms[j]
        if order == 0:
            return self.raw_phi(y, j) @ W
        return [a @ W for a in self.raw_phi(y, j, order)]

    def whiten(self, Y, eig_tol=1e-10):
        """Set W_j = V L^-1/2 from the empirical Gram of each site's raw dictionary.

        Pools equivalent pairs (same type), since the projected model sees every pair
        of a type at every site of that type. Near-dependent directions are dropped.
        """
        types = np.array(self.pair_types)
        for j in range(self.n_sites):
            pooled = Y[:, types == self.pair_types[j]].ravel()
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
    def predict(self, Y, order=0, chunk=4096):
        """f(Y); order 1 adds df/dy[b, d]; order 2 adds d2f/dy2[b, d, d]."""
        step = chunk if order < 2 else 256
        parts = [self._predict_chunk(Y[s:s + step], order) for s in range(0, len(Y), step)]
        out = [np.concatenate([p[k] for p in parts]) * self.y_scale for k in range(order + 1)]
        return out[0] if order == 0 else tuple(out)

    def _predict_chunk(self, Y, order):
        B, d = len(Y), self.n_sites
        f = np.zeros(B)
        g = np.zeros_like(Y) if order >= 1 else None
        H = np.zeros((B, d, d)) if order == 2 else None
        for P in self.pair_perms:
            Yp = Y[:, P]
            feats = [self.phi(Yp[:, j], j, order) for j in range(d)]
            val = [ph if order == 0 else ph[0] for ph in feats]
            left = [np.ones((B, 1))]
            for j in range(d):
                left.append(_step(left[-1], val[j], self.cores[j]))
            f += left[-1][:, 0]
            if order == 0:
                continue
            right = [None] * d + [np.ones((B, 1))]          # right[j]: sites j..d-1
            for j in range(d - 1, -1, -1):
                right[j] = _step_right(val[j], self.cores[j], right[j + 1])
            for j in range(d):
                g[:, P[j]] += np.einsum("bc,bc->b", _step(left[j], feats[j][1], self.cores[j]), right[j + 1])
            if order == 2:
                for i in range(d):
                    H[:, P[i], P[i]] += np.einsum("bc,bc->b", _step(left[i], feats[i][2], self.cores[i]),
                                                  right[i + 1])
                    u = _step(left[i], feats[i][1], self.cores[i])
                    for j in range(i + 1, d):
                        hij = np.einsum("bc,bc->b", _step(u, feats[j][1], self.cores[j]), right[j + 1])
                        H[:, P[i], P[j]] += hij
                        H[:, P[j], P[i]] += hij
                        u = _step(u, val[j], self.cores[j])
        n = len(self.pair_perms)
        return (f / n,) + ((g / n,) if order >= 1 else ()) + ((H / n,) if order == 2 else ())

    # ------------------------------------------------------------------ fitting
    @classmethod
    def initialize(cls, Y, pair_perms, pair_types, n_gauss=6, width_factor=1.0, rank=2, noise=1e-2,
                   seed=0):
        """Quantile centres per pair type (pooled over equivalent pairs), near-constant cores."""
        rng = np.random.default_rng(seed)
        types = np.array(pair_types)
        centers, widths = [], []
        for j in range(Y.shape[1]):
            pooled = Y[:, types == pair_types[j]].ravel()
            q = np.quantile(pooled, (np.arange(n_gauss) + 0.5) / n_gauss)
            q = np.maximum.accumulate(q + 1e-9 * np.arange(n_gauss))
            gaps = np.diff(q)
            spacing = np.r_[gaps[0], 0.5 * (gaps[:-1] + gaps[1:]), gaps[-1]]
            centers.append(q)
            widths.append(width_factor * np.maximum(spacing, 1e-4))
        tt = cls(pair_perms, pair_types, centers, widths, [None] * Y.shape[1])
        tt.whiten(Y)
        d = tt.n_sites
        for j in range(d):
            rl = 1 if j == 0 else rank
            rr = 1 if j == d - 1 else rank
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
        d, NP = self.n_sites, len(self.pair_perms)
        phis = [[self.phi(Y[:, P[j]], j) for j in range(d)] for P in self.pair_perms]
        self._right_canonical()
        R_env = self._right_envs(phis)
        L_env = [np.ones((NP, len(Y), 1))] + [None] * (d - 1)

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
                sites = range(d - 1) if forward else range(d - 2, -1, -1)
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
        for row in phis:
            v = np.ones((len(s), 1))
            for j, ph in enumerate(row):
                v = _step(v, ph, self.cores[j])
            out += v[:, 0]
        return s * out / len(phis)

    def _right_canonical(self):
        for j in range(self.n_sites - 1, 0, -1):
            rl, b, rr = self.cores[j].shape
            q, t = np.linalg.qr(self.cores[j].reshape(rl, b * rr).T)
            self.cores[j] = q.T.reshape(-1, b, rr)
            self.cores[j - 1] = np.tensordot(self.cores[j - 1], t.T, axes=1)

    def _right_envs(self, phis):
        n, d = len(phis[0][0]), self.n_sites
        R = [None] * d
        R[d - 1] = np.ones((len(phis), n, 1))
        for j in range(d - 2, -1, -1):
            R[j] = _env_right(R[j + 1], phis, j + 1, self.cores[j + 1])
        return R

    # -------------------------------------------------------------------- state
    def state(self, prefix="tt_"):
        s = {f"{prefix}n": np.array(self.n_sites), f"{prefix}y_scale": np.array(self.y_scale),
             f"{prefix}perms": self.pair_perms, f"{prefix}types": np.array(self.pair_types)}
        for j in range(self.n_sites):
            s[f"{prefix}core{j}"] = self.cores[j]
            s[f"{prefix}centers{j}"] = self.centers[j]
            s[f"{prefix}widths{j}"] = self.widths[j]
            s[f"{prefix}whiten{j}"] = self.transforms[j]
        return s

    @classmethod
    def from_state(cls, s, prefix="tt_"):
        n = int(s[f"{prefix}n"])
        return cls(s[f"{prefix}perms"], [str(x) for x in s[f"{prefix}types"]],
                   [s[f"{prefix}centers{j}"] for j in range(n)],
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
    return np.stack([_step(L[p], phis[p][j], G) for p in range(len(phis))])


def _env_right(R, phis, j, G):
    return np.stack([_step_right(phis[p][j], G, R[p]) for p in range(len(phis))])


def _with_site_left(L, phis, j):
    """(|G|, n, rl) x site-j features -> (|G|, n, rl*b) row-wise Kronecker products."""
    return np.stack([(L[p][:, :, None] * phis[p][j][:, None, :]).reshape(len(L[p]), -1)
                     for p in range(len(phis))])


def _with_site_right(phis, j, R):
    """site-j features x (|G|, n, rr) -> (|G|, n, b*rr)."""
    return np.stack([(phis[p][j][:, :, None] * R[p][:, None, :]).reshape(len(R[p]), -1)
                     for p in range(len(phis))])


def _normal(left, right, r, w, s, chunk, gram=False):
    """Normal equations for the projected design D = s/|G| sum_P left_P (x) right_P.

    With gram=True also return the weighted Gram matrices of the left and right
    environment functions (metric for the truncation)."""
    NP, n, pd, qd = left.shape[0], left.shape[1], left.shape[2], right.shape[2]
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
