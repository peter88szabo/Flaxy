"""Vectorized forward-mode duals of first or second order.

A Dual holds values v[b], exact gradients g[b, n] and, for second order, exact
Hessians h[b, n, n] with respect to n seed variables (the Cartesian coordinates).
Used for the small pieces of the model (tail, origins) where hand-written
derivatives would be error prone. Second order costs O(n^2) per operation, so it
is meant for small batches (Hessians on demand).
"""
import numpy as np
from scipy.special import gammainc, gammaln


def _c2(a):
    a = np.asarray(a, dtype=float)
    return a[:, None] if a.ndim == 1 else a


def _c3(a):
    a = np.asarray(a, dtype=float)
    return a[:, None, None] if a.ndim == 1 else a


def _outer(a, b):
    return a[:, :, None] * b[:, None, :]


class Dual:
    __slots__ = ("v", "g", "h")
    __array_priority__ = 1000

    def __init__(self, v, g, h=None):
        self.v, self.g, self.h = v, g, h

    def __add__(self, o):
        if isinstance(o, Dual):
            return Dual(self.v + o.v, self.g + o.g, None if self.h is None else self.h + o.h)
        return Dual(self.v + o, self.g, self.h)

    __radd__ = __add__

    def __neg__(self):
        return Dual(-self.v, -self.g, None if self.h is None else -self.h)

    def __sub__(self, o):
        return self + (-o)

    def __rsub__(self, o):
        return (-self) + o

    def __mul__(self, o):
        if isinstance(o, Dual):
            g = self.g * _c2(o.v) + o.g * _c2(self.v)
            h = None
            if self.h is not None:
                cross = _outer(self.g, o.g)
                h = self.h * _c3(o.v) + o.h * _c3(self.v) + cross + cross.transpose(0, 2, 1)
            return Dual(self.v * o.v, g, h)
        return Dual(self.v * o, self.g * _c2(o), None if self.h is None else self.h * _c3(o))

    __rmul__ = __mul__

    def __truediv__(self, o):
        if isinstance(o, Dual):
            return self * o.reciprocal()
        return self * (1.0 / np.asarray(o, dtype=float))

    def __rtruediv__(self, o):
        return self.reciprocal() * o

    def unary(self, f, fp, fpp):
        """Chain rule for a scalar function with values f, f', f'' at self.v."""
        h = None
        if self.h is not None:
            h = self.h * _c3(fp) + _c3(fpp) * _outer(self.g, self.g)
        return Dual(f, self.g * _c2(fp), h)

    def reciprocal(self):
        v = self.v
        return self.unary(1.0 / v, -1.0 / v**2, 2.0 / v**3)

    def __pow__(self, p):
        v = self.v
        return self.unary(v**p, p * v**(p - 1), p * (p - 1) * v**(p - 2))

    def sqrt(self):
        s = np.sqrt(self.v)
        return self.unary(s, 0.5 / s, -0.25 / (s * self.v))

    def exp(self):
        e = np.exp(self.v)
        return self.unary(e, e, e)


def seed(X, order=1):
    """Turn X[b, atoms, 3] into a nested list [atom][k] of Duals over all coordinates."""
    b, n, _ = X.shape
    eye = np.eye(3 * n)
    zero = np.broadcast_to(np.zeros((1, 3 * n, 3 * n)), (b, 3 * n, 3 * n)) if order == 2 else None
    return [[Dual(X[:, i, k].copy(), np.broadcast_to(eye[3 * i + k], (b, 3 * n)), zero)
             for k in range(3)] for i in range(n)]


def vadd(a, b):
    return [x + y for x, y in zip(a, b)]


def vsub(a, b):
    return [x - y for x, y in zip(a, b)]


def vscale(a, s):
    return [x * s for x in a]


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def matvec(M, v):
    return [dot(row, v) for row in M]


def outer_traceless(vectors, scale=1.0):
    """sum_v (v v^T - |v|^2 I / 3) * scale, as a 3x3 nested list."""
    Q = [[None] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(i, 3):
            acc = None
            for v in vectors:
                term = v[i] * v[j]
                if i == j:
                    term = term - dot(v, v) * (1.0 / 3.0)
                acc = term if acc is None else acc + term
            Q[i][j] = Q[j][i] = acc * scale
    return Q


def contract(A, B):
    """Full contraction A:B of two 3x3 nested lists."""
    acc = None
    for i in range(3):
        for j in range(3):
            t = A[i][j] * B[i][j]
            acc = t if acc is None else acc + t
    return acc


def damped_inverse_power(R, n, beta):
    """Tang-Toennies damped u(R) = f_n(beta R) / R^n, f_n = P(n+1, beta R), with u', u''."""
    r = R.v
    x = beta * r
    F = gammainc(n + 1, x)
    p = np.exp(n * np.log(x) - x - gammaln(n + 1))   # x^n e^-x / n!
    F1 = beta * p
    F2 = beta**2 * p * (n / x - 1.0)
    u = F / r**n
    u1 = F1 / r**n - n * F / r**(n + 1)
    u2 = F2 / r**n - 2 * n * F1 / r**(n + 1) + n * (n + 1) * F / r**(n + 2)
    return R.unary(u, u1, u2)
