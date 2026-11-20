"""Vectorized first-order forward-mode duals.

A Dual holds values v[b] and exact derivatives g[b, n] with respect to n seed
variables (here the 18 Cartesian coordinates). Used for the small pieces of the
model (tail, origins) where hand-written derivatives would be error prone.
"""
import numpy as np
from scipy.special import gammainc, gammaln


def _col(a):
    a = np.asarray(a, dtype=float)
    return a[:, None] if a.ndim == 1 else a


class Dual:
    __slots__ = ("v", "g")
    __array_priority__ = 1000

    def __init__(self, v, g):
        self.v, self.g = v, g

    def __add__(self, o):
        if isinstance(o, Dual):
            return Dual(self.v + o.v, self.g + o.g)
        return Dual(self.v + o, self.g)

    __radd__ = __add__

    def __neg__(self):
        return Dual(-self.v, -self.g)

    def __sub__(self, o):
        return self + (-o)

    def __rsub__(self, o):
        return (-self) + o

    def __mul__(self, o):
        if isinstance(o, Dual):
            return Dual(self.v * o.v, self.g * _col(o.v) + o.g * _col(self.v))
        return Dual(self.v * o, self.g * _col(o))

    __rmul__ = __mul__

    def __truediv__(self, o):
        if isinstance(o, Dual):
            return self * o.reciprocal()
        return self * (1.0 / np.asarray(o, dtype=float))

    def __rtruediv__(self, o):
        return self.reciprocal() * o

    def unary(self, f, fp):
        return Dual(f, self.g * _col(fp))

    def reciprocal(self):
        return self.unary(1.0 / self.v, -1.0 / self.v**2)

    def __pow__(self, p):
        return self.unary(self.v**p, p * self.v**(p - 1))

    def sqrt(self):
        s = np.sqrt(self.v)
        return self.unary(s, 0.5 / s)

    def exp(self):
        e = np.exp(self.v)
        return self.unary(e, e)


def seed(X):
    """Turn X[b, atoms, 3] into a nested list [atom][k] of Duals over all coordinates."""
    b, n, _ = X.shape
    eye = np.eye(3 * n)
    return [[Dual(X[:, i, k].copy(), np.broadcast_to(eye[3 * i + k], (b, 3 * n)))
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
    """Tang-Toennies damped f_n(beta R) / R^n, with f_n = P(n+1, beta R)."""
    x = beta * R.v
    f = gammainc(n + 1, x)
    fp = np.exp(n * np.log(x) - x - gammaln(n + 1))
    val = f / R.v**n
    der = beta * fp / R.v**n - n * f / R.v**(n + 1)
    return R.unary(val, der)
