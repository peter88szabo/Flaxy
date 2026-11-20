"""C2 quintic step used to hand the short-range model over to the baseline."""
import numpy as np


def smoothstep(t):
    """h(t) = t^3 (10 - 15 t + 6 t^2) on [0, 1], 0 below and 1 above; returns h, h', h''."""
    t = np.clip(t, 0.0, 1.0)
    h = t**3 * (10.0 + t * (-15.0 + 6.0 * t))
    dh = 30.0 * t**2 * (1.0 - t) ** 2
    d2h = 60.0 * t * (1.0 - t) * (1.0 - 2.0 * t)
    return h, dh, d2h


def rising(R, r_in, r_out):
    """g(R) = h((R - r_in)/(r_out - r_in)) with dg/dR and d2g/dR2."""
    width = r_out - r_in
    h, dh, d2h = smoothstep((R - r_in) / width)
    return h, dh / width, d2h / width**2
