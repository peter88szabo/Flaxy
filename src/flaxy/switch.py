"""C2 quintic step used to hand the short-range model over to the baseline."""
import numpy as np


def smoothstep(t):
    """h(t) = t^3 (10 - 15 t + 6 t^2) on [0, 1], 0 below and 1 above; returns (h, h')."""
    t = np.clip(t, 0.0, 1.0)
    h = t**3 * (10.0 + t * (-15.0 + 6.0 * t))
    dh = 30.0 * t**2 * (1.0 - t) ** 2
    return h, dh


def rising(R, r_in, r_out):
    """g(R) = h((R - r_in)/(r_out - r_in)) and dg/dR."""
    width = r_out - r_in
    h, dh = smoothstep((R - r_in) / width)
    return h, dh / width
