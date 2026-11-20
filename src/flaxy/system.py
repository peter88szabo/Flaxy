"""The NH3 + CH system: atom order, fragments, pair order and the S3 group.

Atom order (fixed by the data files): 0 N, 1 H, 2 H, 3 H, 4 C, 5 H.
Capture only: the NH3 hydrogens never exchange with the CH hydrogen, so the
declared symmetry group is S3 acting on atoms 1, 2, 3 (|G| = 6).

Pairs are listed in tensor-train site order. Equivalent pairs (same type) are
adjacent and share their feature scales.
"""
import itertools

import numpy as np

from .units import MASS

SYMBOLS = ("N", "H", "H", "H", "C", "H")
N_ATOMS = 6
FRAG_A = (0, 1, 2, 3)  # NH3
FRAG_B = (4, 5)        # CH
MASSES = np.array([MASS[s] for s in SYMBOLS])

PAIRS = (
    (1, 2), (1, 3), (2, 3),   # HH in NH3
    (0, 1), (0, 2), (0, 3),   # NH in NH3
    (1, 4), (2, 4), (3, 4),   # H(NH3)...C
    (0, 4),                   # N...C
    (0, 5),                   # N...H(CH)
    (1, 5), (2, 5), (3, 5),   # H(NH3)...H(CH)
    (4, 5),                   # CH
)
PAIR_TYPE = ("HH_A",) * 3 + ("NH_A",) * 3 + ("HC",) * 3 + ("NC", "NH5") \
    + ("HH5",) * 3 + ("CH_B",)
N_PAIRS = len(PAIRS)
PAIR_I = np.array([p[0] for p in PAIRS])
PAIR_J = np.array([p[1] for p in PAIRS])
NH3_PAIRS = np.arange(6)        # sites 0..5 are internal NH3 distances
CH_PAIR = 14
INTER_PAIRS = np.arange(6, 14)

ATOM_PERMS = np.array([[0, *p, 4, 5] for p in itertools.permutations((1, 2, 3))])


def _pair_gather(perm):
    """Index map with y(P x) = y(x)[gather], where (P x)_i = x_{perm[i]}."""
    lookup = {p: e for e, p in enumerate(PAIRS)}
    return np.array([lookup[tuple(sorted((perm[i], perm[j])))] for i, j in PAIRS])


PAIR_PERMS = np.array([_pair_gather(p) for p in ATOM_PERMS])  # (6, 15)


def type_array(values):
    """Expand a {pair_type: value} mapping to a per-pair array in site order."""
    return np.array([float(values[t]) for t in PAIR_TYPE])
