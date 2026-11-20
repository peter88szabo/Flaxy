"""Atoms, fragments, pair order and permutation group of a molecular system.

One fragment (fragments=None): any molecule or complex, V = baseline + f_G.
Two fragments: bimolecular capture surfaces with a physical long-range tail.

The permutation group is generated, not hand-written:

group="bonds" (default)
    element-preserving permutations that map the bond graph onto itself, i.e. the
    feasible permutations of a fixed-connectivity domain (for C5H8 + O3: 48 instead
    of 5!*8!*3! = 29 030 400). Bonds of `mobile` atoms are ignored, so a mobile atom
    may swap with any mobile atom of the same element in its fragment.
group="full"
    all permutations of identical nuclei within each fragment.

With two fragments only fragment-preserving permutations are generated: the
surface has a single arrangement channel (fragment A + fragment B), so atoms never
change fragment.
Interchange of identical fragments and inter-fragment exchange need the
multi-channel switch of notes/main_ideas.tex and are not supported yet.

`active` atoms (reaction centre) select the pair distances seen by the short-range
model: every pair with at least one active atom. The active set must be a union
of atom orbits so that the selection is closed under the group.
"""
import itertools
import json
import math

import numpy as np

from .units import MASS

COVALENT_RADIUS = {"H": 0.31, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
                   "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02}  # Angstrom


def detect_bonds(symbols, X_ang, scale=1.25):
    """Bonds where the distance is below scale x (sum of covalent radii)."""
    X = np.asarray(X_ang, dtype=float)
    return [(i, j) for i, j in itertools.combinations(range(len(symbols)), 2)
            if np.linalg.norm(X[i] - X[j]) < scale * (COVALENT_RADIUS[symbols[i]] + COVALENT_RADIUS[symbols[j]])]


class System:
    def __init__(self, symbols, fragments=None, bonds=None, X_ref=None, mobile=(), active=None,
                 group="bonds", pairs=None, max_group=20000, bond_scale=1.25):
        self.symbols = tuple(symbols)
        n = self.n_atoms = len(self.symbols)
        self.masses = np.array([MASS[s] for s in self.symbols])
        if fragments is None:
            self.frag_a, self.frag_b = tuple(range(n)), ()
        else:
            self.frag_a, self.frag_b = (tuple(sorted(int(i) for i in f)) for f in fragments)
            if sorted(self.frag_a + self.frag_b) != list(range(n)) or not self.frag_a or not self.frag_b:
                raise ValueError("fragments must partition all atoms into two non-empty sets")
        self.n_fragments = 1 if not self.frag_b else 2
        self.fragment_of = np.array([0 if i in self.frag_a else 1 for i in range(n)])
        self.mobile = tuple(sorted(int(i) for i in mobile))
        if bonds is None:
            if X_ref is None:
                raise ValueError("give bonds or a reference geometry X_ref (Angstrom)")
            bonds = detect_bonds(self.symbols, X_ref, bond_scale)
        self.bonds = tuple(sorted(tuple(sorted((int(i), int(j)))) for i, j in bonds))
        if any(self.fragment_of[i] != self.fragment_of[j] for i, j in self.bonds):
            raise ValueError("a bond connects the two fragments; check fragments or bonds")
        self.group = group

        perms = _bond_automorphisms(self, max_group) if group == "bonds" else _full_group(self, max_group)
        identity = tuple(range(n))
        perms = [identity] + sorted(p for p in set(perms) if p != identity)
        self.atom_perms = np.array(perms)

        self.pairs = tuple(tuple(sorted(p)) for p in (pairs or self._default_pairs()))
        if sorted(self.pairs) != list(itertools.combinations(range(n), 2)):
            raise ValueError("pairs must list every unordered atom pair exactly once")
        self.n_pairs = len(self.pairs)
        self.pair_i = np.array([p[0] for p in self.pairs])
        self.pair_j = np.array([p[1] for p in self.pairs])
        self.pair_type = tuple(self._pair_type(i, j) for i, j in self.pairs)
        self.pair_perms = np.array([self._pair_gather(p) for p in self.atom_perms])
        types = np.array(self.pair_type)
        if any((types[P] != types).any() for P in self.pair_perms):
            raise ValueError("pair types are not invariant under the group")

        self.active = tuple(range(n)) if active is None else tuple(sorted(int(i) for i in active))
        act = set(self.active)
        if any(set(p[list(self.active)]) != act for p in self.atom_perms):
            raise ValueError("active atoms must be a union of atom orbits (closed under the group)")
        self.active_pairs = np.array([e for e, (i, j) in enumerate(self.pairs) if i in act or j in act])
        local = {int(e): k for k, e in enumerate(self.active_pairs)}
        self.active_pair_perms = np.array([[local[int(P[e])] for e in self.active_pairs]
                                           for P in self.pair_perms])
        self.active_pair_type = tuple(self.pair_type[e] for e in self.active_pairs)

    # ------------------------------------------------------------------ queries
    @property
    def group_order(self):
        return len(self.atom_perms)

    def fragment(self, which):
        return self.frag_a if which in (0, "A") else self.frag_b

    def fragment_pairs(self, which):
        """Indices of the pairs inside fragment A (0) or B (1)."""
        f = 0 if which in (0, "A") else 1
        return np.array([e for e, (i, j) in enumerate(self.pairs)
                         if self.fragment_of[i] == f and self.fragment_of[j] == f], dtype=int)

    def atom_orbits(self, which):
        """Orbits of the atoms of one fragment under the group, in atom order."""
        seen, orbits = set(), []
        for i in self.fragment(which):
            if i not in seen:
                orbit = sorted({int(p[i]) for p in self.atom_perms})
                seen.update(orbit)
                orbits.append(tuple(orbit))
        return orbits

    def summary(self):
        frags = f"fragments {self.frag_a} + {self.frag_b}" if self.n_fragments == 2 else "one fragment"
        return (f"{''.join(self.symbols)}: {self.n_atoms} atoms, {frags}, "
                f"group '{self.group}' of order {self.group_order}, {self.n_pairs} pairs "
                f"({len(self.active_pairs)} active), {len(set(self.pair_type))} pair types")

    # ---------------------------------------------------------------- internals
    def _pair_gather(self, perm):
        """Index map with y(P x) = y(x)[gather], where (P x)_i = x_{perm[i]}."""
        lookup = {p: e for e, p in enumerate(self.pairs)}
        return np.array([lookup[tuple(sorted((int(perm[i]), int(perm[j]))))] for i, j in self.pairs])

    def _pair_type(self, i, j):
        a, b = sorted((self.symbols[i], self.symbols[j]))
        if self.n_fragments == 1:
            loc = "M"
        else:
            loc = "AB" if self.fragment_of[i] != self.fragment_of[j] else "AB"[self.fragment_of[i]]
        if self.group != "bonds":
            kind = "any"
        elif i in self.mobile or j in self.mobile:
            kind = "mobile"
        else:
            kind = "bond" if (i, j) in set(self.bonds) else "nonbond"
        return f"{a}{b}:{loc}:{kind}"

    def _default_pairs(self):
        """Intra-A, inter-fragment, intra-B; equivalent pairs adjacent (tensor-train order)."""
        all_pairs = list(itertools.combinations(range(self.n_atoms), 2))
        rank = {"M": 0, "A": 0, "AB": 1, "B": 2}
        return sorted(all_pairs, key=lambda p: (rank[self._pair_type(*p).split(":")[1]],
                                                self._pair_type(*p), p))

    # -------------------------------------------------------------------- state
    def state(self, prefix="sys_"):
        spec = {"symbols": self.symbols, "frag_a": self.frag_a, "frag_b": self.frag_b,
                "bonds": self.bonds, "mobile": self.mobile, "active": self.active,
                "group": self.group, "pairs": self.pairs}
        return {f"{prefix}spec": np.array(json.dumps(spec)), f"{prefix}atom_perms": self.atom_perms}

    @classmethod
    def from_state(cls, s, prefix="sys_"):
        spec = json.loads(str(s[f"{prefix}spec"]))
        frags = (spec["frag_a"], spec["frag_b"]) if spec["frag_b"] else None
        obj = cls(spec["symbols"], frags, bonds=spec["bonds"],
                  mobile=spec["mobile"], active=spec["active"], group=spec["group"],
                  pairs=[tuple(p) for p in spec["pairs"]])
        if not np.array_equal(obj.atom_perms, s[f"{prefix}atom_perms"]):
            raise ValueError("stored permutation group differs from the regenerated one")
        return obj


def _colour(system, i):
    return (system.symbols[i], int(system.fragment_of[i]), i in system.mobile)


def _bond_automorphisms(system, max_group):
    """All colour-preserving automorphisms of the bond graph without mobile-atom bonds."""
    n = system.n_atoms
    adj = [set() for _ in range(n)]
    for i, j in system.bonds:
        if i not in system.mobile and j not in system.mobile:
            adj[i].add(j)
            adj[j].add(i)
    # Refined colours (two rounds of neighbour colouring) prune the search.
    col = [_colour(system, i) for i in range(n)]
    for _ in range(2):
        col = [(col[i], tuple(sorted(col[j] for j in adj[i]))) for i in range(n)]
    candidates = [[c for c in range(n) if col[c] == col[i]] for i in range(n)]
    out = []

    def extend(m, used):
        i = len(m)
        if i == n:
            out.append(tuple(m))
            if len(out) > max_group:
                raise ValueError(f"group larger than max_group={max_group}; declare fewer mobile "
                                 "atoms or use permutation-invariant input features")
            return
        for c in candidates[i]:
            if c in used:
                continue
            if all((m[j] in adj[c]) == (j in adj[i]) for j in range(i)):
                used.add(c)
                extend(m + [c], used)
                used.discard(c)

    extend([], set())
    return out


def _full_group(system, max_group):
    """Product of symmetric groups over identical atoms within each fragment."""
    classes = {}
    for i in range(system.n_atoms):
        classes.setdefault((system.symbols[i], int(system.fragment_of[i])), []).append(i)
    order = math.prod(math.factorial(len(c)) for c in classes.values())
    if order > max_group:
        raise ValueError(f"full group has order {order:,} > max_group={max_group}; use group='bonds'")
    out = []
    for choice in itertools.product(*[itertools.permutations(c) for c in classes.values()]):
        perm = list(range(system.n_atoms))
        for c, image in zip(classes.values(), choice):
            for src, dst in zip(c, image):
                perm[src] = dst
        out.append(tuple(perm))
    return out


def nh3_ch(active=None):
    """NH3 + CH, atom order N H H H C H, pair order of the first exploratory fits."""
    pairs = ((1, 2), (1, 3), (2, 3), (0, 1), (0, 2), (0, 3), (1, 4), (2, 4), (3, 4),
             (0, 4), (0, 5), (1, 5), (2, 5), (3, 5), (4, 5))
    bonds = ((0, 1), (0, 2), (0, 3), (4, 5))
    return System(("N", "H", "H", "H", "C", "H"), ((0, 1, 2, 3), (4, 5)), bonds=bonds,
                  pairs=pairs, active=active)
