import numpy as np
import pytest

from conftest import ISOPRENE_O3
from src.system import System, nh3_ch

ISO = ISOPRENE_O3["symbols"]
BONDS = ISOPRENE_O3["bonds"]
FRAGS = (range(13), range(13, 16))


def _is_group(perms):
    s = {tuple(p) for p in perms}
    return all(tuple(np.asarray(a)[list(b)]) in s for a in s for b in s)


def test_nh3_ch_group_and_types():
    s = nh3_ch()
    assert s.group_order == 6 and _is_group(s.atom_perms)
    assert set(s.atom_perms[:, 0]) == {0} and set(s.atom_perms[:, 5]) == {5}
    assert len(set(s.pair_type)) == 7


def test_bonds_from_geometry_match_explicit():
    X = ISOPRENE_O3["X"]
    a = System(ISO, FRAGS, X_ref=X)
    assert set(a.bonds) == set(tuple(sorted(b)) for b in BONDS)


def test_isoprene_ozone_bond_group_is_48():
    s = System(ISO, FRAGS, bonds=BONDS)
    assert s.group_order == 48 and _is_group(s.atom_perms)
    assert (13, 15) in s.atom_orbits(1) and (10, 11, 12) in s.atom_orbits(0)
    types = np.array(s.pair_type)
    assert all((types[P] == types).all() for P in s.pair_perms)


def test_mobile_atoms_enlarge_group():
    s = System(ISO, FRAGS, bonds=BONDS, mobile=(5, 6, 8, 9))
    assert s.group_order == 288 and _is_group(s.atom_perms)


def test_full_group_is_refused_when_huge():
    with pytest.raises(ValueError, match="full group"):
        System(ISO, FRAGS, bonds=BONDS, group="full")
    assert System(ISO[13:], None, bonds=[(0, 1), (1, 2)], group="full").group_order == 6


def test_active_selection_closed_under_group():
    s = System(ISO, FRAGS, bonds=BONDS, active=(0, 1, 2, 3, 13, 14, 15))
    assert len(s.active_pairs) == 84
    for P in s.active_pair_perms:
        assert sorted(P) == list(range(84))
    with pytest.raises(ValueError, match="active"):
        System(ISO, FRAGS, bonds=BONDS, active=(10,))


def test_unimolecular_methane():
    s = System(["C", "H", "H", "H", "H"], bonds=[(0, 1), (0, 2), (0, 3), (0, 4)])
    assert s.n_fragments == 1 and s.group_order == 24


def test_state_roundtrip():
    s = System(ISO, FRAGS, bonds=BONDS, active=(0, 1, 2, 3, 13, 14, 15))
    t = System.from_state(s.state())
    assert t.pairs == s.pairs and np.array_equal(t.atom_perms, s.atom_perms)
    assert np.array_equal(t.active_pairs, s.active_pairs)
