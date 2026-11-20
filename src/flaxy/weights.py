"""Fitting domain and training weights.

E_rel is the reference energy relative to the NH3 + CH dissociation limit,
estimated as the lowest energy among the R >= 9 Angstrom training geometries.

Domain (user decision): low-energy physics of the NH3 ... CH channel only.
Geometries more than 200 kJ/mol above the dissociation limit are not fitted
(they stay in the validation split and are reported separately as extrapolation).

Inside the domain, E_rel spans about
  -18 .. 0 kcal/mol   H3N-CH adduct well and attractive entrance (10 K capture)
    0 .. 20           thermally distorted reactants (300 K sampling)
   20 .. 47.8         hot monomers (1500 K sampling) and short-range repulsion
"""
import numpy as np

KJ_PER_KCAL = 4.184
ENERGY_CUTOFF_KJ = 200.0


def in_domain(E_rel_kcal, cutoff_kj=ENERGY_CUTOFF_KJ):
    return np.asarray(E_rel_kcal) < cutoff_kj / KJ_PER_KCAL


def energy_weights(E_rel_kcal, temperature):
    """Return one positive weight per in-domain geometry.

    TODO(human): choose the in-domain weighting. PROVISIONAL body below
    (Bowman-type, dE = 10 kcal/mol, above the asymptote only) so the pipeline runs.
    """
    excess = np.maximum(np.asarray(E_rel_kcal, dtype=float), 0.0)
    return 10.0 / (10.0 + excess)
