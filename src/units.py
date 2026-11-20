"""Physical constants (CODATA 2018) and nuclear masses used for fragment origins."""
BOHR_ANG = 0.529177210903
ANG_BOHR = 1.0 / BOHR_ANG
HARTREE_CM = 219474.6313705
HARTREE_KCAL = 627.5094740631
AMU_ME = 1822.888486209          # electron masses per dalton

# Most abundant isotopes (Da). Origins use these fixed weights for every isotopologue.
MASS = {"H": 1.00782503207, "B": 11.0093054, "C": 12.0, "N": 14.0030740048,
        "O": 15.99491461956, "F": 18.99840322, "Si": 27.9769265325, "P": 30.97376163,
        "S": 31.97207100, "Cl": 34.96885268}
