# Flaxy
Flaxy builds analytic potential energy surfaces (PES) from ab initio energies, for any
fixed set of nuclei: a single molecule or complex, or two colliding fragments.

The main application is bimolecular capture surfaces for classical trajectories at very
low collision energies, as in cold or astrochemical (~10 K) environments. There the
outcome is decided at large separations, so the long range must be physically correct
and smooth, while the short range must be accurate where the data are.

Under development. Pure NumPy/SciPy. Method notes: `notes/main_ideas.tex`. Fit reports: `reports/`.

## What it can do
- **Fit a PES from energies.** No gradients are required.
- **Analytic derivatives.**
  - Forces and full Cartesian Hessians are exact analytic derivatives of one scalar energy.
  - Energies and forces are vectorized over batches of geometries.
  - Hessians are computed on demand, e.g. for harmonic frequencies (`analysis.py`).
- **Exact permutation invariance**, with the group generated automatically from the bond graph (see below):
  - the energy is unchanged under the group;
  - forces and Hessians transform accordingly.
- **Any system.** One fragment: a molecule, complex or reactive system. Two fragments: a capture surface with fragment energies and a physical long-range tail.
- **Physical long range** (two fragments):
  - The interaction is a multipole expansion: dipole–dipole R⁻³, dipole–quadrupole R⁻⁴, quadrupole–quadrupole R⁻⁵, and R⁻⁶ dispersion/induction with angular terms.
  - Its coefficients are fitted to the long-range data, with standard errors.
  - It extrapolates correctly beyond the data.
- **Two interchangeable short-range models:**
  - a group-symmetrized kernel ridge regression, the accurate reference;
  - a Gaussian tensor train (TT), compact and fast to evaluate.
- **Diagnostics:**
  - error tables by file, R shell and energy window;
  - radial cuts through the switch;
  - tests for forces, Hessians, invariance and switch limits.

## How the PES is partitioned
```
one fragment:   V(x) = U(x) +            f_G(y_active(x))
two fragments:  V(x) = U(x) + (1 − g(R)) f_G(y_active(x))

U = d (+ invariant polynomial of the molecule)                     one fragment
U = d + U_A(q_A) + U_B(q_B) + L(R, orientation, q)                 two fragments
```

```
 R (fragment centre-of-mass distance) →
 |---- short range: data-driven f + baseline ----|-- switch --|---- long range: physics only ----→ ∞
                                               R_in         R_out
```

- **Fragment energies `U_A`, `U_B`** (or the whole molecule's energy).
  - Polynomials in Morse variables exp(−r/a) of the internal distances, summed over permutation orbits.
  - For two fragments, they hold the energy of distorting each molecule, so a fragment-fit error never turns into a spurious force along R.
- **Long range `L`** (two fragments).
  - Shapes come from Cartesian interaction tensors contracted with per-orbit dipole and quadrupole tensors of each fragment (equivalent atoms share one effective charge). No body frames are needed.
  - Each term is Tang–Toennies damped. Its coefficients are fitted with `U_A + U_B` in one linear least-squares problem.
- **Switch `g(R)`** (two fragments).
  - A C² quintic step between R_in and R_out.
  - Beyond R_out, V is exactly the physical baseline.
- **Short range `f_G`.**
  - Learns the residual E − U.
  - Inputs are the active pair features exp(−r/a), averaged over the permutation group G, which makes it exactly invariant.

## Permutation groups for larger systems
`System(symbols, fragments=None, bonds=None | X_ref=..., mobile=(), active=None, group="bonds")`

- **`group="bonds"`** (default). All element-preserving permutations that keep the bond graph. These are the feasible permutations of a fixed-connectivity domain.
  - Bonds are given explicitly, or detected from a reference geometry via covalent radii.
  - Example, C5H8O3 as isoprene + O₃: 48 permutations instead of 5!·8!·3! = 29,030,400.
  - Averaging over the group stays cheap.
- **`mobile`.** Atoms that may change bonding partners, e.g. migrating H. Their bonds are ignored, so they permute freely among mobile atoms of the same element. In the isoprene example, making the C1/C4 hydrogens mobile gives 288.

**What the bond constraint keeps and what it removes.** The group is not "permutations of the mobile atoms only". It contains two kinds of permutations:

1. **Equivalent positions in the fixed bond graph.** These need no mobile atoms. For isoprene + O₃ with nothing mobile the group is already 48:

   | Equivalent atoms | Permutations |
   |---|---:|
   | 3 methyl H's | 3! = 6 |
   | 2 H's of each =CH₂ | 2 × 2 |
   | 2 terminal O's of O₃ | 2 |

   These swaps keep every bond, so the surface must be invariant under them.
2. **Mobile atoms.** Their bonds are ignored, so they may also swap with any mobile atom of the same element. This enlarges the group, e.g. 48 → 288.

**What is removed.** Permutations that would require breaking bonds of non-mobile atoms, e.g. a methyl H with the C3–H hydrogen, or C1 with C4. These are unreachable inside the fitted region (the feasible-permutation idea of Longuet-Higgins). That is what cuts 29,030,400 down to 48.

**Caveat.** Declaring atoms non-mobile is a promise about where trajectories can go. If an H can migrate during the dynamics but is not marked mobile, the surface is not required to treat its old and new positions alike, and it will be wrong once it gets there.
- **`active`.** The reaction centre.
  - The short-range model sees only the pairs with at least one active atom, e.g. 84 of 120 pairs for isoprene C1–C4 + O₃.
  - Fewer input dimensions means less data needed.
  - The set must be closed under the group; this is checked.
- **`group="full"`.** All identical-nuclei permutations within each fragment. It refuses groups beyond `max_group`.
- **Restriction with two fragments.** Only fragment-preserving permutations are used, i.e. a single arrangement channel. Exchange of atoms between fragments needs the multi-channel switch of the notes.

`nh3_ch()` is the NH₃ + CH system used in the first fit.

### The tensor train (TT)
A TT, also called a matrix product state, writes a function of d coordinates
(the active pair features) as a chain of small matrices:
`f(y) = A_1(y_1) A_2(y_2) ··· A_d(y_d)`, with `A_j(y) = Σ_k G_j[:, k, :] φ_jk(y)`.

- **Basis.** φ_jk are 1-D Gaussians plus a constant, whitened per site.
- **Parameters.** The full expansion would need b^d coefficients (7^15 ≈ 5·10^12). The chain needs only about d·b·r², where the rank r sets how strongly coordinates couple.
- **Cost.** Energy, gradient and Hessian come from left/right matrix products. They cost about d·b·r² operations per geometry (d²·b·r² for the Hessian), independent of the training-set size.
- **Fitting.** Uses DMRG-like two-site sweeps:
  1. freeze all cores except two neighbours;
  2. solve the linear least-squares problem for their merged core;
  3. split it by an SVD in the data metric, which adapts the rank;
  4. re-fit the neighbouring core;
  5. move along the chain.

## Code map (`src/`)
| Module | Role |
|---|---|
| `system.py` | `System`: atoms, fragments, bond graph, generated permutation group, pair order/types, active pairs |
| `features.py` | pair features exp(−r/a), ranges tied by pair type |
| `monomer.py`, `tail.py`, `baseline.py` | invariant polynomials; multipole tail; reference energy U and its linear fit |
| `switch.py` | quintic switching function |
| `krr.py`, `tt.py` | short-range models: group-projected kernel ridge regression, Gaussian tensor train |
| `surface.py` | assembled PES: `energy`, `energy_forces`, `energy_forces_hessian`, `save`/`load` |
| `analysis.py` | harmonic frequencies (mass-weighted, translations/rotations projected out) |
| `data.py`, `weights.py`, `report.py` | xyz input and splits; fit domain and weights; error tables |
| `dual.py`, `geometry.py`, `units.py` | forward-mode derivatives (1st/2nd order), chain rules, constants |

## Usage
- Conventions:
  - internal units are bohr and Hartree;
  - forces are −dV/dX in Hartree/bohr;
  - Hessians are in Hartree/bohr²;
  - geometries are `X[batch, atoms, 3]` in the atom order of the `System`;
  - xyz input is in Ångström.

Run from the repository root; modules import as `src.*`.

```python
from src.system import System
from src.data import load_dataset
from src.features import PairFeatures
from src.geometry import pair_vectors
from src.baseline import Baseline
from src.krr import SymKRR
from src.surface import Surface

# 1. System: symbols in file order; bonds from a reference geometry (Angstrom).
system = System(symbols, fragments=(frag_A, frag_B), X_ref=X_eq_ang, active=reaction_centre)
ds = load_dataset("data_dir", system, key="E_Bps")      # every *.xyz in data_dir; X in bohr, E in Hartree

# 2. Reference energy U (two fragments: fragment polynomials + long-range tail).
feats = PairFeatures(system)
base = Baseline.build(system, ds.X, feats, degrees=(6, 6))
base.fit(ds.X[long_range], ds.E[long_range])             # or fit_two_stage(...)

# 3. Short-range model on the residual (one fragment: omit r_in, r_out).
pes = Surface(system, base, feats, None, r_in=5.0 / 0.529177, r_out=7.0 / 0.529177)
target, s = pes.inner_residual_target(ds.X, ds.E)
_, r = pair_vectors(ds.X, system)
Y = feats(r)[0][:, system.active_pairs]
pes.inner = SymKRR(system.active_pair_perms, length=0.8).fit(Y, target, weights, s)

# 4. Use.
E, F = pes.energy_forces(X)              # Hartree, Hartree/bohr
E, F, H = pes.energy_forces_hessian(X)   # + Hessian, Hartree/bohr^2
pes.save("model.npz");  pes = Surface.load("model.npz")
```

**Data format.** Extended xyz:
- one frame per geometry;
- the comment line starts with a frame name and carries `key=value` fields, e.g. `000047_PES_sample_48 E_Bps=-94.73 status=PASS`;
- the atom order must equal `System.symbols`.

**Scripts:**
- `scripts/fit_capture.py` runs the whole NH₃ + CH capture fit: data, fixed validation split, energy window, two-stage baseline, kernel model, tensor train, error tables and saved surfaces.
- `scripts/plot_cuts.py` plots rigid radial cuts of the energy and radial force through the switch.

```bash
venv                    # ~/.venvs/science: numpy, scipy, pytest, matplotlib
python -m pytest -q     # 27 tests: derivatives vs finite differences, invariance, groups, switch limits, I/O
```

## Status
First application: NH₃ + CH capture, 13,422 MRCC energies. Details in `reports/2026-11-20_nh3-ch_capture_exploratory_fit.md`.

**Validated**
- **Long range.** The dipole–dipole coefficient is −0.333 ± 0.004 a.u., against −μ(NH₃)μ(CH) ≈ −0.33 from experimental dipoles.
- **Kernel model.** 223 cm⁻¹ (0.64 kcal/mol) on validation geometries within 200 kJ/mol of the dissociation limit.
- **Analytic Hessian.** CH stretch 2857.1 cm⁻¹ on the fitted surface, against ω_e = 2858.5; the degenerate NH₃ modes come out exactly degenerate.

**Open**
- **Tensor train.** It trails the kernel model with about 5k energies.
- **Large fragments.** A big fragment's own energy (e.g. isoprene, 78 distances) needs a kernel model; the polynomial only works at low degree.
- **Inter-fragment exchange** needs the multi-channel switch of the notes.
- **Unsampled compressed geometries** can be too soft; a repulsive wall in U is planned.
