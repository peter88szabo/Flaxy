# Flaxy
Flaxy builds analytic potential energy surfaces (PES) for molecular collisions, from
a set of ab initio energies. The target application is classical trajectories at very
low collision energies, as in cold or astrochemical (~10 K) environments. There the
outcome is decided at large separations, so the long range must be physically correct
and smooth, while the short range must be accurate where the data are.

Under development. Pure NumPy/SciPy. Method notes: `notes/main_ideas.tex`. Fit reports: `reports/`.

## What it can do
- **Fit a PES from energies.** No gradients are required; forces could be added as extra rows later.
- **Analytic forces** everywhere, for MD or trajectory codes. Energies and forces are vectorized over batches of geometries.
- **Exact permutation invariance** under a declared group of identical-atom swaps: the energy is unchanged and the forces swap accordingly.
- **Physical long-range behaviour.**
  - The interaction between the two fragments is a multipole expansion: dipole–dipole R⁻³, dipole–quadrupole R⁻⁴, quadrupole–quadrupole R⁻⁵, and R⁻⁶ dispersion/induction with angular terms.
  - Its coefficients are fitted to the long-range data, with standard errors.
  - It extrapolates correctly beyond the data, e.g. to 30+ Å.
- **Flexible fragments.** Each molecule's own energy is a separate permutation-invariant fit, so vibrationally hot or distorted fragments are allowed at any separation.
- **Two interchangeable short-range models:**
  - a symmetrized kernel ridge regression, the accurate reference;
  - a Gaussian tensor train (TT), compact and fast to evaluate, fitted by two-site sweeps.
- **Diagnostics:**
  - errors by file, R shell and energy window;
  - radial cuts of the energy and radial force through the switch;
  - tests for forces, invariance and switch limits.

## How the PES is partitioned
The total energy is split by what each part is responsible for:

```
V(x) = d + U_A(q_A) + U_B(q_B)            isolated fragments (internal coordinates only)
         + L(R, orientation, q)           long-range interaction between the fragments
         + (1 − g(R)) · f(x)              short-range correction, switched off at long range
```

```
 R (fragment centre-of-mass distance) →
 |---- short range: data-driven f + baseline ----|-- switch --|---- long range: physics only ----→ ∞
                                               R_in         R_out
```

- **Fragments, `U_A` and `U_B`.**
  - These are polynomials in Morse variables exp(−r/a) of the internal distances, summed over permutation orbits.
  - They hold the energy of distorting each molecule, which can be far larger than the interaction.
  - Keeping them separate means a fragment-fit error never turns into a spurious force along R.
- **Long range, `L`.**
  - Shapes come from Cartesian interaction tensors contracted with fragment dipole and quadrupole tensors. These are built from bond vectors, so no body frames are needed.
  - Each term is damped (Tang–Toennies) so it stays finite at short range.
  - The coefficients are fitted on the long-range data. Together with `U_A + U_B` this forms the *baseline*, which is linear in all its parameters.
- **Switch, `g(R)`.**
  - A C² quintic step between R_in and R_out.
  - For R > R_out the surface is exactly the physical baseline, and the ML part cannot leak outward.
- **Short range, `f`.**
  - Learns the residual (E − baseline) only where 1 − g > 0: chemical bonding, exchange repulsion and corrections to the tail.
  - Inputs are the pair features exp(−r/a). It is averaged over the permutation group, which makes it exactly invariant.

### The tensor train (TT)
A TT, also called a matrix product state, writes a function of d coordinates
(here the pair features) as a chain of small matrices:
`f(y) = A_1(y_1) A_2(y_2) ··· A_d(y_d)`, with `A_j(y) = Σ_k G_j[:, k, :] φ_jk(y)`.

- **Basis.** φ_jk are 1-D Gaussians plus a constant.
- **Parameters.** The full expansion would need b^d coefficients (7^15 ≈ 5·10^12). The chain needs only about d·b·r², where the rank r sets how strongly coordinates couple.
- **Cost.** Evaluating f, and its gradient by left/right products, costs about d·b·r² operations per geometry, independent of the training-set size. A kernel model's cost grows with the number of training points.
- **Fitting.** Uses DMRG-like two-site sweeps:
  1. freeze all cores except two neighbours;
  2. solve the linear least-squares problem for their merged core;
  3. split it by an SVD in the data metric, which adapts the rank;
  4. move along the chain.

## Code map
| Module | Role |
|---|---|
| `system.py` | atoms, fragments, pair order, permutation group (system-specific) |
| `tail.py` | long-range multipole terms (fragment-specific shapes) |
| `monomer.py`, `baseline.py` | fragment polynomials; linear baseline fit (`fit`, `fit_two_stage`) |
| `switch.py` | quintic switching function |
| `features.py`, `krr.py`, `tt.py` | pair features; kernel model; tensor-train model |
| `surface.py` | assembled PES: `energy`, `energy_forces`, `save`/`load` |
| `data.py`, `weights.py`, `report.py` | xyz input and splits; fit domain and weights; error tables |
| `dual.py`, `geometry.py`, `units.py` | forward-mode derivatives, distances and origins, constants |

To fit a new two-fragment system, adapt `system.py` and `tail.py`; the rest is generic.

## Usage
- Conventions:
  - internal units are bohr and Hartree;
  - forces are −dV/dX in Hartree/bohr;
  - geometries are `X[batch, atoms, 3]` in the atom order of `system.py`;
  - xyz input is in Ångström.
- Fitting steps:
  1. Load the data and make the split (`data.py`).
  2. Choose the energy domain and weights (`weights.py`).
  3. Fit the baseline.
  4. Fit `f` on `Surface.inner_residual_target(X, E)`.
  5. Assemble the `Surface`.
- `scripts/fit_capture.py` runs these steps end to end, and `scripts/plot_cuts.py` plots radial cuts.
- Using a fitted surface:

```python
from flaxy.surface import Surface
pes = Surface.load("model.npz")
E, F = pes.energy_forces(X_bohr)         # Hartree, Hartree/bohr
E, F = pes.energy_forces_angstrom(X_A)   # same, positions in Angstrom
```

```bash
venv                    # ~/.venvs/science: numpy, scipy, pytest, matplotlib
python -m pytest -q     # forces vs finite differences, invariance, switch limits, I/O
```
`src/main.py` is the standalone synthetic reference fixture from the notes.
