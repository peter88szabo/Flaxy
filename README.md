# Flaxy
Kernel / tensor-train PES fitting with physical long-range tails (under development).

Current target: NH3 + CH capture (no H transfer), `data_to_fit/*.xyz`, label `E_Bps`.
Method notes: `notes/main_ideas.tex`; design and findings:
`docs/superpowers/specs/2026-11-20-nh3-ch-capture-fit-design.md`.

## Model
V = d + U_NH3 + U_CH + L_damp(R, orientation) + (1 - g(R)) f_S3(pair features)

- `U_NH3`, `U_CH`: permutation-invariant polynomials in Morse variables.
- `L_damp`: multipole-shaped long-range tail (R^-3 ... R^-6), Tang-Toennies damped,
  coefficients fitted to the 300 K data at R >= 9 Angstrom.
- `g`: C2 switch in the centre-of-mass distance (default 5 -> 7 Angstrom); beyond
  R_out the surface is exactly the physical baseline.
- `f_S3`: short-range model, exactly invariant under the NH3 hydrogen permutations:
  `krr` (S3-kernel ridge reference) or `tt` (Gaussian tensor train).
- Analytic forces everywhere; fitted only below 200 kJ/mol above the NH3 + CH limit.

## Use
```bash
venv                                   # ~/.venvs/science (numpy, scipy, pytest, matplotlib)
python -m pytest -q                    # derivative, invariance, gate and I/O tests
python scripts/fit_capture.py --out runs/my_fit --models krr tt
python scripts/plot_cuts.py runs/my_fit/surface_krr.npz runs/my_fit/surface_tt.npz
```
```python
from flaxy.surface import Surface
pes = Surface.load("runs/my_fit/surface_krr.npz")
E, F = pes.energy_forces(X_bohr)        # X[b, 6, 3], atom order N H H H C H; Hartree, Hartree/bohr
E, F = pes.energy_forces_angstrom(X_A)  # same, positions in Angstrom
```
`src/main.py` is the synthetic H3O reference fixture from the notes.
