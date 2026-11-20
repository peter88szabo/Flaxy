# NH3 + CH capture PES — exploratory Flaxy fit (design)

## Intent
Test whether the Flaxy construction (notes/main_ideas.tex) works globally on a real
reactive system and measure the accuracy it reaches. Target: the NH3 + CH(X²Π)
entrance/capture region only (no H transfer), for later 10 K trajectories launched
from long range. Exploratory: data-only tail, no extra electronic-structure runs.

## Data
`data_to_fit/*.xyz`, 13 422 geometries, Å / Hartree, atom order N H H H C H.
Label = `E_Bps` (MRCC hybrid method). Four files: R_COM ∈ [1.5, 4.5] Å and
[4.5, 12] Å at 300 K and 1500 K normal-mode sampling, 15 fixed R shells per file,
each geometry an independent orientation × monomer sample (no scans).
No forces. Split: 300 validation geometries per file, stratified by R shell
(20 per shell), fixed seed, manifest saved. Hyperparameters are tuned on an
internal split of the training set, never on validation.

## Goals / non-goals
- Goals: analytic forces; exact invariance under the S3 permutations of the NH3
  hydrogens; smooth, physically shaped long range; errors reported by file,
  R shell and energy window; kernel reference vs tensor-train comparison.
- Non-goals (now): Hessians/HVP at runtime, H-transfer channels (S4), multistate
  ²Π treatment, physical monomer properties from new ab initio runs, trajectories.
- Anti-goal: a tail that fits 9–12 Å but has a wrong R⁻³ coefficient, or any
  short-range model leaking beyond R_out.

## Model
V(x) = d + U_NH3(q_A) + U_CH(r_CH) + L_damp(R, Ω, q) + (1 − g(x)) · (P_S3 f)(x)

- U_NH3: S3-invariant polynomial (orbit sums of monomials) in Morse variables of
  the 6 NH3 distances. U_CH: polynomial in a Morse variable of r_CH.
- L_damp: multipole-shaped basis, Tang–Toennies damped, built from covariant
  bond vectors t_A = Σ_i (X_Hi − X_N), t_B = X_H5 − X_C (no planar singularity):
  R⁻³ dipole–dipole, R⁻⁴ dipole–quadrupole (both ways), R⁻⁵ quad–quad,
  R⁻⁶ isotropic + two P2-type anisotropies. Coefficients b fitted.
- Joint linear fit of (d, U_NH3, U_CH, b) on training points with R ≥ 9 Å
  (user choice), weighted least squares; coefficient standard errors reported;
  R⁻³ coefficient checked against μ(NH3)μ(CH) ≈ 0.33 a.u. (experimental dipoles).
- Gate g = h_{R_in,R_out}(R) · v_A · v_B (quintic C² step; monomer validity
  factors set outside the sampled monomer range). Defaults R_in = 7 Å, R_out = 9 Å.
- f: (1) KRR reference, Matérn-5/2 on y_e = exp(−r_e/a_type), S3-summed kernel,
  gate operator (1−g) in the design, energy-only; (2) Gaussian tensor train over
  the 15 pair features, Gaussian dictionaries + constant feature per site,
  S3-projected design, one-site ALS then two-site DMRG with SVD rank adaptation,
  ridge + proximal term, best-validation checkpoint (internal split).

## Approaches considered
1. Spec build order (chosen): monomers + tail → KRR reference → TT; KRR gives the
   accuracy ceiling for the features, TT gap = compression cost.
2. TT only: faster, but no yardstick.
3. One global fit without channel split: rejected, long range would be extrapolated.

## Builder's calls
Python = `~/.venvs/science` (alias `venv`; numpy 2.4, scipy 1.17, pytest), package in
`src/flaxy/`, NumPy-only, vectorized forward-mode duals for the baseline pieces,
hand-written gradients for KRR/TT. pytest: permutation/rotation/translation
invariance, Richardson FD gradients, zero net force and torque, gate limits,
save/load. In-domain energy weighting function: user contribution.

## Findings from the first exploratory runs (2026-11-20)
Domain: geometries more than 200 kJ/mol above the NH3 + CH limit are not fitted
(user decision); 7 856 training and 824 in-domain validation geometries remain.

- Tail from the 300 K, R >= 9 A data: dipole-dipole coefficient -0.333 +/- 0.004 a.u.
  versus -mu(NH3) mu(CH) = -0.33 a.u. (experimental dipoles, C-H+ polarity). Stage-1
  RMSE 1.5 cm^-1. R^-6 terms are poorly identified (narrow R window), harmless at 10 K.
- Monomer misfit (degree-8 PIP NH3, degree-8 CH): ~4 cm^-1 at 300 K, ~12 cm^-1 in
  domain overall; it is the floor of the long-range accuracy (data only).
- Baseline alone is good to ~10 cm^-1 for R >= 6 A at 300 K; switch moved to 5-7 A so
  that R > 7 A is purely physical (KRR wiggles in a 6-9 A switch gave ~20 cm^-1/A
  of spurious radial force, comparable to the true dipole-dipole force there).
- Tang-Toennies beta = 1.0 / bohr: same accuracy as 1.5 with a much smaller
  short-range tail (7 vs 30 kcal/mol rms at R < 2.6 A); beta = 2.5 is harmful.
- S3-KRR reference (Matern 5/2, Morse features, inter-fragment range 4 bohr):
  225 cm^-1 RMSE on in-domain validation, train ~42 cm^-1 -> data-limited; error is
  concentrated at R_COM < 2.5 A (adduct region, few in-domain points).
- S3-projected Gaussian TT fitted directly to the energies: 1 200-1 600 cm^-1 at
  rank 6, overfits at rank 12 (train 170 cm^-1, tune 4 400 cm^-1). The projection
  plus ridge selects the symmetrised representative, whose TT rank is up to |G| = 6
  times larger; with ~5 k energies in 15-D the TT has no smoothness prior across
  sites. Fixes made: per-site whitening, data-metric SVD truncation, one-site repair,
  monotone acceptance.
- Rank 12 fitted directly: tune 2 644 cm^-1 with 10 440 parameters (overfits).
- Distillation (TT trained on the real data plus ~23.6 k KRR-labelled perturbed
  in-domain geometries, weight 0.5): rank 8, 4 872 parameters, tune 659 cm^-1 against
  the data and 678 cm^-1 against the KRR teacher (KRR itself: 279 cm^-1 on that split).
  Dense labels halve the TT error, confirming the data limit; the remaining gap to
  the teacher is compression error at rank 8.
- Data gap: only ~70 training (3 validation) geometries lie below the dissociation
  limit, i.e. in the attractive funnel and the H3N-CH well (-18 kcal/mol).
- Reference surface: runs/krr_sw_5.0_7.0/surface_krr.npz (225 cm^-1 in-domain
  validation, 8 cm^-1 on the 300 K 4.5-12 A file, 1.6 ms/geometry batched energy).
