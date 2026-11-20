#!/usr/bin/env python
"""Rigid radial cuts through the switch: energy and radial force versus R.

Takes validation geometries from the 300 K long-range file, keeps both monomers
rigid and moves CH along the COM vector from R_min to R_max. Plots, per cut,
R^3 [V(R) - V(inf)] (dipole-dipole scale) and the radial force -dV/dR, for the
fitted surfaces and the baseline alone. Also prints the largest deviation of
-dV/dR from the baseline inside the switch (a spurious-force diagnostic).

Example:
  python scripts/plot_cuts.py runs/explore_krr/surface_krr.npz runs/explore_tt_r8/surface_tt.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_dataset, load_split, validation_mask  # noqa: E402
from src.geometry import com_vector  # noqa: E402
from src.surface import Surface  # noqa: E402
from src.units import ANG_BOHR, HARTREE_CM  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def cut(X0, R_ang, system):
    """Copies of X0 (bohr) with fragment B translated so that |R_COM| = R (Angstrom)."""
    Rv = com_vector(X0[None], system)[0]
    n = Rv / np.linalg.norm(Rv)
    shift = (R_ang * ANG_BOHR - np.linalg.norm(Rv))[:, None] * n[None, :]
    X = np.repeat(X0[None], len(R_ang), axis=0)
    X[:, list(system.frag_b)] += shift[:, None, :]
    return X, n


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("surfaces", nargs="+")
    p.add_argument("--n-cuts", type=int, default=6)
    p.add_argument("--rmin", type=float, default=3.0)
    p.add_argument("--rmax", type=float, default=30.0)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surfs = {Path(s).stem.replace("surface_", ""): Surface.load(s) for s in a.surfaces}
    base = next(iter(surfs.values()))
    system = base.system
    ds = load_dataset(ROOT / "data_to_fit", system)
    valid = validation_mask(ds, load_split(ROOT / "splits/validation_seed20261120.json"))
    pick = np.flatnonzero(valid & (ds.temperature == 300) & (ds.R > 8.5))
    pick = pick[np.random.default_rng(0).choice(len(pick), a.n_cuts, replace=False)]
    R = np.linspace(a.rmin, a.rmax, 600)
    surfs = {"baseline": Surface(system, base.baseline, base.features, None, base.r_in, base.r_out), **surfs}

    fig, axes = plt.subplots(2, a.n_cuts, figsize=(3.2 * a.n_cuts, 6.4), sharex=True)
    worst = {k: 0.0 for k in surfs}
    for c, idx in enumerate(pick):
        X, n = cut(ds.X[idx], R, system)
        far, _ = cut(ds.X[idx], np.array([1e4]), system)
        for name, s in surfs.items():
            E, F = s.energy_forces(X)
            E_inf = s.energy(far)[0]
            # -dV/dR for a rigid translation of CH along n: sum of CH forces along n.
            FR = np.einsum("bik,k->b", F[:, list(system.frag_b)], n)
            axes[0, c].plot(R, (E - E_inf) * HARTREE_CM * (R * ANG_BOHR) ** 3 / 1e4, label=name)
            axes[1, c].plot(R, FR * HARTREE_CM * ANG_BOHR, label=name)  # Eh/bohr -> cm-1/Angstrom
            if name != "baseline":
                _, Fb = surfs["baseline"].energy_forces(X)
                FRb = np.einsum("bik,k->b", Fb[:, list(system.frag_b)], n)
                sw = (R > base.r_in / ANG_BOHR) & (R < base.r_out / ANG_BOHR)
                worst[name] = max(worst[name], np.max(np.abs(FR - FRb)[sw]) * HARTREE_CM * ANG_BOHR)
        for ax in axes[:, c]:
            ax.axvspan(base.r_in / ANG_BOHR, base.r_out / ANG_BOHR, color="0.9")
            ax.set_xscale("log")
        axes[0, c].set_title(f"{ds.names[idx][:6]} (R0 {ds.R[idx]:.1f} Å)", fontsize=9)
        axes[1, c].set_xlabel("R_COM (Å)")
    axes[0, 0].set_ylabel("R³ [V − V∞]  (10⁴ cm⁻¹ bohr³)")
    axes[1, 0].set_ylabel("−dV/dR  (cm⁻¹/Å)")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    out = a.out or str(Path(a.surfaces[0]).with_name("radial_cuts.png"))
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")
    for k, v in worst.items():
        if k != "baseline":
            print(f"{k}: max |dF_R| vs baseline inside switch {v:.2f} cm-1/Angstrom")


if __name__ == "__main__":
    main()
