#!/usr/bin/env python
"""Relaxed reorientation profiles near the channel crossing (rigid NH3 + CH).

At fixed R the CH axis is tilted from the approach axis (theta_CH = 0: H end
towards NH3, collinear channel) to side-on; at every tilt the NH3 orientation (3
angles) and the CH azimuth are relaxed. The profile E(theta_CH) shows the two
orientation minima and the ridge between them. Energies relative to NH3 + CH at
infinity (cm-1). Two sweeps (up from the collinear minimum, down from the side-on
minimum); the lower energy is kept at each tilt.

Saves reorientation.dat / .npz and fig_reorientation.png/.pdf next to scans.npz.

Example:
  python scripts/reorientation_scan.py runs/lowE_all6_cut120/surface_krr.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as Rot

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "surface.py").is_file())  # repo root, also from copies
sys.path.insert(0, str(ROOT))

from approach_scans import Scanner, align  # noqa: E402
from src.surface import Surface  # noqa: E402
from src.units import ANG_BOHR, HARTREE_CM  # noqa: E402


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("surface")
    p.add_argument("--R", type=float, nargs="+", default=[3.8, 3.9, 4.0, 4.1, 4.2, 4.4])
    p.add_argument("--tilt-max", type=float, default=110.0)
    p.add_argument("--tilt-step", type=float, default=2.0)
    return p.parse_args()


def main():
    a = parse()
    sc = Scanner(Surface.load(a.surface))
    out = Path(a.surface).with_name("scans")
    d = np.load(out / "scans.npz")
    XA, XB = d["XA_body_ang"][0] * ANG_BOHR, d["XB_body_ang"][0] * ANG_BOHR
    ch = XB[0] - XB[1]                                    # H -> C
    E_inf = sc.base.energy(sc.build(XA, XB, np.eye(3), np.eye(3), 1e4))[0]
    chan = {k: np.loadtxt(out / f"min_path_{k}.dat") for k in ("collinear", "side_on")}

    def rotB(tilt, phi):
        n = np.array([np.sin(tilt) * np.cos(phi), np.sin(tilt) * np.sin(phi), np.cos(tilt)])
        return align(ch, (1.0, 0, 0), n)

    def energy(q, tilt, R):
        """q = (NH3 rotation vector, CH azimuth); cm-1 relative to the asymptote."""
        X = np.concatenate([sc.build(XA, XB, Rot.from_rotvec(v[:3]).as_matrix(), rotB(tilt, v[3]), R) for v in q])
        return (sc.pes.energy(X) - E_inf) * HARTREE_CM

    def fg(v, tilt, R, h=1e-4):
        P = np.concatenate([v[None], v[None] + h * np.eye(4), v[None] - h * np.eye(4)])
        E = energy(P, tilt, R)
        return E[0], (E[1:5] - E[5:]) / (2 * h)

    def relax(v, tilt, R):
        r = minimize(fg, v, args=(tilt, R), jac=True, method="L-BFGS-B",
                     options={"maxiter": 2000, "gtol": 1e-6, "ftol": 1e-15})
        return r.x, r.fun

    def seed(name, R):
        row = chan[name][np.argmin(np.abs(chan[name][:, 0] - R))]
        rB = Rot.from_rotvec(row[8:11]).apply(ch / np.linalg.norm(ch))
        return np.concatenate([row[5:8], [np.arctan2(rB[1], rB[0])]])

    tilts = np.radians(np.arange(0.0, a.tilt_max + 1e-9, a.tilt_step))
    prof = {}
    for R in a.R:
        up, v = [], seed("collinear", R)
        for t in tilts:
            v, E = relax(v, t, R)
            up.append(E)
        down, v = [], seed("side_on", R)
        for t in tilts[::-1]:
            v, E = relax(v, t, R)
            down.append(E)
        prof[R] = np.minimum(up, down[::-1])
        E = prof[R]
        mins = [i for i in range(1, len(E) - 1) if E[i] <= E[i - 1] and E[i] <= E[i + 1]] + ([0] if E[0] <= E[1] else [])
        mins = sorted(set(mins))
        msg = ", ".join(f"min {np.degrees(tilts[i]):5.1f} deg {E[i]:8.1f}" for i in mins)
        if len(mins) >= 2:
            i, j = mins[0], mins[-1]
            top = i + int(np.argmax(E[i:j + 1]))
            msg += (f";  ridge {np.degrees(tilts[top]):5.1f} deg {E[top]:8.1f} cm-1 "
                    f"(above collinear {E[top] - E[i]:6.1f}, above side-on {E[top] - E[j]:6.1f})")
        print(f"R {R:4.2f} A:  {msg}", flush=True)

    deg = np.degrees(tilts)
    np.savetxt(out / "reorientation.dat", np.column_stack([deg] + [prof[R] for R in a.R]), fmt="%.6f",
               header="relaxed reorientation profiles: E (cm-1, relative to NH3 + CH at infinity) vs CH tilt "
                      "theta_CH (deg; 0 = H end towards NH3, collinear); other orientation angles relaxed\n"
                      "theta_CH_deg  " + "  ".join(f"E_R{R:.2f}A" for R in a.R))
    np.savez(out / "reorientation.npz", tilt_deg=deg, R=np.array(a.R), E=np.array([prof[R] for R in a.R]))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 13, "axes.labelsize": 13, "legend.fontsize": 11, "lines.linewidth": 2.2})
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    cmap = plt.get_cmap("viridis")
    for k, R in enumerate(a.R):
        ax.plot(deg, prof[R], color=cmap(k / max(1, len(a.R) - 1)), label=f"R = {R:.1f} Å")
    ax.axhline(0, color="0.4", lw=1.0)
    ax.text(deg[-1], 15, "NH$_3$ + CH dissociation limit", ha="right", va="bottom", fontsize=11, color="0.3")
    ax.set(xlabel="CH tilt from the approach axis (deg)   [0°: collinear N ··· H–C]",
           ylabel="E − E(NH$_3$ + CH)  (cm$^{-1}$)",
           title="Reorientation between the two channels (other angles relaxed)",
           xlim=(0, deg[-1]), ylim=(min(p.min() for p in prof.values()) * 1.1, 120))
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_reorientation.{ext}", dpi=220)
    print(f"wrote {out / 'fig_reorientation.png'} and .pdf")


if __name__ == "__main__":
    main()
