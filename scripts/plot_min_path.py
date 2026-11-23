#!/usr/bin/env python
"""Meeting figure: the lowest-energy approach of rigid NH3 + CH, channel by channel.

The lowest energy over orientations, E_min(R), is the lower envelope of smooth
channels. Each channel is followed continuously (orientation re-optimised at every
R from the previous R, tight convergence, energies in cm-1):

  collinear  started at R_max from a global orientation search, followed inwards
             (N lone pair ... H-C, head-to-tail dipoles)
  side-on    started at R_min from a global orientation search, followed outwards
             (CH side-on over the N lone pair, the H3N-CH adduct channel)

Monomers are rigid at their equilibrium geometries (from scans.npz written by
approach_scans.py). Saves min_path_channels.dat / .npz and fig_min_path_channels.png/.pdf.

Example:
  python scripts/plot_min_path.py runs/lowE_all6_cut120/surface_krr.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as Rot

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "surface.py").is_file())  # repo root, also from copies
sys.path.insert(0, str(ROOT))

from approach_scans import FORCE, Scanner  # noqa: E402
from src.surface import Surface  # noqa: E402
from src.units import ANG_BOHR, HARTREE_CM  # noqa: E402

KJ_PER_CM = 0.01196266


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("surface")
    p.add_argument("--scans", default=None, help="scans.npz of approach_scans.py (default: <surface dir>/scans)")
    p.add_argument("--rmax", type=float, default=15.0)
    p.add_argument("--rmin", type=float, default=3.0)
    p.add_argument("--step", type=float, default=0.05)
    p.add_argument("--n-search", type=int, default=200, help="random starts of the global searches")
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--plot-only", action="store_true", help="replot from min_path_channels.npz")
    return p.parse_args()


def main():
    a = parse()
    sc = Scanner(Surface.load(a.surface))
    out = Path(a.scans or Path(a.surface).with_name("scans"))
    switch = (sc.pes.r_in / ANG_BOHR, sc.r_out_ang)
    if a.plot_only:
        plot(out, *switch)
        return
    d = np.load(out / "scans.npz")
    XA = d["XA_body_ang"][0] * ANG_BOHR              # named scans use the equilibrium monomers
    XB = d["XB_body_ang"][0] * ANG_BOHR
    E_inf = sc.base.energy(sc.build(XA, XB, np.eye(3), np.eye(3), 1e4))[0]
    rng = np.random.default_rng(a.seed)

    def geoms(P, R):
        return np.concatenate([sc.build(XA, XB, Rot.from_rotvec(p[:3]).as_matrix(),
                                        Rot.from_rotvec(p[3:]).as_matrix(), R) for p in P])

    def fg(p, R, h=1e-4):
        P = np.concatenate([p[None], p[None] + h * np.eye(6), p[None] - h * np.eye(6)])
        E = (sc.pes.energy(geoms(P, R)) - E_inf) * HARTREE_CM
        return E[0], (E[1:7] - E[7:]) / (2 * h)

    def relax(p, R):
        """L-BFGS in cm-1, then a BFGS polish; returns orientation, energy, final |gradient|."""
        r = minimize(fg, p, args=(R,), jac=True, method="L-BFGS-B",
                     options={"maxiter": 3000, "gtol": 1e-7, "ftol": 1e-16})
        r = minimize(fg, r.x, args=(R,), jac=True, method="BFGS", options={"gtol": 1e-6, "maxiter": 500})
        return r.x, r.fun, float(np.abs(fg(r.x, R)[1]).max())

    def search(R):
        """Global orientation search at one R: random starts, best few relaxed."""
        P = np.concatenate([Rot.random(a.n_search, random_state=rng).as_rotvec(),
                            Rot.random(a.n_search, random_state=rng).as_rotvec()], 1)
        E = sc.pes.energy(geoms(P, R))
        return min((relax(P[i], R) for i in np.argsort(E)[:12]), key=lambda t: t[1])

    grad_max = {}

    def follow(R_seq, p, name):
        rows, g = [], []
        for R in R_seq:
            p, E, gn = relax(p, R)
            _, F = sc.pes.energy_forces(geoms(p[None], R))
            rows.append((R, E, F[0, sc.B, 2].sum() * FORCE, *p))
            g.append(gn)
        grad_max[name] = max(g)
        return np.array(rows)

    R_in = np.round(np.arange(a.rmax, a.rmin - 1e-9, -a.step), 6)
    coll = follow(R_in, search(a.rmax)[0], "collinear")[::-1]      # ascending R
    side = follow(R_in[::-1], search(a.rmin)[0], "side_on")        # ascending R
    print("largest remaining orientation gradient (cm-1/rad):",
          {k: f"{v:.1e}" for k, v in grad_max.items()})
    if side[0, 1] >= coll[0, 1]:
        print("note: the global search at R_min found the same channel as the inward branch")
    env = np.minimum(coll[:, 1], side[:, 1])
    cross = np.flatnonzero(np.diff(np.sign(coll[:, 1] - side[:, 1])))
    R_cross = coll[cross[0], 0] if len(cross) else None

    c3, ch = XA[0] - XA[1:].mean(0), XB[0] - XB[1]
    ang = lambda rows, v, s: np.degrees(np.arccos(  # noqa: E731
        np.clip(Rot.from_rotvec(rows[:, s]).apply(v / np.linalg.norm(v))[:, 2], -1, 1)))
    cols = ["R_A", "E_cm", "F_R_cm_per_A", "theta_NH3_deg", "theta_CH_deg", "rvA_x", "rvA_y", "rvA_z",
            "rvB_x", "rvB_y", "rvB_z"]
    for name, rows in [("collinear", coll), ("side_on", side)]:
        tab = np.column_stack([rows[:, :3], ang(rows, c3, slice(3, 6)), ang(rows, ch, slice(6, 9)), rows[:, 3:]])
        np.savetxt(out / f"min_path_{name}.dat", tab, fmt="%.6f",
                   header=f"channel '{name}' of the lowest-energy rigid approach (equilibrium monomers); "
                          "E relative to E(inf) in cm-1; F_R = -dE/dR (< 0 attractive); theta_NH3: C3 axis "
                          "(N - centroid H) vs approach axis, 0 = lone pair towards CH; theta_CH: H->C axis "
                          "vs approach axis, 0 = H end towards NH3\n" + "  ".join(cols))
    np.savez(out / "min_path_channels.npz", collinear=coll, side_on=side, envelope=env, R_cross=R_cross or np.nan,
             theta_collinear=np.stack([ang(coll, c3, slice(3, 6)), ang(coll, ch, slice(6, 9))], 1),
             theta_side_on=np.stack([ang(side, c3, slice(3, 6)), ang(side, ch, slice(6, 9))], 1))

    # Checks printed for the record.
    for name, rows in [("collinear", coll), ("side_on", side)]:
        rise = np.max(rows[:-1, 1] - rows[1:, 1])
        print(f"{name:9s}: E(15) {rows[-1, 1]:8.2f}  E(7) {np.interp(7, rows[:, 0], rows[:, 1]):8.1f}  "
              f"E(5) {np.interp(5, rows[:, 0], rows[:, 1]):8.1f}  E(3) {rows[0, 1]:9.1f} cm-1;  "
              f"largest rise inward {max(rise, 0):.4f} cm-1;  max F_R {rows[:, 2].max():.3f} cm-1/A")
    print(f"channels cross at R = {R_cross} A" if R_cross else "channels do not cross")

    plot(out, *switch)


def channel_range(theta, inward, jump_deg=5.0):
    """Mask of the points where a followed channel is still the same orientation minimum:
    stop at the first orientation jump (> jump_deg per step) in the following direction."""
    d = np.max(np.abs(np.diff(theta, axis=0)), axis=1)
    keep = np.ones(len(theta), bool)
    if inward:            # followed from large R (last index) to small R
        j = np.flatnonzero(d[::-1] > jump_deg)
        if len(j):
            keep[: len(theta) - 1 - j[0]] = False
    else:                 # followed from small R (index 0) outwards
        j = np.flatnonzero(d > jump_deg)
        if len(j):
            keep[j[0] + 1:] = False
    return keep


def plot(out, r_in, r_out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(out / "min_path_channels.npz")
    coll, side, thc, ths = d["collinear"], d["side_on"], d["theta_collinear"], d["theta_side_on"]
    R = coll[:, 0]
    kc, ks = channel_range(thc, inward=True), channel_range(ths, inward=False)
    both = kc & ks
    diff = np.where(both, coll[:, 1] - side[:, 1], np.nan)
    i = np.flatnonzero(np.diff(np.sign(diff[both])))
    R_cross = float(R[both][i[0]]) if len(i) else None
    print(f"collinear channel {R[kc].min():.2f}-{R[kc].max():.2f} A, side-on channel {R[ks].min():.2f}-"
          f"{R[ks].max():.2f} A, crossing {R_cross} A")
    for name, rows, k in [("collinear", coll, kc), ("side-on", side, ks)]:
        e = rows[k]
        print(f"  {name}: monotonic inward: {bool(np.all(np.diff(e[:, 1]) > 0))}, max F_R {e[:, 2].max():.3f} cm-1/A")

    plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13, "legend.fontsize": 11,
                         "xtick.labelsize": 12, "ytick.labelsize": 12, "lines.linewidth": 2.4})
    C1, C2 = "#1f5fa8", "#d1495b"
    lab1, lab2 = "collinear:  H$_3$N ··· H–C", "side-on:  CH over the N lone pair"
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.6))

    def frame(a_):
        a_.axvspan(r_in, r_out, color="0.93", zorder=0, lw=0)
        if R_cross:
            a_.axvline(R_cross, color="0.45", ls=":", lw=1.3)
        a_.set_xlim(R[0], R[-1])
        a_.set_xlabel("centre-of-mass distance R (Å)")

    # (a) energies of the two channels
    a0 = ax[0]
    # bold where the channel is the lowest path, thin where it is a higher-lying minimum
    lc = kc & (~ks | (coll[:, 1] <= side[:, 1]))
    ls_ = ks & (~kc | (side[:, 1] < coll[:, 1]))

    def draw(a_, y1, y2):
        a_.plot(R[kc], y1[kc], color=C1, lw=1.1, alpha=0.55)
        a_.plot(R[ks], y2[ks], color=C2, lw=1.1, alpha=0.55)
        a_.plot(np.where(lc, R, np.nan), np.where(lc, y1, np.nan), color=C1, label=lab1)
        a_.plot(np.where(ls_, R, np.nan), np.where(ls_, y2, np.nan), color=C2, label=lab2)

    draw(a0, coll[:, 1], side[:, 1])
    a0.axhline(0, color="0.5", lw=0.8)
    lo = min(coll[kc, 1].min(), side[ks, 1].min())
    a0.set(ylabel="E − E(NH$_3$ + CH)  (cm$^{-1}$)", ylim=(lo * 1.06, 0.12 * abs(lo)),
           title="(a) Lowest energy over orientations")
    a0.legend(loc="lower right", frameon=False)
    sec = a0.secondary_yaxis("right", functions=(lambda e: e * KJ_PER_CM, lambda k: k / KJ_PER_CM))
    sec.set_ylabel("kJ/mol")
    ins = a0.inset_axes([0.40, 0.30, 0.55, 0.40])
    m = kc & (R >= 7.0)
    ins.plot(R[m], coll[m, 1], color=C1)
    ins.axhline(0, color="0.5", lw=0.6)
    ins.set_xlim(7, R[-1])
    ins.set_title("7–15 Å (physical multipole tail)", fontsize=10)
    ins.tick_params(labelsize=9)
    frame(a0)

    # (b) slope dE/dR > 0 everywhere = attractive, no barrier
    a1 = ax[1]
    draw(a1, -coll[:, 2], -side[:, 2])
    a1.set(yscale="log", ylabel="dE/dR  (cm$^{-1}$/Å)", title="(b) Slope > 0 at every R: no barrier")
    frame(a1)

    # (c) orientation along each channel
    a2 = ax[2]
    for th, k, low, c, nm in [(thc, kc, lc, C1, "collinear"), (ths, ks, ls_, C2, "side-on")]:
        for j, sty, ax_name in [(0, "-", "NH$_3$ C$_3$ axis"), (1, "--", "CH axis")]:
            a2.plot(R[k], th[k, j], color=c, ls=sty, lw=1.1, alpha=0.55)
            a2.plot(np.where(low, R, np.nan), np.where(low, th[:, j], np.nan), color=c, ls=sty,
                    label=f"{ax_name}, {nm}")
    a2.set(ylabel="angle to the approach axis (deg)", ylim=(-5, 185), yticks=[0, 45, 90, 135, 180],
           title="(c) Orientation")
    a2.text(0.98, 0.97, "0°: N lone pair and H end of CH face each other\n"
            "collinear: both axes at 0°\nbold: lowest path, thin: higher-lying minimum", transform=a2.transAxes,
            ha="right", va="top", fontsize=10, color="0.3")
    a2.legend(loc="center right", frameon=False, fontsize=10)
    frame(a2)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_min_path_channels.{ext}", dpi=220)
    print(f"wrote {out / 'fig_min_path_channels.png'} and .pdf")


if __name__ == "__main__":
    main()
