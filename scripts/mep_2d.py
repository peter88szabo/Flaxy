#!/usr/bin/env python
"""Relaxed 2D scan E(R, CH tilt) and the minimum energy path of rigid NH3 + CH.

2D scan  On a grid of the centre-of-mass distance R and the CH tilt theta_CH
         (angle between the H->C axis and the approach axis; 0 = H end towards NH3,
         ~75 = side-on, 180 = C end towards NH3) the remaining orientation angles
         (NH3: 3, CH azimuth: 1) are relaxed. Two sweeps per R (from the collinear and
         from the side-on minimum); the lower energy is kept.
MEP      Steepest-descent path on this relaxed surface in mass-weighted coordinates
         (sqrt(mu) R, sqrt(I_CH) theta). It follows the collinear valley (theta = 0) until
         the collinear arrangement turns unstable to bending (d2E/dtheta2 = 0), then
         leaves it with a 0.1 degree starting bend and rolls into the side-on valley.
         Beyond the grid it continues along the collinear channel to 15 Angstrom. The
         scan is mirrored to negative tilt so the spline is symmetric about theta = 0.

Saves mep_2d_scan.dat / .npz, mep.dat and fig_mep_2d.png/.pdf, fig_mep.png/.pdf next to
scans.npz. Monomers rigid at equilibrium (from approach_scans.py).

Example:
  python scripts/mep_2d.py runs/lowE_all6_cut120/surface_krr.npz --procs 16
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")     # many small evaluations: one thread per process

import argparse  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402
from scipy.interpolate import RectBivariateSpline  # noqa: E402
from scipy.optimize import brentq, minimize  # noqa: E402
from scipy.spatial.transform import Rotation as Rot  # noqa: E402

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "surface.py").is_file())  # repo root, also from copies
sys.path.insert(0, str(ROOT))

from approach_scans import Scanner, align  # noqa: E402
from src.surface import Surface  # noqa: E402
from src.units import ANG_BOHR, HARTREE_CM  # noqa: E402

AMU = {"H": 1.00782503, "C": 12.0, "N": 14.0030740}
KJ_PER_CM = 0.01196266
G = {}   # per-process state


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("surface")
    p.add_argument("--rmin", type=float, default=3.0)
    p.add_argument("--rmax", type=float, default=6.0)
    p.add_argument("--r-step", type=float, default=0.025)
    p.add_argument("--tilt-step", type=float, default=3.0)
    p.add_argument("--procs", type=int, default=16)
    p.add_argument("--plot-only", action="store_true")
    return p.parse_args()


def setup(surface):
    sc = Scanner(Surface.load(surface))
    out = Path(surface).with_name("scans")
    d = np.load(out / "scans.npz")
    XA, XB = d["XA_body_ang"][0] * ANG_BOHR, d["XB_body_ang"][0] * ANG_BOHR
    G.update(sc=sc, out=out, XA=XA, XB=XB, ch=XB[0] - XB[1],
             E_inf=sc.base.energy(sc.build(XA, XB, np.eye(3), np.eye(3), 1e4))[0],
             chan={k: np.loadtxt(out / f"min_path_{k}.dat") for k in ("collinear", "side_on")})


def energy(q, tilt, R):
    """q rows = (NH3 rotation vector, CH azimuth); cm-1 relative to NH3 + CH at infinity."""
    sc, XA, XB, ch = G["sc"], G["XA"], G["XB"], G["ch"]
    X = []
    for v in q:
        n = np.array([np.sin(tilt) * np.cos(v[3]), np.sin(tilt) * np.sin(v[3]), np.cos(tilt)])
        X.append(sc.build(XA, XB, Rot.from_rotvec(v[:3]).as_matrix(), align(ch, (1.0, 0, 0), n), R))
    return (sc.pes.energy(np.concatenate(X)) - G["E_inf"]) * HARTREE_CM


def fg(v, tilt, R, h=1e-4):
    P = np.concatenate([v[None], v[None] + h * np.eye(4), v[None] - h * np.eye(4)])
    E = energy(P, tilt, R)
    return E[0], (E[1:5] - E[5:]) / (2 * h)


def relax(v, tilt, R):
    r = minimize(fg, v, args=(tilt, R), jac=True, method="L-BFGS-B",
                 options={"maxiter": 2000, "gtol": 1e-5, "ftol": 1e-14})
    return r.x, r.fun


def seed(name, R):
    """Orientation (NH3 rotvec, CH azimuth) of a channel minimum at the nearest valid R."""
    t = G["chan"][name]
    lo, hi = (3.75, 15.0) if name == "collinear" else (3.0, 4.40)
    row = t[np.argmin(np.abs(t[:, 0] - np.clip(R, lo, hi)))]
    rB = Rot.from_rotvec(row[8:11]).apply(G["ch"] / np.linalg.norm(G["ch"]))
    return np.concatenate([row[5:8], [np.arctan2(rB[1], rB[0])]])


def scan_row(args):
    R, tilts = args
    best = np.full(len(tilts), np.inf)
    thA = np.zeros(len(tilts))
    c3 = G["XA"][0] - G["XA"][1:].mean(0)
    for name, order in (("collinear", range(len(tilts))), ("side_on", range(len(tilts) - 1, -1, -1))):
        v = seed(name, R)
        for i in order:
            v, E = relax(v, tilts[i], R)
            if E < best[i]:
                best[i] = E
                thA[i] = np.degrees(np.arccos(np.clip(Rot.from_rotvec(v[:3]).apply(c3 / np.linalg.norm(c3))[2], -1, 1)))
    return best, thA


# ------------------------------------------------------------------ MEP
def steepest_descent(spl, R0, th0, metric, R_stop, s_max=60.0):
    """Steepest-descent path dx/ds = -g/|g| in mass-weighted coordinates
    x = (sqrt(mu) R, sqrt(I) theta), from (R0, th0) until R = R_stop. Downhill by
    construction; s is the mass-weighted arc length (amu^1/2 Angstrom)."""
    m = np.asarray(metric, float)

    def rhs(_, x):
        R, th = x[0] / m[0], x[1] / m[1]
        g = np.array([spl(R, th, dx=1, grid=False) / m[0], spl(R, th, dy=1, grid=False) / m[1]], float)
        return -g / np.linalg.norm(g)

    def hit(_, x):
        return x[0] / m[0] - R_stop
    hit.terminal = True
    sol = solve_ivp(rhs, (0.0, s_max), [R0 * m[0], th0 * m[1]], method="DOP853", rtol=1e-10, atol=1e-12,
                    max_step=0.01, events=hit)
    R, th = sol.y[0] / m[0], sol.y[1] / m[1]
    return R, th, spl(R, th, grid=False), sol.t


def main():
    a = parse()
    setup(a.surface)
    out = G["out"]
    if not a.plot_only:
        R_grid = np.round(np.arange(a.rmin, a.rmax + 1e-9, a.r_step), 6)
        tilt_deg = np.arange(0.0, 180.0 + 1e-9, a.tilt_step)
        tilts = np.radians(tilt_deg)
        t0 = time.time()
        with mp.get_context("fork").Pool(a.procs) as pool:
            rows = pool.map(scan_row, [(R, tilts) for R in R_grid], chunksize=1)
        E = np.array([r[0] for r in rows])
        thA = np.array([r[1] for r in rows])
        print(f"2D scan {E.shape} in {time.time() - t0:.0f}s")
        np.savez(out / "mep_2d_scan.npz", R=R_grid, tilt_deg=tilt_deg, E=E, theta_NH3_deg=thA)
        np.savetxt(out / "mep_2d_scan.dat", np.column_stack([np.repeat(R_grid, len(tilt_deg)),
                                                             np.tile(tilt_deg, len(R_grid)), E.ravel(), thA.ravel()]),
                   fmt="%.6f", header="relaxed 2D scan (rigid monomers): E (cm-1, relative to NH3 + CH at infinity) "
                   "vs R and CH tilt (0 = H end towards NH3); NH3 orientation and CH azimuth relaxed\n"
                   "R_A  theta_CH_deg  E_cm  theta_NH3_deg")

    d = np.load(out / "mep_2d_scan.npz")
    R_grid, tilt_deg, E = d["R"], d["tilt_deg"], d["E"]
    # E(R, -theta) = E(R, theta): mirror the scan so the spline is exactly symmetric about the
    # collinear line (dE/dtheta = 0 there, as it must be).
    t_sym = np.radians(np.r_[-tilt_deg[:0:-1], tilt_deg])
    spl = RectBivariateSpline(R_grid, t_sym, np.hstack([E[:, :0:-1], E]), kx=3, ky=3)

    # Mass-weighted metric: reduced mass of NH3 + CH, moment of inertia of CH (amu, Angstrom).
    sym = G["sc"].sys.symbols
    mA = sum(AMU[s] for s in sym[:4])
    mB = sum(AMU[s] for s in sym[4:])
    mu = mA * mB / (mA + mB)
    I_ch = AMU["C"] * AMU["H"] / mB * (np.linalg.norm(G["ch"]) / ANG_BOHR) ** 2
    metric = (np.sqrt(mu), np.sqrt(I_ch))

    # Where the collinear arrangement stops being a minimum against bending: d2E/dtheta2 at
    # theta = 0 changes sign (valley-ridge inflection). Below it the collinear line is a ridge.
    Rf = np.linspace(R_grid[0], R_grid[-1], 6001)
    curv = spl(Rf, np.zeros_like(Rf), dy=2, grid=False)
    flip = np.flatnonzero(np.diff(np.sign(curv)))
    R_vri = float(Rf[flip[-1]])
    print(f"bending curvature at theta = 0 changes sign at R = {R_vri:.3f} A")

    # Saddle point on the collinear line: dE/dR = 0 at theta = 0 (minimum along R, maximum along the
    # CH bend). The steepest-descent path from the asymptote runs down the collinear line into it.
    Rc = np.linspace(R_grid[0], R_vri, 4001)
    dER = spl(Rc, np.zeros_like(Rc), dx=1, grid=False)
    z = np.flatnonzero(np.diff(np.sign(dER)))
    R_sad = brentq(lambda r: float(spl(r, 0.0, dx=1, grid=False)), Rc[z[-1]], Rc[z[-1] + 1])
    E_sad = float(spl(R_sad, 0.0, grid=False))
    print(f"saddle point on the collinear line: R = {R_sad:.4f} A, E = {E_sad:.1f} cm-1 "
          f"({E_sad * KJ_PER_CM:.2f} kJ/mol); d2E/dR2 {float(spl(R_sad, 0.0, dx=2, grid=False)):.0f} cm-1/A2, "
          f"d2E/dtilt2 {float(spl(R_sad, 0.0, dy=2, grid=False)):.0f} cm-1/rad2")

    # MEP inside the grid: the collinear line down to the saddle, then the steepest-descent path
    # (IRC) from the saddle along the unstable CH bend into the side-on valley.
    th0 = np.radians(0.1)
    Rs, ths, Es, ss = steepest_descent(spl, R_sad, th0, metric, R_grid[0])
    for t0 in (0.01, 1.0):                       # sensitivity to the starting displacement
        r_, t_, _, _ = steepest_descent(spl, R_sad, np.radians(t0), metric, R_grid[0])
        print(f"  start {t0:5.2f} deg: tilt passes 45 deg at R = {np.interp(45, np.degrees(t_), r_):.4f} A")
    print(f"  start  0.10 deg: tilt passes 45 deg at R = {np.interp(45, np.degrees(ths), Rs):.4f} A (used)")
    R_line = np.linspace(R_grid[-1], R_sad, 400, endpoint=False)
    Rm = np.r_[R_line, Rs]
    thm = np.r_[np.zeros_like(R_line), ths]
    Em = spl(Rm, thm, grid=False)
    rise = np.max(Em[1:] - Em[:-1])
    print(f"MEP inside the grid: E {Em[0]:.1f} -> {Em[-1]:.1f} cm-1; largest rise along the path {max(rise, 0):.4f} cm-1")

    # Full MEP: collinear channel from 15 A down to the grid edge, then the path found above.
    coll = G["chan"]["collinear"]
    outer = coll[coll[:, 0] > R_grid[-1] + 1e-9]
    R_full = np.r_[outer[::-1, 0], Rm]
    E_full = np.r_[outer[::-1, 1], Em]
    th_full = np.r_[np.zeros(len(outer)), np.degrees(thm)]
    s_mw = np.r_[0, np.cumsum(np.hypot(np.sqrt(mu) * np.diff(R_full), np.sqrt(I_ch) * np.diff(np.radians(th_full))))]
    np.savetxt(out / "mep.dat", np.column_stack([s_mw, R_full, th_full, E_full]), fmt="%.6f",
               header="minimum energy path, rigid NH3 + CH: s = mass-weighted arc length (amu^1/2 A), R (A), "
                      "CH tilt (deg, 0 = H end towards NH3), E (cm-1 relative to NH3 + CH at infinity)\n"
                      "s  R_A  theta_CH_deg  E_cm")
    plots(out, R_grid, tilt_deg, E, spl, R_full, th_full, E_full, s_mw, (R_sad, E_sad, R_vri))


def plots(out, R_grid, tilt_deg, E, spl, R_full, th_full, E_full, s_mw, marks):
    R_sad, E_sad, R_vri = marks
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13, "legend.fontsize": 11,
                         "xtick.labelsize": 12, "ytick.labelsize": 12, "lines.linewidth": 2.4})
    C_MEP = "white"

    # ---- Figure A: heatmap + MEP, full and zoom
    fig, ax = plt.subplots(1, 2, figsize=(17, 6.4), gridspec_kw={"width_ratios": [1.25, 1]})
    for a_, (r0, r1, t0, t1), title in [(ax[0], (R_grid[0], R_grid[-1], 0, 180), "(a) Relaxed 2D scan"),
                                        (ax[1], (3.5, 4.6, 0, 110), "(b) Zoom: the reorientation")]:
        Rf = np.linspace(r0, r1, 400)
        Tf = np.linspace(t0, t1, 400)
        Z = spl(Rf, np.radians(Tf))
        vmax = min(np.percentile(Z, 98), 1500.0) if a_ is ax[0] else np.percentile(Z, 95)
        im = a_.pcolormesh(Rf, Tf, Z.T, shading="auto", cmap="viridis", vmin=Z.min(), vmax=vmax)
        step = 100 if a_ is ax[0] else 20
        lev = np.arange(np.floor(Z.min() / step) * step, vmax + step, step)
        cs = a_.contour(Rf, Tf, Z.T, levels=lev, colors="k", linewidths=0.5, alpha=0.5)
        a_.clabel(cs, cs.levels[::2], fontsize=8, fmt="%d")
        if a_ is ax[0]:
            a_.contour(Rf, Tf, Z.T, levels=[0.0], colors="w", linewidths=1.2, linestyles="--")
        m = (R_full >= r0) & (R_full <= r1)
        a_.plot(R_full[m], th_full[m], color=C_MEP, lw=2.8, label="MEP", clip_on=False, zorder=4)
        a_.plot(R_full[m], th_full[m], color="k", lw=0.8, ls="--")
        if r0 <= R_sad <= r1:
            a_.annotate(f"saddle point  R = {R_sad:.2f} Å\n{E_sad:.0f} cm$^{{-1}}$ ({E_sad * KJ_PER_CM:.1f} kJ/mol)\n"
                        "MEP turns from R into the CH bend", (R_sad, 0), (R_sad + 0.08 * (r1 - r0), 0.30 * t1),
                        color="w", fontsize=10, arrowprops=dict(arrowstyle="->", color="w"))
        a_.set(xlim=(r0, r1), ylim=(t0 - 2.5, t1), xlabel="centre-of-mass distance R (Å)",
               ylabel="CH tilt from the approach axis (deg)", title=title)
        cb = fig.colorbar(im, ax=a_, pad=0.01)
        cb.set_label("E − E(NH$_3$ + CH)  (cm$^{-1}$)")
    ax[0].text(0.98, 0.04, "white dashed: E = dissociation limit\n0°: H end of CH towards NH$_3$\n"
               "180°: C end towards NH$_3$\nother angles relaxed", transform=ax[0].transAxes,
               ha="right", va="bottom", fontsize=10, color="w")
    ax[0].legend(loc="upper right", frameon=False, labelcolor="w")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_mep_2d.{ext}", dpi=220)

    # ---- Figure A2: mirrored tilt (-180..180): the collinear valley runs through the middle, the two
    # equivalent side-on wells lie at +tilt and -tilt, the saddle is the pass between them.
    fig, ax = plt.subplots(1, 2, figsize=(17, 6.6), gridspec_kw={"width_ratios": [1.25, 1]})
    for a_, (r0, r1, t1), title in [(ax[0], (R_grid[0], R_grid[-1], 180), "(a) Relaxed 2D scan, CH tilt to both sides"),
                                    (ax[1], (3.5, 4.6, 110), "(b) Zoom: the pass between the two side-on wells")]:
        Rf = np.linspace(r0, r1, 400)
        Tf = np.linspace(-t1, t1, 600)
        Z = spl(Rf, np.radians(Tf))
        vmax = min(np.percentile(Z, 98), 1500.0) if a_ is ax[0] else np.percentile(Z, 95)
        im = a_.pcolormesh(Rf, Tf, Z.T, shading="auto", cmap="viridis", vmin=Z.min(), vmax=vmax)
        step = 100 if a_ is ax[0] else 20
        lev = np.arange(np.floor(Z.min() / step) * step, vmax + step, step)
        cs = a_.contour(Rf, Tf, Z.T, levels=lev, colors="k", linewidths=0.5, alpha=0.5)
        a_.clabel(cs, cs.levels[::2], fontsize=8, fmt="%d")
        if a_ is ax[0]:
            a_.contour(Rf, Tf, Z.T, levels=[0.0], colors="w", linewidths=1.2, linestyles="--")
        m = (R_full >= r0) & (R_full <= r1)
        a_.plot(R_full[m], th_full[m], color="w", lw=2.6, label="MEP")
        a_.plot(R_full[m], -th_full[m], color="w", lw=1.4, ls="--", label="MEP, mirror image")
        a_.axhline(0, color="w", lw=0.6, alpha=0.5)
        a_.annotate(f"saddle point (pass)\nR = {R_sad:.2f} Å, {E_sad:.0f} cm$^{{-1}}$", (R_sad, 0),
                    (R_sad + 0.12 * (r1 - r0), -0.45 * t1), color="w", fontsize=10,
                    arrowprops=dict(arrowstyle="->", color="w"))
        a_.set(xlim=(r0, r1), ylim=(-t1, t1), xlabel="centre-of-mass distance R (Å)",
               ylabel="CH tilt from the approach axis (deg)", title=title)
        a_.set_yticks(np.arange(-180, 181, 45) if t1 == 180 else np.arange(-100, 101, 25))
        cb = fig.colorbar(im, ax=a_, pad=0.01)
        cb.set_label("E − E(NH$_3$ + CH)  (cm$^{-1}$)")
    ax[0].text(0.98, 0.535, "collinear N··· H–C\n(MEP comes in along 0°)", transform=ax[0].transAxes, ha="right",
               va="bottom", fontsize=10, color="w")
    ax[0].text(0.02, 0.81, "side-on well", transform=ax[0].transAxes, fontsize=10, color="w")
    ax[0].text(0.02, 0.16, "side-on well (mirror)", transform=ax[0].transAxes, fontsize=10, color="w")
    ax[0].legend(loc="upper right", frameon=False, labelcolor="w")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_mep_2d_mirror.{ext}", dpi=220)

    # ---- Figure B: the MEP: overview against R, the reorientation against the path length s
    i0 = int(np.argmin(np.abs(R_full - R_sad)))
    s_rel = s_mw - s_mw[i0]                      # s = 0 at the saddle point, where the path turns
    to_R = lambda x: np.interp(x, s_rel, R_full)                    # noqa: E731  (R decreases along s)
    to_s = lambda r: np.interp(r, R_full[::-1], s_rel[::-1])        # noqa: E731
    C_E, C_T = "#1f5fa8", "#d1495b"
    fig, ax = plt.subplots(1, 3, figsize=(18.5, 5.8))

    a0 = ax[0]                                   # (a) overview
    a0.plot(R_full, E_full, color=C_E)
    a0.axhline(0, color="0.5", lw=0.8)
    a0.axvline(R_sad, color="0.45", ls=":", lw=1.3)
    a0.annotate(f"saddle point R = {R_sad:.2f} Å, {E_sad:.0f} cm$^{{-1}}$:\npath turns into the CH bend (zoom in b)",
                (R_sad, E_sad), (R_sad + 1.3, 0.86 * E_full.min()), fontsize=10, color="0.3",
                arrowprops=dict(arrowstyle="->", color="0.4"))
    a0.set(xlim=(R_full.min(), R_full.max()), ylim=(E_full.min() * 1.06, 0.12 * abs(E_full.min())),
           xlabel="centre-of-mass distance R (Å)", ylabel="E − E(NH$_3$ + CH)  (cm$^{-1}$)",
           title="(a) Energy along the minimum energy path")
    sec = a0.secondary_yaxis("right", functions=(lambda e: e * KJ_PER_CM, lambda k: k / KJ_PER_CM))
    sec.set_ylabel("kJ/mol")
    ins = a0.inset_axes([0.40, 0.30, 0.55, 0.40])
    m7 = R_full >= 7.0
    ins.plot(R_full[m7], E_full[m7], color=C_E)
    ins.axhline(0, color="0.5", lw=0.6)
    ins.set_xlim(7, R_full.max())
    ins.set_title("7–15 Å (physical multipole tail)", fontsize=10)
    ins.tick_params(labelsize=9)

    a1 = ax[1]                                   # (b) zoom on the reorientation, against s
    z = (s_rel >= -2.5) & (s_rel <= s_rel.max())
    a1.plot(s_rel[z], E_full[z], color=C_E, label="energy")
    a1.annotate("saddle point:\npath turns from R\ninto the CH bend", (0, E_sad),
                (-2.35, E_sad - 0.30 * (E_full[z].max() - E_full[z].min())), fontsize=10,
                color="0.3", arrowprops=dict(arrowstyle="->", color="0.4"))
    a1.set(xlabel="path length s from the saddle point  (amu$^{1/2}$ Å)", ylabel="E − E(NH$_3$ + CH)  (cm$^{-1}$)",
           xlim=(-2.5, s_rel.max()), title="(b) Zoom: the reorientation along the path")
    a1.axvline(0, color="0.45", ls=":", lw=1.3)
    tw = a1.twinx()
    tw.plot(s_rel[z], th_full[z], color=C_T, label="CH tilt")
    tw.set_ylabel("CH tilt from the approach axis (deg)", color=C_T)
    tw.tick_params(axis="y", colors=C_T)
    tw.set_ylim(-3, 90)
    top = a1.secondary_xaxis("top", functions=(to_R, to_s))
    top.set_xlabel("R (Å)")
    top.set_xticks([4.6, 4.2, 3.9, 3.7, 3.5, 3.3, 3.1])
    a1.text(0.03, 0.06, "collinear N··· H–C:\nCH tilt 0°", transform=a1.transAxes, fontsize=10, color="0.3")
    a1.text(0.97, 0.94, "CH turns side-on\nover the N lone pair", transform=a1.transAxes, ha="right", va="top",
            fontsize=10, color=C_T)
    h1, l1 = a1.get_legend_handles_labels()
    h2, l2 = tw.get_legend_handles_labels()
    a1.legend(h1 + h2, l1 + l2, loc="lower left", bbox_to_anchor=(0.0, 0.14), frameon=False)

    a2 = ax[2]                                   # (c) slope along the whole path
    a2.plot(s_rel, -np.gradient(E_full, s_mw), color=C_E)
    a2.axvline(0, color="0.45", ls=":", lw=1.3)
    a2.set(yscale="log", ylim=(0.3, None), xlim=(s_rel.min(), s_rel.max()),
           xlabel="path length s from the saddle point  (amu$^{1/2}$ Å)",
           ylabel="−dE/ds  (cm$^{-1}$ amu$^{-1/2}$ Å$^{-1}$)",
           title="(c) Downhill all the way: no barrier")
    a2.annotate("slope 0 only at\nthe saddle point", (0, 0.45), (-14, 0.45), fontsize=10, color="0.3", va="center",
                arrowprops=dict(arrowstyle="->", color="0.4"))
    top2 = a2.secondary_xaxis("top", functions=(to_R, to_s))
    top2.set_xlabel("R (Å)")
    top2.set_xticks([14, 10, 8, 6, 5, 4, 3])
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_mep.{ext}", dpi=220)
    print(f"wrote {out / 'fig_mep_2d.png'}, {out / 'fig_mep_2d_mirror.png'}, {out / 'fig_mep.png'} (+ .pdf)")


if __name__ == "__main__":
    main()
