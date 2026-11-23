#!/usr/bin/env python
"""Rigid approach scans of NH3 + CH: is there any bump or kink on the way in?

Both monomers are kept rigid; CH moves along the centre-of-mass axis from R_max
(15 Angstrom) to R_min (3 Angstrom) in 0.01 Angstrom steps. Scans:

  named      six fixed orientations with equilibrium monomers
             lp_C    NH3 lone pair towards CH, C end first   (H3N...CH)
             lp_H    NH3 lone pair towards CH, H end first   (H3N...HC)
             H3_C    NH3 H3 face towards CH, C end first     (NH3 H's...CH)
             H3_H    NH3 H3 face towards CH, H end first
             lp_side NH3 lone pair towards CH, CH side-on
             side_C  NH3 C3 axis perpendicular, C end first
  min_path   lowest energy over all orientations at each R (equilibrium monomers)
  random     random orientations, half with equilibrium monomers ("eq"), half with
             rigid monomer geometries from the 300 K data ("zpe": ZPE-like distortion)

Energies are E(R) - E(inf) of the same rigid monomers (the interaction energy on
the surface, cm-1); F_R = -dE/dR (cm-1/Angstrom, > 0 repulsive) is analytic. The
baseline-only surface (multipole tail, no short-range model) is saved alongside:
beyond r_out the two are identical; local maxima present in the full surface but
not in the baseline come from the short-range model (whether they are physical
needs ab initio points along that scan). Bump size = topographic prominence.

Outputs in --out: scans.npz (everything), text tables (*.dat), start geometries
(*.xyz, Angstrom), summary.txt and figures (fig_*.png).

Example:
  python scripts/approach_scans.py runs/lowE_all6_cut120/surface_krr.npz
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation as Rot

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "surface.py").is_file())  # repo root, also from copies
sys.path.insert(0, str(ROOT))

from src.data import load_dataset  # noqa: E402
from src.surface import Surface  # noqa: E402
from src.units import ANG_BOHR, HARTREE_CM  # noqa: E402

FORCE = HARTREE_CM * ANG_BOHR          # Hartree/bohr -> cm-1/Angstrom
NAMED = {  # (NH3 C3 axis N - centroid(H), CH axis C - H) in the lab frame; CH sits at +z
    "lp_C": ((0, 0, 1), (0, 0, -1)),
    "lp_H": ((0, 0, 1), (0, 0, 1)),
    "H3_C": ((0, 0, -1), (0, 0, -1)),
    "H3_H": ((0, 0, -1), (0, 0, 1)),
    "lp_side": ((0, 0, 1), (1, 0, 0)),
    "side_C": ((1, 0, 0), (0, 0, -1)),
}


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("surface")
    p.add_argument("--out", default=None, help="default: <surface dir>/scans")
    p.add_argument("--rmax", type=float, default=15.0)
    p.add_argument("--rmin", type=float, default=3.0)
    p.add_argument("--step", type=float, default=0.01)
    p.add_argument("--n-random", type=int, default=1000)
    p.add_argument("--path-step", type=float, default=0.1, help="R step of the orientation-minimised path")
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


class Scanner:
    def __init__(self, pes):
        self.pes = pes
        self.sys = pes.system
        self.base = Surface(self.sys, pes.baseline, pes.features, None, pes.r_in, pes.r_out)
        self.A, self.B = list(self.sys.frag_a), list(self.sys.frag_b)
        self.m = np.asarray(self.sys.masses, dtype=float)
        self.r_out_ang = pes.r_out / ANG_BOHR

    def centred(self, X, idx):
        return X[idx] - (self.m[idx, None] * X[idx]).sum(0) / self.m[idx].sum()

    def build(self, XA, XB, rotA, rotB, R_ang):
        """Geometries [nR, N, 3] in bohr: A's COM at the origin, B's COM at (0, 0, R)."""
        R = np.atleast_1d(np.asarray(R_ang, dtype=float)) * ANG_BOHR
        X = np.empty((len(R), self.sys.n_atoms, 3))
        X[:, self.A] = XA @ rotA.T
        X[:, self.B] = (XB @ rotB.T)[None] + np.stack([0 * R, 0 * R, R], 1)[:, None, :]
        return X

    def scan(self, XA, XB, rotA, rotB, R):
        """E - E(inf) (cm-1) and F_R (cm-1/A) on the full surface and the baseline."""
        X = self.build(XA, XB, rotA, rotB, R)
        E_inf = self.base.energy(self.build(XA, XB, rotA, rotB, 1e4))[0]
        Eb, Fb = self.base.energy_forces(X)
        Ef, Ff = Eb.copy(), Fb.copy()
        inner = R < self.r_out_ang + 1e-9          # beyond r_out the surface is the baseline exactly
        if inner.any():
            Ef[inner], Ff[inner] = self.pes.energy_forces(X[inner])
        fr = lambda F: F[:, self.B, 2].sum(1) * FORCE  # noqa: E731  (-dE/dR for a rigid shift of B along z)
        return (Ef - E_inf) * HARTREE_CM, fr(Ff), (Eb - E_inf) * HARTREE_CM, fr(Fb), E_inf


def frame(u, w):
    """Orthonormal columns (u, w_perp, u x w_perp)."""
    u = np.asarray(u, float) / np.linalg.norm(u)
    w = np.asarray(w, float) - np.dot(w, u) * u
    if np.linalg.norm(w) < 1e-8:
        w = np.cross(u, [1.0, 0, 0]) if abs(u[0]) < 0.9 else np.cross(u, [0, 1.0, 0])
    w /= np.linalg.norm(w)
    return np.stack([u, w, np.cross(u, w)], 1)


def align(body_axis, body_ref, lab_axis, lab_ref=(1.0, 0.3, 0.2)):
    """Rotation taking body_axis to lab_axis (and body_ref's perpendicular part towards lab_ref)."""
    return frame(lab_axis, lab_ref) @ frame(body_axis, body_ref).T


def local_maxima(E):
    """Local maxima of E(R) on an ascending R grid as (index, prominence, climb).

    prominence: topographic prominence (how far E dips on both sides before rising
                above the maximum again): the size of the bump itself.
    climb:      E(R*) - min over R >= R* of E: what a partner arriving from R_max on
                this rigid scan must climb (includes any long-range repulsion)."""
    k, props = find_peaks(E, prominence=0.0)
    return [(int(i), float(p), float(E[i] - E[i:].min())) for i, p in zip(k, props["prominences"])]


def write_table(path, header, cols, names):
    np.savetxt(path, np.column_stack(cols), fmt="%.6f", header=header + "\n" + "  ".join(names))


def write_xyz(path, frames, symbols):
    with open(path, "w") as f:
        for title, X in frames:
            f.write(f"{len(symbols)}\n{title}\n")
            for s, x in zip(symbols, X / ANG_BOHR):
                f.write(f"{s:2s} {x[0]:14.8f} {x[1]:14.8f} {x[2]:14.8f}\n")


def main():
    a = parse()
    out = Path(a.out or Path(a.surface).with_name("scans"))
    out.mkdir(parents=True, exist_ok=True)
    sc = Scanner(Surface.load(a.surface))
    sysm, A, B = sc.sys, sc.A, sc.B
    R = np.round(np.arange(a.rmin, a.rmax + 1e-9, a.step), 6)
    rng = np.random.default_rng(a.seed)
    log = open(out / "summary.txt", "w")

    def say(*msg):
        print(*msg, flush=True)
        print(*msg, file=log, flush=True)

    say(f"surface {a.surface}; rigid scans R = {a.rmax} -> {a.rmin} A, step {a.step} A; "
        f"switch {sc.pes.r_in / ANG_BOHR:.2f}-{sc.r_out_ang:.2f} A")

    # ------------------------------------------------ equilibrium monomers
    ds = load_dataset(ROOT / "data_to_fit", sysm)
    i0 = np.flatnonzero((ds.temperature == 300) & (ds.R > 11))[0]
    X0 = sc.build(sc.centred(ds.X[i0], A), sc.centred(ds.X[i0], B), np.eye(3), np.eye(3), 100.0)[0]
    res = minimize(lambda x: sc.base.energy(x.reshape(1, -1, 3))[0], X0.ravel(),
                   jac=lambda x: -sc.base.energy_forces(x.reshape(1, -1, 3))[1].ravel(),
                   method="BFGS", options={"gtol": 1e-10, "maxiter": 5000})
    Xeq = res.x.reshape(-1, 3)
    XA_eq, XB_eq = sc.centred(Xeq, A), sc.centred(Xeq, B)
    N, H_A = XA_eq[0], XA_eq[1:]
    c3 = N - H_A.mean(0)                       # NH3 C3 axis, towards the lone pair
    ch = XB_eq[0] - XB_eq[1]                   # CH axis, H -> C
    say(f"equilibrium monomers: max |grad| {np.abs(res.jac).max():.1e} Eh/bohr; "
        f"r_CH {np.linalg.norm(ch) / ANG_BOHR:.4f} A; "
        f"r_NH {np.mean(np.linalg.norm(H_A - N, axis=1)) / ANG_BOHR:.4f} A")

    scans = []   # dicts: label, kind, XA, XB, rotA, rotB, Ef, Ff, Eb, Fb, E_inf

    def add(label, kind, XA, XB, rotA, rotB):
        Ef, Ff, Eb, Fb, E_inf = sc.scan(XA, XB, rotA, rotB, R)
        scans.append(dict(label=label, kind=kind, XA=XA, XB=XB, rotA=rotA, rotB=rotB,
                          Ef=Ef, Ff=Ff, Eb=Eb, Fb=Fb, E_inf=E_inf))

    # ------------------------------------------------------- named scans
    for label, (u_lab, v_lab) in NAMED.items():
        rotA = align(c3, H_A[0] - N, u_lab)
        rotB = align(ch, (1.0, 0, 0), v_lab)
        add(label, "named", XA_eq, XB_eq, rotA, rotB)

    # ------------------------------------------------------ random scans
    pool = np.flatnonzero((ds.temperature == 300) & (ds.R > 8))
    t0 = time.time()
    for c in range(a.n_random):
        if c % 2 == 0:
            XA, XB, kind, src = XA_eq, XB_eq, "eq", "equilibrium"
        else:
            j = rng.choice(pool)
            XA, XB, kind, src = sc.centred(ds.X[j], A), sc.centred(ds.X[j], B), "zpe", f"{ds.files[ds.file[j]]}:{ds.names[j]}"
        rotA = Rot.random(random_state=rng).as_matrix()
        rotB = Rot.random(random_state=rng).as_matrix()
        add(f"random_{c:04d}", kind, XA, XB, rotA, rotB)
        scans[-1]["source"] = src
        if (c + 1) % 100 == 0:
            print(f"  {c + 1} random scans ({time.time() - t0:.0f}s)", flush=True)

    # -------------------------------------- orientation-minimised path
    E_inf_eq = sc.base.energy(sc.build(XA_eq, XB_eq, np.eye(3), np.eye(3), 1e4))[0]

    def fg(p, Rv, h=1e-5):
        """Energy and orientation gradient in cm-1 (well scaled even where E ~ 1 cm-1)."""
        P = np.concatenate([p[None], p[None] + h * np.eye(6), p[None] - h * np.eye(6)])
        Xs = np.concatenate([sc.build(XA_eq, XB_eq, Rot.from_rotvec(q[:3]).as_matrix(),
                                      Rot.from_rotvec(q[3:]).as_matrix(), Rv) for q in P])
        E = (sc.pes.energy(Xs) - E_inf_eq) * HARTREE_CM
        return E[0], (E[1:7] - E[7:]) / (2 * h)

    eq_idx = [i for i, s in enumerate(scans) if s["kind"] == "eq"]
    E_eq = np.array([scans[i]["Ef"] for i in eq_idx])
    R_path = np.round(np.arange(a.rmax, a.rmin - 1e-9, -a.path_step), 6)
    path, best = [], None
    t0 = time.time()
    for Rv in R_path:
        k = int(np.argmin(np.abs(R - Rv)))
        seeds = [np.concatenate([Rot.from_matrix(scans[eq_idx[i]]["rotA"]).as_rotvec(),
                                 Rot.from_matrix(scans[eq_idx[i]]["rotB"]).as_rotvec()])
                 for i in np.argsort(E_eq[:, k])[:3]]
        if best is not None:
            seeds.insert(0, best)
        trials = [minimize(fg, s, args=(Rv,), jac=True, method="L-BFGS-B",
                           options={"maxiter": 2000, "gtol": 1e-7, "ftol": 1e-14}) for s in seeds]
        t = min(trials, key=lambda r: r.fun)
        best = t.x
        rotA, rotB = Rot.from_rotvec(best[:3]).as_matrix(), Rot.from_rotvec(best[3:]).as_matrix()
        _, F = sc.pes.energy_forces(sc.build(XA_eq, XB_eq, rotA, rotB, Rv))
        path.append((Rv, t.fun, F[0, B, 2].sum() * FORCE, *best))
    path = np.array(path[::-1])                      # ascending R
    say(f"orientation-minimised path: {len(path)} points in {time.time() - t0:.0f}s")

    # ------------------------------------------------------- diagnostics
    say("\n1. Named orientations and the capture path (local maxima of E on the way in from "
        f"{a.rmax} A):")
    for s in scans[:len(NAMED)]:
        mx = local_maxima(s["Ef"])
        txt = ", ".join(f"R {R[k]:.2f} A bump {p:.2f} cm-1 (climb from {a.rmax:.0f} A {c:.1f})"
                        for k, p, c in mx) or "none"
        say(f"   {s['label']:8s} E(15) {s['Ef'][-1]:7.2f}  E(7) {np.interp(7, R, s['Ef']):8.1f}  "
            f"E(5) {np.interp(5, R, s['Ef']):8.1f}  E(3) {s['Ef'][0]:9.1f} cm-1;  maxima: {txt}")
    rise = np.max(path[:-1, 1] - path[1:, 1])        # ascending R: E(R_i) > E(R_i+1) is a rise inward
    say(f"   min_path: E(15) {path[-1, 1]:.2f}, E(7) {np.interp(7, path[:, 0], path[:, 1]):.1f}, "
        f"E(5) {np.interp(5, path[:, 0], path[:, 1]):.1f}, E(3) {path[0, 1]:.1f} cm-1; "
        f"largest rise inward {max(rise, 0.0):.3f} cm-1; F_R max {path[:, 2].max():.3f} cm-1/A "
        f"(< 0 everywhere = attractive all the way)")

    rand = [s for s in scans if s["kind"] in ("eq", "zpe")]
    say(f"\n2. {len(rand)} random rigid scans: local maxima of E(R) by window of R*")
    say(f"   {'window (A)':11s} {'scans: full':>11s} {'baseline':>9s}   bump size (prominence), full surface (cm-1)")
    maxima = {"full": [], "base": []}   # (scan, R*, prominence, climb)
    for i, s in enumerate(rand):
        maxima["full"] += [(i, R[k], p, c) for k, p, c in local_maxima(s["Ef"])]
        maxima["base"] += [(i, R[k], p, c) for k, p, c in local_maxima(s["Eb"])]
    for lo, hi in [(3, 4.5), (4.5, 5), (5, 7), (7, 10), (10, 15)]:
        f = [m for m in maxima["full"] if lo <= m[1] < hi]
        b = [m for m in maxima["base"] if lo <= m[1] < hi]
        p = np.array([m[2] for m in f])
        desc = (f"median {np.median(p):7.3f}  90% {np.percentile(p, 90):7.3f}  max {p.max():8.3f}"
                if len(p) else "-")
        say(f"   {lo:>4.1f}-{hi:<6.1f} {len({m[0] for m in f}):11d} {len({m[0] for m in b}):9d}   {desc}")
    # Bumps the short-range model adds: maxima of the full surface from 4.5 A to r_out with no
    # baseline maximum within 0.5 A on the same scan.
    lo_sw = sc.pes.r_in / ANG_BOHR - 0.5
    spur = [m for m in maxima["full"] if lo_sw <= m[1] < sc.r_out_ang
            and not any(b[0] == m[0] and abs(b[1] - m[1]) < 0.5 for b in maxima["base"])]
    if spur:
        big = max(spur, key=lambda m: m[2])
        p = np.array([m[2] for m in spur])
        say(f"   maxima at {lo_sw:.1f}-{sc.r_out_ang:.1f} A not present in the baseline: {len(spur)} on "
            f"{len({m[0] for m in spur})} scans; bump size median {np.median(p):.3f}, "
            f"> 1 cm-1: {np.sum(p > 1)}, > 5 cm-1: {np.sum(p > 5)}; largest {big[2]:.2f} cm-1 at R {big[1]:.2f} A "
            f"({rand[big[0]]['label']}, climb from {a.rmax:.0f} A {big[3]:.1f} cm-1)")
    else:
        say(f"   maxima at {lo_sw:.1f}-{sc.r_out_ang:.1f} A not present in the baseline: none")
    E15 = np.array([s["Ef"][-1] for s in rand])
    say(f"   E(15 A) over random orientations: {E15.min():.2f} .. {E15.max():.2f} cm-1 "
        f"({np.mean(E15 > 0):.0%} start repulsive)")

    say(f"\n3. Kinks: max over all scans of |F_R(R+h) - 2 F_R(R) + F_R(R-h)|, h = {a.step} A (cm-1/A); "
        "a kink would show as a spike at the switch edges")
    for lo, hi in [(4.0, 4.9), (4.9, 5.1), (5.1, 6.9), (6.9, 7.1), (7.1, 15.0)]:
        sel = (R[1:-1] >= lo) & (R[1:-1] < hi)
        d2 = max(np.abs(np.diff(s["Ff"], 2))[sel].max() for s in rand + scans[:len(NAMED)])
        say(f"   R {lo:4.1f}-{hi:<4.1f} A: {d2:9.4f}")

    # ------------------------------------------------------------ saving
    lab = np.array([s["label"] for s in scans])
    np.savez_compressed(
        out / "scans.npz", R=R, label=lab, kind=np.array([s["kind"] for s in scans]),
        E_full=np.array([s["Ef"] for s in scans]), F_full=np.array([s["Ff"] for s in scans]),
        E_base=np.array([s["Eb"] for s in scans]), F_base=np.array([s["Fb"] for s in scans]),
        E_inf_hartree=np.array([s["E_inf"] for s in scans]),
        rotA=np.array([s["rotA"] for s in scans]), rotB=np.array([s["rotB"] for s in scans]),
        XA_body_ang=np.array([s["XA"] for s in scans]) / ANG_BOHR,
        XB_body_ang=np.array([s["XB"] for s in scans]) / ANG_BOHR,
        path=path, path_columns=np.array(["R_A", "E_cm", "F_R_cm_per_A", "rotvecA(3)", "rotvecB(3)"]),
        symbols=np.array(sysm.symbols), surface=np.array(str(a.surface)))
    cols = ["R_A", "E_full_cm", "E_baseline_cm", "F_R_full_cm_per_A", "F_R_baseline_cm_per_A"]
    (out / "named").mkdir(exist_ok=True)
    for s in scans[:len(NAMED)]:
        start = sc.build(s["XA"], s["XB"], s["rotA"], s["rotB"], a.rmax)[0]
        geo = "\n".join(f"{e:2s} {x[0]:12.6f} {x[1]:12.6f} {x[2]:12.6f}" for e, x in zip(sysm.symbols, start / ANG_BOHR))
        write_table(out / "named" / f"{s['label']}.dat",
                    f"rigid scan '{s['label']}', CH moves along +z; E relative to the same monomers at "
                    f"infinity; F_R = -dE/dR (> 0 repulsive)\nstart geometry at R = {a.rmax} A (Angstrom):\n{geo}",
                    [R, s["Ef"], s["Eb"], s["Ff"], s["Fb"]], cols)
    write_table(out / "min_path.dat", "lowest energy over orientations at each R (equilibrium monomers); "
                "rotation vectors (rad) of NH3 and CH relative to the body frames in scans.npz",
                list(path.T), ["R_A", "E_min_cm", "F_R_cm_per_A", "rvA_x", "rvA_y", "rvA_z", "rvB_x", "rvB_y", "rvB_z"])
    for key, name in [("Ef", "E_full"), ("Eb", "E_baseline"), ("Ff", "F_R_full"), ("Fb", "F_R_baseline")]:
        write_table(out / f"random_{name}.dat", f"{name} of the random rigid scans (cm-1 or cm-1/A); one column "
                    "per scan, labels as in random_orientations.dat", [R] + [s[key] for s in rand],
                    ["R_A"] + [s["label"] for s in rand])
    with open(out / "random_orientations.dat", "w") as f:
        f.write("# label kind monomer_source rotvecA_x rotvecA_y rotvecA_z rotvecB_x rotvecB_y rotvecB_z\n")
        for s in rand:
            rv = np.concatenate([Rot.from_matrix(s["rotA"]).as_rotvec(), Rot.from_matrix(s["rotB"]).as_rotvec()])
            f.write(f"{s['label']} {s['kind']} {s['source']} " + " ".join(f"{v:.8f}" for v in rv) + "\n")
    write_xyz(out / "start_geometries.xyz",
              [(f"{s['label']} {s['kind']} R={a.rmax}A", sc.build(s["XA"], s["XB"], s["rotA"], s["rotB"], a.rmax)[0])
               for s in scans], sysm.symbols)
    say(f"\nsaved {len(scans)} scans + min_path in {out}")
    log.close()
    plots(out, R, scans, path, rand, maxima, sc.pes.r_in / ANG_BOHR, sc.r_out_ang, a.rmax)


def plots(out, R, scans, path, rand, maxima, r_in, r_out, rmax):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def shade(ax):
        ax.axvspan(r_in, r_out, color="0.92", zorder=0)

    named = [s for s in scans if s["kind"] == "named"]
    # Figure 1: named orientations + capture path.
    fig, ax = plt.subplots(2, 2, figsize=(12, 8.5))
    for s in named:
        l, = ax[0, 0].plot(R, s["Ef"], lw=1.3, label=s["label"])
        ax[0, 1].plot(R, s["Ef"], lw=1.3, color=l.get_color())
        ax[0, 1].plot(R, s["Eb"], lw=0.9, ls="--", color=l.get_color())
        ax[1, 0].plot(R, s["Ff"], lw=1.3, color=l.get_color())
        ax[1, 1].plot(R, s["Ff"], lw=1.3, color=l.get_color())
        ax[1, 1].plot(R, s["Fb"], lw=0.9, ls="--", color=l.get_color())
    for a_ in (ax[0, 0], ax[0, 1]):
        a_.plot(path[:, 0], path[:, 1], "k", lw=2.2, label="lowest over orientations")
    for a_ in (ax[1, 0], ax[1, 1]):
        a_.plot(path[:, 0], path[:, 2], "k", lw=2.2)
    zoom = R >= 5.0
    lo = min(min(s["Ef"][zoom].min() for s in named), path[path[:, 0] >= 5, 1].min())
    hi = max(s["Ef"][zoom].max() for s in named)
    ax[0, 1].set(xlim=(5, rmax), ylim=(lo - 5, hi + 5))
    flo = min(min(s["Ff"][zoom].min() for s in named), path[path[:, 0] >= 5, 2].min())
    fhi = max(s["Ff"][zoom].max() for s in named)
    ax[1, 1].set(xlim=(5, rmax), ylim=(flo - 2, fhi + 2))
    ax[0, 0].set(ylim=(max(-6000, path[:, 1].min() - 200), 1500), xlim=(R[0], rmax))
    ax[1, 0].set(xlim=(R[0], rmax), ylim=(-3000, 3000))
    for a_ in ax.ravel():
        shade(a_)
        a_.axhline(0, color="0.6", lw=0.6)
        a_.set_xlabel("R_COM (Å)")
    ax[0, 0].set_ylabel("E − E(∞)  (cm⁻¹)")
    ax[1, 0].set_ylabel("F_R = −dE/dR  (cm⁻¹/Å),  > 0 repulsive")
    ax[0, 0].set_title("Rigid approach, named orientations (solid: full surface)")
    ax[0, 1].set_title("Zoom 5–15 Å (dashed: baseline tail only)")
    ax[0, 0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "fig_named.png", dpi=140)

    # Figure 1b: the lowest-energy path and the orientation along it.
    XA, XB = named[0]["XA"], named[0]["XB"]
    c3, ch = XA[0] - XA[1:].mean(0), XB[0] - XB[1]
    ang = lambda rv, v: np.degrees(np.arccos(Rot.from_rotvec(rv).apply(v / np.linalg.norm(v))[2]))  # noqa: E731
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    ax[0].plot(path[:, 0], path[:, 1], "k.-", ms=3)
    ax[0].set(ylabel="E_min − E(∞)  (cm⁻¹)", yscale="symlog", title="Lowest energy over orientations, rigid monomers")
    ax[1].plot(path[:, 0], path[:, 2], "k.-", ms=3)
    ax[1].set(ylabel="F_R = −dE/dR  (cm⁻¹/Å)", yscale="symlog", title="Radial force along it (< 0: attractive)")
    ax[2].plot(path[:, 0], [ang(p[3:6], c3) for p in path], ".-", ms=3, label="NH₃ C₃ axis (0° = lone pair towards CH)")
    ax[2].plot(path[:, 0], [ang(p[6:9], ch) for p in path], ".-", ms=3, label="CH axis H→C (180° = C end towards NH₃)")
    ax[2].set(ylabel="angle to the approach axis (deg)", ylim=(-5, 185), title="Orientation along the path")
    ax[2].legend(fontsize=8)
    for a_ in ax:
        shade(a_)
        a_.set(xlabel="R_COM (Å)", xlim=(R[0], rmax))
    fig.tight_layout()
    fig.savefig(out / "fig_min_path.png", dpi=140)

    # Figure 2: all random scans.
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for s in rand:
        c = "tab:blue" if s["kind"] == "eq" else "tab:orange"
        ax[0].plot(R, s["Ef"], lw=0.3, alpha=0.25, color=c)
        ax[1].plot(R, s["Ff"], lw=0.3, alpha=0.25, color=c)
    ax[0].plot(path[:, 0], path[:, 1], "k", lw=2)
    ax[1].plot(path[:, 0], path[:, 2], "k", lw=2)
    ax[0].set(xlim=(5, rmax), ylim=(-250, 150), ylabel="E − E(∞)  (cm⁻¹)",
              title=f"{len(rand)} random rigid scans (blue: equilibrium, orange: ZPE-like monomers)")
    ax[1].set(xlim=(5, rmax), ylim=(-120, 80), ylabel="F_R = −dE/dR  (cm⁻¹/Å)", title="radial force")
    for a_ in ax:
        shade(a_)
        a_.axhline(0, color="0.6", lw=0.6)
        a_.set_xlabel("R_COM (Å)")
    fig.tight_layout()
    fig.savefig(out / "fig_random.png", dpi=140)

    # Figure 3: where local maxima sit, full surface vs baseline.
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for key, mk, c in [("base", "o", "0.5"), ("full", "x", "tab:red")]:
        m = np.array([(r, p) for _, r, p, _ in maxima[key]]) if maxima[key] else np.zeros((0, 2))
        ax.scatter(m[:, 0], np.maximum(m[:, 1], 1e-4), marker=mk, s=14, color=c,
                   label=f"{'full surface' if key == 'full' else 'baseline tail only'} ({len(m)})")
    shade(ax)
    ax.set(yscale="log", xlim=(R[0], rmax), xlabel="R* of the local maximum (Å)",
           ylabel="bump size: prominence (cm⁻¹)", title="Local maxima along the random rigid scans")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_maxima.png", dpi=140)

    # Figure 4: eight individual random scans, full vs baseline.
    pick = [s for s in rand if s["kind"] == "eq"][:4] + [s for s in rand if s["kind"] == "zpe"][:4]
    fig, ax = plt.subplots(2, 4, figsize=(15, 6.5))
    for a_, s in zip(ax.ravel(), pick):
        a_.plot(R, s["Ef"], lw=1.4, label="full")
        a_.plot(R, s["Eb"], lw=1.0, ls="--", label="baseline")
        sel = R >= 4.5
        span = s["Ef"][sel]
        a_.set(xlim=(4.5, rmax), ylim=(span.min() - 0.1 * np.ptp(span) - 1, span.max() + 0.1 * np.ptp(span) + 1),
               title=f"{s['label']} ({s['kind']})")
        shade(a_)
        a_.axhline(0, color="0.6", lw=0.6)
    ax[0, 0].legend(fontsize=8)
    for a_ in ax[1]:
        a_.set_xlabel("R_COM (Å)")
    for a_ in ax[:, 0]:
        a_.set_ylabel("E − E(∞)  (cm⁻¹)")
    fig.tight_layout()
    fig.savefig(out / "fig_examples.png", dpi=140)
    print(f"figures: {', '.join(p.name for p in sorted(out.glob('fig_*.png')))}")


if __name__ == "__main__":
    main()
