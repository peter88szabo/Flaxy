#!/usr/bin/env python
"""Fit the NH3 + CH capture surface and compare short-range models.

Steps: load data and the fixed validation split -> energy domain -> two-stage
baseline (tail from clean long range, monomers from R >= mono_rmin) -> S3 KRR
reference -> S3 Gaussian tensor train -> error tables, models, report.

Example:
  python scripts/fit_capture.py --out runs/explore1 --models krr tt
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baseline import Baseline  # noqa: E402
from src.data import load_dataset, load_split, validation_mask  # noqa: E402
from src.features import PairFeatures  # noqa: E402
from src.geometry import pair_vectors  # noqa: E402
from src.krr import SymKRR  # noqa: E402
from src.report import error_table, format_table  # noqa: E402
from src.surface import Surface  # noqa: E402
from src.system import nh3_ch  # noqa: E402
from src.tt import GaussianTT  # noqa: E402
from src.units import ANG_BOHR, HARTREE_CM, HARTREE_KCAL  # noqa: E402
from src.weights import KJ_PER_KCAL, energy_weights, in_domain  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(ROOT / "data_to_fit"))
    p.add_argument("--split", default=str(ROOT / "splits/validation_seed20261120.json"))
    p.add_argument("--out", required=True)
    p.add_argument("--models", nargs="+", default=["krr", "tt"], choices=["krr", "tt"])
    p.add_argument("--ecut-kj", type=float, default=200.0, help="fit only E_rel below this (kJ/mol)")
    p.add_argument("--tail-rmin", type=float, default=9.0, help="Angstrom; stage-1 tail rows (300 K)")
    p.add_argument("--mono-rmin", type=float, default=7.0, help="Angstrom; stage-2 monomer rows")
    p.add_argument("--nh3-degree", type=int, default=8)
    p.add_argument("--ch-degree", type=int, default=8)
    p.add_argument("--beta", type=float, default=1.0, help="Tang-Toennies damping, 1/bohr")
    p.add_argument("--r-in", type=float, default=5.0, help="Angstrom")
    p.add_argument("--r-out", type=float, default=7.0, help="Angstrom")
    p.add_argument("--inter-range", type=float, default=4.0, help="bohr, Morse range of intermolecular pairs")
    p.add_argument("--tune-frac", type=float, default=0.12)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--krr-lengths", type=float, nargs="+", default=[0.6, 0.8, 1.1])
    p.add_argument("--krr-ridge", type=float, default=1e-6)
    p.add_argument("--tt-gauss", type=int, default=6)
    p.add_argument("--tt-rank", type=int, default=8)
    p.add_argument("--tt-sweeps", type=int, default=6)
    p.add_argument("--tt-ridge", type=float, default=1e-7)
    p.add_argument("--tt-prox", type=float, default=1e-5)
    p.add_argument("--tt-width", type=float, default=1.0)
    return p.parse_args()


class Log:
    def __init__(self, path):
        self.f = open(path, "a")
        self.t0 = time.time()

    def __call__(self, *msg):
        line = f"[{time.time() - self.t0:7.1f}s] " + " ".join(str(m) for m in msg)
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def wrmse_cm(pred, target, w):
    return float(np.sqrt(np.sum(w * (pred - target) ** 2) / np.sum(w)) * HARTREE_CM)


def main():
    a = parse()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = Log(out / "fit.log")
    log("config", json.dumps(vars(a)))

    system = nh3_ch()
    log(system.summary())
    ds = load_dataset(a.data, system)
    valid = validation_mask(ds, load_split(a.split))
    e_asym = float(ds.E[~valid & (ds.R >= a.tail_rmin)].min())
    e_rel = (ds.E - e_asym) * HARTREE_KCAL
    dom = in_domain(e_rel, a.ecut_kj)
    w = np.where(dom, energy_weights(e_rel, ds.temperature), 0.0)
    train = ~valid & dom
    log(f"data {len(ds)}; dissociation reference {e_asym:.8f} Eh; cut {a.ecut_kj} kJ/mol = "
        f"{a.ecut_kj / KJ_PER_KCAL:.2f} kcal/mol; train {train.sum()}, valid in-domain "
        f"{(valid & dom).sum()}, valid out-of-domain {(valid & ~dom).sum()}")

    # ------------------------------------------------------------- baseline
    T, R = ds.temperature, ds.R
    feats = PairFeatures(system, inter=a.inter_range)
    base = Baseline.build(system, ds.X[train], feats, degrees=(a.nh3_degree, a.ch_degree), beta=a.beta)
    tr = np.flatnonzero(train)
    binfo = base.fit_two_stage(ds.X[tr], ds.E[tr], w[tr], tail_rows=(T[tr] == 300) & (R[tr] >= a.tail_rmin),
                               mono_rows=R[tr] >= a.mono_rmin, ridge=1e-12)
    log("baseline tail (coef, stderr):", {k: (round(v[0], 5), round(v[1], 5)) for k, v in binfo["tail"].items()})
    log(f"baseline stage-1 RMSE {binfo['rmse'] * HARTREE_CM:.2f} cm-1 (n {binfo['n']}), "
        f"stage-2 monomer RMSE {binfo['mono_rmse'] * HARTREE_CM:.2f} cm-1 (n {binfo['mono_n']})")

    r_in, r_out = a.r_in * ANG_BOHR, a.r_out * ANG_BOHR
    surf0 = Surface(system, base, feats, None, r_in, r_out)
    target, scale = surf0.inner_residual_target(ds.X, ds.E)
    _, r = pair_vectors(ds.X, system)
    Y = feats(r)[0][:, system.active_pairs]
    perms = system.active_pair_perms
    inner = train & (scale > 0)
    tune = np.random.default_rng(a.seed).random(len(ds)) < a.tune_frac
    fit_rows, tune_rows = inner & ~tune, inner & tune
    log(f"inner rows {inner.sum()} (fit {fit_rows.sum()}, tune {tune_rows.sum()}); "
        f"residual std {np.std(target[inner]) * HARTREE_CM:.0f} cm-1")

    report = {"config": vars(a), "e_asym": e_asym, "baseline": binfo, "models": {}}
    surfaces = {"baseline": surf0}

    # ------------------------------------------------------------------ KRR
    if "krr" in a.models:
        scores = {}
        for length in a.krr_lengths:
            t0 = time.time()
            k = SymKRR(perms, length, a.krr_ridge).fit(Y[fit_rows], target[fit_rows], w[fit_rows], scale[fit_rows])
            scores[length] = wrmse_cm(scale[tune_rows] * k.predict(Y[tune_rows]), target[tune_rows], w[tune_rows])
            log(f"KRR length {length}: tune wRMSE {scores[length]:.1f} cm-1 ({time.time() - t0:.0f}s)")
        best = min(scores, key=scores.get)
        krr = SymKRR(perms, best, a.krr_ridge).fit(Y[inner], target[inner], w[inner], scale[inner])
        surfaces["krr"] = Surface(system, base, feats, krr, r_in, r_out)
        report["models"]["krr"] = {"length": best, "tune_scores": scores, "n_centers": int(inner.sum()),
                                   "n_params": int(inner.sum())}
        log(f"KRR refit on all inner rows with length {best}")

    # ------------------------------------------------------------------- TT
    if "tt" in a.models:
        tt = GaussianTT.initialize(Y[inner], perms, system.active_pair_type, n_gauss=a.tt_gauss,
                                   width_factor=a.tt_width, rank=2, seed=a.seed)
        hist = tt.fit(Y[fit_rows], target[fit_rows], w[fit_rows], scale[fit_rows],
                      Y[tune_rows], target[tune_rows], w[tune_rows], scale[tune_rows],
                      sweeps=a.tt_sweeps, rank_cap=a.tt_rank, ridge=a.tt_ridge, prox=a.tt_prox,
                      log=lambda m: log(m))
        surfaces["tt"] = Surface(system, base, feats, tt, r_in, r_out)
        report["models"]["tt"] = {"ranks": tt.ranks, "n_params": tt.n_params,
                                  "history": [(h[0], h[1], h[2]) for h in hist]}
        log(f"TT ranks {tt.ranks}, {tt.n_params} parameters")

    # ------------------------------------------------------------ evaluation
    md = [f"# Flaxy NH3 + CH capture fit: {out.name}", "",
          f"Dissociation reference {e_asym:.8f} Eh; domain E_rel < {a.ecut_kj} kJ/mol. "
          f"Switch {a.r_in}-{a.r_out} Angstrom. Errors are model minus reference.", ""]
    for name, s in surfaces.items():
        t0 = time.time()
        pred = np.concatenate([s.energy(ds.X[i:i + 2000]) for i in range(0, len(ds), 2000)])
        report["models"].setdefault(name, {})["timing_s_per_geom"] = (time.time() - t0) / len(ds)
        tables = {"valid_in_domain": error_table(ds.subset(valid & dom), pred[valid & dom], e_asym),
                  "train_in_domain": error_table(ds.subset(train), pred[train], e_asym),
                  "valid_out_of_domain": error_table(ds.subset(valid & ~dom), pred[valid & ~dom], e_asym)}
        report["models"][name]["errors"] = tables
        for k in ("valid_in_domain", "train_in_domain"):
            md.append(format_table(f"{name}: {k}", tables[k]))
        log(f"{name}: valid in-domain RMSE {tables['valid_in_domain']['all']['rmse_cm']:.1f} cm-1, "
            f"train {tables['train_in_domain']['all']['rmse_cm']:.1f} cm-1")
        s.save(out / f"surface_{name}.npz", meta={"name": name, "e_asym": e_asym, "config": vars(a)})

    (out / "report.json").write_text(json.dumps(report, indent=1, default=float))
    (out / "report.md").write_text("\n".join(md))
    log(f"wrote {out}/report.md, report.json and surface_*.npz")


if __name__ == "__main__":
    main()
