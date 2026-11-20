"""Error tables by file, R shell and energy window (cm^-1 and kcal/mol)."""
import numpy as np

from .units import HARTREE_CM, HARTREE_KCAL

WINDOWS_KCAL = [(-60, 0), (0, 10), (10, 30), (30, 60), (60, 120), (120, 1e9)]


def stats(err_hartree):
    e = np.asarray(err_hartree) * HARTREE_CM
    if len(e) == 0:
        return {"n": 0}
    return {"n": int(len(e)), "rmse_cm": float(np.sqrt(np.mean(e**2))),
            "mae_cm": float(np.mean(np.abs(e))), "max_cm": float(np.max(np.abs(e))),
            "rmse_kcal": float(np.sqrt(np.mean(e**2)) / HARTREE_CM * HARTREE_KCAL)}


def error_table(ds, pred, e_asym):
    """Nested dict of error statistics for predictions on a Dataset."""
    err = pred - ds.E
    e_rel = (ds.E - e_asym) * HARTREE_KCAL
    out = {"all": stats(err), "by_file": {}, "by_window_kcal": {}, "by_shell": {}}
    for f, stem in enumerate(ds.files):
        m = ds.file == f
        if m.any():
            out["by_file"][stem] = stats(err[m])
            out["by_shell"][stem] = {f"{np.mean(ds.R[m & (ds.shell == s)]):.2f}": stats(err[m & (ds.shell == s)])
                                     for s in np.unique(ds.shell[m])}
    for lo, hi in WINDOWS_KCAL:
        m = (e_rel >= lo) & (e_rel < hi)
        out["by_window_kcal"][f"{lo}..{hi if hi < 1e8 else 'inf'}"] = stats(err[m])
    return out


def format_table(name, table):
    lines = [f"### {name}", "", "| subset | n | RMSE (cm⁻¹) | MAE (cm⁻¹) | max (cm⁻¹) | RMSE (kcal/mol) |",
             "|---|---:|---:|---:|---:|---:|"]

    def row(label, s):
        if s["n"]:
            lines.append(f"| {label} | {s['n']} | {s['rmse_cm']:.2f} | {s['mae_cm']:.2f} | "
                         f"{s['max_cm']:.1f} | {s['rmse_kcal']:.4f} |")
    row("all", table["all"])
    for k, s in table["by_file"].items():
        row(k, s)
    for k, s in table["by_window_kcal"].items():
        row(f"E_rel {k} kcal/mol", s)
    return "\n".join(lines) + "\n"
