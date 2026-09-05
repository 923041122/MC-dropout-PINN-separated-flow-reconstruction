#!/usr/bin/env python3
"""
Final publication plotting script for NASA hump audit figures N4/N5/N6.

IMPORTANT
---------
- No training.
- No model evaluation.
- Reads only frozen CSV evidence.
- N6 uses explicit column names from the validated B-PINN held-out Cp files.
- N5 removes empty panels and uses a shared color scale.
- N4 avoids connecting unordered boundary point clouds into misleading diagonal lines.

Run from:
    /hy-tmp/nasa-hump

Outputs:
    paper_ready_nasa_audit_final/
"""

from pathlib import Path
import re, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.tri import Triangulation

ROOT = Path.cwd()
OUT = ROOT / "paper_ready_nasa_audit_final"
OUT.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Utilities
# ----------------------------
def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())

def pick_col(df, candidates):
    cmap = {norm(c): c for c in df.columns}
    for cand in candidates:
        nc = norm(cand)
        if nc in cmap:
            return cmap[nc]
    for cand in candidates:
        nc = norm(cand)
        for k, v in cmap.items():
            if nc in k or k in nc:
                return v
    return None

def xy_cols(df):
    x = pick_col(df, ["x", "x_coord", "coord_x", "x_coordinate"])
    y = pick_col(df, ["y", "y_coord", "coord_y", "y_coordinate"])
    return x, y

def save(fig, stem):
    p1 = OUT / f"{stem}.png"
    p2 = OUT / f"{stem}.pdf"
    fig.savefig(p1, dpi=400, bbox_inches="tight")
    fig.savefig(p2, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {p1}")
    print(f"[saved] {p2}")

def panel(ax, label):
    ax.text(0.015, 0.985, label, transform=ax.transAxes,
            ha="left", va="top", fontsize=12, fontweight="bold")

# ----------------------------
# N4 — sampler audit
# ----------------------------
def make_n4():
    pdir = ROOT / "results_smoke_v12" / "protocol"
    legacy = pd.read_csv(pdir / "legacy_random_box_diagnostic_all_points.csv")
    bad = pd.read_csv(pdir / "legacy_outside_local_reference_window_points.csv")
    valid = pd.read_csv(pdir / "valid_fluid_collocation_points.csv")

    lx, ly = xy_cols(legacy)
    bx, by = xy_cols(bad)
    vx, vy = xy_cols(valid)

    # classify 129/147 if a reason-like field exists
    reason = pick_col(bad, ["reason", "status", "invalid_reason", "classification", "category"])
    below_mask = above_mask = None
    if reason:
        s = bad[reason].astype(str).str.lower()
        below_mask = s.str.contains("below|solid|wall", regex=True)
        above_mask = s.str.contains("above|window|top", regex=True)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), constrained_layout=True)

    ax = axes[0]
    ax.scatter(legacy[lx], legacy[ly], s=8, alpha=0.25, label="Legacy samples")
    if below_mask is not None and below_mask.any():
        ax.scatter(bad.loc[below_mask, bx], bad.loc[below_mask, by],
                   s=22, marker="x", label=f"Below wall / solid-side ({below_mask.sum()})")
    if above_mask is not None and above_mask.any():
        ax.scatter(bad.loc[above_mask, bx], bad.loc[above_mask, by],
                   s=22, marker="s", label=f"Above local reference window ({above_mask.sum()})")
    if below_mask is None and above_mask is None:
        ax.scatter(bad[bx], bad[by], s=22, marker="x",
                   label=f"Outside local reference window ({len(bad)})")

    # IMPORTANT: plot boundary CSVs as point clouds, not connected lines.
    boundary_files = [
        ("Boundary", pdir / "subdomain_boundary_points.csv"),
        ("Left boundary", pdir / "left_subdomain_boundary.csv"),
        ("Right boundary", pdir / "right_subdomain_boundary.csv"),
        ("Top boundary", pdir / "top_subdomain_boundary.csv"),
    ]
    for label, bp in boundary_files:
        if bp.exists():
            d = pd.read_csv(bp)
            xx, yy = xy_cols(d)
            if xx and yy:
                ax.scatter(d[xx], d[yy], s=10, alpha=0.75, label=label)

    ax.set_title("Legacy rectangular sampling")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.15)
    ax.legend(frameon=False, fontsize=8, loc="best")
    panel(ax, "(a)")

    ax = axes[1]
    ax.scatter(valid[vx], valid[vy], s=10, alpha=0.55,
               label=f"Valid-fluid collocation ({len(valid)})")
    for label, bp in boundary_files:
        if bp.exists():
            d = pd.read_csv(bp)
            xx, yy = xy_cols(d)
            if xx and yy:
                ax.scatter(d[xx], d[yy], s=10, alpha=0.75, label=label)
    ax.set_title("Geometry-aware sampling")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.15)
    ax.legend(frameon=False, fontsize=8, loc="best")
    panel(ax, "(b)")

    xs = np.concatenate([legacy[lx].to_numpy(), valid[vx].to_numpy()])
    ys = np.concatenate([legacy[ly].to_numpy(), valid[vy].to_numpy()])
    xpad = 0.02 * max(np.ptp(xs), 1e-12)
    ypad = 0.04 * max(np.ptp(ys), 1e-12)
    for ax in axes:
        ax.set_xlim(xs.min()-xpad, xs.max()+xpad)
        ax.set_ylim(ys.min()-ypad, ys.max()+ypad)

    fig.suptitle("NASA Hump Collocation Sampling Audit", fontsize=14, fontweight="bold")
    fig.text(0.5, -0.015,
             "Legacy diagnostic: 2,000 points; 276 (13.80%) outside the local reference window. "
             "Geometry-aware sampling: 2,000 valid-fluid collocation points.",
             ha="center", fontsize=9)
    save(fig, "Fig_N4_sampler_spatial_map_FINAL")

# ----------------------------
# N5 — residual maps, 1x2 only
# ----------------------------
def find_residual_file(model_token):
    base = ROOT / "reviewer21_cross_model_residual_audit"
    cands = list(base.rglob("*.csv"))
    scored = []
    for p in cands:
        name = p.name.lower()
        if model_token not in name:
            continue
        if "point" not in name and "residual" not in name:
            continue
        try:
            h = pd.read_csv(p, nrows=20)
        except Exception:
            continue
        x, y = xy_cols(h)
        q = pick_col(h, [
            "physics_residual_magnitude", "residual_magnitude",
            "physics_vector", "physics_residual", "residual_vector"
        ])
        if x and y and q:
            scored.append((0 if "pointwise" in name else 1, len(str(p)), p))
    if not scored:
        return None
    scored.sort()
    return scored[0][2]

def make_n5():
    paths = [
        ("Data-only NN", find_residual_file("data_only")),
        ("Standard PINN", find_residual_file("standard_pinn")),
    ]
    paths = [(name, p) for name, p in paths if p is not None]
    if len(paths) != 2:
        print("[N5 skip] Expected Data-only and Standard pointwise residual files.")
        return

    loaded = []
    allz = []
    for name, p in paths:
        df = pd.read_csv(p)
        x, y = xy_cols(df)
        q = pick_col(df, [
            "physics_residual_magnitude", "residual_magnitude",
            "physics_vector", "physics_residual", "residual_vector"
        ])
        xx = pd.to_numeric(df[x], errors="coerce").to_numpy()
        yy = pd.to_numeric(df[y], errors="coerce").to_numpy()
        zz = pd.to_numeric(df[q], errors="coerce").to_numpy()
        mask = np.isfinite(xx) & np.isfinite(yy) & np.isfinite(zz)
        loaded.append((name, p, xx[mask], yy[mask], zz[mask], q))
        allz.append(zz[mask])

    zcat = np.concatenate(allz)
    vmin, vmax = np.nanpercentile(zcat, [2, 98])
    normc = Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), constrained_layout=True)
    mappable = None
    for i, (ax, item) in enumerate(zip(axes, loaded)):
        name, p, x, y, z, q = item
        tri = Triangulation(x, y)
        levels = np.linspace(vmin, vmax, 28)
        mappable = ax.tricontourf(tri, z, levels=levels, norm=normc)
        ax.set_title(name)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(alpha=0.08)
        panel(ax, f"({chr(97+i)})")

    cbar = fig.colorbar(mappable, ax=axes, pad=0.02, shrink=0.96)
    cbar.set_label("Physics-residual magnitude")

    fig.suptitle("NASA Hump Physics-Residual Spatial Diagnostics",
                 fontsize=14, fontweight="bold")
    save(fig, "Fig_N5_residual_field_map_FINAL")

# ----------------------------
# N6 — validated B-PINN Cp gauge audit
# ----------------------------
def make_n6():
    hdir = ROOT / "baseline_suite" / "nasa_hump" / "bpinn_dropout" / "formal" / "archive" / "heldout"
    point_path = hdir / "heldout_cp_pointwise_predictions.csv"
    err_path = hdir / "heldout_cp_errors_raw_and_gauge_aligned.csv"

    if not point_path.exists() or not err_path.exists():
        print("[N6 skip] Required validated held-out Cp files not found.")
        return

    pp = pd.read_csv(point_path)
    ee = pd.read_csv(err_path)

    # Explicit, validated semantic columns.
    # Adjust ONLY if your exact file headers differ; do not use heuristic guessing for N6.
    xcol = pick_col(pp, ["x"])
    ycol = pick_col(pp, ["y"])
    cp_true_col = pick_col(pp, ["cp", "cp_true", "cp_reference"])
    raw_pred_col = pick_col(pp, ["bpinn_dropout_cp_pred_raw"])
    aligned_pred_col = pick_col(pp, ["bpinn_dropout_cp_pred_gauge_aligned"])

    if not all([xcol, ycol, cp_true_col, raw_pred_col, aligned_pred_col]):
        print("[N6 skip] Explicit validated Cp columns were not found.")
        print("pointwise columns:", list(pp.columns))
        return

    x = pd.to_numeric(pp[xcol], errors="coerce").to_numpy()
    y = pd.to_numeric(pp[ycol], errors="coerce").to_numpy()
    cp_true = pd.to_numeric(pp[cp_true_col], errors="coerce").to_numpy()
    raw_pred = pd.to_numeric(pp[raw_pred_col], errors="coerce").to_numpy()
    ali_pred = pd.to_numeric(pp[aligned_pred_col], errors="coerce").to_numpy()

    raw_err = raw_pred - cp_true
    ali_err = ali_pred - cp_true

    raw_rmse = float(np.sqrt(np.mean(raw_err**2)))
    ali_rmse = float(np.sqrt(np.mean(ali_err**2)))
    shift = float(np.mean(ali_pred - raw_pred))
    shift_std = float(np.std(ali_pred - raw_pred))

    print(f"[N6 check] raw RMSE = {raw_rmse:.9f}")
    print(f"[N6 check] gauge RMSE = {ali_rmse:.9f}")
    print(f"[N6 check] gauge shift = {shift:.9f}, std = {shift_std:.3e}")

    # Stop if the frozen metrics are not reproduced.
    if abs(raw_rmse - 0.054260) > 5e-4 or abs(ali_rmse - 0.051172) > 5e-4:
        raise RuntimeError("N6 frozen metric check failed. Figure not generated.")

    vmax = np.nanpercentile(np.abs(np.concatenate([raw_err, ali_err])), 98)
    vmin = -vmax
    levels = np.linspace(vmin, vmax, 25)
    shared_norm = Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    for i, (ax, z, title) in enumerate([
        (axes[0], raw_err, r"Raw $C_p$ error"),
        (axes[1], ali_err, r"Gauge-aligned $C_p$ error"),
    ]):
        tri = Triangulation(x, y)
        m = ax.tricontourf(tri, z, levels=levels, norm=shared_norm)
        ax.scatter(x, y, s=5, alpha=0.15)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        panel(ax, f"({chr(97+i)})")

    cbar = fig.colorbar(m, ax=axes, pad=0.02, shrink=0.96)
    cbar.set_label(r"$C_p$ prediction error")

    fig.suptitle("NASA Hump Wall-Pressure Gauge Audit",
                 fontsize=14, fontweight="bold")
    fig.text(0.5, -0.015,
             rf"B-PINN held-out wall pressure: raw RMSE = {raw_rmse:.5f}, "
             rf"gauge-aligned RMSE = {ali_rmse:.5f}; constant gauge shift = {shift:.5f}.",
             ha="center", fontsize=9)
    save(fig, "Fig_N6_pressure_gauge_error_map_FINAL")

def main():
    print("=== FINAL NASA publication figures ===")
    print("No training. No model evaluation. Frozen evidence only.\n")
    make_n4()
    make_n5()
    make_n6()
    print("\nDone. Output directory:")
    print(OUT)

if __name__ == "__main__":
    main()
