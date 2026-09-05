#!/usr/bin/env python3
"""
Publication-grade NASA hump audit visualizations.

Purpose
-------
This script DOES NOT train, evaluate, or re-run any model.
It only reads already-existing CSV evidence and creates spatial visualizations.

Outputs (when supported by available pointwise evidence)
---------------------------------------------------------
paper_ready_nasa_audit_publication/
  Fig_N4_sampler_spatial_map.png/.pdf
  Fig_N5_residual_field_map.png/.pdf          [only if real x/y + residual point data exist]
  Fig_N6_pressure_gauge_error_map.png/.pdf    [only if real x/y + pressure point data exist]
  publication_visualization_manifest.txt

Canonical fixed evidence used by N4
-----------------------------------
results_smoke_v12/protocol/
  legacy_random_box_diagnostic_all_points.csv
  legacy_outside_local_reference_window_points.csv
  valid_fluid_collocation_points.csv
  left_subdomain_boundary.csv
  right_subdomain_boundary.csv
  top_subdomain_boundary.csv
  subdomain_boundary_points.csv

Important scientific rule
-------------------------
No cloud/contour map is synthesized from aggregate RMSE values. N5/N6 are created
only if actual pointwise coordinates and pointwise physical/error quantities are found.
"""

from pathlib import Path
import sys, re, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

ROOT = Path.cwd()
OUT = ROOT / "paper_ready_nasa_audit_publication"
OUT.mkdir(parents=True, exist_ok=True)

def log(msg):
    print(msg)

def read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        log(f"[skip] cannot read {path}: {e}")
        return None

def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())

def pick_col(df, candidates):
    if df is None:
        return None
    cmap = {norm(c): c for c in df.columns}
    for cand in candidates:
        nc = norm(cand)
        if nc in cmap:
            return cmap[nc]
    # partial matches
    for cand in candidates:
        nc = norm(cand)
        for k, v in cmap.items():
            if nc in k or k in nc:
                return v
    return None

def xy_cols(df):
    x = pick_col(df, ["x", "x_coord", "x_coordinate", "coord_x", "x_nd", "x_over_c"])
    y = pick_col(df, ["y", "y_coord", "y_coordinate", "coord_y", "y_nd", "y_over_c"])
    return x, y

def save(fig, stem):
    png = OUT / f"{stem}.png"
    pdf = OUT / f"{stem}.pdf"
    fig.savefig(png, dpi=350, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    log(f"[saved] {png}")
    log(f"[saved] {pdf}")
    return [png, pdf]

def annotate_panel(ax, label):
    ax.text(0.015, 0.985, label, transform=ax.transAxes, ha="left", va="top",
            fontsize=12, fontweight="bold")

# ----------------------------------------------------------------------
# N4: spatial sampler audit — deterministic, canonical, no model inference
# ----------------------------------------------------------------------
def make_sampler_map():
    pdir = ROOT / "results_smoke_v12" / "protocol"
    all_path = pdir / "legacy_random_box_diagnostic_all_points.csv"
    bad_path = pdir / "legacy_outside_local_reference_window_points.csv"
    valid_path = pdir / "valid_fluid_collocation_points.csv"

    required = [all_path, bad_path, valid_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        log("[N4 skip] missing canonical sampler files:")
        for p in missing: log("  " + p)
        return []

    legacy = read_csv(all_path)
    bad = read_csv(bad_path)
    valid = read_csv(valid_path)
    lx, ly = xy_cols(legacy)
    bx, by = xy_cols(bad)
    vx, vy = xy_cols(valid)
    if not all([lx, ly, bx, by, vx, vy]):
        log("[N4 skip] x/y columns could not be identified.")
        log(f" legacy columns={list(legacy.columns)}")
        log(f" bad columns={list(bad.columns)}")
        log(f" valid columns={list(valid.columns)}")
        return []

    # classify invalid points if explicit flags exist
    below_col = pick_col(bad, ["below_wall", "inside_solid", "belowwall", "solid"])
    above_col = pick_col(bad, ["above_local_reference_window", "above_reference_window",
                              "above_window", "outside_top", "above_local_window"])

    # Prefer explicit status/reason text if boolean flags are absent.
    reason_col = pick_col(bad, ["reason", "status", "invalid_reason", "classification", "category"])

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), constrained_layout=True)

    # Boundaries, if available
    boundary_files = [
        pdir / "subdomain_boundary_points.csv",
        pdir / "left_subdomain_boundary.csv",
        pdir / "right_subdomain_boundary.csv",
        pdir / "top_subdomain_boundary.csv",
    ]
    boundaries = []
    for bp in boundary_files:
        if bp.exists():
            d = read_csv(bp)
            if d is not None:
                xx, yy = xy_cols(d)
                if xx and yy:
                    boundaries.append((d[xx].to_numpy(), d[yy].to_numpy()))

    # (a) Legacy
    ax = axes[0]
    ax.scatter(legacy[lx], legacy[ly], s=8, alpha=0.30, label="Legacy random-box samples")
    # Plot invalid categories from real evidence only
    if reason_col:
        reason = bad[reason_col].astype(str).str.lower()
        below_mask = reason.str.contains("below|solid|wall", regex=True)
        above_mask = reason.str.contains("above|window|top", regex=True)
        other_mask = ~(below_mask | above_mask)
        if below_mask.any():
            ax.scatter(bad.loc[below_mask, bx], bad.loc[below_mask, by], s=18, marker="x",
                       label=f"Below wall / solid ({below_mask.sum()})")
        if above_mask.any():
            ax.scatter(bad.loc[above_mask, bx], bad.loc[above_mask, by], s=18, marker="^",
                       label=f"Above local reference window ({above_mask.sum()})")
        if other_mask.any():
            ax.scatter(bad.loc[other_mask, bx], bad.loc[other_mask, by], s=18, marker="s",
                       label=f"Other outside-window ({other_mask.sum()})")
    elif below_col or above_col:
        plotted = np.zeros(len(bad), dtype=bool)
        if below_col:
            mask = bad[below_col].astype(bool).to_numpy()
            plotted |= mask
            ax.scatter(bad.loc[mask, bx], bad.loc[mask, by], s=18, marker="x",
                       label=f"Below wall / solid ({mask.sum()})")
        if above_col:
            mask = bad[above_col].astype(bool).to_numpy()
            plotted |= mask
            ax.scatter(bad.loc[mask, bx], bad.loc[mask, by], s=18, marker="^",
                       label=f"Above local reference window ({mask.sum()})")
        if (~plotted).any():
            ax.scatter(bad.loc[~plotted, bx], bad.loc[~plotted, by], s=18, marker="s",
                       label=f"Other outside-window ({(~plotted).sum()})")
    else:
        ax.scatter(bad[bx], bad[by], s=18, marker="x",
                   label=f"Outside local reference window ({len(bad)})")

    for xline, yline in boundaries:
        ax.plot(xline, yline, linewidth=1.2)

    ax.set_title("Legacy rectangular/random-box sampling")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.grid(alpha=0.15)
    annotate_panel(ax, "(a)")

    # (b) Geometry aware
    ax = axes[1]
    ax.scatter(valid[vx], valid[vy], s=9, alpha=0.55, label=f"Valid-fluid collocation ({len(valid)})")
    for xline, yline in boundaries:
        ax.plot(xline, yline, linewidth=1.2)
    ax.set_title("Geometry-aware valid-fluid sampling")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.grid(alpha=0.15)
    annotate_panel(ax, "(b)")

    # same limits for a fair visual comparison
    xs = np.concatenate([legacy[lx].to_numpy(), valid[vx].to_numpy()])
    ys = np.concatenate([legacy[ly].to_numpy(), valid[vy].to_numpy()])
    xpad = 0.02 * max(np.ptp(xs), 1e-12)
    ypad = 0.04 * max(np.ptp(ys), 1e-12)
    for ax in axes:
        ax.set_xlim(np.nanmin(xs)-xpad, np.nanmax(xs)+xpad)
        ax.set_ylim(np.nanmin(ys)-ypad, np.nanmax(ys)+ypad)

    fig.suptitle("NASA Hump Collocation Audit: Legacy vs Geometry-Aware Sampling",
                 fontsize=14, fontweight="bold")
    fig.text(0.5, -0.015,
             "Legacy diagnostic: 2,000 points; 276 (13.80%) outside the local reference window. "
             "Geometry-aware sampling: 2,000 valid-fluid collocation points.",
             ha="center", fontsize=9)
    return save(fig, "Fig_N4_sampler_spatial_map")

# ----------------------------------------------------------------------
# Helper: discover genuine pointwise CSVs for N5/N6.
# The script never invents fields from aggregate values.
# ----------------------------------------------------------------------
def candidate_csvs(base_dirs, max_mb=200):
    out = []
    for b in base_dirs:
        if not b.exists():
            continue
        for p in b.rglob("*.csv"):
            try:
                if p.stat().st_size <= max_mb * 1024 * 1024:
                    out.append(p)
            except OSError:
                pass
    return out

def inspect_columns(path, nrows=50):
    try:
        return pd.read_csv(path, nrows=nrows)
    except Exception:
        return None

def find_xy_quantity_csvs(base_dirs, quantity_groups):
    """
    quantity_groups: list of (logical_name, candidate_columns)
    returns list of dicts, one per file with x/y and >=1 quantity
    """
    hits = []
    for p in candidate_csvs(base_dirs):
        h = inspect_columns(p)
        if h is None:
            continue
        x, y = xy_cols(h)
        if not x or not y:
            continue
        qhits = {}
        for logical, cands in quantity_groups:
            c = pick_col(h, cands)
            if c:
                qhits[logical] = c
        if qhits:
            hits.append({"path": p, "x": x, "y": y, "quantities": qhits})
    return hits

def robust_contour(ax, x, y, z, title, cbar_label):
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[mask], y[mask], z[mask]
    if len(z) < 10:
        return False
    # Robust symmetric or positive scale is handled by data itself.
    try:
        tri = Triangulation(x, y)
        lo, hi = np.nanpercentile(z, [2, 98])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = np.nanmin(z), np.nanmax(z)
        levels = np.linspace(lo, hi, 24)
        m = ax.tricontourf(tri, z, levels=levels)
    except Exception:
        m = ax.scatter(x, y, c=z, s=10)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("auto")
    cb = plt.colorbar(m, ax=ax, pad=0.02)
    cb.set_label(cbar_label)
    return True

# ----------------------------------------------------------------------
# N5: residual field — only if pointwise residual evidence exists
# ----------------------------------------------------------------------
def make_residual_map():
    bases = [
        ROOT / "reviewer21_cross_model_residual_audit",
        ROOT / "results_reference_audit_m4",
        ROOT / "results_reference_audit_m3",
    ]
    quantity_groups = [
        ("physics_vector_rmse", ["physics_vector", "physics_residual", "residual_vector",
                                 "physics_residual_magnitude", "residual_magnitude"]),
        ("continuity", ["continuity_residual", "continuity", "mass_residual"]),
        ("fx", ["fx", "momentum_x_residual", "x_momentum_residual", "residual_x"]),
        ("fy", ["fy", "momentum_y_residual", "y_momentum_residual", "residual_y"]),
    ]
    hits = find_xy_quantity_csvs(bases, quantity_groups)
    if not hits:
        log("[N5 skip] No real pointwise x/y + residual CSV found.")
        log("          Aggregate reviewer21_cross_model_residual_summary.csv is not enough for a cloud map.")
        return []

    # Prefer files whose names suggest pointwise/field/residual.
    hits.sort(key=lambda h: (
        0 if re.search(r"point|field|residual", h["path"].name, re.I) else 1,
        len(str(h["path"]))
    ))

    selected = hits[:4]
    n = len(selected)
    fig, axes = plt.subplots(1, n, figsize=(5.2*n, 4.8), constrained_layout=True)
    if n == 1:
        axes = [axes]

    plotted = 0
    for i, (ax, hit) in enumerate(zip(axes, selected)):
        df = read_csv(hit["path"])
        if df is None:
            continue
        xcol, ycol = hit["x"], hit["y"]
        qname, qcol = next(iter(hit["quantities"].items()))
        x = pd.to_numeric(df[xcol], errors="coerce").to_numpy()
        y = pd.to_numeric(df[ycol], errors="coerce").to_numpy()
        z = pd.to_numeric(df[qcol], errors="coerce").to_numpy()
        title = hit["path"].parent.name + " / " + hit["path"].stem
        if robust_contour(ax, x, y, z, title, qcol):
            annotate_panel(ax, f"({chr(97+i)})")
            plotted += 1

    if plotted == 0:
        plt.close(fig)
        log("[N5 skip] Candidate files were found, but none contained usable numeric pointwise residuals.")
        return []

    fig.suptitle("NASA Hump Physics-Residual Spatial Diagnostics",
                 fontsize=14, fontweight="bold")
    return save(fig, "Fig_N5_residual_field_map")

# ----------------------------------------------------------------------
# N6: pressure gauge field — only if pointwise pressure evidence exists
# ----------------------------------------------------------------------
def make_pressure_map():
    bases = [
        ROOT / "baseline_suite" / "nasa_hump",
        ROOT / "results_smoke_v12",
        ROOT / "secondary_locked_protocol",
    ]
    pressure_groups = [
        ("p_true", ["p_true", "cp_true", "cp_reference", "pressure_true", "reference_cp"]),
        ("p_pred", ["p_pred", "cp_pred", "pressure_pred", "predicted_cp"]),
        ("raw_error", ["p_raw_error", "cp_raw_error", "raw_pressure_error", "raw_cp_error"]),
        ("gauge_error", ["p_gauge_error", "cp_gauge_error", "gauge_aligned_error",
                         "gauge_aligned_pressure_error"]),
    ]
    hits = find_xy_quantity_csvs(bases, pressure_groups)
    if not hits:
        log("[N6 skip] No real pointwise x/y + pressure prediction/error CSV found.")
        log("          nasa_six_model_final_rmse.csv contains aggregate metrics only, so no error cloud is fabricated.")
        return []

    # Prefer pointwise/heldout/final files.
    hits.sort(key=lambda h: (
        0 if re.search(r"pointwise|heldout|final", h["path"].name, re.I) else 1,
        len(str(h["path"]))
    ))
    hit = hits[0]
    df = read_csv(hit["path"])
    if df is None:
        return []
    xcol, ycol = hit["x"], hit["y"]
    x = pd.to_numeric(df[xcol], errors="coerce").to_numpy()
    y = pd.to_numeric(df[ycol], errors="coerce").to_numpy()

    q = hit["quantities"]
    raw = None
    gau = None

    # Direct error columns are preferred.
    if "raw_error" in q:
        raw = pd.to_numeric(df[q["raw_error"]], errors="coerce").to_numpy()
    if "gauge_error" in q:
        gau = pd.to_numeric(df[q["gauge_error"]], errors="coerce").to_numpy()

    # If only reference/prediction are present, derive RAW error only.
    # Gauge-aligned error is derived only when a clearly named gauge-shift/aligned prediction exists;
    # otherwise we do not guess the alignment convention.
    if raw is None and "p_true" in q and "p_pred" in q:
        ref = pd.to_numeric(df[q["p_true"]], errors="coerce").to_numpy()
        pred = pd.to_numeric(df[q["p_pred"]], errors="coerce").to_numpy()
        raw = pred - ref

    aligned_pred_col = pick_col(df, ["p_pred_gauge_aligned", "cp_pred_gauge_aligned",
                                     "gauge_aligned_pred", "aligned_pressure_pred"])
    if gau is None and aligned_pred_col and "p_true" in q:
        ref = pd.to_numeric(df[q["p_true"]], errors="coerce").to_numpy()
        ap = pd.to_numeric(df[aligned_pred_col], errors="coerce").to_numpy()
        gau = ap - ref

    panels = []
    if raw is not None:
        panels.append(("Raw pressure error", raw))
    if gau is not None:
        panels.append(("Gauge-aligned pressure error", gau))

    if not panels:
        log(f"[N6 skip] Found {hit['path']} but cannot form a scientifically traceable raw/aligned error field.")
        return []

    fig, axes = plt.subplots(1, len(panels), figsize=(6.2*len(panels), 5.0), constrained_layout=True)
    if len(panels) == 1:
        axes = [axes]
    for i, (ax, (title, z)) in enumerate(zip(axes, panels)):
        robust_contour(ax, x, y, z, title, "Pressure error")
        annotate_panel(ax, f"({chr(97+i)})")
    fig.suptitle("NASA Hump Pressure-Gauge Spatial Diagnostics",
                 fontsize=14, fontweight="bold")
    fig.text(0.5, -0.015, f"Source: {hit['path']}", ha="center", fontsize=8)
    return save(fig, "Fig_N6_pressure_gauge_error_map")

def main():
    manifest = []
    log("=== NASA publication visualization layer ===")
    log("No training. No model evaluation. No synthetic cloud maps from aggregate metrics.\n")

    for name, fn in [
        ("N4 sampler spatial map", make_sampler_map),
        ("N5 residual field map", make_residual_map),
        ("N6 pressure-gauge field map", make_pressure_map),
    ]:
        log(f"\n--- {name} ---")
        files = fn()
        manifest.append((name, files))

    mp = OUT / "publication_visualization_manifest.txt"
    with mp.open("w", encoding="utf-8") as f:
        f.write("NASA Hump publication visualization manifest\n")
        f.write("="*48 + "\n\n")
        for name, files in manifest:
            f.write(name + "\n")
            if files:
                for p in files:
                    f.write(f"  CREATED: {p}\n")
            else:
                f.write("  NOT CREATED: required genuine pointwise evidence was not found.\n")
            f.write("\n")
        f.write("Scientific safeguard: aggregate RMSE/count summaries are never converted into fake spatial fields.\n")
    log(f"\n[saved] {mp}")
    log("\nDone.")

if __name__ == "__main__":
    main()
