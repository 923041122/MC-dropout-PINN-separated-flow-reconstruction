#!/usr/bin/env python3
"""
Create paper-ready NASA Audit figures for the 8/30 audit day.
No training, no checkpoint loading, no model evaluation.

Run from the NASA project root, e.g.:
    cd /hy-tmp/nasa-hump
    python make_nasa_audit_figures.py

Outputs:
    paper_ready_nasa_audit/
      Fig_N1_sampler_audit.png/.pdf
      Fig_N2_reference_residual_audit.png/.pdf
      Fig_N3_pressure_gauge_audit.png/.pdf
      Fig_N0_nasa_audit_master.png/.pdf
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(".")
OUT = ROOT / "paper_ready_nasa_audit"
OUT.mkdir(parents=True, exist_ok=True)

# Canonical evidence paths already locked during the audit.
RESIDUAL_CSV = ROOT / "reviewer21_cross_model_residual_audit" / "reviewer21_cross_model_residual_summary.csv"
PRESSURE_CSV = ROOT / "baseline_suite" / "nasa_hump" / "final_comparison" / "nasa_six_model_final_rmse.csv"

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})

def save_both(fig, stem):
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)

def find_col(df, candidates, required=True):
    norm = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in norm:
            return norm[cand.lower()]
    # relaxed substring match
    for cand in candidates:
        cl = cand.lower()
        for k, original in norm.items():
            if cl in k:
                return original
    if required:
        raise KeyError(
            f"Could not find any of {candidates}. Available columns: {list(df.columns)}"
        )
    return None

def clean_method_names(s):
    return (
        s.astype(str)
         .str.replace("_", " ", regex=False)
         .str.replace("MC-dropout B-PINN deterministic", "B-PINN deterministic", regex=False)
         .str.replace("MC-dropout B-PINN MC50 predictive mean", "B-PINN MC50 mean", regex=False)
    )

# ---------------------------------------------------------------------
# Fig N1 — Reviewer #20: sampler / geometry audit
# Locked counts from the canonical protocol audit:
# legacy random-box = 2000
# below-wall/solid = 129
# above local reference window = 147
# total outside local reference window = 276
# corrected geometry-aware valid-fluid points = 2000
# ---------------------------------------------------------------------
legacy_total = 2000
below_wall = 129
above_window = 147
legacy_valid = legacy_total - below_wall - above_window
corrected_valid = 2000

fig, ax = plt.subplots(figsize=(7.2, 4.5))
labels = [
    "Legacy: valid\nwithin window",
    "Legacy: below\nwall / solid",
    "Legacy: above\nreference window",
    "Geometry-aware:\nvalid fluid",
]
values = [legacy_valid, below_wall, above_window, corrected_valid]
bars = ax.bar(labels, values)

for bar, val in zip(bars, values):
    pct = val / legacy_total * 100
    ax.text(
        bar.get_x() + bar.get_width()/2,
        val + legacy_total*0.015,
        f"{val:,}\n({pct:.2f}%)",
        ha="center", va="bottom", fontsize=9
    )

ax.set_ylabel("Number of collocation points")
ax.set_title("NASA Hump Sampler Audit (Reviewer #20)")
ax.set_ylim(0, 2250)
ax.text(
    0.02, 0.96,
    "Legacy outside local reference window: 276 / 2,000 = 13.80%",
    transform=ax.transAxes, ha="left", va="top", fontsize=9
)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
save_both(fig, "Fig_N1_sampler_audit")

# ---------------------------------------------------------------------
# Fig N2 — Reviewer #21: cross-model residual compatibility audit
# Reads the canonical CSV; does not recompute residuals.
# ---------------------------------------------------------------------
if not RESIDUAL_CSV.exists():
    raise FileNotFoundError(f"Missing canonical residual table: {RESIDUAL_CSV}")

rdf = pd.read_csv(RESIDUAL_CSV)
method_col = find_col(rdf, ["method", "model", "name"])

continuity_col = find_col(
    rdf,
    ["continuity_rmse", "continuity rmse"],
    required=False
)
physics_col = find_col(
    rdf,
    ["physics_vector_rmse", "physics vector rmse", "vector_rmse"],
    required=False
)
mag_mean_col = find_col(
    rdf,
    ["residual_magnitude_mean", "residual magnitude mean", "physics_magnitude_mean"],
    required=False
)
mag_p95_col = find_col(
    rdf,
    ["residual_magnitude_p95", "residual magnitude p95", "physics_magnitude_p95"],
    required=False
)

metric_specs = [
    (continuity_col, "Continuity RMSE"),
    (physics_col, "Physics-vector RMSE"),
    (mag_mean_col, "Residual magnitude mean"),
    (mag_p95_col, "Residual magnitude P95"),
]
metric_specs = [(c, label) for c, label in metric_specs if c is not None]

if not metric_specs:
    raise KeyError(
        "Residual CSV was found, but none of the expected residual metric columns "
        f"were detected. Columns: {list(rdf.columns)}"
    )

methods = clean_method_names(rdf[method_col])
n = len(metric_specs)
fig, axes = plt.subplots(1, n, figsize=(4.3*n, max(4.8, 0.55*len(rdf)+2)), squeeze=False)
axes = axes.ravel()

for ax, (col, label) in zip(axes, metric_specs):
    vals = pd.to_numeric(rdf[col], errors="coerce")
    y = np.arange(len(rdf))
    ax.barh(y, vals)
    ax.set_yticks(y)
    ax.set_yticklabels(methods)
    ax.invert_yaxis()
    ax.set_xlabel(label)
    ax.grid(axis="x", alpha=0.25)
    for yi, val in zip(y, vals):
        if pd.notna(val):
            ax.text(val, yi, f" {val:.4g}", va="center", fontsize=8)

fig.suptitle("NASA Hump Reference-Field Residual Compatibility Audit (Reviewer #21)", y=1.02)
fig.tight_layout()
save_both(fig, "Fig_N2_reference_residual_audit")

# ---------------------------------------------------------------------
# Fig N3 — Reviewer #23: pressure raw vs gauge-aligned audit
# Reads the frozen six-model final comparison table.
# ---------------------------------------------------------------------
if not PRESSURE_CSV.exists():
    raise FileNotFoundError(f"Missing canonical pressure table: {PRESSURE_CSV}")

pdf = pd.read_csv(PRESSURE_CSV)
p_method = find_col(pdf, ["method", "model", "name"])
raw_col = find_col(pdf, ["cp_raw_rmse", "cp raw rmse", "raw_cp_rmse"])
gauge_col = find_col(pdf, ["cp_gauge_aligned_rmse", "cp gauge-aligned rmse", "gauge_aligned_cp_rmse"])

methods_p = clean_method_names(pdf[p_method])
raw = pd.to_numeric(pdf[raw_col], errors="coerce")
gauge = pd.to_numeric(pdf[gauge_col], errors="coerce")

y = np.arange(len(pdf))
h = 0.36
fig, ax = plt.subplots(figsize=(8.0, max(4.8, 0.62*len(pdf)+2)))
ax.barh(y - h/2, raw, height=h, label="Raw Cp RMSE")
ax.barh(y + h/2, gauge, height=h, label="Gauge-aligned Cp RMSE")
ax.set_yticks(y)
ax.set_yticklabels(methods_p)
ax.invert_yaxis()
ax.set_xlabel("Cp RMSE")
ax.set_title("NASA Hump Pressure-Gauge Audit (Reviewer #23)")
ax.legend()
ax.grid(axis="x", alpha=0.25)

for yi, rv, gv in zip(y, raw, gauge):
    if pd.notna(rv):
        ax.text(rv, yi-h/2, f" {rv:.5f}", va="center", fontsize=8)
    if pd.notna(gv):
        ax.text(gv, yi+h/2, f" {gv:.5f}", va="center", fontsize=8)

fig.tight_layout()
save_both(fig, "Fig_N3_pressure_gauge_audit")

# ---------------------------------------------------------------------
# Fig N0 — compact master summary for audit/reporting.
# This is an overview figure, not a replacement for canonical CSV tables.
# ---------------------------------------------------------------------
fig = plt.figure(figsize=(12.0, 8.0))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15])

# Panel (a): sampler counts
ax1 = fig.add_subplot(gs[0, 0])
labels_short = ["Legacy valid", "Below wall", "Above window", "Corrected valid"]
vals_short = [legacy_valid, below_wall, above_window, corrected_valid]
ax1.bar(labels_short, vals_short)
ax1.set_ylabel("Points")
ax1.set_title("(a) Sampler audit — Reviewer #20")
ax1.tick_params(axis="x", rotation=18)
ax1.grid(axis="y", alpha=0.25)
ax1.text(0.02, 0.94, "Outside window = 13.80%", transform=ax1.transAxes, va="top")

# Panel (b): one primary residual metric
ax2 = fig.add_subplot(gs[0, 1])
primary_col = physics_col if physics_col is not None else metric_specs[0][0]
primary_label = "Physics-vector RMSE" if physics_col is not None else metric_specs[0][1]
vals2 = pd.to_numeric(rdf[primary_col], errors="coerce")
y2 = np.arange(len(rdf))
ax2.barh(y2, vals2)
ax2.set_yticks(y2)
ax2.set_yticklabels(methods, fontsize=8)
ax2.invert_yaxis()
ax2.set_xlabel(primary_label)
ax2.set_title("(b) Residual compatibility — Reviewer #21")
ax2.grid(axis="x", alpha=0.25)

# Panel (c): pressure gauge
ax3 = fig.add_subplot(gs[1, :])
ax3.barh(y - h/2, raw, height=h, label="Raw Cp RMSE")
ax3.barh(y + h/2, gauge, height=h, label="Gauge-aligned Cp RMSE")
ax3.set_yticks(y)
ax3.set_yticklabels(methods_p)
ax3.invert_yaxis()
ax3.set_xlabel("Cp RMSE")
ax3.set_title("(c) Pressure-gauge audit — Reviewer #23")
ax3.legend()
ax3.grid(axis="x", alpha=0.25)

fig.suptitle("NASA Hump Audit Summary — 30 August", fontsize=14, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.97])
save_both(fig, "Fig_N0_nasa_audit_master")

print("\nDONE. Generated:")
for p in sorted(OUT.glob("Fig_N*")):
    print(" ", p)
