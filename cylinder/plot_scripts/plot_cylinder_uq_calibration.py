from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# ============================================================
# Paths
# ============================================================

BPINN = Path(
    "./cylinder_bpinn_final_p0002_all/heldout_mc50_final/"
    "final_heldout_mc50_summary.csv"
)

ENS_RAW = Path(
    "./cylinder_deep_ensemble_final_heldout/"
    "deep_ensemble_final_heldout_uq_raw.csv"
)

ENS_CAL = Path(
    "./cylinder_deep_ensemble_final_heldout/"
    "deep_ensemble_final_heldout_uq_calibrated.csv"
)

OUTDIR = Path("./evaluation/final_uq_figures")
OUTDIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load
# ============================================================

bp = pd.read_csv(BPINN)
ens_raw = pd.read_csv(ENS_RAW)
ens_cal = pd.read_csv(ENS_CAL)

print("B-PINN rows:", len(bp))
print("Deep Ensemble raw rows:", len(ens_raw))
print("Deep Ensemble calibrated rows:", len(ens_cal))


# ============================================================
# Provenance checks
# ============================================================

if "test_points" in bp.columns:
    assert (bp["test_points"] == 200000).all(), \
        "B-PINN is not the expected 200,000-point held-out result."

assert (ens_raw["split"] == "heldout_test").all(), \
    "Deep Ensemble raw UQ is not heldout_test."

assert (ens_cal["split"] == "heldout_test").all(), \
    "Deep Ensemble calibrated UQ is not heldout_test."

assert (ens_cal["frozen_from"] == "validation").all(), \
    "Deep Ensemble calibration is not frozen from validation."

print("PASS: final-heldout UQ provenance checks.")


# ============================================================
# Extract B-PINN
# ============================================================

bp_levels = np.array([0.50, 0.80, 0.95])

bp_raw = {}
bp_cal = {}

for var in ["u", "v", "p"]:
    row = bp.loc[bp["variable"] == var].iloc[0]

    bp_raw[var] = np.array([
        row["raw_picp_50"],
        row["raw_picp_80"],
        row["raw_picp_95"],
    ], dtype=float)

    bp_cal[var] = np.array([
        row["calibrated_picp_50"],
        row["calibrated_picp_80"],
        row["calibrated_picp_95"],
    ], dtype=float)


# ============================================================
# Extract Deep Ensemble
# ============================================================

ens_levels = np.array([0.50, 0.80, 0.90, 0.95])

ens_raw_cov = {}
ens_cal_cov = {}

for var in ["u", "v", "p"]:
    r = (
        ens_raw.loc[ens_raw["variable"] == var]
        .sort_values("nominal_level")
    )

    c = (
        ens_cal.loc[ens_cal["variable"] == var]
        .sort_values("nominal_level")
    )

    assert np.allclose(
        r["nominal_level"].to_numpy(dtype=float),
        ens_levels
    )

    assert np.allclose(
        c["nominal_level"].to_numpy(dtype=float),
        ens_levels
    )

    ens_raw_cov[var] = r["picp"].to_numpy(dtype=float)
    ens_cal_cov[var] = c["picp"].to_numpy(dtype=float)


# ============================================================
# Plot
# ============================================================

fig, axes = plt.subplots(
    1, 2,
    figsize=(11.2, 5.0),
    sharex=True,
    sharey=True
)

variables = ["u", "v", "p"]

# ------------------------------------------------------------
# Panel (a): MC-dropout B-PINN
# ------------------------------------------------------------

ax = axes[0]

ax.plot(
    [0.45, 1.0],
    [0.45, 1.0],
    linestyle="--",
    linewidth=1.2,
    label="Ideal calibration"
)

for var in variables:
    ax.plot(
        bp_levels,
        bp_raw[var],
        marker="o",
        linestyle=":",
        linewidth=1.4,
        label=f"{var} raw"
    )

    ax.plot(
        bp_levels,
        bp_cal[var],
        marker="s",
        linestyle="-",
        linewidth=1.6,
        label=f"{var} calibrated"
    )

ax.set_title("(a) MC-dropout B-PINN")
ax.set_xlabel("Nominal coverage")
ax.set_ylabel("Empirical coverage")
ax.grid(True, alpha=0.25)


# ------------------------------------------------------------
# Panel (b): Deep Ensemble
# ------------------------------------------------------------

ax = axes[1]

ax.plot(
    [0.45, 1.0],
    [0.45, 1.0],
    linestyle="--",
    linewidth=1.2,
    label="Ideal calibration"
)

for var in variables:
    ax.plot(
        ens_levels,
        ens_raw_cov[var],
        marker="o",
        linestyle=":",
        linewidth=1.4,
        label=f"{var} raw"
    )

    ax.plot(
        ens_levels,
        ens_cal_cov[var],
        marker="s",
        linestyle="-",
        linewidth=1.6,
        label=f"{var} calibrated"
    )

ax.set_title("(b) Deep Ensemble")
ax.set_xlabel("Nominal coverage")
ax.grid(True, alpha=0.25)


# ============================================================
# Common formatting
# ============================================================

for ax in axes:
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(0.25, 1.0)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0])
    ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0])

# One shared legend
handles, labels = axes[1].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.08),
    ncol=4,
    frameon=False
)

fig.tight_layout(rect=[0, 0.10, 1, 1])


# ============================================================
# Save actual image files
# ============================================================

png_path = OUTDIR / "cylinder_final_uq_calibration.png"
pdf_path = OUTDIR / "cylinder_final_uq_calibration.pdf"

fig.savefig(
    png_path,
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    pdf_path,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# Also save exact plotted numbers
# ============================================================

rows = []

for var in variables:
    for level, raw, cal in zip(
        bp_levels, bp_raw[var], bp_cal[var]
    ):
        rows.append({
            "method": "MC-dropout B-PINN",
            "variable": var,
            "nominal_coverage": level,
            "raw_picp": raw,
            "calibrated_picp": cal,
        })

for var in variables:
    for level, raw, cal in zip(
        ens_levels,
        ens_raw_cov[var],
        ens_cal_cov[var]
    ):
        rows.append({
            "method": "Deep Ensemble",
            "variable": var,
            "nominal_coverage": level,
            "raw_picp": raw,
            "calibrated_picp": cal,
        })

plot_data = pd.DataFrame(rows)

csv_path = OUTDIR / "cylinder_final_uq_calibration_plot_data.csv"
plot_data.to_csv(csv_path, index=False)


print()
print("===== SAVED FINAL UQ FIGURE =====")
print(png_path)
print(pdf_path)
print(csv_path)

print()
print("===== PLOTTED DATA =====")
print(
    plot_data.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)
