from pathlib import Path
import re
import shutil

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(".")
OUT = ROOT / "evaluation" / "paper_ready_cylinder_diagnostics"
OUT.mkdir(parents=True, exist_ok=True)

TIME_SOURCES = {
    "Standard PINN": ROOT / "cylinder_final_lambda_0.1" /
        "heldout_evaluation" / "heldout_time_resolved_metrics.csv",

    "Weight-decay PINN": ROOT / "baseline_suite" / "cylinder" /
        "weight_decay" / "final_heldout" /
        "heldout_evaluation" / "heldout_time_resolved_metrics.csv",

    "Adaptive-weight PINN": ROOT / "baseline_suite" / "cylinder" /
        "adaptive_weight" / "final_heldout" /
        "heldout_evaluation" / "heldout_time_resolved_metrics.csv",
}

REGIONAL = (
    ROOT / "evaluation" / "final_heldout_regional" /
    "cylinder_regional_final_heldout_time_resolved.csv"
)

BPINN = (
    ROOT / "cylinder_bpinn_final_p0002_all" /
    "heldout_mc50_final" /
    "final_heldout_mc50_summary.csv"
)

ENS_RAW = (
    ROOT / "cylinder_deep_ensemble_final_heldout" /
    "deep_ensemble_final_heldout_uq_raw.csv"
)

ENS_CAL = (
    ROOT / "cylinder_deep_ensemble_final_heldout" /
    "deep_ensemble_final_heldout_uq_calibrated.csv"
)


# ============================================================
# Load time-resolved canonical data
# ============================================================

dfs = {}

for method, path in TIME_SOURCES.items():
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = {
        "time_index",
        "time_value",
        "variable",
        "rmse",
        "relative_l2",
        "max_absolute_error",
    }

    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"{method}: missing columns {sorted(missing)}"
        )

    dfs[method] = df

print("PASS: canonical time-resolved sources loaded.")


# ============================================================
# S1 — Extended temporal errors
# relative-L2 + max absolute error
# ============================================================

variables = [
    ("u", r"$u$"),
    ("v", r"$v$"),
    ("p_gauge_aligned", "Gauge-aligned pressure"),
]

fig, axes = plt.subplots(
    2, 3,
    figsize=(14.0, 7.2),
    sharex="col"
)

linestyles = {
    "Standard PINN": "-",
    "Weight-decay PINN": "--",
    "Adaptive-weight PINN": "-.",
}

s1_rows = []

for j, (var, label) in enumerate(variables):

    for method, df in dfs.items():

        sub = (
            df[df["variable"] == var]
            .sort_values("time_index")
        )

        if len(sub) != 100:
            raise RuntimeError(
                f"{method}/{var}: expected 100 snapshots, "
                f"found {len(sub)}"
            )

        axes[0, j].plot(
            sub["time_value"],
            sub["relative_l2"],
            linestyle=linestyles[method],
            linewidth=1.5,
            label=method,
        )

        axes[1, j].plot(
            sub["time_value"],
            sub["max_absolute_error"],
            linestyle=linestyles[method],
            linewidth=1.5,
            label=method,
        )

        for _, r in sub.iterrows():
            s1_rows.append({
                "method": method,
                "variable": var,
                "time_index": int(r["time_index"]),
                "time_value": float(r["time_value"]),
                "relative_l2": float(r["relative_l2"]),
                "max_absolute_error":
                    float(r["max_absolute_error"]),
            })

    axes[0, j].set_title(
        f"({chr(97+j)}) {label}"
    )

    axes[0, j].set_ylabel(
        r"Relative $L_2$ error"
    )

    axes[1, j].set_ylabel(
        "Maximum absolute error"
    )

    axes[1, j].set_xlabel("Time")

    for i in range(2):
        axes[i, j].grid(alpha=0.20)
        axes[i, j].set_ylim(bottom=0)

handles, labels = axes[0, 0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.015),
    ncol=3,
    frameon=False,
)

fig.tight_layout(rect=[0, 0.07, 1, 1])

fig.savefig(
    OUT / "Fig_S1_temporal_extended_errors.png",
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    OUT / "Fig_S1_temporal_extended_errors.pdf",
    bbox_inches="tight",
)

plt.close(fig)

pd.DataFrame(s1_rows).to_csv(
    OUT / "Fig_S1_temporal_extended_errors_plot_data.csv",
    index=False,
)

print("PASS: Fig. S1 generated.")


# ============================================================
# S2 — Raw vs gauge-aligned pressure
# ============================================================

fig, axes = plt.subplots(
    1, 3,
    figsize=(13.4, 4.3),
    sharey=True,
)

s2_rows = []

for ax, (method, df) in zip(axes, dfs.items()):

    raw = (
        df[df["variable"] == "p_raw"]
        .sort_values("time_index")
    )

    gauge = (
        df[df["variable"] == "p_gauge_aligned"]
        .sort_values("time_index")
    )

    if len(raw) != 100 or len(gauge) != 100:
        raise RuntimeError(
            f"{method}: pressure time series incomplete."
        )

    if not np.allclose(
        raw["time_value"].to_numpy(),
        gauge["time_value"].to_numpy()
    ):
        raise RuntimeError(
            f"{method}: raw/gauge time grids differ."
        )

    ax.plot(
        raw["time_value"],
        raw["rmse"],
        linewidth=1.6,
        label="Raw pressure",
    )

    ax.plot(
        gauge["time_value"],
        gauge["rmse"],
        linewidth=1.6,
        linestyle="--",
        label="Gauge-aligned pressure",
    )

    ax.set_title(method)
    ax.set_xlabel("Time")
    ax.set_ylabel("Pressure RMSE")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.20)

    for (_, rr), (_, gg) in zip(
        raw.iterrows(),
        gauge.iterrows()
    ):
        s2_rows.append({
            "method": method,
            "time_index": int(rr["time_index"]),
            "time_value": float(rr["time_value"]),
            "raw_pressure_rmse": float(rr["rmse"]),
            "gauge_aligned_pressure_rmse":
                float(gg["rmse"]),
        })

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.02),
    ncol=2,
    frameon=False,
)

fig.tight_layout(rect=[0, 0.08, 1, 1])

fig.savefig(
    OUT / "Fig_S2_pressure_raw_vs_gauge.png",
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    OUT / "Fig_S2_pressure_raw_vs_gauge.pdf",
    bbox_inches="tight",
)

plt.close(fig)

pd.DataFrame(s2_rows).to_csv(
    OUT / "Fig_S2_pressure_raw_vs_gauge_plot_data.csv",
    index=False,
)

print("PASS: Fig. S2 generated.")


# ============================================================
# S3 — Regional time-resolved RMSE heatmaps
# ============================================================

if not REGIONAL.exists():
    raise FileNotFoundError(REGIONAL)

reg = pd.read_csv(REGIONAL)

required = {
    "method",
    "time_index",
    "time_value",
    "region",
    "variable",
    "rmse",
}

missing = required - set(reg.columns)

if missing:
    raise RuntimeError(
        f"Regional CSV missing {sorted(missing)}"
    )

methods = [
    "Standard PINN",
    "Weight-decay PINN",
    "Adaptive-weight PINN",
]

regions = [
    "separation_zone",
    "shear_layer",
    "vortex_core",
    "near_wake",
    "far_wake",
]

region_labels = [
    "Separation zone",
    "Shear layer",
    "Vortex core",
    "Near wake",
    "Far wake",
]

reg_variables = [
    ("u", r"$u$ RMSE"),
    ("v", r"$v$ RMSE"),
    ("p_gauge_aligned", "Gauge-aligned pressure RMSE"),
]

fig, axes = plt.subplots(
    3, 3,
    figsize=(14.0, 10.2),
    sharex=True,
    sharey=True,
)

for j, (var, var_label) in enumerate(reg_variables):

    all_var = reg[
        (reg["variable"] == var) &
        (reg["method"].isin(methods)) &
        (reg["region"].isin(regions))
    ]

    vmin = float(all_var["rmse"].min())
    vmax = float(all_var["rmse"].max())

    last_im = None

    for i, method in enumerate(methods):

        sub = reg[
            (reg["method"] == method) &
            (reg["variable"] == var)
        ].copy()

        pivot = (
            sub.pivot_table(
                index="region",
                columns="time_index",
                values="rmse",
                aggfunc="first",
            )
            .reindex(regions)
        )

        if pivot.shape[1] != 100:
            raise RuntimeError(
                f"{method}/{var}: expected 100 time points."
            )

        time_lookup = (
            sub[["time_index", "time_value"]]
            .drop_duplicates()
            .sort_values("time_index")
        )

        tmin = float(time_lookup["time_value"].min())
        tmax = float(time_lookup["time_value"].max())

        last_im = axes[i, j].imshow(
            pivot.to_numpy(),
            aspect="auto",
            origin="lower",
            extent=[tmin, tmax, -0.5, 4.5],
            vmin=vmin,
            vmax=vmax,
        )

        if i == 0:
            axes[i, j].set_title(var_label)

        if j == 0:
            axes[i, j].set_ylabel(method)

        if i == 2:
            axes[i, j].set_xlabel("Time")

        axes[i, j].set_yticks(range(5))
        axes[i, j].set_yticklabels(region_labels)

    cbar = fig.colorbar(
        last_im,
        ax=axes[:, j],
        shrink=0.82,
        pad=0.02,
    )

    cbar.set_label("RMSE")

fig.suptitle(
    "Time-resolved regional reconstruction errors",
    fontsize=14,
)

fig.savefig(
    OUT / "Fig_S3_regional_time_resolved.png",
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    OUT / "Fig_S3_regional_time_resolved.pdf",
    bbox_inches="tight",
)

plt.close(fig)

reg.to_csv(
    OUT / "Fig_S3_regional_time_resolved_plot_data.csv",
    index=False,
)

print("PASS: Fig. S3 generated.")


# ============================================================
# Table S1 — exact locked C2 numerical audit
# ============================================================

C2_AUDIT = (
    OUT / "Fig_C2_refined_spectrum_phase_audit.csv"
)

if not C2_AUDIT.exists():
    raise FileNotFoundError(
        "Locked C2 audit CSV not found: "
        f"{C2_AUDIT}"
    )

shutil.copy2(
    C2_AUDIT,
    OUT / "Table_S1_spectral_phase.csv"
)

print("PASS: Table S1 generated from locked C2 audit.")


# ============================================================
# Table S2 — UQ calibration long-form table
# ============================================================

rows = []

if not BPINN.exists():
    raise FileNotFoundError(BPINN)

bp = pd.read_csv(BPINN)

for _, r in bp.iterrows():

    var = r["variable"]

    # Parse raw/calibrated PICP and MPIW columns.
    values = {}

    for c in bp.columns:

        m = re.match(
            r"^(raw|calibrated)_(picp|mpiw)_(\d+)$",
            c
        )

        if m:
            status, metric, level = m.groups()
            values[(status, metric, int(level))] = r[c]

    levels = sorted({
        k[2] for k in values
        if k[1] == "picp"
    })

    for level in levels:

        for status in ["raw", "calibrated"]:

            rows.append({
                "method": "MC-dropout B-PINN",
                "variable": var,
                "status": status,
                "nominal_coverage": level / 100.0,
                "picp": values.get(
                    (status, "picp", level),
                    np.nan
                ),
                "mpiw": values.get(
                    (status, "mpiw", level),
                    np.nan
                ),
            })


for status, path in [
    ("raw", ENS_RAW),
    ("calibrated", ENS_CAL),
]:

    if not path.exists():
        raise FileNotFoundError(path)

    d = pd.read_csv(path)

    required_ens = {
        "variable",
        "nominal_level",
        "picp",
    }

    missing = required_ens - set(d.columns)

    if missing:
        raise RuntimeError(
            f"{path}: missing {sorted(missing)}"
        )

    mpiw_col = None

    for candidate in [
        "mpiw",
        "mean_prediction_interval_width",
        "prediction_interval_width",
    ]:
        if candidate in d.columns:
            mpiw_col = candidate
            break

    for _, r in d.iterrows():

        rows.append({
            "method": "Deep Ensemble",
            "variable": r["variable"],
            "status": status,
            "nominal_coverage":
                float(r["nominal_level"]),
            "picp": float(r["picp"]),
            "mpiw": (
                float(r[mpiw_col])
                if mpiw_col is not None
                else np.nan
            ),
        })


uq = pd.DataFrame(rows)

uq = uq.sort_values([
    "method",
    "variable",
    "nominal_coverage",
    "status",
])

uq.to_csv(
    OUT / "Table_S2_uq_calibration.csv",
    index=False,
)

print("PASS: Table S2 generated.")


# ============================================================
# README / provenance manifest
# ============================================================

README = OUT / "README_CYLINDER_DIAGNOSTICS_FIGURES.txt"

with open(README, "w", encoding="utf-8") as f:

    f.write(
        "Cylinder Diagnostics — Paper-Ready Evidence Package\n"
        "===================================================\n\n"

        "STATUS\n"
        "------\n"
        "Frozen evidence only.\n"
        "No training or new model inference was performed "
        "during figure packaging.\n\n"

        "MAIN FIGURES\n"
        "------------\n"
        "Fig_C1_time_resolved_rmse.*\n"
        "Fig_C2_refined_spectrum_phase.*\n"
        "Fig_C3_regional_rmse.*\n"
        "Fig_C4_uq_calibration.*\n\n"

        "SUPPLEMENT FIGURES\n"
        "------------------\n"
        "Fig_S1_temporal_extended_errors.*\n"
        "Fig_S2_pressure_raw_vs_gauge.*\n"
        "Fig_S3_regional_time_resolved.*\n\n"

        "SUPPLEMENT TABLES\n"
        "-----------------\n"
        "Table_S1_spectral_phase.csv\n"
        "Table_S2_uq_calibration.csv\n\n"

        "CANONICAL SOURCES\n"
        "-----------------\n"
        "Time resolved:\n"
        "  cylinder_final_lambda_0.1/heldout_evaluation/"
        "heldout_time_resolved_metrics.csv\n"
        "  baseline_suite/cylinder/weight_decay/final_heldout/"
        "heldout_evaluation/heldout_time_resolved_metrics.csv\n"
        "  baseline_suite/cylinder/adaptive_weight/final_heldout/"
        "heldout_evaluation/heldout_time_resolved_metrics.csv\n\n"

        "Spectral / phase:\n"
        "  cylinder_final_lambda_0.1/"
        "spectral_phase_analysis_refined/\n\n"

        "Regional:\n"
        "  evaluation/final_heldout_regional/\n\n"

        "UQ:\n"
        "  cylinder_bpinn_final_p0002_all/heldout_mc50_final/\n"
        "  cylinder_deep_ensemble_final_heldout/\n\n"

        "NON-CANONICAL / DO NOT USE\n"
        "--------------------------\n"
        "evaluation/local_region_errors.csv\n"
        "  Historical reviewer diagnostic; not the final "
        "frozen held-out regional result.\n\n"

        "PRESSURE POLICY\n"
        "---------------\n"
        "Gauge-aligned pressure is used as the canonical "
        "pressure result in primary model comparisons.\n"
        "Raw pressure is retained for pressure-gauge audit "
        "and Supplement diagnostics.\n"
    )

print("PASS: README/provenance manifest generated.")


# ============================================================
# Final package inventory
# ============================================================

print()
print("=" * 72)
print("CYLINDER SUPPLEMENT PACKAGE FINISHED")
print("=" * 72)

for p in sorted(OUT.iterdir()):
    if p.is_file():
        print(p.name)

print()
print("NO TRAINING.")
print("NO MODEL INFERENCE.")
print("FROZEN RESULTS ONLY.")
