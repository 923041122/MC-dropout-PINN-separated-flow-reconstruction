from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# ============================================================
# Canonical final-heldout sources
# ============================================================

SOURCES = {
    "Standard PINN": Path(
        "./cylinder_final_lambda_0.1/"
        "heldout_evaluation/heldout_time_resolved_metrics.csv"
    ),
    "Weight-decay PINN": Path(
        "./baseline_suite/cylinder/weight_decay/final_heldout/"
        "heldout_evaluation/heldout_time_resolved_metrics.csv"
    ),
    "Adaptive-weight PINN": Path(
        "./baseline_suite/cylinder/adaptive_weight/final_heldout/"
        "heldout_evaluation/heldout_time_resolved_metrics.csv"
    ),
}

OUTDIR = Path("./evaluation/paper_ready_cylinder_diagnostics")
OUTDIR.mkdir(parents=True, exist_ok=True)

PNG = OUTDIR / "Fig_C1_time_resolved_rmse.png"
PDF = OUTDIR / "Fig_C1_time_resolved_rmse.pdf"
DATA = OUTDIR / "Fig_C1_time_resolved_rmse_plot_data.csv"


# ============================================================
# Load + provenance checks
# ============================================================

dfs = {}

required_variables = {
    "u",
    "v",
    "p_raw",
    "p_gauge_aligned",
}

for method, path in SOURCES.items():

    if not path.exists():
        raise FileNotFoundError(
            f"Missing canonical source for {method}: {path}"
        )

    df = pd.read_csv(path)

    required_columns = {
        "time_index",
        "time_value",
        "heldout_points",
        "variable",
        "rmse",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise RuntimeError(
            f"{method}: missing columns {sorted(missing)}"
        )

    variables = set(df["variable"].unique())

    if variables != required_variables:
        raise RuntimeError(
            f"{method}: unexpected variables {sorted(variables)}"
        )

    counts = df["variable"].value_counts()

    if not (counts == 100).all():
        raise RuntimeError(
            f"{method}: expected 100 snapshots per variable.\n{counts}"
        )

    # Each variable must have exactly the same time grid.
    grids = []

    for var in [
        "u",
        "v",
        "p_raw",
        "p_gauge_aligned",
    ]:
        x = (
            df.loc[df["variable"] == var]
            .sort_values("time_index")
        )

        grids.append(
            x[["time_index", "time_value"]]
            .to_numpy(dtype=float)
        )

    for grid in grids[1:]:
        if not np.allclose(grid, grids[0]):
            raise RuntimeError(
                f"{method}: inconsistent time grids between variables."
            )

    dfs[method] = df

    print()
    print(method)
    print("  rows:", len(df))
    print("  snapshots:", df["time_index"].nunique())
    print(
        "  heldout_points:",
        sorted(df["heldout_points"].unique().tolist())
    )
    print(
        "  time range:",
        f'{df["time_value"].min():.6f}',
        "to",
        f'{df["time_value"].max():.6f}'
    )


# ============================================================
# Cross-model time-grid check
# ============================================================

reference_grid = None

for method, df in dfs.items():

    x = (
        df.loc[df["variable"] == "u"]
        .sort_values("time_index")
    )

    grid = x[
        ["time_index", "time_value"]
    ].to_numpy(dtype=float)

    if reference_grid is None:
        reference_grid = grid

    elif not np.allclose(grid, reference_grid):
        raise RuntimeError(
            f"{method}: time grid differs from the other models."
        )

print()
print(
    "PASS: all three methods use the same "
    "100-snapshot time grid."
)


# ============================================================
# Prepare exact plotted data
# ============================================================

plot_rows = []

for method, df in dfs.items():

    for var in ["u", "v", "p_gauge_aligned"]:

        sub = (
            df.loc[df["variable"] == var]
            .sort_values("time_index")
        )

        for _, row in sub.iterrows():

            plot_rows.append({
                "method": method,
                "variable": var,
                "time_index": int(row["time_index"]),
                "time_value": float(row["time_value"]),
                "heldout_points": int(row["heldout_points"]),
                "rmse": float(row["rmse"]),
            })

plot_df = pd.DataFrame(plot_rows)
plot_df.to_csv(DATA, index=False)


# ============================================================
# Figure
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(14.2, 4.35),
    sharex=True
)

panels = [
    ("u", "(a) $u$", r"$u$ RMSE"),
    ("v", "(b) $v$", r"$v$ RMSE"),
    (
        "p_gauge_aligned",
        "(c) Gauge-aligned pressure",
        "Gauge-aligned pressure RMSE",
    ),
]

linestyles = {
    "Standard PINN": "-",
    "Weight-decay PINN": "--",
    "Adaptive-weight PINN": "-.",
}

for ax, (variable, title, ylabel) in zip(
    axes,
    panels
):

    for method in SOURCES:

        sub = (
            plot_df[
                (plot_df["method"] == method)
                & (plot_df["variable"] == variable)
            ]
            .sort_values("time_index")
        )

        ax.plot(
            sub["time_value"],
            sub["rmse"],
            linestyle=linestyles[method],
            linewidth=1.7,
            label=method,
        )

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)

    # RMSE is non-negative.
    ax.set_ylim(bottom=0)


# ============================================================
# Shared legend
# ============================================================

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.055),
    ncol=3,
    frameon=False,
)

fig.tight_layout(
    rect=[0, 0.10, 1, 1]
)


# ============================================================
# Save
# ============================================================

fig.savefig(
    PNG,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    PDF,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# Numerical audit corresponding to the figure
# ============================================================

summary = (
    plot_df
    .groupby(["method", "variable"])["rmse"]
    .agg(
        mean="mean",
        p95=lambda x: np.quantile(x, 0.95),
        maximum="max",
    )
    .reset_index()
)

print()
print("===== C1 NUMERICAL AUDIT =====")

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

print()
print("===== SAVED FIG. C1 =====")
print(PNG)
print(PDF)
print(DATA)
print()
print("C1 generation finished successfully.")
