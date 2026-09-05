from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SRC = Path(
    "./evaluation/final_heldout_regional/"
    "cylinder_regional_final_heldout_summary.csv"
)

OUTDIR = Path("./evaluation/paper_ready_cylinder_diagnostics")
OUTDIR.mkdir(parents=True, exist_ok=True)

PNG = OUTDIR / "Fig_C3_regional_rmse.png"
PDF = OUTDIR / "Fig_C3_regional_rmse.pdf"
AUDIT = OUTDIR / "Fig_C3_regional_rmse_audit.csv"

df = pd.read_csv(SRC)

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
    "Separation\nzone",
    "Shear\nlayer",
    "Vortex\ncore",
    "Near\nwake",
    "Far\nwake",
]

metrics = [
    ("u_rmse", "(a) Regional u RMSE", "RMSE"),
    ("v_rmse", "(b) Regional v RMSE", "RMSE"),
    (
        "p_gauge_aligned_rmse",
        "(c) Regional gauge-aligned pressure RMSE",
        "RMSE",
    ),
]

# Strict provenance / schema checks
required = {
    "method",
    "region",
    "u_rmse",
    "v_rmse",
    "p_gauge_aligned_rmse",
}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

plot_df = df[
    df["method"].isin(methods) &
    df["region"].isin(regions)
].copy()

expected_rows = len(methods) * len(regions)
if len(plot_df) != expected_rows:
    raise ValueError(
        f"Expected {expected_rows} method-region rows, "
        f"found {len(plot_df)}"
    )

for method in methods:
    sub = plot_df[plot_df["method"] == method]
    missing_regions = set(regions) - set(sub["region"])
    if missing_regions:
        raise ValueError(
            f"{method} missing regions: {sorted(missing_regions)}"
        )

# Save the exact rows used for Fig. C3
audit = plot_df[
    [
        "method",
        "region",
        "u_rmse",
        "v_rmse",
        "p_gauge_aligned_rmse",
    ]
].copy()

audit["method"] = pd.Categorical(
    audit["method"], methods, ordered=True
)
audit["region"] = pd.Categorical(
    audit["region"], regions, ordered=True
)

audit = audit.sort_values(["method", "region"])
audit.to_csv(AUDIT, index=False)

# Figure
fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.5))

x = np.arange(len(regions))
width = 0.24

for ax, (metric, title, ylabel) in zip(axes, metrics):
    for i, method in enumerate(methods):
        sub = (
            plot_df[plot_df["method"] == method]
            .set_index("region")
            .loc[regions]
        )

        offset = (i - 1) * width

        ax.bar(
            x + offset,
            sub[metric].to_numpy(),
            width=width,
            label=method,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(region_labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)

# One common legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.04),
    ncol=3,
    frameon=False,
)

fig.tight_layout(rect=[0, 0, 1, 0.93])

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

print()
print("===== FIG. C3 SOURCE CHECK =====")
print(f"Source: {SRC}")
print(f"Rows used: {len(audit)}")
print(f"Methods: {methods}")
print(f"Regions: {regions}")

print()
print("===== FIG. C3 NUMERICAL AUDIT =====")
print(audit.to_string(index=False))

print()
print("===== SAVED FIG. C3 =====")
print(PNG)
print(PDF)
print(AUDIT)

print()
print("C3 generation finished successfully.")
