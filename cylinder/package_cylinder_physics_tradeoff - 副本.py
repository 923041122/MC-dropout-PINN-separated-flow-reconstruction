from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(".")
OUT = ROOT / "evaluation" / "paper_ready_cylinder_diagnostics"
OUT.mkdir(parents=True, exist_ok=True)

BASE = (
    ROOT /
    "cylinder_physics_weight_tradeoff" /
    "cylinder_physics_weight_formal_tradeoff"
)

SUMMARY = BASE / "cylinder_physics_weight_formal_tradeoff_summary.csv"
PHYSICS = BASE / "independent_physics_residual_metrics.csv"

if not SUMMARY.exists():
    raise FileNotFoundError(SUMMARY)

if not PHYSICS.exists():
    raise FileNotFoundError(PHYSICS)

summary = pd.read_csv(SUMMARY)
physics = pd.read_csv(PHYSICS)

print("SUMMARY columns:")
print(list(summary.columns))
print()
print("PHYSICS columns:")
print(list(physics.columns))

key = "equation_loss_weight"

if key not in summary.columns or key not in physics.columns:
    raise RuntimeError("equation_loss_weight column missing")

# Keep the formal candidates only.
formal_weights = [0.0, 0.1, 1.0]

summary = summary[
    summary[key].isin(formal_weights)
].copy()

physics = physics[
    physics[key].isin(formal_weights)
].copy()

merged = pd.merge(
    summary,
    physics,
    on=key,
    how="inner",
    suffixes=("_summary", "_independent"),
)

merged = merged.sort_values(key)

if len(merged) != 3:
    raise RuntimeError(
        f"Expected 3 formal candidates, found {len(merged)}"
    )

merged.to_csv(
    OUT / "Table_S3_physics_weight_tradeoff.csv",
    index=False,
)

# ------------------------------------------------------------
# Identify canonical columns
# ------------------------------------------------------------

def pick(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None

u_col = pick(merged, [
    "val_u_rmse",
    "validation_u_rmse",
])

v_col = pick(merged, [
    "val_v_rmse",
    "validation_v_rmse",
])

p_col = pick(merged, [
    "val_p_rmse",
    "validation_p_rmse",
])

vec_col = pick(merged, [
    "physics_vector_rmse_independent",
    "physics_vector_rmse",
])

p95_col = pick(merged, [
    "physics_magnitude_p95_independent",
    "physics_magnitude_p95",
])

if vec_col is None or p95_col is None:
    raise RuntimeError(
        "Independent physics-vector RMSE/P95 columns not found"
    )

# ------------------------------------------------------------
# Figure S4
# ------------------------------------------------------------

labels = [
    f"{x:g}" for x in merged[key].to_numpy()
]

x = range(len(labels))

fig, axes = plt.subplots(
    1, 2,
    figsize=(10.8, 4.3),
)

# Panel a: validation reconstruction accuracy
if u_col and v_col and p_col:

    axes[0].plot(
        x, merged[u_col],
        marker="o",
        label=r"$u$",
    )

    axes[0].plot(
        x, merged[v_col],
        marker="o",
        label=r"$v$",
    )

    axes[0].plot(
        x, merged[p_col],
        marker="o",
        label="Pressure",
    )

    axes[0].set_ylabel("Validation RMSE")
    axes[0].legend(frameon=False)

else:
    axes[0].text(
        0.5, 0.5,
        "Validation RMSE columns unavailable",
        ha="center",
        va="center",
        transform=axes[0].transAxes,
    )

axes[0].set_title("(a) Reconstruction accuracy")
axes[0].set_xlabel(r"Equation-loss weight $\lambda_f$")
axes[0].set_xticks(list(x))
axes[0].set_xticklabels(labels)
axes[0].grid(alpha=0.20)

# Panel b: independent physics diagnostics
axes[1].plot(
    x,
    merged[vec_col],
    marker="o",
    label="Physics-vector RMSE",
)

axes[1].plot(
    x,
    merged[p95_col],
    marker="o",
    label="Residual magnitude P95",
)

axes[1].set_title("(b) Independent physics residual")
axes[1].set_xlabel(r"Equation-loss weight $\lambda_f$")
axes[1].set_ylabel("Residual metric")
axes[1].set_xticks(list(x))
axes[1].set_xticklabels(labels)
axes[1].grid(alpha=0.20)
axes[1].legend(frameon=False)

# Mark the selected formal setting λ_f = 0.1
selected_index = labels.index("0.1")

for ax in axes:
    ax.axvline(
        selected_index,
        linestyle=":",
        linewidth=1.2,
    )

fig.tight_layout()

fig.savefig(
    OUT / "Fig_S4_physics_weight_tradeoff.png",
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    OUT / "Fig_S4_physics_weight_tradeoff.pdf",
    bbox_inches="tight",
)

plt.close(fig)

print()
print("=" * 68)
print("PHYSICS TRADE-OFF PACKAGING FINISHED")
print("=" * 68)

print(
    OUT / "Fig_S4_physics_weight_tradeoff.png"
)
print(
    OUT / "Fig_S4_physics_weight_tradeoff.pdf"
)
print(
    OUT / "Table_S3_physics_weight_tradeoff.csv"
)

print()
print("NO TRAINING.")
print("NO MODEL INFERENCE.")
print("EXISTING FORMAL EVIDENCE ONLY.")
