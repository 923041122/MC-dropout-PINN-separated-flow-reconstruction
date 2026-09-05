from pathlib import Path
import numpy as np
import pandas as pd

from benchmark_tools import make_wake_region_masks


FILES = {
    "Standard PINN":
        Path("./cylinder_final_lambda_0.1/heldout_evaluation/"
             "heldout_pointwise_predictions.csv"),

    "Adaptive-weight PINN":
        Path("./baseline_suite/cylinder/adaptive_weight/final_heldout/"
             "heldout_evaluation/heldout_pointwise_predictions.csv"),

    "Weight-decay PINN":
        Path("./baseline_suite/cylinder/weight_decay/final_heldout/"
             "heldout_evaluation/heldout_pointwise_predictions.csv"),
}

OUTDIR = Path("./evaluation/final_heldout_regional")
OUTDIR.mkdir(parents=True, exist_ok=True)


def metrics(ref, pred):
    ref = np.asarray(ref, dtype=float)
    pred = np.asarray(pred, dtype=float)

    err = pred - ref

    return {
        "n_points": len(ref),
        "mae": np.mean(np.abs(err)),
        "rmse": np.sqrt(np.mean(err ** 2)),
        "max_abs_error": np.max(np.abs(err)),
        "relative_l2": (
            np.linalg.norm(err) /
            (np.linalg.norm(ref) + 1e-12)
        ),
    }


# ------------------------------------------------------------
# 1. Load frozen final-heldout predictions
# ------------------------------------------------------------

dfs = {}

for method, path in FILES.items():
    if not path.exists():
        raise FileNotFoundError(f"{method}: {path}")

    df = pd.read_csv(path)
    dfs[method] = df

    print(f"{method}: {len(df):,} held-out points")


# ------------------------------------------------------------
# 2. Strict provenance checks
# ------------------------------------------------------------

reference_method = "Standard PINN"
base = dfs[reference_method]

required = {
    "flat_index", "spatial_index", "time_index",
    "x", "y", "t",
    "u_ref", "u_pred",
    "v_ref", "v_pred",
    "p_ref",
    "p_pred_raw",
    "p_pred_gauge_aligned",
}

for method, df in dfs.items():

    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"{method}: missing columns: {sorted(missing)}"
        )

    if len(df) != len(base):
        raise RuntimeError(
            f"{method}: held-out row count differs from Standard"
        )

    if not np.array_equal(
        df["flat_index"].to_numpy(),
        base["flat_index"].to_numpy()
    ):
        raise RuntimeError(
            f"{method}: flat_index does not match Standard held-out set"
        )

    for col in ["x", "y", "t", "u_ref", "v_ref", "p_ref"]:
        if not np.allclose(
            df[col].to_numpy(),
            base[col].to_numpy(),
            rtol=0.0,
            atol=1e-10,
        ):
            raise RuntimeError(
                f"{method}: reference column '{col}' differs from Standard"
            )

print()
print("PASS: all three methods use the same frozen held-out points.")
print(f"Held-out size: {len(base):,}")
print(
    f"Snapshots represented: "
    f"{base['time_index'].nunique()}"
)


# ------------------------------------------------------------
# 3. Apply the EXISTING frozen wake-region definitions
# ------------------------------------------------------------

masks = make_wake_region_masks(
    base["x"].to_numpy(),
    base["y"].to_numpy(),
)

print()
print("Regional held-out counts:")

for region, mask in masks.items():
    print(
        f"  {region:20s}: "
        f"{int(np.sum(mask)):,}"
    )


# ------------------------------------------------------------
# 4. Compute all-time regional metrics
#
# Pressure policy:
#   Main/canonical pressure metric = gauge-aligned pressure.
#   Raw pressure is retained separately for audit/provenance.
# ------------------------------------------------------------

rows = []

variable_map = {
    "u": ("u_ref", "u_pred"),
    "v": ("v_ref", "v_pred"),
    "p_gauge_aligned": (
        "p_ref",
        "p_pred_gauge_aligned",
    ),
    "p_raw": (
        "p_ref",
        "p_pred_raw",
    ),
}

for method, df in dfs.items():

    for region, mask in masks.items():

        mask = np.asarray(mask, dtype=bool)

        if mask.sum() == 0:
            raise RuntimeError(
                f"Region '{region}' contains zero held-out points"
            )

        for variable, (ref_col, pred_col) in variable_map.items():

            result = metrics(
                df.loc[mask, ref_col],
                df.loc[mask, pred_col],
            )

            rows.append({
                "method": method,
                "region": region,
                "variable": variable,
                **result,
            })


regional = pd.DataFrame(rows)

regional_path = (
    OUTDIR / "cylinder_regional_final_heldout_metrics.csv"
)

regional.to_csv(regional_path, index=False)


# ------------------------------------------------------------
# 5. Paper-oriented RMSE summary
# ------------------------------------------------------------

paper = regional[
    regional["variable"].isin(
        ["u", "v", "p_gauge_aligned"]
    )
].copy()

paper_table = paper.pivot_table(
    index=["method", "region"],
    columns="variable",
    values="rmse",
).reset_index()

paper_table = paper_table.rename(columns={
    "u": "u_rmse",
    "v": "v_rmse",
    "p_gauge_aligned": "p_gauge_aligned_rmse",
})

paper_path = (
    OUTDIR / "cylinder_regional_final_heldout_summary.csv"
)

paper_table.to_csv(paper_path, index=False)


# ------------------------------------------------------------
# 6. Time-resolved regional RMSE
#    Uses ALL 100 held-out snapshots; no cherry-picking.
# ------------------------------------------------------------

time_rows = []

for method, df in dfs.items():

    for time_index in sorted(df["time_index"].unique()):

        tm = df["time_index"].to_numpy() == time_index
        time_value = float(
            df.loc[tm, "t"].iloc[0]
        )

        for region, region_mask in masks.items():

            combined = tm & np.asarray(region_mask, dtype=bool)

            if combined.sum() == 0:
                continue

            for variable, (ref_col, pred_col) in variable_map.items():

                result = metrics(
                    df.loc[combined, ref_col],
                    df.loc[combined, pred_col],
                )

                time_rows.append({
                    "method": method,
                    "time_index": int(time_index),
                    "time_value": time_value,
                    "region": region,
                    "variable": variable,
                    **result,
                })


time_df = pd.DataFrame(time_rows)

time_path = (
    OUTDIR /
    "cylinder_regional_final_heldout_time_resolved.csv"
)

time_df.to_csv(time_path, index=False)


# ------------------------------------------------------------
# 7. README / provenance record
# ------------------------------------------------------------

readme = OUTDIR / "README_REGIONAL_FINAL_HELDOUT.txt"

with open(readme, "w", encoding="utf-8") as f:
    f.write(
        "Cylinder regional final-heldout diagnostics\n"
        "===========================================\n\n"
        "Source:\n"
        "Existing frozen heldout_pointwise_predictions.csv files.\n"
        "No training and no new model inference were performed.\n\n"
        f"Held-out points: {len(base)}\n"
        f"Snapshots represented: {base['time_index'].nunique()}\n\n"
        "Region definitions:\n"
        "Imported unchanged from "
        "benchmark_tools.make_wake_region_masks().\n\n"
        "Pressure policy:\n"
        "Gauge-aligned pressure is the canonical pressure result.\n"
        "Raw pressure is retained only as an audit quantity.\n\n"
        "Methods:\n"
        "- Standard PINN\n"
        "- Adaptive-weight PINN\n"
        "- Weight-decay PINN\n"
    )


print()
print("===== PAPER-ORIENTED REGIONAL RMSE =====")
print(
    paper_table.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

print()
print("Saved:")
print(f"  {regional_path}")
print(f"  {paper_path}")
print(f"  {time_path}")
print(f"  {readme}")
