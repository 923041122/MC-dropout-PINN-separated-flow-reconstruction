
"""Post-process final cylinder held-out time-resolved metrics.

No training and no model loading are performed.

Input
-----
cylinder_final_lambda_0.1/heldout_evaluation/heldout_time_resolved_metrics.csv

Outputs
-------
<results-root>/time_resolved_analysis/
    time_resolved_summary.csv
    velocity_rmse_vs_time.png
    velocity_relative_l2_vs_time.png
    velocity_max_abs_error_vs_time.png
    pressure_rmse_raw_vs_gauge_aligned.png
    pressure_relative_l2_raw_vs_gauge_aligned.png
    README_TIME_RESOLVED.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input-csv",
        default="./cylinder_final_lambda_0.1/heldout_evaluation/"
                "heldout_time_resolved_metrics.csv",
    )
    p.add_argument(
        "--results-root",
        default="./cylinder_final_lambda_0.1",
    )
    return p.parse_args()


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = [
        "mae",
        "rmse",
        "nrmse_by_reference_range",
        "relative_l2",
        "max_absolute_error",
    ]

    for variable in sorted(df["variable"].unique()):
        sub = df[df["variable"] == variable]

        for metric in metrics:
            values = sub[metric].to_numpy(dtype=float)
            rows.append({
                "variable": variable,
                "metric": metric,
                "time_mean": float(np.mean(values)),
                "time_std": float(np.std(values, ddof=0)),
                "time_median": float(np.median(values)),
                "time_min": float(np.min(values)),
                "time_max": float(np.max(values)),
                "time_p05": float(np.quantile(values, 0.05)),
                "time_p95": float(np.quantile(values, 0.95)),
                "time_of_min": float(
                    sub.iloc[int(np.argmin(values))]["time_value"]
                ),
                "time_of_max": float(
                    sub.iloc[int(np.argmax(values))]["time_value"]
                ),
                "snapshots": int(len(values)),
            })

    return pd.DataFrame(rows)


def get_series(df, variable, metric):
    sub = df[df["variable"] == variable].sort_values("time_value")
    return (
        sub["time_value"].to_numpy(dtype=float),
        sub[metric].to_numpy(dtype=float),
    )


def plot_two(df, variables, metric, ylabel, title, path, labels=None):
    fig, ax = plt.subplots(figsize=(8.8, 5.0))

    if labels is None:
        labels = variables

    for variable, label in zip(variables, labels):
        t, y = get_series(df, variable, metric)
        ax.plot(t, y, label=label)

    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=250)
    plt.close(fig)


def main():
    args = parse_args()

    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(input_csv)

    df = pd.read_csv(input_csv)

    required = {
        "time_index",
        "time_value",
        "heldout_points",
        "variable",
        "mae",
        "rmse",
        "nrmse_by_reference_range",
        "relative_l2",
        "max_absolute_error",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    output = Path(args.results_root) / "time_resolved_analysis"
    output.mkdir(parents=True, exist_ok=True)

    summary = summarize(df)
    summary.to_csv(
        output / "time_resolved_summary.csv",
        index=False,
    )

    plot_two(
        df,
        ["u", "v"],
        "rmse",
        "RMSE",
        "Time-resolved held-out velocity reconstruction errors",
        output / "velocity_rmse_vs_time.png",
        ["u", "v"],
    )

    plot_two(
        df,
        ["u", "v"],
        "relative_l2",
        "Relative L2 error",
        "Time-resolved held-out velocity relative L2 errors",
        output / "velocity_relative_l2_vs_time.png",
        ["u", "v"],
    )

    plot_two(
        df,
        ["u", "v"],
        "max_absolute_error",
        "Maximum absolute error",
        "Time-resolved held-out velocity maximum absolute errors",
        output / "velocity_max_abs_error_vs_time.png",
        ["u", "v"],
    )

    plot_two(
        df,
        ["p_raw", "p_gauge_aligned"],
        "rmse",
        "RMSE",
        "Effect of pressure-gauge alignment on held-out pressure error",
        output / "pressure_rmse_raw_vs_gauge_aligned.png",
        ["Raw pressure", "Gauge-aligned pressure"],
    )

    plot_two(
        df,
        ["p_raw", "p_gauge_aligned"],
        "relative_l2",
        "Relative L2 error",
        "Effect of pressure-gauge alignment on held-out pressure relative L2 error",
        output / "pressure_relative_l2_raw_vs_gauge_aligned.png",
        ["Raw pressure", "Gauge-aligned pressure"],
    )

    readme = """Cylinder final held-out time-resolved analysis

This analysis uses only the final held-out test results after the physics-loss
weight was fixed. No hyperparameter is changed based on these results.

The fixed spatial held-out mask is evaluated at every time snapshot, enabling
complete-time error curves and complete-time averages/extrema.

Outputs:
- time_resolved_summary.csv
- velocity_rmse_vs_time.png
- velocity_relative_l2_vs_time.png
- velocity_max_abs_error_vs_time.png
- pressure_rmse_raw_vs_gauge_aligned.png
- pressure_relative_l2_raw_vs_gauge_aligned.png
"""
    (output / "README_TIME_RESOLVED.txt").write_text(
        readme,
        encoding="utf-8",
    )

    print("=" * 88)
    print("Cylinder complete-time held-out analysis finished")
    print(f"Snapshots: {df['time_index'].nunique()}")
    print(f"Output: {output.resolve()}")
    print("=" * 88)

    show = summary[
        (summary["metric"] == "rmse")
        & summary["variable"].isin(
            ["u", "v", "p_raw", "p_gauge_aligned"]
        )
    ][
        [
            "variable",
            "time_mean",
            "time_std",
            "time_min",
            "time_max",
            "time_p95",
            "time_of_max",
            "snapshots",
        ]
    ]

    print("\n=== Complete-time RMSE summary ===")
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
