"""Dropout-rate and Monte-Carlo-sample ablation for B-PINN uncertainty.

This script evaluates how dropout rate and the number of MC samples affect
uncertainty calibration for B-PINN / MC-dropout.

Main outputs:
    1. dropout_mc_ablation.csv
    2. dropout_mc_ablation_summary.csv
    3. coverage_ablation_u/v/p.png
    4. interval_width_ablation_u/v/p.png
    5. correlation_ablation_u/v/p.png
    6. calibration_error_ablation_u/v/p.png

Important change:
    The 95% prediction interval is computed from MC quantiles:

        lower = quantile(samples, 0.025)
        upper = quantile(samples, 0.975)

    instead of mean ± 1.96 * std.
"""

import argparse
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from benchmark_config import DEFAULT_EVAL, EVAL_ROOT, LAYER_MAT_PSI, METHODS
from benchmark_evaluate import load_data_stack, prepare_snapshot
from benchmark_tools import build_model, get_device, predict_uvp_numpy, safe_load_state

mpl.use("Agg")


def parse_args():
    parser = argparse.ArgumentParser(description="Run B-PINN UQ ablation.")

    parser.add_argument(
        "--data-path",
        default="./2d_cylinder_Re3900_100x100_kw_sst.mat",
        help="Path to the Re=3900 cylinder-flow .mat data file.",
    )

    parser.add_argument(
        "--checkpoint",
        default=str(METHODS["bpinn_dropout"]["checkpoint"]),
        help="Checkpoint path of the trained B-PINN / MC-dropout model.",
    )

    parser.add_argument(
        "--time-indices",
        nargs="+",
        type=int,
        default=DEFAULT_EVAL["time_indices"],
        help="Snapshot indices used for uncertainty ablation.",
    )

    parser.add_argument(
        "--dropout-rates",
        nargs="+",
        type=float,
        default=DEFAULT_EVAL["dropout_rates"],
        help="Dropout rates used for ablation.",
    )

    parser.add_argument(
        "--mc-samples",
        nargs="+",
        type=int,
        default=DEFAULT_EVAL["mc_sample_grid"],
        help="Monte-Carlo sample numbers used for ablation.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_EVAL["batch_size"],
        help="Batch size used during full-field inference.",
    )

    return parser.parse_args()


def mc_metrics(model, x_flat, y_flat, t_flat, truths, device, samples, batch_size):
    """Compute MC-dropout uncertainty metrics for u, v and p.

    The 95% interval is computed using MC quantiles:
        [2.5%, 97.5%]

    This is more appropriate for MC-dropout than assuming Gaussian uncertainty.
    """
    stacks = {
        "u": [],
        "v": [],
        "p": [],
    }

    was_training = model.training
    model.train()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()

    for _ in range(samples):
        pred, _ = predict_uvp_numpy(
            model,
            x_flat,
            y_flat,
            t_flat,
            device,
            batch_size=batch_size,
            eval_mode=False,
        )

        for quantity in stacks:
            stacks[quantity].append(
                pred[quantity].reshape(truths[quantity].shape)
            )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    mc_inference_time = time.perf_counter() - start

    if was_training:
        model.train()
    else:
        model.eval()

    rows = []

    for quantity, values in stacks.items():
        arr = np.stack(values, axis=0)

        mean = arr.mean(axis=0)
        std = arr.std(axis=0, ddof=0)

        error = np.abs(mean - truths[quantity])

        lower = np.quantile(arr, 0.025, axis=0)
        upper = np.quantile(arr, 0.975, axis=0)

        covered = (truths[quantity] >= lower) & (truths[quantity] <= upper)

        coverage_95 = float(covered.mean())
        mpiw_95 = float((upper - lower).mean())
        calibration_error_95 = float(abs(coverage_95 - 0.95))

        if np.std(std) > 0:
            corr = np.corrcoef(error.ravel(), std.ravel())[0, 1]
        else:
            corr = np.nan

        rows.append(
            {
                "variable": quantity,
                "mae": float(error.mean()),
                "rmse": float(np.sqrt(np.mean((mean - truths[quantity]) ** 2))),
                "coverage_95": coverage_95,
                "mpiw_95": mpiw_95,
                "calibration_error_95": calibration_error_95,
                "error_uncertainty_correlation": float(corr),
                "mc_inference_time_seconds": float(mc_inference_time),
            }
        )

    return rows


def plot_coverage_ablation(df, dropout_rates, save_dir):
    for quantity in ["u", "v", "p"]:
        fig, ax = plt.subplots(figsize=(6, 4))

        sub = df[df["variable"] == quantity]

        for dropout_rate in dropout_rates:
            rate_df = sub[sub["dropout_rate"] == dropout_rate]

            coverage = (
                rate_df.groupby("mc_samples")["coverage_95"]
                .mean()
                .sort_index()
            )

            ax.plot(
                coverage.index,
                coverage.values,
                marker="o",
                linewidth=2,
                label=f"dropout={dropout_rate}",
            )

        ax.axhline(
            0.95,
            color="k",
            linestyle="--",
            linewidth=1,
            label="ideal 95%",
        )

        ax.set_xlabel("MC samples")
        ax.set_ylabel("Empirical 95% coverage")
        ax.set_title(f"Coverage ablation for {quantity}")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        fig.tight_layout()

        fig.savefig(
            save_dir / f"coverage_ablation_{quantity}.png",
            dpi=300,
        )

        fig.savefig(
            save_dir / f"coverage_ablation_{quantity}.pdf",
            dpi=300,
        )

        plt.close(fig)


def plot_width_ablation(df, dropout_rates, save_dir):
    for quantity in ["u", "v", "p"]:
        fig, ax = plt.subplots(figsize=(6, 4))

        sub = df[df["variable"] == quantity]

        for dropout_rate in dropout_rates:
            rate_df = sub[sub["dropout_rate"] == dropout_rate]

            width = (
                rate_df.groupby("mc_samples")["mpiw_95"]
                .mean()
                .sort_index()
            )

            ax.plot(
                width.index,
                width.values,
                marker="o",
                linewidth=2,
                label=f"dropout={dropout_rate}",
            )

        ax.set_xlabel("MC samples")
        ax.set_ylabel("Mean prediction interval width, 95%")
        ax.set_title(f"Interval-width ablation for {quantity}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        fig.tight_layout()

        fig.savefig(
            save_dir / f"interval_width_ablation_{quantity}.png",
            dpi=300,
        )

        fig.savefig(
            save_dir / f"interval_width_ablation_{quantity}.pdf",
            dpi=300,
        )

        plt.close(fig)


def plot_correlation_ablation(df, dropout_rates, save_dir):
    for quantity in ["u", "v", "p"]:
        fig, ax = plt.subplots(figsize=(6, 4))

        sub = df[df["variable"] == quantity]

        for dropout_rate in dropout_rates:
            rate_df = sub[sub["dropout_rate"] == dropout_rate]

            corr = (
                rate_df.groupby("mc_samples")["error_uncertainty_correlation"]
                .mean()
                .sort_index()
            )

            ax.plot(
                corr.index,
                corr.values,
                marker="o",
                linewidth=2,
                label=f"dropout={dropout_rate}",
            )

        ax.set_xlabel("MC samples")
        ax.set_ylabel("Error-uncertainty correlation")
        ax.set_title(f"Correlation ablation for {quantity}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        fig.tight_layout()

        fig.savefig(
            save_dir / f"correlation_ablation_{quantity}.png",
            dpi=300,
        )

        fig.savefig(
            save_dir / f"correlation_ablation_{quantity}.pdf",
            dpi=300,
        )

        plt.close(fig)


def plot_calibration_error_ablation(df, dropout_rates, save_dir):
    for quantity in ["u", "v", "p"]:
        fig, ax = plt.subplots(figsize=(6, 4))

        sub = df[df["variable"] == quantity]

        for dropout_rate in dropout_rates:
            rate_df = sub[sub["dropout_rate"] == dropout_rate]

            calibration_error = (
                rate_df.groupby("mc_samples")["calibration_error_95"]
                .mean()
                .sort_index()
            )

            ax.plot(
                calibration_error.index,
                calibration_error.values,
                marker="o",
                linewidth=2,
                label=f"dropout={dropout_rate}",
            )

        ax.set_xlabel("MC samples")
        ax.set_ylabel("|Empirical coverage - 0.95|")
        ax.set_title(f"Calibration-error ablation for {quantity}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        fig.tight_layout()

        fig.savefig(
            save_dir / f"calibration_error_ablation_{quantity}.png",
            dpi=300,
        )

        fig.savefig(
            save_dir / f"calibration_error_ablation_{quantity}.pdf",
            dpi=300,
        )

        plt.close(fig)


def save_summary_table(df, save_dir):
    """Save a compact summary averaged over selected time snapshots."""
    summary = (
        df.groupby(["dropout_rate", "mc_samples", "variable"])
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            coverage_95_mean=("coverage_95", "mean"),
            coverage_95_std=("coverage_95", "std"),
            mpiw_95_mean=("mpiw_95", "mean"),
            mpiw_95_std=("mpiw_95", "std"),
            calibration_error_95_mean=("calibration_error_95", "mean"),
            calibration_error_95_std=("calibration_error_95", "std"),
            error_uncertainty_correlation_mean=(
                "error_uncertainty_correlation",
                "mean",
            ),
            error_uncertainty_correlation_std=(
                "error_uncertainty_correlation",
                "std",
            ),
            mc_inference_time_seconds_mean=("mc_inference_time_seconds", "mean"),
            mc_inference_time_seconds_std=("mc_inference_time_seconds", "std"),
        )
        .reset_index()
    )

    summary_path = save_dir / "dropout_mc_ablation_summary.csv"
    summary.to_csv(summary_path, index=False)

    return summary_path


def main():
    args = parse_args()

    device = get_device()
    data_stack = load_data_stack(args.data_path)

    save_dir = EVAL_ROOT / "uq_ablation"
    save_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for dropout_rate in args.dropout_rates:
        cfg = dict(METHODS["bpinn_dropout"])
        cfg["dropout_rate"] = dropout_rate

        model = build_model(cfg, LAYER_MAT_PSI).to(device)

        model.load_state_dict(
            safe_load_state(args.checkpoint, device),
            strict=False,
        )

        model.eval()

        for mc_samples in args.mc_samples:
            for time_index in args.time_indices:
                (
                    select_time,
                    _,
                    _,
                    x_flat,
                    y_flat,
                    t_flat,
                    truths,
                ) = prepare_snapshot(data_stack, time_index)

                metrics = mc_metrics(
                    model=model,
                    x_flat=x_flat,
                    y_flat=y_flat,
                    t_flat=t_flat,
                    truths=truths,
                    device=device,
                    samples=mc_samples,
                    batch_size=args.batch_size,
                )

                for row in metrics:
                    row.update(
                        {
                            "dropout_rate": float(dropout_rate),
                            "mc_samples": int(mc_samples),
                            "time_index": int(time_index),
                            "time_value": float(select_time),
                            "nominal_coverage": 0.95,
                            "interval_method": "mc_quantile",
                        }
                    )

                    rows.append(row)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    df = pd.DataFrame(rows)

    csv_path = save_dir / "dropout_mc_ablation.csv"
    df.to_csv(csv_path, index=False)

    summary_path = save_summary_table(df, save_dir)

    plot_coverage_ablation(
        df=df,
        dropout_rates=args.dropout_rates,
        save_dir=save_dir,
    )

    plot_width_ablation(
        df=df,
        dropout_rates=args.dropout_rates,
        save_dir=save_dir,
    )

    plot_correlation_ablation(
        df=df,
        dropout_rates=args.dropout_rates,
        save_dir=save_dir,
    )

    plot_calibration_error_ablation(
        df=df,
        dropout_rates=args.dropout_rates,
        save_dir=save_dir,
    )

    print(f"Ablation results saved to {save_dir.resolve()}")
    print(f"Ablation table saved to: {csv_path}")
    print(f"Ablation summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
