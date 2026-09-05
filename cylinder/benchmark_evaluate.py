"""Evaluate baselines, plot absolute-error maps, local errors and UQ calibration.

This script is designed for reviewer-response experiments. It evaluates
multiple PINN variants on u, v, and p fields, generates pointwise absolute-error
maps, computes local-region errors, and performs MC-dropout uncertainty
calibration for B-PINN.

Main outputs:
    1. global_uvp_errors.csv
    2. local_region_errors.csv
    3. timing_and_parameters.csv
    4. uncertainty_calibration_metrics.csv
    5. absolute-error maps
    6. uncertainty maps
    7. calibration curves
"""

import argparse
import time
from statistics import NormalDist

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from benchmark_config import (
    ACTIVE_METHODS,
    DEFAULT_EVAL,
    EVAL_ROOT,
    LAYER_MAT_PSI,
    METHODS,
)
from benchmark_tools import (
    count_parameters,
    get_device,
    load_trained_model,
    make_wake_region_masks,
    metric_dict,
    predict_uvp_numpy,
)
from pinn_model import read_2D_data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reviewer-response benchmark evaluation."
    )

    parser.add_argument(
        "--data-path",
        default="./2d_cylinder_Re3900_100x100_kw_sst.mat",
        help="Path to the Re=3900 cylinder-flow .mat data file.",
    )

    parser.add_argument(
        "--methods",
        nargs="+",
        default=ACTIVE_METHODS,
        help="Methods to evaluate. Names must exist in benchmark_config.METHODS.",
    )

    parser.add_argument(
        "--time-indices",
        nargs="+",
        type=int,
        default=DEFAULT_EVAL["time_indices"],
        help="Snapshot indices used for error maps and UQ calibration.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_EVAL["batch_size"],
        help="Batch size used during full-field inference.",
    )

    parser.add_argument(
        "--mc-samples",
        type=int,
        default=DEFAULT_EVAL["mc_samples"],
        help="Number of MC-dropout samples for uncertainty quantification.",
    )

    parser.add_argument(
        "--skip-uq",
        action="store_true",
        help="Skip MC-dropout uncertainty calibration and uncertainty maps.",
    )

    return parser.parse_args()


def _to_numpy(array):
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()

    return np.asarray(array)


def load_data_stack(data_path):
    x, y, t, u, v, p, _ = read_2D_data(data_path)

    x = _to_numpy(x)
    y = _to_numpy(y)
    t = _to_numpy(t)
    u = _to_numpy(u)
    v = _to_numpy(v)
    p = _to_numpy(p)

    data_stack = np.concatenate(
        (
            x,
            y,
            t,
            u,
            v,
            p,
        ),
        axis=1,
    )

    return data_stack


def prepare_snapshot(data_stack, time_index):
    x = data_stack[:, 0:1]
    y = data_stack[:, 1:2]
    t = data_stack[:, 2:3]

    t_unique = np.unique(t).reshape(-1, 1)

    if time_index >= len(t_unique):
        raise IndexError(
            f"time_index {time_index} exceeds available snapshots {len(t_unique)}"
        )

    select_time = float(t_unique[time_index, 0])

    idx = np.where(t == select_time)[0]

    x_unique = np.unique(x).reshape(-1, 1)
    y_unique = np.unique(y).reshape(-1, 1)

    mesh_x, mesh_y = np.meshgrid(x_unique, y_unique)

    x_flat = mesh_x.reshape(-1, 1)
    y_flat = mesh_y.reshape(-1, 1)
    t_flat = np.ones_like(x_flat) * select_time

    truths = {
        "u": data_stack[idx, 3].reshape(mesh_x.shape),
        "v": data_stack[idx, 4].reshape(mesh_x.shape),
        "p": data_stack[idx, 5].reshape(mesh_x.shape),
    }

    return select_time, mesh_x, mesh_y, x_flat, y_flat, t_flat, truths


def plot_absolute_error_maps(error_maps, method_labels, select_time, quantity, save_dir):
    """Plot pointwise absolute-error maps for all methods.

    The same color scale is used for all methods under the same variable and
    time instant.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    vmax = max(float(np.percentile(err, 99)) for err in error_maps.values())
    vmax = max(vmax, 1e-12)

    fig, axes = plt.subplots(
        1,
        len(error_maps),
        figsize=(5 * len(error_maps), 4),
        squeeze=False,
    )

    im = None

    for ax, (method, err) in zip(axes.ravel(), error_maps.items()):
        im = ax.imshow(
            err,
            cmap="magma",
            vmin=0.0,
            vmax=vmax,
            aspect="auto",
        )

        ax.set_title(method_labels[method])
        ax.set_xlabel("X grid")
        ax.set_ylabel("Y grid")

    fig.suptitle(
        f"Absolute error of {quantity} at t = {select_time:.2f}",
        fontsize=14,
    )

    fig.colorbar(
        im,
        ax=axes.ravel().tolist(),
        shrink=0.85,
        label="Absolute error",
    )

    fig.savefig(
        save_dir / f"{quantity}_absolute_error_t_{select_time:.2f}.png",
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        save_dir / f"{quantity}_absolute_error_t_{select_time:.2f}.pdf",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def local_error_rows(method, time_index, select_time, quantity, abs_error, masks):
    rows = []

    for region, mask in masks.items():
        values = abs_error[mask]

        rows.append(
            {
                "method": method,
                "time_index": time_index,
                "time_value": select_time,
                "variable": quantity,
                "region": region,
                "local_mae": float(values.mean()) if values.size else np.nan,
                "local_rmse": float(np.sqrt(np.mean(values ** 2)))
                if values.size
                else np.nan,
                "local_max_abs_error": float(values.max()) if values.size else np.nan,
                "points": int(values.size),
            }
        )

    return rows


def normal_interval_multiplier(level):
    """Return the two-sided normal z multiplier for a central interval.

    Example:
        level = 0.95 gives z = 1.95996 for mean ± z * std.

    This function is kept for optional Gaussian-interval diagnostics.
    In the main UQ calibration below, MC quantile intervals are used instead.
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be between 0 and 1, got {level}")

    return NormalDist().inv_cdf((1.0 + level) / 2.0)


def mc_dropout_stats(model, x_np, y_np, t_np, device, samples, batch_size):
    """Run MC-dropout inference and return mean, std, samples, and timing.

    Dropout is activated by setting model.train() during sampling. The original
    training/evaluation mode is restored afterwards.
    """
    if samples < 2:
        raise ValueError(f"MC-dropout requires at least 2 samples, got {samples}")

    all_u = []
    all_v = []
    all_p = []

    was_training = model.training
    model.train()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()

    for _ in range(samples):
        preds, _ = predict_uvp_numpy(
            model,
            x_np,
            y_np,
            t_np,
            device,
            batch_size=batch_size,
            eval_mode=False,
        )

        all_u.append(preds["u"])
        all_v.append(preds["v"])
        all_p.append(preds["p"])

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    if was_training:
        model.train()
    else:
        model.eval()

    stacks = {
        "u": np.stack(all_u, axis=0),
        "v": np.stack(all_v, axis=0),
        "p": np.stack(all_p, axis=0),
    }

    return {
        q: {
            "mean": arr.mean(axis=0),
            "std": arr.std(axis=0, ddof=0),
            "samples": arr,
            "time": elapsed,
        }
        for q, arr in stacks.items()
    }


def uq_rows(
    method,
    time_index,
    select_time,
    quantity,
    truth,
    mean,
    std,
    samples,
    levels,
    mc_samples=None,
    mc_inference_time_seconds=None,
):
    """Compute UQ calibration metrics using MC quantile intervals.

    For MC-dropout, prediction intervals are computed directly from MC samples:

        lower = quantile(samples, (1 - c) / 2)
        upper = quantile(samples, (1 + c) / 2)

    where c is the nominal coverage level.
    """
    rows = []

    truth_flat = np.asarray(truth).reshape(-1)
    mean_flat = np.asarray(mean).reshape(-1)
    std_flat = np.asarray(std).reshape(-1)

    sample_arr = np.asarray(samples)
    sample_arr = sample_arr.reshape(sample_arr.shape[0], -1)

    error = np.abs(mean_flat - truth_flat)

    if np.std(std_flat) > 0:
        corr = np.corrcoef(error.ravel(), std_flat.ravel())[0, 1]
    else:
        corr = np.nan

    for level in levels:
        level = float(level)

        if not 0.0 < level < 1.0:
            raise ValueError(f"nominal coverage must be between 0 and 1, got {level}")

        lower_q = (1.0 - level) / 2.0
        upper_q = (1.0 + level) / 2.0

        lower = np.quantile(sample_arr, lower_q, axis=0)
        upper = np.quantile(sample_arr, upper_q, axis=0)

        covered = (truth_flat >= lower) & (truth_flat <= upper)

        empirical_coverage = float(covered.mean())
        mpiw = float((upper - lower).mean())
        calibration_error = float(abs(empirical_coverage - level))

        rows.append(
            {
                "method": method,
                "time_index": time_index,
                "time_value": select_time,
                "variable": quantity,
                "nominal_coverage": level,
                "empirical_coverage": empirical_coverage,
                "mean_prediction_interval_width": mpiw,
                "calibration_error": calibration_error,
                "error_uncertainty_correlation": float(corr),
                "mc_samples": int(mc_samples) if mc_samples is not None else None,
                "mc_inference_time_seconds": float(mc_inference_time_seconds)
                if mc_inference_time_seconds is not None
                else None,
                "interval_method": "mc_quantile",
            }
        )

    return rows


def plot_calibration_curve(uq_df, save_dir):
    """Plot calibration curves for u, v, and p."""
    save_dir.mkdir(parents=True, exist_ok=True)

    for quantity in uq_df["variable"].unique():
        fig, ax = plt.subplots(figsize=(5, 4))

        sub = uq_df[uq_df["variable"] == quantity]

        for method in sub["method"].unique():
            method_df = sub[sub["method"] == method]

            coverage = (
                method_df.groupby("nominal_coverage")["empirical_coverage"]
                .mean()
                .sort_index()
            )

            ax.plot(
                coverage.index,
                coverage.values,
                marker="o",
                linewidth=2,
                label=METHODS[method]["label"],
            )

        ax.plot(
            [0.5, 1.0],
            [0.5, 1.0],
            "k--",
            linewidth=1,
            label="ideal",
        )

        ax.set_xlabel("Nominal coverage")
        ax.set_ylabel("Empirical coverage")
        ax.set_title(f"Calibration curve for {quantity}")
        ax.set_xlim(0.5, 1.0)
        ax.set_ylim(0.5, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        fig.tight_layout()

        fig.savefig(
            save_dir / f"calibration_curve_{quantity}.png",
            dpi=300,
        )

        fig.savefig(
            save_dir / f"calibration_curve_{quantity}.pdf",
            dpi=300,
        )

        plt.close(fig)


def plot_uncertainty_maps(uq_stats, truths, select_time, method, save_dir):
    """Plot true field, predictive mean, absolute error, and predictive std.

    This layout directly compares pointwise error and predictive uncertainty.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    for quantity, stats in uq_stats.items():
        mean = stats["mean"].reshape(truths[quantity].shape)
        std = stats["std"].reshape(truths[quantity].shape)
        truth = truths[quantity]
        abs_error = np.abs(mean - truth)

        field_vmin = float(min(np.min(truth), np.min(mean)))
        field_vmax = float(max(np.max(truth), np.max(mean)))

        error_vmax = float(np.percentile(abs_error, 99))
        error_vmax = max(error_vmax, 1e-12)

        std_vmax = float(np.percentile(std, 99))
        std_vmax = max(std_vmax, 1e-12)

        fig, axes = plt.subplots(1, 4, figsize=(17, 4))

        im0 = axes[0].imshow(
            truth,
            cmap="jet",
            vmin=field_vmin,
            vmax=field_vmax,
            aspect="auto",
        )
        axes[0].set_title(f"Reference {quantity}")

        im1 = axes[1].imshow(
            mean,
            cmap="jet",
            vmin=field_vmin,
            vmax=field_vmax,
            aspect="auto",
        )
        axes[1].set_title(f"Predictive mean {quantity}")

        im2 = axes[2].imshow(
            abs_error,
            cmap="magma",
            vmin=0.0,
            vmax=error_vmax,
            aspect="auto",
        )
        axes[2].set_title(f"Absolute error; MAE={abs_error.mean():.3e}")

        im3 = axes[3].imshow(
            std,
            cmap="magma",
            vmin=0.0,
            vmax=std_vmax,
            aspect="auto",
        )
        axes[3].set_title("Predictive std")

        for ax in axes:
            ax.set_xlabel("X grid")
            ax.set_ylabel("Y grid")

        fig.colorbar(im0, ax=axes[0], fraction=0.046)
        fig.colorbar(im1, ax=axes[1], fraction=0.046)
        fig.colorbar(im2, ax=axes[2], fraction=0.046, label="Absolute error")
        fig.colorbar(im3, ax=axes[3], fraction=0.046, label="Std")

        fig.suptitle(
            f"{METHODS[method]['label']} uncertainty and error at t = {select_time:.2f}",
            fontsize=14,
        )

        fig.tight_layout()

        fig.savefig(
            save_dir / f"{method}_{quantity}_uncertainty_error_t_{select_time:.2f}.png",
            dpi=300,
            bbox_inches="tight",
        )

        fig.savefig(
            save_dir / f"{method}_{quantity}_uncertainty_error_t_{select_time:.2f}.pdf",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)


def main():
    args = parse_args()

    for method in args.methods:
        if method not in METHODS:
            raise KeyError(
                f"Unknown method: {method}. Available methods: {list(METHODS.keys())}"
            )

    device = get_device()

    EVAL_ROOT.mkdir(parents=True, exist_ok=True)

    data_stack = load_data_stack(args.data_path)

    metric_rows = []
    local_rows = []
    timing_rows = []
    uq_metric_rows = []

    method_labels = {
        method: METHODS[method]["label"]
        for method in args.methods
    }

    loaded_models = {}

    for method in args.methods:
        loaded_models[method] = load_trained_model(
            METHODS[method],
            LAYER_MAT_PSI,
            device,
        )

    for time_index in args.time_indices:
        (
            select_time,
            mesh_x,
            mesh_y,
            x_flat,
            y_flat,
            t_flat,
            truths,
        ) = prepare_snapshot(data_stack, time_index)

        masks = make_wake_region_masks(mesh_x, mesh_y)

        predictions = {}

        for method, model in loaded_models.items():
            preds, inference_time = predict_uvp_numpy(
                model,
                x_flat,
                y_flat,
                t_flat,
                device,
                batch_size=args.batch_size,
            )

            predictions[method] = {
                quantity: preds[quantity].reshape(mesh_x.shape)
                for quantity in ["u", "v", "p"]
            }

            timing_rows.append(
                {
                    "method": method,
                    "time_index": time_index,
                    "time_value": select_time,
                    "inference_time_seconds": inference_time,
                    "points": int(x_flat.shape[0]),
                    "parameters": count_parameters(model),
                }
            )

            for quantity in ["u", "v", "p"]:
                global_metrics = metric_dict(
                    predictions[method][quantity],
                    truths[quantity],
                )

                global_metrics.update(
                    {
                        "method": method,
                        "time_index": time_index,
                        "time_value": select_time,
                        "variable": quantity,
                    }
                )

                metric_rows.append(global_metrics)

                abs_error = np.abs(
                    predictions[method][quantity] - truths[quantity]
                )

                local_rows.extend(
                    local_error_rows(
                        method=method,
                        time_index=time_index,
                        select_time=select_time,
                        quantity=quantity,
                        abs_error=abs_error,
                        masks=masks,
                    )
                )

        for quantity in ["u", "v", "p"]:
            error_maps = {
                method: np.abs(predictions[method][quantity] - truths[quantity])
                for method in args.methods
            }

            plot_absolute_error_maps(
                error_maps=error_maps,
                method_labels=method_labels,
                select_time=select_time,
                quantity=quantity,
                save_dir=EVAL_ROOT / "absolute_error_maps",
            )

        if not args.skip_uq:
            for method, model in loaded_models.items():
                if METHODS[method].get("uncertainty") != "dropout":
                    continue

                uq_stats = mc_dropout_stats(
                    model=model,
                    x_np=x_flat,
                    y_np=y_flat,
                    t_np=t_flat,
                    device=device,
                    samples=args.mc_samples,
                    batch_size=args.batch_size,
                )

                plot_uncertainty_maps(
                    uq_stats=uq_stats,
                    truths=truths,
                    select_time=select_time,
                    method=method,
                    save_dir=EVAL_ROOT / "uncertainty_maps",
                )

                for quantity in ["u", "v", "p"]:
                    uq_metric_rows.extend(
                        uq_rows(
                            method=method,
                            time_index=time_index,
                            select_time=select_time,
                            quantity=quantity,
                            truth=truths[quantity],
                            mean=uq_stats[quantity]["mean"].reshape(mesh_x.shape),
                            std=uq_stats[quantity]["std"].reshape(mesh_x.shape),
                            samples=uq_stats[quantity]["samples"],
                            levels=DEFAULT_EVAL["interval_levels"],
                            mc_samples=args.mc_samples,
                            mc_inference_time_seconds=uq_stats[quantity]["time"],
                        )
                    )

    global_error_df = pd.DataFrame(metric_rows)
    local_error_df = pd.DataFrame(local_rows)
    timing_df = pd.DataFrame(timing_rows)

    global_error_path = EVAL_ROOT / "global_uvp_errors.csv"
    local_error_path = EVAL_ROOT / "local_region_errors.csv"
    timing_path = EVAL_ROOT / "timing_and_parameters.csv"

    global_error_df.to_csv(global_error_path, index=False)
    local_error_df.to_csv(local_error_path, index=False)
    timing_df.to_csv(timing_path, index=False)

    if uq_metric_rows:
        uq_df = pd.DataFrame(uq_metric_rows)

        uq_path = EVAL_ROOT / "uncertainty_calibration_metrics.csv"
        uq_df.to_csv(uq_path, index=False)

        plot_calibration_curve(
            uq_df,
            EVAL_ROOT / "calibration_curves",
        )

    print(f"Evaluation finished. Results saved to {EVAL_ROOT.resolve()}")
    print(f"Global errors saved to: {global_error_path}")
    print(f"Local region errors saved to: {local_error_path}")
    print(f"Timing and parameters saved to: {timing_path}")

    if uq_metric_rows:
        print(f"Uncertainty calibration metrics saved to: {uq_path}")


if __name__ == "__main__":
    main()
