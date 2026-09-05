
"""Independent evaluation on the exact fixed v1.2 held-out interior NASA hump set.

This script reads the split manifests written by hump_train.py. It never recreates
a random split. Boundary points are excluded from the velocity test set by design.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping

import numpy as np
import pandas as pd

from benchmark_tools import get_device
from hump_validation import (
    ACTIVE_HUMP_METHODS,
    HUMP_METHODS,
    load_hump_models,
    mc_dropout_stats,
    predict_on_points,
    uq_metric_rows,
)


METHOD_PREFIXES = {
    "standard_pinn": "standard",
    "weight_decay_pinn": "weight_decay",
    "adaptive_weight_pinn": "adaptive_weight",
    "bpinn_dropout": "bpinn",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate NASA hump models only on fixed held-out interior test data."
    )
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--models-root", default="./results_2pct_v12/models")
    parser.add_argument("--results-root", default="./results_2pct_v12")
    parser.add_argument("--protocol-root", default=None)
    parser.add_argument("--methods", nargs="+", default=ACTIVE_HUMP_METHODS)
    parser.add_argument("--batch-size", type=int, default=20000)
    parser.add_argument("--timing-repeats", type=int, default=10)
    parser.add_argument("--mc-samples", type=int, default=50)
    parser.add_argument(
        "--uq-levels",
        nargs="+",
        type=float,
        default=[0.50, 0.68, 0.80, 0.90, 0.95, 0.99],
    )
    parser.add_argument("--dropout-rate", type=float, default=0.002)
    parser.add_argument("--pressure-to-cp-scale", type=float, default=2.0)
    parser.add_argument("--skip-uq", action="store_true")
    parser.add_argument("--require-models", action="store_true")
    return parser.parse_args()


def metric_dict(pred: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
    pred = np.asarray(pred, dtype=float).reshape(-1)
    truth = np.asarray(truth, dtype=float).reshape(-1)
    mask = np.isfinite(pred) & np.isfinite(truth)
    pred = pred[mask]
    truth = truth[mask]

    if truth.size == 0:
        return {
            "relative_l2": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
            "max_abs_error": np.nan,
            "nrmse_range": np.nan,
            "points": 0,
        }

    err = pred - truth
    rmse = float(np.sqrt(np.mean(err ** 2)))
    span = float(np.max(truth) - np.min(truth))
    denom = float(np.linalg.norm(truth))

    return {
        "relative_l2": float(np.linalg.norm(err) / denom) if denom > 0 else np.nan,
        "mae": float(np.mean(np.abs(err))),
        "rmse": rmse,
        "max_abs_error": float(np.max(np.abs(err))),
        "nrmse_range": float(rmse / span) if span > 0 else np.nan,
        "points": int(truth.size),
    }


def method_label(method: str) -> str:
    return HUMP_METHODS.get(method, {}).get("label", method)


def method_prefix(method: str) -> str:
    return METHOD_PREFIXES.get(
        method,
        method.replace("_pinn", "").replace("-", "_").replace(" ", "_").lower(),
    )


def load_manifest(path: Path, required: Iterable[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Required protocol manifest not found: {path}. "
            "Run the revised hump_train.py --protocol-only first."
        )
    df = pd.read_csv(path)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{path.name} is missing columns: {missing}")
    return df


def add_deterministic_columns(
    pointwise: pd.DataFrame,
    method: str,
    preds: MutableMapping[str, np.ndarray],
) -> None:
    prefix = method_prefix(method)
    for q in ("u", "v"):
        values = np.asarray(preds[q], dtype=float).reshape(-1)
        truth = pointwise[f"{q}_ref"].to_numpy(dtype=float)
        if len(values) != len(pointwise):
            raise ValueError(f"{method}: wrong number of {q} predictions.")
        pointwise[f"{prefix}_{q}_pred"] = values
        pointwise[f"{prefix}_{q}_abs_error"] = np.abs(values - truth)


def add_mc_columns(
    pointwise: pd.DataFrame,
    stats: MutableMapping[str, MutableMapping[str, np.ndarray]],
    interval_levels: Iterable[float],
) -> None:
    for q in ("u", "v"):
        samples = np.asarray(stats[q]["samples"], dtype=float)
        samples = samples.reshape(samples.shape[0], -1)
        mean = samples.mean(axis=0)
        std = samples.std(axis=0, ddof=0)
        truth = pointwise[f"{q}_ref"].to_numpy(dtype=float)

        pointwise[f"bpinn_{q}_mean"] = mean
        pointwise[f"bpinn_{q}_std"] = std
        pointwise[f"bpinn_{q}_abs_error"] = np.abs(mean - truth)

        for level in interval_levels:
            alpha = (1.0 - float(level)) / 2.0
            suffix = str(int(round(100 * level)))
            pointwise[f"bpinn_{q}_lower_{suffix}"] = np.quantile(
                samples, alpha, axis=0
            )
            pointwise[f"bpinn_{q}_upper_{suffix}"] = np.quantile(
                samples, 1.0 - alpha, axis=0
            )


def evaluate_velocity_test(models, test_df, *, device, args, output_dir: Path):
    x = test_df["x"].to_numpy(dtype=float)
    y = test_df["y"].to_numpy(dtype=float)

    pointwise = pd.DataFrame({
        "original_index": test_df["original_index"].to_numpy(dtype=int),
        "i_index": test_df["i_index"].to_numpy(dtype=int),
        "j_index": test_df["j_index"].to_numpy(dtype=int),
        "x": x,
        "y": y,
        "split": "test",
        "u_ref": test_df["u"].to_numpy(dtype=float),
        "v_ref": test_df["v"].to_numpy(dtype=float),
    })

    error_rows: List[Dict[str, object]] = []
    timing_rows: List[Dict[str, object]] = []

    for method, model in models.items():
        predict_on_points(model, x, y, device, args.batch_size, train_mode=False)

        times = []
        preds = None
        for _ in range(args.timing_repeats):
            preds, elapsed = predict_on_points(
                model, x, y, device, args.batch_size, train_mode=False
            )
            times.append(float(elapsed))

        assert preds is not None
        add_deterministic_columns(pointwise, method, preds)

        for q in ("u", "v"):
            row = metric_dict(preds[q], test_df[q].to_numpy(dtype=float))
            row.update({
                "method": method,
                "label": method_label(method),
                "variable": q,
                "evaluation_set": "fixed_strict_interior_heldout_test",
            })
            error_rows.append(row)

        timing_rows.append({
            "method": method,
            "label": method_label(method),
            "evaluation_set": "fixed_strict_interior_heldout_test",
            "evaluation_point_count": int(len(test_df)),
            "timing_repeats": int(args.timing_repeats),
            "evaluation_seconds_mean": float(np.mean(times)),
            "evaluation_seconds_std": float(np.std(times, ddof=0)),
            "evaluation_seconds_median": float(np.median(times)),
            "evaluation_seconds_min": float(np.min(times)),
            "evaluation_seconds_max": float(np.max(times)),
        })

    pd.DataFrame(error_rows).to_csv(
        output_dir / "heldout_global_uv_errors.csv", index=False
    )
    pd.DataFrame(timing_rows).to_csv(
        output_dir / "heldout_inference_timing.csv", index=False
    )

    if not args.skip_uq and "bpinn_dropout" in models:
        stats = mc_dropout_stats(
            model=models["bpinn_dropout"],
            x=x,
            y=y,
            device=device,
            samples=args.mc_samples,
            batch_size=args.batch_size,
        )
        uq_rows: List[Dict[str, object]] = []
        for q in ("u", "v"):
            rows = uq_metric_rows(
                method="bpinn_dropout",
                quantity=q,
                truth=test_df[q].to_numpy(dtype=float),
                samples=stats[q]["samples"],
                levels=args.uq_levels,
                mc_samples=args.mc_samples,
                mc_inference_time_seconds=stats[q]["time"],
                region_name="fixed_strict_interior_heldout_test",
            )
            for row in rows:
                row["evaluation_set"] = "fixed_strict_interior_heldout_test"
                row["dropout_rate"] = float(args.dropout_rate)
            uq_rows.extend(rows)

        pd.DataFrame(uq_rows).to_csv(
            output_dir / "heldout_uncertainty_calibration_metrics.csv", index=False
        )
        add_mc_columns(pointwise, stats, args.uq_levels)

    pointwise.to_csv(
        output_dir / "heldout_pointwise_predictions.csv", index=False
    )


def evaluate_cp_test(models, cp_test, *, device, args, output_dir: Path):
    if cp_test.empty:
        return

    x = cp_test["x"].to_numpy(dtype=float)
    y = cp_test["y"].to_numpy(dtype=float)
    truth = cp_test["cp"].to_numpy(dtype=float)
    rows = []
    pointwise = cp_test.copy()

    for method, model in models.items():
        preds, _ = predict_on_points(
            model, x, y, device, args.batch_size, train_mode=False
        )
        raw = float(args.pressure_to_cp_scale) * np.asarray(preds["p"], dtype=float)
        offset = float(np.nanmean(truth - raw))
        aligned = raw + offset

        pointwise[f"{method}_cp_pred_raw"] = raw
        pointwise[f"{method}_cp_pred_gauge_aligned"] = aligned

        for treatment, values, added in [
            ("raw", raw, 0.0),
            ("gauge_aligned", aligned, offset),
        ]:
            row = metric_dict(values, truth)
            row.update({
                "method": method,
                "label": method_label(method),
                "variable": "Cp_wall",
                "evaluation_set": "fixed_heldout_cp_test",
                "gauge_treatment": treatment,
                "pressure_to_cp_scale": float(args.pressure_to_cp_scale),
                "constant_offset_added": float(added),
            })
            rows.append(row)

    pd.DataFrame(rows).to_csv(
        output_dir / "heldout_cp_errors_raw_and_gauge_aligned.csv", index=False
    )

    if not args.skip_uq and "bpinn_dropout" in models:
        stats = mc_dropout_stats(
            model=models["bpinn_dropout"],
            x=x,
            y=y,
            device=device,
            samples=args.mc_samples,
            batch_size=args.batch_size,
        )
        raw_samples = (
            float(args.pressure_to_cp_scale)
            * np.asarray(stats["p"]["samples"], dtype=float)
        )
        mean_raw = raw_samples.mean(axis=0)
        offset = float(np.nanmean(truth - mean_raw))
        aligned_samples = raw_samples + offset

        uq_rows = []
        for treatment, samples in [
            ("raw", raw_samples),
            ("gauge_aligned", aligned_samples),
        ]:
            part = uq_metric_rows(
                method="bpinn_dropout",
                quantity="Cp_wall",
                truth=truth,
                samples=samples,
                levels=args.uq_levels,
                mc_samples=args.mc_samples,
                mc_inference_time_seconds=stats["p"]["time"],
                region_name="fixed_heldout_cp_test",
            )
            for row in part:
                row["gauge_treatment"] = treatment
                row["constant_offset_added"] = (
                    0.0 if treatment == "raw" else offset
                )
            uq_rows.extend(part)

        pd.DataFrame(uq_rows).to_csv(
            output_dir / "heldout_cp_uncertainty_metrics.csv", index=False
        )

    pointwise.to_csv(
        output_dir / "heldout_cp_pointwise_predictions.csv", index=False
    )


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    protocol_root = (
        Path(args.protocol_root)
        if args.protocol_root is not None
        else results_root / "protocol"
    )
    output_dir = results_root / "heldout_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    velocity_manifest = load_manifest(
        protocol_root / "interior_velocity_split_manifest.csv",
        required=["original_index", "i_index", "j_index", "x", "y", "u", "v", "split"],
    )
    test_df = velocity_manifest[velocity_manifest["split"] == "test"].copy()
    if test_df.empty:
        raise RuntimeError("Interior velocity manifest contains no test points.")

    cp_manifest = load_manifest(
        protocol_root / "cp_split_manifest.csv",
        required=["x", "y", "cp", "split"],
    )
    cp_test = cp_manifest[cp_manifest["split"] == "test"].copy()

    if "bpinn_dropout" in HUMP_METHODS:
        HUMP_METHODS["bpinn_dropout"]["dropout_rate"] = float(args.dropout_rate)

    device = get_device()
    models = load_hump_models(
        methods=args.methods,
        models_root=Path(args.models_root),
        device=device,
        require_models=args.require_models,
    )
    if not models:
        raise RuntimeError(f"No model checkpoints loaded from {args.models_root}")

    evaluate_velocity_test(
        models, test_df, device=device, args=args, output_dir=output_dir
    )
    evaluate_cp_test(
        models, cp_test, device=device, args=args, output_dir=output_dir
    )

    test_df.to_csv(output_dir / "heldout_velocity_test_points.csv", index=False)
    cp_test.to_csv(output_dir / "heldout_cp_test_points.csv", index=False)

    print(f"Held-out evaluation finished: {output_dir.resolve()}")
    print(f"Strict-interior velocity test points: {len(test_df)}")
    print(f"Cp test points: {len(cp_test)}")


if __name__ == "__main__":
    main()
