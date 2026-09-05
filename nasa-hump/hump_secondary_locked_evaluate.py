#!/usr/bin/env python3
"""
Outcome-independent secondary locked evaluation for the revised NASA hump benchmark.

This script evaluates the already-frozen final NASA hump checkpoints on a secondary
velocity subset whose membership was fixed before running this evaluator.

Protocol:
- Source: rows labelled "train_pool_unused" in interior_velocity_split_manifest.csv.
- Exclusion: any structured i-column that is the same as, or immediately adjacent to,
  a column used by the prior train, validation, or original test partitions.
- Rule: retain all train_pool_unused rows with
      min |i_candidate - i_prior| >= 2.
- For the supplied v1.2 manifest this yields 336 points on 16 structured columns:
      [41, 51, 58, 63, 70, 71, 72, 73, 103, 107, 114, 115, 122, 123, 156, 157].

Important:
This is a locked secondary robustness evaluation. It should not be described as a
"never-viewed pristine test set" unless the authors can independently document that
no earlier full-grid diagnostic exposed these points. The current full-grid
post-processing pipeline is capable of evaluating all LES points.

The script reports:
1. deterministic/dropout-off errors for all four final models;
2. MC50 predictive-mean errors for the MC-dropout B-PINN separately;
3. MC-dropout calibration metrics on the same locked points;
4. Pearson/Spearman error-uncertainty association and empirical-ensemble CRPS;
5. repeated inference timing;
6. exact pointwise predictions and compressed MC samples.
"""

from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import torch

from benchmark_tools import get_device
from hump_validation import (
    ACTIVE_HUMP_METHODS,
    HUMP_METHODS,
    load_hump_models,
    mc_dropout_stats,
    predict_on_points,
    uq_metric_rows,
)

EXPECTED_POINT_COUNT = 336
EXPECTED_COLUMNS = [41, 51, 58, 63, 70, 71, 72, 73, 103, 107, 114, 115, 122, 123, 156, 157]
EXPECTED_INDEX_SHA256 = "19dc4889f6ebc926b52191b5bf8505b47f034af59528ffd18f20ddf53d957c62"
PRIOR_SPLITS = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate frozen NASA hump models on the locked secondary velocity subset."
    )
    p.add_argument(
        "--source-manifest",
        default="./hump_results_revision_r02/protocol/interior_velocity_split_manifest.csv",
    )
    p.add_argument(
        "--frozen-manifest",
        default="./secondary_locked_protocol/secondary_locked_manifest.csv",
    )
    p.add_argument(
        "--models-root",
        default="./hump_results_revision_r02/models",
    )
    p.add_argument(
        "--output-dir",
        default="./hump_results_revision_r02/secondary_locked_evaluation",
    )
    p.add_argument("--batch-size", type=int, default=20000)
    p.add_argument("--timing-repeats", type=int, default=10)
    p.add_argument("--mc-samples", type=int, default=50)
    p.add_argument("--dropout-rate", type=float, default=0.002)
    p.add_argument("--seed", type=int, default=424242)
    p.add_argument(
        "--uq-levels",
        nargs="+",
        type=float,
        default=[0.50, 0.68, 0.80, 0.90, 0.95, 0.99],
    )
    return p.parse_args()


def set_eval_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    rmse = float(np.sqrt(np.mean(err**2)))
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


def index_hash(indices: Iterable[int]) -> str:
    payload = ",".join(str(int(x)) for x in indices).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_locked_protocol(source_path: Path, frozen_path: Path) -> pd.DataFrame:
    required = ["original_index", "i_index", "j_index", "x", "y", "u", "v", "split"]
    source = pd.read_csv(source_path)
    frozen = pd.read_csv(frozen_path)

    missing_source = [c for c in required if c not in source.columns]
    missing_frozen = [c for c in required if c not in frozen.columns]
    if missing_source:
        raise KeyError(f"Source manifest missing columns: {missing_source}")
    if missing_frozen:
        raise KeyError(f"Frozen manifest missing columns: {missing_frozen}")

    if len(frozen) != EXPECTED_POINT_COUNT:
        raise RuntimeError(
            f"Frozen secondary set has {len(frozen)} points; expected {EXPECTED_POINT_COUNT}."
        )

    frozen = frozen.sort_values(["i_index", "j_index", "original_index"]).reset_index(drop=True)
    frozen_columns = sorted(frozen["i_index"].astype(int).unique().tolist())
    if frozen_columns != EXPECTED_COLUMNS:
        raise RuntimeError(
            f"Frozen i-columns differ from the locked protocol.\n"
            f"Expected: {EXPECTED_COLUMNS}\nFound: {frozen_columns}"
        )

    digest = index_hash(frozen["original_index"].astype(int).tolist())
    if digest != EXPECTED_INDEX_SHA256:
        raise RuntimeError(
            "Frozen original-index membership hash mismatch. "
            f"Expected {EXPECTED_INDEX_SHA256}, found {digest}."
        )

    if not (frozen["split"].astype(str) == "train_pool_unused").all():
        raise RuntimeError("Every frozen row must originate from split='train_pool_unused'.")

    s = source.set_index("original_index", drop=False)
    for row in frozen.itertuples(index=False):
        idx = int(row.original_index)
        if idx not in s.index:
            raise RuntimeError(f"Frozen original_index={idx} is missing from source manifest.")
        ref = s.loc[idx]
        if isinstance(ref, pd.DataFrame):
            raise RuntimeError(f"Duplicate original_index={idx} in source manifest.")
        if str(ref["split"]) != "train_pool_unused":
            raise RuntimeError(
                f"original_index={idx} is {ref['split']!r} in source manifest, "
                "not train_pool_unused."
            )
        for col in ["i_index", "j_index"]:
            if int(ref[col]) != int(getattr(row, col)):
                raise RuntimeError(f"Mismatch for original_index={idx}, column={col}.")
        for col in ["x", "y", "u", "v"]:
            if not np.isclose(float(ref[col]), float(getattr(row, col)), rtol=0.0, atol=1e-12):
                raise RuntimeError(f"Mismatch for original_index={idx}, column={col}.")

    prior_cols = np.sort(
        source.loc[source["split"].isin(PRIOR_SPLITS), "i_index"].astype(int).unique()
    )
    if prior_cols.size == 0:
        raise RuntimeError("No prior train/validation/test columns found in source manifest.")

    min_dist = np.array(
        [int(np.min(np.abs(prior_cols - int(i)))) for i in frozen["i_index"].astype(int)],
        dtype=int,
    )
    if np.any(min_dist < 2):
        bad = frozen.loc[min_dist < 2, ["original_index", "i_index"]]
        raise RuntimeError(
            "Frozen set violates min structured-column distance >= 2:\n"
            + bad.head(20).to_string(index=False)
        )

    unused = source[source["split"] == "train_pool_unused"].copy()
    unused["min_i_distance_to_prior_used_column"] = unused["i_index"].astype(int).apply(
        lambda i: int(np.min(np.abs(prior_cols - int(i))))
    )
    regenerated = unused[
        unused["min_i_distance_to_prior_used_column"] >= 2
    ].sort_values(["i_index", "j_index", "original_index"]).reset_index(drop=True)

    if len(regenerated) != EXPECTED_POINT_COUNT:
        raise RuntimeError(
            f"Outcome-independent rule now yields {len(regenerated)} points; "
            f"expected {EXPECTED_POINT_COUNT}. Source protocol may have changed."
        )
    if not np.array_equal(
        regenerated["original_index"].astype(int).to_numpy(),
        frozen["original_index"].astype(int).to_numpy(),
    ):
        raise RuntimeError(
            "Frozen membership is not identical to the complete set generated by the locked rule."
        )

    return frozen


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if len(a) < 2:
        return float("nan")
    ar = pd.Series(a).rank(method="average").to_numpy()
    br = pd.Series(b).rank(method="average").to_numpy()
    if np.std(ar) == 0 or np.std(br) == 0:
        return float("nan")
    return float(np.corrcoef(ar, br)[0, 1])


def empirical_ensemble_crps(samples: np.ndarray, truth: np.ndarray) -> float:
    samples = np.asarray(samples, dtype=float)
    truth = np.asarray(truth, dtype=float).reshape(-1)
    samples = samples.reshape(samples.shape[0], -1)
    if samples.shape[1] != truth.size:
        raise ValueError("CRPS samples/reference size mismatch.")
    first = np.mean(np.abs(samples - truth[None, :]), axis=0)
    pair = np.mean(
        np.abs(samples[:, None, :] - samples[None, :, :]),
        axis=(0, 1),
    )
    return float(np.mean(first - 0.5 * pair))


def main() -> None:
    args = parse_args()
    source_path = Path(args.source_manifest)
    frozen_path = Path(args.frozen_manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frozen = verify_locked_protocol(source_path, frozen_path)
    frozen.to_csv(output_dir / "secondary_locked_points_verified.csv", index=False)

    x = frozen["x"].to_numpy(dtype=float)
    y = frozen["y"].to_numpy(dtype=float)
    u_ref = frozen["u"].to_numpy(dtype=float)
    v_ref = frozen["v"].to_numpy(dtype=float)

    pointwise = frozen[
        ["original_index", "i_index", "j_index", "x", "y", "u", "v"]
    ].rename(columns={"u": "u_ref", "v": "v_ref"}).copy()
    pointwise["evaluation_set"] = "secondary_locked_outcome_independent_subset"

    set_eval_seed(args.seed)
    HUMP_METHODS["bpinn_dropout"]["dropout_rate"] = float(args.dropout_rate)

    device = get_device()
    models = load_hump_models(
        methods=ACTIVE_HUMP_METHODS,
        models_root=Path(args.models_root),
        device=device,
        require_models=True,
    )
    if set(models) != set(ACTIVE_HUMP_METHODS):
        raise RuntimeError(
            f"Expected all final models {ACTIVE_HUMP_METHODS}, loaded {list(models)}."
        )

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

        prefix = method.replace("_pinn", "").replace("-", "_")
        for q, truth in [("u", u_ref), ("v", v_ref)]:
            values = np.asarray(preds[q], dtype=float).reshape(-1)
            pointwise[f"{prefix}_{q}_dropout_off_pred"] = values
            pointwise[f"{prefix}_{q}_dropout_off_abs_error"] = np.abs(values - truth)
            row = metric_dict(values, truth)
            row.update(
                {
                    "method": method,
                    "label": HUMP_METHODS[method]["label"],
                    "variable": q,
                    "prediction_mode": (
                        "dropout_off" if method == "bpinn_dropout" else "deterministic"
                    ),
                    "evaluation_set": "secondary_locked_outcome_independent_subset",
                }
            )
            error_rows.append(row)

        timing_rows.append(
            {
                "method": method,
                "label": HUMP_METHODS[method]["label"],
                "prediction_mode": (
                    "dropout_off" if method == "bpinn_dropout" else "deterministic"
                ),
                "evaluation_set": "secondary_locked_outcome_independent_subset",
                "evaluation_point_count": int(len(frozen)),
                "timing_repeats": int(args.timing_repeats),
                "evaluation_seconds_mean": float(np.mean(times)),
                "evaluation_seconds_std": float(np.std(times, ddof=0)),
                "evaluation_seconds_median": float(np.median(times)),
                "evaluation_seconds_min": float(np.min(times)),
                "evaluation_seconds_max": float(np.max(times)),
            }
        )

    set_eval_seed(args.seed)
    bpinn = models["bpinn_dropout"]
    stats = mc_dropout_stats(
        model=bpinn,
        x=x,
        y=y,
        device=device,
        samples=args.mc_samples,
        batch_size=args.batch_size,
    )

    uq_rows: List[Dict[str, object]] = []
    uq_summary_rows: List[Dict[str, object]] = []

    for q, truth in [("u", u_ref), ("v", v_ref)]:
        samples = np.asarray(stats[q]["samples"], dtype=float).reshape(args.mc_samples, -1)
        mean = samples.mean(axis=0)
        std = samples.std(axis=0, ddof=0)
        abs_error = np.abs(mean - truth)

        pointwise[f"bpinn_{q}_mc{args.mc_samples}_mean"] = mean
        pointwise[f"bpinn_{q}_mc{args.mc_samples}_std"] = std
        pointwise[f"bpinn_{q}_mc{args.mc_samples}_abs_error"] = abs_error

        for level in args.uq_levels:
            suffix = str(int(round(100 * float(level))))
            alpha = (1.0 - float(level)) / 2.0
            pointwise[f"bpinn_{q}_lower_{suffix}"] = np.quantile(samples, alpha, axis=0)
            pointwise[f"bpinn_{q}_upper_{suffix}"] = np.quantile(
                samples, 1.0 - alpha, axis=0
            )

        mean_row = metric_dict(mean, truth)
        mean_row.update(
            {
                "method": "bpinn_dropout",
                "label": HUMP_METHODS["bpinn_dropout"]["label"],
                "variable": q,
                "prediction_mode": f"mc{args.mc_samples}_predictive_mean",
                "evaluation_set": "secondary_locked_outcome_independent_subset",
            }
        )
        error_rows.append(mean_row)

        rows = uq_metric_rows(
            method="bpinn_dropout",
            quantity=q,
            truth=truth,
            samples=samples,
            levels=args.uq_levels,
            mc_samples=args.mc_samples,
            mc_inference_time_seconds=float(stats[q]["time"]),
            region_name="secondary_locked_outcome_independent_subset",
        )
        for row in rows:
            row["evaluation_set"] = "secondary_locked_outcome_independent_subset"
            row["dropout_rate"] = float(args.dropout_rate)
            row["evaluation_seed"] = int(args.seed)
        uq_rows.extend(rows)

        pearson = (
            float(np.corrcoef(abs_error, std)[0, 1])
            if np.std(abs_error) > 0 and np.std(std) > 0
            else float("nan")
        )
        spearman = spearman_corr(abs_error, std)
        crps = empirical_ensemble_crps(samples, truth)

        qdf = pd.DataFrame(rows)
        mean_cal_error = float(qdf["calibration_error"].mean()) if not qdf.empty else np.nan
        row95 = qdf[np.isclose(qdf["nominal_coverage"], 0.95)]
        picp95 = float(row95["empirical_coverage"].iloc[0]) if len(row95) else np.nan
        mpiw95 = float(row95["mean_prediction_interval_width"].iloc[0]) if len(row95) else np.nan

        uq_summary_rows.append(
            {
                "variable": q,
                "evaluation_set": "secondary_locked_outcome_independent_subset",
                "points": int(len(truth)),
                "mc_samples": int(args.mc_samples),
                "dropout_rate": float(args.dropout_rate),
                "evaluation_seed": int(args.seed),
                "picp95": picp95,
                "mpiw95": mpiw95,
                "mean_calibration_error": mean_cal_error,
                "pearson_abs_error_vs_std": pearson,
                "spearman_abs_error_vs_std": spearman,
                "empirical_ensemble_crps": crps,
            }
        )

    pd.DataFrame(error_rows).to_csv(
        output_dir / "secondary_locked_global_uv_errors.csv", index=False
    )
    pd.DataFrame(timing_rows).to_csv(
        output_dir / "secondary_locked_inference_timing.csv", index=False
    )
    pd.DataFrame(uq_rows).to_csv(
        output_dir / "secondary_locked_uncertainty_calibration_metrics.csv", index=False
    )
    pd.DataFrame(uq_summary_rows).to_csv(
        output_dir / "secondary_locked_uncertainty_summary.csv", index=False
    )
    pointwise.to_csv(
        output_dir / "secondary_locked_pointwise_predictions.csv", index=False
    )

    np.savez_compressed(
        output_dir / "secondary_locked_mc_samples.npz",
        original_index=frozen["original_index"].to_numpy(dtype=int),
        u_samples=np.asarray(stats["u"]["samples"], dtype=np.float32),
        v_samples=np.asarray(stats["v"]["samples"], dtype=np.float32),
        p_samples=np.asarray(stats["p"]["samples"], dtype=np.float32),
        evaluation_seed=np.array([args.seed], dtype=np.int64),
        dropout_rate=np.array([args.dropout_rate], dtype=np.float64),
        mc_samples=np.array([args.mc_samples], dtype=np.int64),
    )

    pd.DataFrame([{
        "evaluation_set": "secondary_locked_outcome_independent_subset",
        "selection_source_split": "train_pool_unused",
        "excluded_prior_splits_for_column_distance": ",".join(PRIOR_SPLITS),
        "minimum_i_column_distance": 2,
        "points": int(len(frozen)),
        "structured_i_columns": ",".join(map(str, EXPECTED_COLUMNS)),
        "original_index_sha256": EXPECTED_INDEX_SHA256,
        "models_root": str(Path(args.models_root)),
        "mc_samples": int(args.mc_samples),
        "dropout_rate": float(args.dropout_rate),
        "evaluation_seed": int(args.seed),
    }]).to_csv(output_dir / "secondary_locked_run_summary.csv", index=False)

    print("=" * 88)
    print("Secondary locked evaluation finished.")
    print(f"Verified points: {len(frozen)}")
    print(f"Structured i-columns: {EXPECTED_COLUMNS}")
    print(f"Membership SHA256: {EXPECTED_INDEX_SHA256}")
    print(f"Output directory: {output_dir.resolve()}")
    print("=" * 88)


if __name__ == "__main__":
    main()
