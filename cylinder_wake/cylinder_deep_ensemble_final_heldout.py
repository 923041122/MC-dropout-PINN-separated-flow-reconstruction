#!/usr/bin/env python3
"""
Final held-out evaluation for the frozen cylinder 5-member Deep Ensemble.

Protocol:
- Uses the same deterministic ensemble architecture/checkpoints as
  cylinder_deep_ensemble_validation.py.
- Loads validation-fitted interval scale factors BEFORE accessing held-out labels.
- Never fits or changes any hyperparameter/calibration factor on the held-out set.
- Evaluates the held-out partition exactly once for final reporting.

Deep-Ensemble interval model:
    mean +/- z(level) * scale * sample_std

where scale is frozen from:
    deep_ensemble_validation_calibration_factors.csv

Outputs
-------
<output-dir>/
    deep_ensemble_final_heldout_member_metrics.csv
    deep_ensemble_final_heldout_ensemble_metrics.csv
    deep_ensemble_final_heldout_uq_raw.csv
    deep_ensemble_final_heldout_uq_calibrated.csv
    deep_ensemble_final_heldout_pressure_gauge_summary.csv
    deep_ensemble_final_heldout_pressure_offsets_by_time.csv
    deep_ensemble_final_heldout_timing.csv
    deep_ensemble_final_heldout_cost.csv
    deep_ensemble_final_heldout_summary.csv
    deep_ensemble_final_heldout_pointwise.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch

from benchmark_tools import count_parameters, get_device
from cylinder_train_v2 import load_protocol, load_reference_stack
from cylinder_deep_ensemble_validation import (
    VARIABLES,
    build_member,
    ensemble_crps_pointwise,
    interval_metrics,
    metric_dict,
    pearson_corr,
    safe_load_state,
    spearman_corr,
    timed_prediction,
    z_for_level,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Final held-out evaluation of the frozen cylinder Deep Ensemble."
    )
    p.add_argument(
        "--data-path",
        default="./2d_cylinder_Re3900_100x100_kw_sst.mat",
    )
    p.add_argument(
        "--protocol-root",
        default="./results_cylinder_protocol_v1_1/protocol",
    )
    p.add_argument(
        "--calibration-factors",
        default=(
            "./cylinder_deep_ensemble_validation/"
            "deep_ensemble_validation_calibration_factors.csv"
        ),
    )
    p.add_argument(
        "--output-dir",
        default="./cylinder_deep_ensemble_final_heldout",
    )
    p.add_argument(
        "--member-seeds",
        nargs="+",
        type=int,
        default=[2025, 2026, 2027, 2028, 2029],
    )
    p.add_argument(
        "--member-root-template",
        default="./deep_ensemble_seed_{seed}",
        help="Must contain {seed}; checkpoint is read from models/<checkpoint-name>.",
    )
    p.add_argument(
        "--checkpoint-name",
        default="bpinn_dropout.pth",
    )
    p.add_argument("--supervised-ratio", type=float, default=0.02)
    p.add_argument("--n-equation-points", type=int, default=100000)
    p.add_argument("--batch-size", type=int, default=5000)
    p.add_argument("--warmup-points", type=int, default=1024)
    p.add_argument(
        "--uq-levels",
        nargs="+",
        type=float,
        default=[0.50, 0.80, 0.90, 0.95],
    )
    p.add_argument(
        "--save-pointwise",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return p.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if len(args.member_seeds) < 2:
        raise ValueError("Deep Ensemble requires at least two members.")
    if "{seed}" not in args.member_root_template:
        raise ValueError("--member-root-template must contain '{seed}'.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")
    if args.warmup_points < 0:
        raise ValueError("--warmup-points must be >= 0.")
    if not args.uq_levels:
        raise ValueError("--uq-levels cannot be empty.")
    bad = [x for x in args.uq_levels if not (0.0 < x < 1.0)]
    if bad:
        raise ValueError(f"All --uq-levels must be in (0,1), got {bad}.")


def load_frozen_factors(path: Path) -> Dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing validation calibration file: {path}\n"
            "Run cylinder_deep_ensemble_validation.py first and freeze its factors."
        )

    df = pd.read_csv(path)
    required = {"variable", "scale_factor"}
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(
            f"Calibration CSV is missing columns {sorted(missing)}: {path}"
        )

    if "split_used_for_fit" in df.columns:
        bad = df[
            df["split_used_for_fit"].astype(str).str.lower() != "validation"
        ]
        if not bad.empty:
            raise RuntimeError(
                "Calibration factors are not all marked as validation-fitted."
            )

    factors = {
        str(row["variable"]): float(row["scale_factor"])
        for _, row in df.iterrows()
    }

    if set(factors) != set(VARIABLES):
        raise RuntimeError(
            f"Calibration variables are {sorted(factors)}, "
            f"expected {sorted(VARIABLES)}."
        )

    if not all(np.isfinite(v) and v > 0.0 for v in factors.values()):
        raise RuntimeError(f"Invalid calibration factors: {factors}")

    return factors


def pressure_align_per_time(
    p_mean: np.ndarray,
    p_ref: np.ndarray,
    time_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, List[Dict[str, object]]]:
    """
    Gauge-invariant evaluation diagnostic.

    One additive offset is fitted from held-out reference pressure independently
    at each time snapshot and applied to the ensemble mean and, later, to all
    pressure interval bounds/samples. This is evaluation-only and does not alter
    interval widths or any model/calibration parameter.
    """
    aligned = np.asarray(p_mean, dtype=float).copy()
    offsets = np.zeros_like(aligned, dtype=float)
    rows: List[Dict[str, object]] = []

    for ti in np.unique(time_index):
        mask = time_index == ti
        offset = float(np.mean(p_ref[mask] - p_mean[mask]))
        aligned[mask] += offset
        offsets[mask] = offset
        rows.append(
            {
                "time_index": int(ti),
                "test_points_at_time": int(np.sum(mask)),
                "ensemble_mean_pressure_offset": offset,
            }
        )

    return aligned, offsets, rows


def mean_absolute_calibration_error(rows: Sequence[Dict[str, object]]) -> float:
    vals = [
        float(r["absolute_calibration_error"])
        for r in rows
        if np.isfinite(float(r["absolute_calibration_error"]))
    ]
    return float(np.mean(vals)) if vals else np.nan


def main() -> None:
    args = parse_args()
    validate_args(args)

    np.random.seed(2025)
    torch.manual_seed(2025)

    # CRITICAL: validation-fitted factors are loaded and validated before
    # held-out labels are indexed.
    factor_path = Path(args.calibration_factors)
    factors = load_frozen_factors(factor_path)

    device = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 104)
    print("FINAL cylinder Deep Ensemble held-out evaluation")
    print(f"Device: {device}")
    print(f"Members: {args.member_seeds}")
    print(f"Frozen validation calibration factors: {factors}")
    print(f"Calibration file: {factor_path}")
    print("Hyperparameter/calibration tuning after test access: NONE")
    print("=" * 104)

    train_idx, val_idx, test_idx, eq_np, protocol_summary = load_protocol(
        protocol_root=Path(args.protocol_root),
        ratio=args.supervised_ratio,
        n_equation_points=args.n_equation_points,
    )
    del train_idx, val_idx, eq_np

    expected_test = int(protocol_summary["heldout_test_points"])
    if len(test_idx) != expected_test:
        raise RuntimeError(
            f"Protocol says {expected_test} held-out points, "
            f"but test_indices contains {len(test_idx)}."
        )

    if "flat_index_order" in protocol_summary.index:
        order = str(protocol_summary["flat_index_order"])
        if "time_major" not in order:
            raise RuntimeError(
                f"Unexpected protocol flattening convention: {order}"
            )

    full_data = load_reference_stack(Path(args.data_path))
    expected_total = int(protocol_summary["total_reference_points"])
    if len(full_data) != expected_total:
        raise RuntimeError(
            f"Reference-data size mismatch: got {len(full_data)}, "
            f"protocol expects {expected_total}."
        )

    # FIRST held-out label access occurs here, after factors are frozen.
    test_data = full_data[torch.from_numpy(test_idx)]
    del full_data

    test_np = test_data.detach().cpu().numpy()
    if test_np.ndim != 2 or test_np.shape[1] < 6:
        raise RuntimeError(
            f"Expected held-out data with at least 6 columns, got {test_np.shape}."
        )

    xyz = np.asarray(test_np[:, 0:3], dtype=np.float32)
    truth = {
        "u": np.asarray(test_np[:, 3], dtype=np.float64),
        "v": np.asarray(test_np[:, 4], dtype=np.float64),
        "p": np.asarray(test_np[:, 5], dtype=np.float64),
    }
    n_test = int(len(test_idx))

    nx = int(protocol_summary["nx"])
    ny = int(protocol_summary["ny"])
    n_spatial = nx * ny
    time_index = np.asarray(test_idx // n_spatial, dtype=np.int64)
    spatial_index = np.asarray(test_idx % n_spatial, dtype=np.int64)

    print(f"Held-out test points: {n_test}")
    print(f"Unique held-out time snapshots: {len(np.unique(time_index))}")
    print()

    member_predictions: Dict[str, List[np.ndarray]] = {
        q: [] for q in VARIABLES
    }
    member_rows: List[Dict[str, object]] = []
    timing_rows: List[Dict[str, object]] = []
    backbone_params_expected = None

    pointwise: Dict[str, np.ndarray] = {
        "global_index": np.asarray(test_idx, dtype=np.int64),
        "spatial_index": spatial_index,
        "time_index": time_index,
        "x": xyz[:, 0],
        "y": xyz[:, 1],
        "t": xyz[:, 2],
        "u_ref": truth["u"],
        "v_ref": truth["v"],
        "p_ref": truth["p"],
    }

    for member_index, seed in enumerate(args.member_seeds, start=1):
        member_root = Path(args.member_root_template.format(seed=seed))
        checkpoint = member_root / "models" / args.checkpoint_name
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Missing Deep Ensemble checkpoint for seed {seed}: {checkpoint}"
            )

        print(
            f"[member {member_index}/{len(args.member_seeds)}] "
            f"seed={seed} checkpoint={checkpoint}"
        )

        model = build_member(device)
        state = safe_load_state(checkpoint, device)
        model.load_state_dict(state, strict=True)
        model.eval()

        params = int(count_parameters(model))
        if backbone_params_expected is None:
            backbone_params_expected = params
        elif params != backbone_params_expected:
            raise RuntimeError(
                "Deep Ensemble members do not have identical parameter counts: "
                f"expected {backbone_params_expected}, got {params} for seed {seed}."
            )

        preds, elapsed = timed_prediction(
            model=model,
            xyz=xyz,
            device=device,
            batch_size=args.batch_size,
            warmup_points=args.warmup_points,
        )

        timing_rows.append(
            {
                "member_index": member_index,
                "seed": seed,
                "checkpoint": str(checkpoint),
                "heldout_points": n_test,
                "batch_size": args.batch_size,
                "warmup_points": min(args.warmup_points, n_test),
                "inference_seconds": float(elapsed),
                "seconds_per_point": float(elapsed / n_test),
                "backbone_parameters": params,
            }
        )

        for q in VARIABLES:
            arr = np.asarray(preds[q], dtype=np.float64)
            member_predictions[q].append(arr)

            member_rows.append(
                {
                    "member_index": member_index,
                    "seed": seed,
                    "variable": q,
                    "split": "heldout_test",
                    **metric_dict(arr, truth[q]),
                }
            )

            if args.save_pointwise:
                pointwise[f"seed_{seed}_{q}_pred"] = arr
                pointwise[f"seed_{seed}_{q}_abs_error"] = np.abs(
                    arr - truth[q]
                )

        del model, state, preds
        if device.type == "cuda":
            torch.cuda.empty_cache()

    member_df = pd.DataFrame(member_rows)
    timing_df = pd.DataFrame(timing_rows)

    ensemble_rows: List[Dict[str, object]] = []
    raw_uq_rows: List[Dict[str, object]] = []
    calibrated_uq_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    ensemble_samples: Dict[str, np.ndarray] = {}
    ensemble_means: Dict[str, np.ndarray] = {}
    ensemble_stds: Dict[str, np.ndarray] = {}

    for q in VARIABLES:
        samples = np.stack(member_predictions[q], axis=0)
        mean = np.mean(samples, axis=0)
        std = np.std(samples, axis=0, ddof=1)
        abs_error = np.abs(mean - truth[q])
        crps_pw = ensemble_crps_pointwise(samples=samples, truth=truth[q])

        ensemble_samples[q] = samples
        ensemble_means[q] = mean
        ensemble_stds[q] = std

        accuracy = metric_dict(mean, truth[q])
        pearson = pearson_corr(abs_error, std)
        spearman = spearman_corr(abs_error, std)
        crps = float(np.mean(crps_pw))

        ensemble_rows.append(
            {
                "method": "deep_ensemble",
                "members": samples.shape[0],
                "variable": q,
                "split": "heldout_test",
                **accuracy,
                "mean_predictive_std": float(np.mean(std)),
                "median_predictive_std": float(np.median(std)),
                "crps": crps,
                "pearson_abs_error_vs_std": pearson,
                "spearman_abs_error_vs_std": spearman,
            }
        )

        raw_rows_q = interval_metrics(
            mean=mean,
            std=std,
            truth=truth[q],
            levels=args.uq_levels,
            scale=1.0,
        )
        cal_rows_q = interval_metrics(
            mean=mean,
            std=std,
            truth=truth[q],
            levels=args.uq_levels,
            scale=factors[q],
        )

        for row in raw_rows_q:
            raw_uq_rows.append(
                {
                    "method": "deep_ensemble",
                    "members": samples.shape[0],
                    "variable": q,
                    "split": "heldout_test",
                    "interval_model": "gaussian_mean_plusminus_z_std",
                    **row,
                }
            )

        for row in cal_rows_q:
            calibrated_uq_rows.append(
                {
                    "method": "deep_ensemble",
                    "members": samples.shape[0],
                    "variable": q,
                    "split": "heldout_test",
                    "interval_model": "gaussian_mean_plusminus_z_scaled_std",
                    "frozen_from": "validation",
                    **row,
                }
            )

        summary_rows.append(
            {
                "method": "deep_ensemble",
                "members": samples.shape[0],
                "variable": q,
                "split": "heldout_test",
                "rmse": accuracy["rmse"],
                "mae": accuracy["mae"],
                "relative_l2": accuracy["relative_l2"],
                "max_abs_error": accuracy["max_abs_error"],
                "mean_predictive_std": float(np.mean(std)),
                "crps": crps,
                "pearson_abs_error_vs_std": pearson,
                "spearman_abs_error_vs_std": spearman,
                "raw_mean_absolute_calibration_error": (
                    mean_absolute_calibration_error(raw_rows_q)
                ),
                "validation_frozen_scale_factor": factors[q],
                "calibrated_mean_absolute_calibration_error": (
                    mean_absolute_calibration_error(cal_rows_q)
                ),
            }
        )

        if args.save_pointwise:
            pointwise[f"ensemble_{q}_mean"] = mean
            pointwise[f"ensemble_{q}_std"] = std
            pointwise[f"ensemble_{q}_abs_error"] = abs_error
            pointwise[f"ensemble_{q}_crps"] = crps_pw

            if any(np.isclose(level, 0.95) for level in args.uq_levels):
                z95 = z_for_level(0.95)
                pointwise[f"ensemble_{q}_lower_95_raw"] = mean - z95 * std
                pointwise[f"ensemble_{q}_upper_95_raw"] = mean + z95 * std
                pointwise[f"ensemble_{q}_lower_95_calibrated"] = (
                    mean - z95 * factors[q] * std
                )
                pointwise[f"ensemble_{q}_upper_95_calibrated"] = (
                    mean + z95 * factors[q] * std
                )

    # --------------------------------------------------------------
    # Pressure gauge-invariant diagnostic, matched to the final B-PINN
    # reporting convention: one additive offset per held-out time snapshot.
    # --------------------------------------------------------------
    p_mean = ensemble_means["p"]
    p_std = ensemble_stds["p"]
    p_ref = truth["p"]

    p_aligned, p_offsets, gauge_rows = pressure_align_per_time(
        p_mean=p_mean,
        p_ref=p_ref,
        time_index=time_index,
    )

    p_samples_aligned = ensemble_samples["p"] + p_offsets[None, :]
    p_crps_aligned = ensemble_crps_pointwise(
        samples=p_samples_aligned,
        truth=p_ref,
    )

    gauge_metric = metric_dict(p_aligned, p_ref)
    p_abs_error_aligned = np.abs(p_aligned - p_ref)

    raw_gauge_rows = interval_metrics(
        mean=p_aligned,
        std=p_std,
        truth=p_ref,
        levels=args.uq_levels,
        scale=1.0,
    )
    cal_gauge_rows = interval_metrics(
        mean=p_aligned,
        std=p_std,
        truth=p_ref,
        levels=args.uq_levels,
        scale=factors["p"],
    )

    p_gauge_summary = pd.DataFrame(
        [
            {
                "method": "deep_ensemble",
                "members": len(args.member_seeds),
                "variable": "p_gauge_aligned_per_time",
                "split": "heldout_test",
                **gauge_metric,
                "mean_predictive_std": float(np.mean(p_std)),
                "crps": float(np.mean(p_crps_aligned)),
                "pearson_abs_error_vs_std": pearson_corr(
                    p_abs_error_aligned, p_std
                ),
                "spearman_abs_error_vs_std": spearman_corr(
                    p_abs_error_aligned, p_std
                ),
                "raw_mean_absolute_calibration_error": (
                    mean_absolute_calibration_error(raw_gauge_rows)
                ),
                "validation_frozen_scale_factor": factors["p"],
                "calibrated_mean_absolute_calibration_error": (
                    mean_absolute_calibration_error(cal_gauge_rows)
                ),
                "calibrated_picp_95": next(
                    (
                        r["picp"]
                        for r in cal_gauge_rows
                        if np.isclose(r["nominal_level"], 0.95)
                    ),
                    np.nan,
                ),
                "calibrated_mpiw_95": next(
                    (
                        r["mpiw"]
                        for r in cal_gauge_rows
                        if np.isclose(r["nominal_level"], 0.95)
                    ),
                    np.nan,
                ),
            }
        ]
    )

    if args.save_pointwise:
        pointwise["ensemble_p_mean_gauge_aligned"] = p_aligned
        pointwise["ensemble_p_abs_error_gauge_aligned"] = p_abs_error_aligned
        if any(np.isclose(level, 0.95) for level in args.uq_levels):
            z95 = z_for_level(0.95)
            pointwise["ensemble_p_lower_95_raw_gauge_aligned"] = (
                p_aligned - z95 * p_std
            )
            pointwise["ensemble_p_upper_95_raw_gauge_aligned"] = (
                p_aligned + z95 * p_std
            )
            pointwise["ensemble_p_lower_95_calibrated_gauge_aligned"] = (
                p_aligned - z95 * factors["p"] * p_std
            )
            pointwise["ensemble_p_upper_95_calibrated_gauge_aligned"] = (
                p_aligned + z95 * factors["p"] * p_std
            )

    ensemble_df = pd.DataFrame(ensemble_rows)
    raw_uq_df = pd.DataFrame(raw_uq_rows)
    calibrated_uq_df = pd.DataFrame(calibrated_uq_rows)
    summary_df = pd.DataFrame(summary_rows)

    total_inference_seconds = float(timing_df["inference_seconds"].sum())
    cost_df = pd.DataFrame(
        [
            {
                "evaluation_grid": (
                    "fixed held-out spatial blocks over all time snapshots"
                ),
                "evaluation_point_count": n_test,
                "members": len(args.member_seeds),
                "serial_ensemble_inference_seconds": total_inference_seconds,
                "mean_member_inference_seconds": float(
                    timing_df["inference_seconds"].mean()
                ),
                "backbone_parameters_per_member": backbone_params_expected,
                "total_ensemble_backbone_parameters": (
                    int(backbone_params_expected * len(args.member_seeds))
                    if backbone_params_expected is not None
                    else np.nan
                ),
            }
        ]
    )

    summary_df["heldout_points"] = n_test
    summary_df["serial_ensemble_inference_seconds"] = total_inference_seconds
    summary_df["backbone_parameters_per_member"] = backbone_params_expected
    summary_df["total_ensemble_backbone_parameters"] = (
        int(backbone_params_expected * len(args.member_seeds))
        if backbone_params_expected is not None
        else np.nan
    )
    summary_df["calibration_fit_split"] = "validation"
    summary_df["test_used_for_calibration_fit"] = False

    member_df.to_csv(
        output_dir / "deep_ensemble_final_heldout_member_metrics.csv",
        index=False,
    )
    ensemble_df.to_csv(
        output_dir / "deep_ensemble_final_heldout_ensemble_metrics.csv",
        index=False,
    )
    raw_uq_df.to_csv(
        output_dir / "deep_ensemble_final_heldout_uq_raw.csv",
        index=False,
    )
    calibrated_uq_df.to_csv(
        output_dir / "deep_ensemble_final_heldout_uq_calibrated.csv",
        index=False,
    )
    p_gauge_summary.to_csv(
        output_dir / "deep_ensemble_final_heldout_pressure_gauge_summary.csv",
        index=False,
    )
    pd.DataFrame(gauge_rows).to_csv(
        output_dir / "deep_ensemble_final_heldout_pressure_offsets_by_time.csv",
        index=False,
    )
    timing_df.to_csv(
        output_dir / "deep_ensemble_final_heldout_timing.csv",
        index=False,
    )
    cost_df.to_csv(
        output_dir / "deep_ensemble_final_heldout_cost.csv",
        index=False,
    )
    summary_df.to_csv(
        output_dir / "deep_ensemble_final_heldout_summary.csv",
        index=False,
    )

    if args.save_pointwise:
        pd.DataFrame(pointwise).to_csv(
            output_dir / "deep_ensemble_final_heldout_pointwise.csv",
            index=False,
        )

    print("\n=== FINAL Deep Ensemble held-out metrics ===")
    print(
        summary_df[
            [
                "variable",
                "rmse",
                "mae",
                "relative_l2",
                "mean_predictive_std",
                "crps",
                "spearman_abs_error_vs_std",
                "raw_mean_absolute_calibration_error",
                "validation_frozen_scale_factor",
                "calibrated_mean_absolute_calibration_error",
            ]
        ].to_string(index=False)
    )

    print("\n=== FINAL gauge-invariant pressure diagnostic ===")
    print(p_gauge_summary.to_string(index=False))

    print("\n=== Inference cost ===")
    print(cost_df.to_string(index=False))

    print(f"\nSaved: {output_dir.resolve()}")
    print(
        "FINAL TEST COMPLETE. No model, dropout, ensemble, or calibration "
        "hyperparameter should be changed using these held-out results."
    )


if __name__ == "__main__":
    main()
