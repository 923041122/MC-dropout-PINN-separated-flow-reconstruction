#!/usr/bin/env python3
"""
cylinder_deep_ensemble_validation.py

Validation-only evaluator for the 5-member deterministic Deep Ensemble used as
an independent uncertainty-quantification baseline for the cylinder benchmark.

Important protocol rule:
    This script evaluates ONLY the fixed validation partition. It reads the
    held-out test indices only to report their reserved count and immediately
    discards them; held-out labels are never indexed or evaluated.

The ensemble members are reconstructed with:
    dropout_rate = 0.0
    dropout_placement = "all"

which matches the deterministic member-training commands used for the Deep
Ensemble experiment.

Outputs
-------
<output-dir>/
    deep_ensemble_validation_member_metrics.csv
    deep_ensemble_validation_ensemble_metrics.csv
    deep_ensemble_validation_uq_raw.csv
    deep_ensemble_validation_calibration_factors.csv
    deep_ensemble_validation_uq_calibrated.csv
    deep_ensemble_validation_timing.csv
    deep_ensemble_training_cost.csv
    deep_ensemble_validation_summary.csv
    deep_ensemble_validation_pointwise.csv

Calibration
-----------
For M=5 members, central intervals are constructed from the ensemble predictive
mean and sample standard deviation using a Gaussian approximation:

    mean +/- z(level) * scale * std

One multiplicative scale factor is fitted independently for u, v, and p using
VALIDATION DATA ONLY. The scale minimizes the mean absolute coverage error over
the requested central interval levels (default: 50%, 80%, 90%, 95%). The fitted
factors must be frozen before any held-out test evaluation.

CRPS
----
The proper score is the finite-ensemble CRPS:

    CRPS = mean_m |x_m - y| - 0.5 * mean_{m,k} |x_m - x_k|

averaged over validation points.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from statistics import NormalDist
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from benchmark_config import LAYER_MAT_PSI
from benchmark_tools import count_parameters, get_device
from cylinder_train_v2 import load_protocol, load_reference_stack
from pinn_model_dropout_ablation import PlacementPINNNet


VARIABLES = ("u", "v", "p")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Validation-only Deep Ensemble UQ evaluation for the corrected "
            "cylinder sparse-data protocol."
        )
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
        "--output-dir",
        default="./cylinder_deep_ensemble_validation",
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
        help=(
            "Template for each member result root. It must contain {seed}. "
            "The checkpoint is expected under models/bpinn_dropout.pth."
        ),
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
        "--scale-min",
        type=float,
        default=0.10,
    )
    p.add_argument(
        "--scale-max",
        type=float,
        default=10.0,
    )
    p.add_argument(
        "--scale-step",
        type=float,
        default=0.005,
    )

    p.add_argument("--seed", type=int, default=2025)
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
        raise ValueError("--uq-levels must contain at least one level.")

    bad = [x for x in args.uq_levels if not (0.0 < x < 1.0)]
    if bad:
        raise ValueError(f"All --uq-levels must lie in (0, 1), got {bad}.")

    if args.scale_min <= 0.0:
        raise ValueError("--scale-min must be > 0.")

    if args.scale_max <= args.scale_min:
        raise ValueError("--scale-max must be larger than --scale-min.")

    if args.scale_step <= 0.0:
        raise ValueError("--scale-step must be > 0.")


def safe_load_state(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def build_member(device: torch.device) -> torch.nn.Module:
    model = PlacementPINNNet(
        LAYER_MAT_PSI,
        dropout_rate=0.0,
        dropout_placement="all",
    ).to(device)
    model.eval()
    return model


def forward_psi_p(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Robustly obtain psi and p from the project's PINN model.

    The primary expected signature is model(x, y, t). A concatenated-input
    fallback is included only to make this evaluator tolerant of a small
    forward-signature difference without changing the checkpoint architecture.
    """
    try:
        out = model(x, y, t)
    except TypeError:
        out = model(torch.cat([x, y, t], dim=1))

    if isinstance(out, (tuple, list)):
        if len(out) != 2:
            raise RuntimeError(
                "Model forward returned a tuple/list that does not contain exactly "
                "(psi, p)."
            )
        psi, p = out
    else:
        if out.ndim != 2 or out.shape[1] < 2:
            raise RuntimeError(
                f"Model forward output must have at least two columns, got {tuple(out.shape)}."
            )
        psi = out[:, 0:1]
        p = out[:, 1:2]

    return psi, p


def predict_uvp(
    model: torch.nn.Module,
    xyz: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    """
    Predict u, v, p on xyz points.

    Gradients remain enabled because u and v are recovered from the streamfunction:
        u = d psi / d y
        v = -d psi / d x
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must have shape (N, 3), got {xyz.shape}.")

    all_u: List[np.ndarray] = []
    all_v: List[np.ndarray] = []
    all_p: List[np.ndarray] = []

    model.eval()

    for start in range(0, len(xyz), batch_size):
        stop = min(start + batch_size, len(xyz))
        batch = xyz[start:stop]

        x = torch.tensor(
            batch[:, 0:1],
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        y = torch.tensor(
            batch[:, 1:2],
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        t = torch.tensor(
            batch[:, 2:3],
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )

        with torch.enable_grad():
            psi, p = forward_psi_p(model, x, y, t)

            u = torch.autograd.grad(
                psi.sum(),
                y,
                create_graph=False,
                retain_graph=True,
            )[0]

            v = -torch.autograd.grad(
                psi.sum(),
                x,
                create_graph=False,
                retain_graph=False,
            )[0]

        all_u.append(u.detach().cpu().numpy().reshape(-1))
        all_v.append(v.detach().cpu().numpy().reshape(-1))
        all_p.append(p.detach().cpu().numpy().reshape(-1))

        del x, y, t, psi, p, u, v

    return {
        "u": np.concatenate(all_u).astype(np.float32, copy=False),
        "v": np.concatenate(all_v).astype(np.float32, copy=False),
        "p": np.concatenate(all_p).astype(np.float32, copy=False),
    }


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_prediction(
    model: torch.nn.Module,
    xyz: np.ndarray,
    device: torch.device,
    batch_size: int,
    warmup_points: int,
) -> Tuple[Dict[str, np.ndarray], float]:
    if warmup_points > 0 and len(xyz) > 0:
        n_warm = min(warmup_points, len(xyz))
        _ = predict_uvp(
            model=model,
            xyz=xyz[:n_warm],
            device=device,
            batch_size=min(batch_size, n_warm),
        )
        synchronize(device)

    synchronize(device)
    start = time.perf_counter()
    preds = predict_uvp(
        model=model,
        xyz=xyz,
        device=device,
        batch_size=batch_size,
    )
    synchronize(device)
    elapsed = time.perf_counter() - start

    return preds, float(elapsed)


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
            "points": 0,
        }

    err = pred - truth
    denominator = np.linalg.norm(truth)

    return {
        "relative_l2": (
            float(np.linalg.norm(err) / denominator)
            if denominator > 0.0
            else np.nan
        ),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "max_abs_error": float(np.max(np.abs(err))),
        "points": int(truth.size),
    }


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)

    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]

    if a.size < 2:
        return np.nan

    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return np.nan

    return float(np.corrcoef(a, b)[0, 1])


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)

    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]

    if a.size < 2:
        return np.nan

    ar = pd.Series(a).rank(method="average").to_numpy(dtype=float)
    br = pd.Series(b).rank(method="average").to_numpy(dtype=float)

    return pearson_corr(ar, br)


def ensemble_crps_pointwise(
    samples: np.ndarray,
    truth: np.ndarray,
) -> np.ndarray:
    """
    Finite-ensemble CRPS per point.

    samples shape: (members, points)
    truth shape:   (points,)
    """
    samples = np.asarray(samples, dtype=float)
    truth = np.asarray(truth, dtype=float).reshape(-1)

    if samples.ndim != 2:
        raise ValueError("samples must have shape (members, points).")

    if samples.shape[1] != truth.size:
        raise ValueError("samples and truth point counts do not match.")

    term1 = np.mean(np.abs(samples - truth[None, :]), axis=0)

    pairwise = np.zeros(truth.size, dtype=float)
    m = samples.shape[0]

    for i in range(m):
        for j in range(m):
            pairwise += np.abs(samples[i] - samples[j])

    term2 = 0.5 * pairwise / float(m * m)
    return term1 - term2


def z_for_level(level: float) -> float:
    return float(NormalDist().inv_cdf(0.5 + 0.5 * float(level)))


def interval_metrics(
    mean: np.ndarray,
    std: np.ndarray,
    truth: np.ndarray,
    levels: Sequence[float],
    scale: float,
) -> List[Dict[str, float]]:
    mean = np.asarray(mean, dtype=float).reshape(-1)
    std = np.asarray(std, dtype=float).reshape(-1)
    truth = np.asarray(truth, dtype=float).reshape(-1)

    rows: List[Dict[str, float]] = []

    for level in levels:
        z = z_for_level(level)
        half = z * float(scale) * std
        lower = mean - half
        upper = mean + half

        valid = (
            np.isfinite(lower)
            & np.isfinite(upper)
            & np.isfinite(truth)
        )

        if valid.sum() == 0:
            coverage = np.nan
            width = np.nan
        else:
            covered = (
                (truth[valid] >= lower[valid])
                & (truth[valid] <= upper[valid])
            )
            coverage = float(np.mean(covered))
            width = float(np.mean(upper[valid] - lower[valid]))

        rows.append(
            {
                "nominal_level": float(level),
                "scale_factor": float(scale),
                "picp": coverage,
                "mpiw": width,
                "absolute_calibration_error": (
                    float(abs(coverage - level))
                    if np.isfinite(coverage)
                    else np.nan
                ),
            }
        )

    return rows


def coverage_at_scale_from_sorted_ratios(
    sorted_ratios: np.ndarray,
    scale: float,
) -> float:
    if sorted_ratios.size == 0:
        return np.nan

    count = np.searchsorted(sorted_ratios, scale, side="right")
    return float(count / sorted_ratios.size)


def fit_scale_factor(
    mean: np.ndarray,
    std: np.ndarray,
    truth: np.ndarray,
    levels: Sequence[float],
    scale_min: float,
    scale_max: float,
    scale_step: float,
) -> Tuple[float, float]:
    """
    Fit one multiplicative std scale by minimizing mean absolute coverage error.

    Efficient implementation: for each nominal level, precompute the required
    scale ratio |error| / (z * std), sort it once, and then evaluate coverage by
    binary search for every candidate scale.
    """
    mean = np.asarray(mean, dtype=float).reshape(-1)
    std = np.asarray(std, dtype=float).reshape(-1)
    truth = np.asarray(truth, dtype=float).reshape(-1)

    abs_err = np.abs(mean - truth)

    ratio_sets: Dict[float, np.ndarray] = {}

    for level in levels:
        z = z_for_level(level)
        denom = z * std

        valid = np.isfinite(abs_err) & np.isfinite(denom)
        err_v = abs_err[valid]
        den_v = denom[valid]

        ratios = np.full(err_v.shape, np.inf, dtype=float)

        positive = den_v > 0.0
        ratios[positive] = err_v[positive] / den_v[positive]

        exact = (~positive) & (err_v == 0.0)
        ratios[exact] = 0.0

        ratio_sets[float(level)] = np.sort(ratios)

    n_steps = int(math.floor((scale_max - scale_min) / scale_step)) + 1
    scales = scale_min + np.arange(n_steps, dtype=float) * scale_step
    scales = scales[scales <= scale_max + 1e-12]

    best_scale = float(scales[0])
    best_score = float("inf")

    for scale in scales:
        errors = []

        for level in levels:
            coverage = coverage_at_scale_from_sorted_ratios(
                ratio_sets[float(level)],
                float(scale),
            )
            if np.isfinite(coverage):
                errors.append(abs(coverage - float(level)))

        if not errors:
            continue

        score = float(np.mean(errors))

        # Tie-break toward the smaller factor to avoid needless interval widening.
        if score < best_score - 1e-15:
            best_score = score
            best_scale = float(scale)

    return best_scale, best_score


def load_training_summary(
    member_root: Path,
    seed: int,
) -> Dict[str, object]:
    path = member_root / "training_logs" / "bpinn_dropout_training_summary.csv"

    row: Dict[str, object] = {
        "seed": int(seed),
        "member_root": str(member_root),
        "summary_path": str(path),
        "summary_found": bool(path.exists()),
    }

    if not path.exists():
        return row

    df = pd.read_csv(path)
    if df.empty:
        return row

    src = df.iloc[-1].to_dict()

    wanted = [
        "training_time_seconds",
        "epochs",
        "supervised_ratio",
        "supervised_points",
        "validation_points",
        "heldout_test_points_reserved",
        "heldout_test_accessed_during_training",
        "equation_points",
        "data_loss_weight",
        "equation_loss_weight",
        "dropout_rate",
        "dropout_placement",
        "backbone_parameters",
        "additional_trainable_parameters",
        "total_optimized_parameters",
        "final_total_loss",
        "final_data_loss",
        "final_equation_loss",
    ]

    for key in wanted:
        if key in src:
            row[key] = src[key]

    return row


def main() -> None:
    args = parse_args()
    validate_args(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("Cylinder Deep Ensemble validation-only UQ evaluation")
    print(f"Device: {device}")
    print(f"Protocol: {args.protocol_root}")
    print(f"Supervised ratio key: {args.supervised_ratio:.2%}")
    print(f"Ensemble members: {args.member_seeds}")
    print("=" * 100)

    train_idx, val_idx, test_idx, eq_np, protocol_summary = load_protocol(
        protocol_root=Path(args.protocol_root),
        ratio=args.supervised_ratio,
        n_equation_points=args.n_equation_points,
    )

    heldout_count = int(len(test_idx))
    validation_count = int(len(val_idx))

    # The validation evaluator does not use train labels, test labels, or physics
    # collocation points. Discard those index/point arrays immediately.
    del train_idx, test_idx, eq_np

    full_data = load_reference_stack(Path(args.data_path))

    expected_total = int(protocol_summary["total_reference_points"])
    if len(full_data) != expected_total:
        raise RuntimeError(
            f"Reference-data size mismatch: got {len(full_data)}, "
            f"protocol expects {expected_total}."
        )

    validation_data = full_data[torch.from_numpy(val_idx)]
    del full_data, val_idx

    val_np = validation_data.detach().cpu().numpy()
    if val_np.ndim != 2 or val_np.shape[1] < 6:
        raise RuntimeError(
            f"Expected validation data with at least 6 columns, got {val_np.shape}."
        )

    xyz = np.asarray(val_np[:, 0:3], dtype=np.float32)
    truth = {
        "u": np.asarray(val_np[:, 3], dtype=np.float64),
        "v": np.asarray(val_np[:, 4], dtype=np.float64),
        "p": np.asarray(val_np[:, 5], dtype=np.float64),
    }

    print(f"Validation points: {validation_count}")
    print(
        f"Held-out test points: {heldout_count} "
        "(RESERVED; LABELS NOT INDEXED OR EVALUATED)"
    )
    print(f"Reference total points: {expected_total}")
    print()

    member_predictions: Dict[str, List[np.ndarray]] = {
        q: [] for q in VARIABLES
    }
    member_metric_rows: List[Dict[str, object]] = []
    timing_rows: List[Dict[str, object]] = []
    training_cost_rows: List[Dict[str, object]] = []
    pointwise_columns: Dict[str, np.ndarray] = {
        "x": xyz[:, 0].astype(np.float32, copy=False),
        "y": xyz[:, 1].astype(np.float32, copy=False),
        "t": xyz[:, 2].astype(np.float32, copy=False),
        "u_ref": truth["u"].astype(np.float32),
        "v_ref": truth["v"].astype(np.float32),
        "p_ref": truth["p"].astype(np.float32),
    }

    backbone_params_expected = None

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

        backbone_params = int(count_parameters(model))
        if backbone_params_expected is None:
            backbone_params_expected = backbone_params
        elif backbone_params != backbone_params_expected:
            raise RuntimeError(
                "Deep Ensemble members do not have identical parameter counts: "
                f"expected {backbone_params_expected}, got {backbone_params} "
                f"for seed {seed}."
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
                "validation_points": validation_count,
                "batch_size": args.batch_size,
                "warmup_points": min(args.warmup_points, validation_count),
                "inference_seconds": elapsed,
                "seconds_per_point": elapsed / validation_count,
                "backbone_parameters": backbone_params,
            }
        )

        for q in VARIABLES:
            arr = np.asarray(preds[q], dtype=np.float64)
            member_predictions[q].append(arr)

            values = metric_dict(arr, truth[q])
            member_metric_rows.append(
                {
                    "member_index": member_index,
                    "seed": seed,
                    "variable": q,
                    "split": "validation",
                    **values,
                }
            )

            if args.save_pointwise:
                pointwise_columns[f"seed_{seed}_{q}_pred"] = arr.astype(np.float32)
                pointwise_columns[f"seed_{seed}_{q}_abs_error"] = np.abs(
                    arr - truth[q]
                ).astype(np.float32)

        training_cost_rows.append(
            load_training_summary(member_root=member_root, seed=seed)
        )

        del model, state, preds
        if device.type == "cuda":
            torch.cuda.empty_cache()

    member_df = pd.DataFrame(member_metric_rows)
    member_df.to_csv(
        output_dir / "deep_ensemble_validation_member_metrics.csv",
        index=False,
    )

    timing_df = pd.DataFrame(timing_rows)
    timing_df.to_csv(
        output_dir / "deep_ensemble_validation_timing.csv",
        index=False,
    )

    training_cost_df = pd.DataFrame(training_cost_rows)
    training_cost_df.to_csv(
        output_dir / "deep_ensemble_training_cost.csv",
        index=False,
    )

    ensemble_metric_rows: List[Dict[str, object]] = []
    raw_uq_rows: List[Dict[str, object]] = []
    calibrated_uq_rows: List[Dict[str, object]] = []
    calibration_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    for q in VARIABLES:
        samples = np.stack(member_predictions[q], axis=0)
        mean = np.mean(samples, axis=0)
        std = np.std(samples, axis=0, ddof=1)
        abs_error = np.abs(mean - truth[q])
        crps_pw = ensemble_crps_pointwise(samples=samples, truth=truth[q])

        accuracy = metric_dict(mean, truth[q])
        pearson = pearson_corr(abs_error, std)
        spearman = spearman_corr(abs_error, std)
        crps = float(np.mean(crps_pw))

        ensemble_metric_rows.append(
            {
                "method": "deep_ensemble",
                "members": samples.shape[0],
                "variable": q,
                "split": "validation",
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

        for row in raw_rows_q:
            raw_uq_rows.append(
                {
                    "method": "deep_ensemble",
                    "members": samples.shape[0],
                    "variable": q,
                    "split": "validation",
                    "interval_model": "gaussian_mean_plusminus_z_std",
                    **row,
                }
            )

        fitted_scale, fitted_mace = fit_scale_factor(
            mean=mean,
            std=std,
            truth=truth[q],
            levels=args.uq_levels,
            scale_min=args.scale_min,
            scale_max=args.scale_max,
            scale_step=args.scale_step,
        )

        calibration_rows.append(
            {
                "method": "deep_ensemble",
                "members": samples.shape[0],
                "variable": q,
                "split_used_for_fit": "validation",
                "scale_factor": fitted_scale,
                "objective": "mean_absolute_coverage_error",
                "fit_levels": ";".join(f"{x:g}" for x in args.uq_levels),
                "mean_absolute_calibration_error_after_fit": fitted_mace,
                "scale_min": args.scale_min,
                "scale_max": args.scale_max,
                "scale_step": args.scale_step,
            }
        )

        calibrated_rows_q = interval_metrics(
            mean=mean,
            std=std,
            truth=truth[q],
            levels=args.uq_levels,
            scale=fitted_scale,
        )

        for row in calibrated_rows_q:
            calibrated_uq_rows.append(
                {
                    "method": "deep_ensemble",
                    "members": samples.shape[0],
                    "variable": q,
                    "split": "validation",
                    "interval_model": "gaussian_mean_plusminus_z_scaled_std",
                    **row,
                }
            )

        raw_mace = float(
            np.mean(
                [
                    r["absolute_calibration_error"]
                    for r in raw_rows_q
                    if np.isfinite(r["absolute_calibration_error"])
                ]
            )
        )

        summary_rows.append(
            {
                "method": "deep_ensemble",
                "members": samples.shape[0],
                "variable": q,
                "split": "validation",
                "rmse": accuracy["rmse"],
                "mae": accuracy["mae"],
                "relative_l2": accuracy["relative_l2"],
                "max_abs_error": accuracy["max_abs_error"],
                "mean_predictive_std": float(np.mean(std)),
                "crps": crps,
                "pearson_abs_error_vs_std": pearson,
                "spearman_abs_error_vs_std": spearman,
                "raw_mean_absolute_calibration_error": raw_mace,
                "validation_fitted_scale_factor": fitted_scale,
                "calibrated_mean_absolute_calibration_error": fitted_mace,
            }
        )

        if args.save_pointwise:
            pointwise_columns[f"ensemble_{q}_mean"] = mean.astype(np.float32)
            pointwise_columns[f"ensemble_{q}_std"] = std.astype(np.float32)
            pointwise_columns[f"ensemble_{q}_abs_error"] = abs_error.astype(np.float32)
            pointwise_columns[f"ensemble_{q}_crps"] = crps_pw.astype(np.float32)

            # Always materialize the 95% raw and validation-calibrated intervals
            # when 0.95 is requested; these columns are useful for later plotting.
            if any(np.isclose(level, 0.95) for level in args.uq_levels):
                z95 = z_for_level(0.95)
                pointwise_columns[f"ensemble_{q}_lower_95_raw"] = (
                    mean - z95 * std
                ).astype(np.float32)
                pointwise_columns[f"ensemble_{q}_upper_95_raw"] = (
                    mean + z95 * std
                ).astype(np.float32)
                pointwise_columns[f"ensemble_{q}_lower_95_calibrated"] = (
                    mean - z95 * fitted_scale * std
                ).astype(np.float32)
                pointwise_columns[f"ensemble_{q}_upper_95_calibrated"] = (
                    mean + z95 * fitted_scale * std
                ).astype(np.float32)

    ensemble_df = pd.DataFrame(ensemble_metric_rows)
    raw_uq_df = pd.DataFrame(raw_uq_rows)
    calibrated_uq_df = pd.DataFrame(calibrated_uq_rows)
    calibration_df = pd.DataFrame(calibration_rows)
    summary_df = pd.DataFrame(summary_rows)

    ensemble_df.to_csv(
        output_dir / "deep_ensemble_validation_ensemble_metrics.csv",
        index=False,
    )
    raw_uq_df.to_csv(
        output_dir / "deep_ensemble_validation_uq_raw.csv",
        index=False,
    )
    calibration_df.to_csv(
        output_dir / "deep_ensemble_validation_calibration_factors.csv",
        index=False,
    )
    calibrated_uq_df.to_csv(
        output_dir / "deep_ensemble_validation_uq_calibrated.csv",
        index=False,
    )

    total_inference_seconds = float(timing_df["inference_seconds"].sum())

    training_time_values = pd.to_numeric(
        training_cost_df.get(
            "training_time_seconds",
            pd.Series(dtype=float),
        ),
        errors="coerce",
    )
    total_training_seconds = (
        float(training_time_values.sum(min_count=1))
        if len(training_time_values) > 0
        else np.nan
    )

    summary_df["validation_points"] = validation_count
    summary_df["heldout_test_points_reserved"] = heldout_count
    summary_df["heldout_test_labels_accessed"] = False
    summary_df["serial_ensemble_inference_seconds"] = total_inference_seconds
    summary_df["total_member_training_seconds"] = total_training_seconds
    summary_df["backbone_parameters_per_member"] = backbone_params_expected
    summary_df["total_ensemble_backbone_parameters"] = (
        int(backbone_params_expected * len(args.member_seeds))
        if backbone_params_expected is not None
        else np.nan
    )

    summary_df.to_csv(
        output_dir / "deep_ensemble_validation_summary.csv",
        index=False,
    )

    if args.save_pointwise:
        pd.DataFrame(pointwise_columns).to_csv(
            output_dir / "deep_ensemble_validation_pointwise.csv",
            index=False,
        )

    print()
    print("=== Deep Ensemble validation: member accuracy ===")
    print(
        member_df[
            ["seed", "variable", "rmse", "mae", "relative_l2"]
        ].to_string(index=False)
    )

    print()
    print("=== Deep Ensemble validation: ensemble-mean accuracy / UQ ===")
    print(
        ensemble_df[
            [
                "variable",
                "rmse",
                "mae",
                "relative_l2",
                "mean_predictive_std",
                "crps",
                "pearson_abs_error_vs_std",
                "spearman_abs_error_vs_std",
            ]
        ].to_string(index=False)
    )

    print()
    print("=== Raw validation intervals ===")
    print(
        raw_uq_df[
            [
                "variable",
                "nominal_level",
                "picp",
                "mpiw",
                "absolute_calibration_error",
            ]
        ].to_string(index=False)
    )

    print()
    print("=== Validation-fitted interval scale factors ===")
    print(
        calibration_df[
            [
                "variable",
                "scale_factor",
                "mean_absolute_calibration_error_after_fit",
            ]
        ].to_string(index=False)
    )

    print()
    print("=== Calibrated validation intervals ===")
    print(
        calibrated_uq_df[
            [
                "variable",
                "nominal_level",
                "picp",
                "mpiw",
                "absolute_calibration_error",
            ]
        ].to_string(index=False)
    )

    print()
    print("=== Deep Ensemble cost ===")
    print(
        timing_df[
            ["seed", "inference_seconds", "backbone_parameters"]
        ].to_string(index=False)
    )
    print(f"Serial 5-member inference seconds: {total_inference_seconds:.6f}")

    if np.isfinite(total_training_seconds):
        print(f"Total member training seconds: {total_training_seconds:.3f}")
    else:
        print(
            "Total member training seconds: unavailable "
            "(one or more training summaries missing the field)."
        )

    print()
    print(f"Saved outputs to: {output_dir.resolve()}")
    print(
        "Held-out test labels were NOT indexed or evaluated. "
        "Freeze the validation-fitted scale factors before held-out testing."
    )


if __name__ == "__main__":
    main()
