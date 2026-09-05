
"""Validation-only MC-dropout evaluator for cylinder dropout-rate screening.

Purpose
-------
Evaluate independently trained dropout-rate checkpoints on the SAME fixed
validation partition, while keeping the held-out test set sealed.

For every dropout rate, this script reports:
1. deterministic dropout-OFF prediction;
2. one reproducible single stochastic dropout prediction;
3. MC predictive means for N = 5, 10, 20, 50, 100;
4. epistemic predictive standard deviation;
5. empirical central prediction-interval coverage and width;
6. calibration error over 50%, 80%, 90%, and 95% central intervals;
7. empirical-ensemble CRPS as a proper scoring rule;
8. Pearson and Spearman correlations between absolute error and uncertainty;
9. stochastic inference cost.

Important interpretation
------------------------
These MC-dropout intervals quantify approximate EPISTEMIC uncertainty only.
No observation-noise likelihood or heteroscedastic output is trained here, so
the intervals must not be described as explicit aleatoric uncertainty.

The held-out test partition is never loaded by this script.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from benchmark_config import LAYER_MAT_PSI
from benchmark_tools import get_device
from pinn_model import read_2D_data
from pinn_model_dropout_ablation import PlacementPINNNet

try:
    from scipy.stats import pearsonr, spearmanr
except Exception as exc:
    raise ImportError("scipy is required for correlation metrics.") from exc


RATE_STRINGS = ["0.002", "0.005", "0.01", "0.02", "0.05"]
MC_COUNTS = [5, 10, 20, 50, 100]
NOMINAL_COVERAGES = [0.50, 0.80, 0.90, 0.95]
VARIABLES = ("u", "v", "p")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-path",
        default="./2d_cylinder_Re3900_100x100_kw_sst.mat",
    )
    p.add_argument(
        "--protocol-root",
        default="./results_cylinder_protocol_v1_1/protocol",
    )
    p.add_argument(
        "--root",
        default=".",
        help="Parent directory containing dropout_rate_screen_p*/ folders.",
    )
    p.add_argument(
        "--output-dir",
        default="./dropout_rate_screen_mc_validation",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=5000,
    )
    p.add_argument(
        "--inference-seed",
        type=int,
        default=9091,
    )
    return p.parse_args()


def safe_load_state(path: Path, device):
    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        return torch.load(
            path,
            map_location=device,
        )


def load_validation_only(data_path: Path, protocol_root: Path):
    idx_path = protocol_root / "protocol_indices.npz"
    summary_path = protocol_root / "protocol_summary.csv"

    if not idx_path.exists():
        raise FileNotFoundError(idx_path)
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    # Intentionally access ONLY validation_indices.
    protocol_npz = np.load(idx_path)
    if "validation_indices" not in protocol_npz.files:
        raise KeyError("validation_indices not found in protocol_indices.npz")
    validation_idx = np.asarray(
        protocol_npz["validation_indices"],
        dtype=np.int64,
    )

    protocol_summary = pd.read_csv(summary_path).iloc[0]

    x, y, t, u, v, p, _ = read_2D_data(str(data_path))
    full = torch.cat([x, y, t, u, v, p], dim=1).float()

    if len(full) != int(protocol_summary["total_reference_points"]):
        raise RuntimeError("Reference-data size does not match protocol.")

    val = full[torch.from_numpy(validation_idx)]

    expected = int(protocol_summary["validation_points"])
    if len(val) != expected:
        raise RuntimeError(
            f"Validation count mismatch: got {len(val)}, expected {expected}."
        )

    return val, protocol_summary


def predict_batch(
    model: torch.nn.Module,
    batch: torch.Tensor,
    device: torch.device,
    stochastic: bool,
) -> np.ndarray:
    x = batch[:, 0:1].to(device)
    y = batch[:, 1:2].to(device)
    t = batch[:, 2:3].to(device)

    u, v, p = model.predict_fields_safe(
        x,
        y,
        t,
        create_graph=False,
        train_mode=bool(stochastic),
    )

    return np.stack(
        [
            u.detach().cpu().numpy().reshape(-1),
            v.detach().cpu().numpy().reshape(-1),
            p.detach().cpu().numpy().reshape(-1),
        ],
        axis=-1,
    )


def empirical_crps_per_point(samples: np.ndarray, target: np.ndarray):
    """Empirical ensemble CRPS, vectorized over points.

    samples: shape (M, B)
    target:  shape (B,)
    """
    m = samples.shape[0]

    term1 = np.mean(
        np.abs(samples - target[None, :]),
        axis=0,
    )

    ordered = np.sort(samples, axis=0)
    coeff = (
        2 * np.arange(1, m + 1, dtype=np.float64)
        - m
        - 1
    )[:, None]

    pair_term = np.sum(
        coeff * ordered,
        axis=0,
    ) / float(m * m)

    return term1 - pair_term


def init_accumulator():
    acc = {}
    for n in MC_COUNTS:
        acc[n] = {}
        for j, var in enumerate(VARIABLES):
            acc[n][var] = {
                "count": 0,
                "sq_error_sum": 0.0,
                "abs_error_sum": 0.0,
                "std_sum": 0.0,
                "crps_sum": 0.0,
                "coverage_counts": {
                    c: 0 for c in NOMINAL_COVERAGES
                },
                "width_sums": {
                    c: 0.0 for c in NOMINAL_COVERAGES
                },
                "abs_error_vectors": [],
                "std_vectors": [],
            }
    return acc


def update_mc_metrics(
    acc,
    samples_all: np.ndarray,
    target: np.ndarray,
):
    """Update all requested MC-prefix metrics for one validation batch.

    samples_all shape: (100, B, 3)
    target shape:      (B, 3)
    """
    for n in MC_COUNTS:
        s_prefix = samples_all[:n]

        for j, var in enumerate(VARIABLES):
            s = s_prefix[:, :, j]
            y = target[:, j]

            mean = np.mean(s, axis=0)
            if n > 1:
                std = np.std(s, axis=0, ddof=1)
            else:
                std = np.zeros_like(mean)

            err = mean - y
            abs_err = np.abs(err)

            d = acc[n][var]
            d["count"] += len(y)
            d["sq_error_sum"] += float(
                np.sum(err**2, dtype=np.float64)
            )
            d["abs_error_sum"] += float(
                np.sum(abs_err, dtype=np.float64)
            )
            d["std_sum"] += float(
                np.sum(std, dtype=np.float64)
            )

            crps = empirical_crps_per_point(s, y)
            d["crps_sum"] += float(
                np.sum(crps, dtype=np.float64)
            )

            for nominal in NOMINAL_COVERAGES:
                alpha = 1.0 - nominal
                lo = np.quantile(
                    s,
                    alpha / 2.0,
                    axis=0,
                )
                hi = np.quantile(
                    s,
                    1.0 - alpha / 2.0,
                    axis=0,
                )
                covered = (y >= lo) & (y <= hi)
                width = hi - lo

                d["coverage_counts"][nominal] += int(
                    np.sum(covered)
                )
                d["width_sums"][nominal] += float(
                    np.sum(width, dtype=np.float64)
                )

            d["abs_error_vectors"].append(
                abs_err.astype(np.float32)
            )
            d["std_vectors"].append(
                std.astype(np.float32)
            )


def finalize_mc_metrics(
    acc,
    rate_string: str,
    dropout_rate: float,
    stochastic_forward_seconds: float,
):
    rows = []

    mean_seconds_per_stochastic_pass = (
        stochastic_forward_seconds / 100.0
    )

    for n in MC_COUNTS:
        for var in VARIABLES:
            d = acc[n][var]
            count = d["count"]

            abs_err = np.concatenate(
                d["abs_error_vectors"]
            ).astype(np.float64)
            std = np.concatenate(
                d["std_vectors"]
            ).astype(np.float64)

            if np.std(abs_err) > 0 and np.std(std) > 0:
                pearson = float(
                    pearsonr(abs_err, std).statistic
                )
                spearman = float(
                    spearmanr(abs_err, std).statistic
                )
            else:
                pearson = np.nan
                spearman = np.nan

            coverages = {}
            widths = {}
            for nominal in NOMINAL_COVERAGES:
                coverages[nominal] = (
                    d["coverage_counts"][nominal]
                    / float(count)
                )
                widths[nominal] = (
                    d["width_sums"][nominal]
                    / float(count)
                )

            calibration_error = float(
                np.mean([
                    abs(coverages[c] - c)
                    for c in NOMINAL_COVERAGES
                ])
            )

            rows.append({
                "dropout_rate_string": rate_string,
                "dropout_rate": dropout_rate,
                "dropout_placement": "all",
                "mc_samples": n,
                "variable": var,
                "validation_points": count,
                "mc_mean_rmse": math.sqrt(
                    d["sq_error_sum"] / float(count)
                ),
                "mc_mean_mae": (
                    d["abs_error_sum"] / float(count)
                ),
                "mean_epistemic_std": (
                    d["std_sum"] / float(count)
                ),
                "picp_50": coverages[0.50],
                "mpiw_50": widths[0.50],
                "picp_80": coverages[0.80],
                "mpiw_80": widths[0.80],
                "picp_90": coverages[0.90],
                "mpiw_90": widths[0.90],
                "picp_95": coverages[0.95],
                "mpiw_95": widths[0.95],
                "mean_absolute_calibration_error": calibration_error,
                "empirical_crps": (
                    d["crps_sum"] / float(count)
                ),
                "pearson_abs_error_vs_std": pearson,
                "spearman_abs_error_vs_std": spearman,
                "mc100_stochastic_forward_seconds": (
                    stochastic_forward_seconds
                ),
                "mean_seconds_per_stochastic_pass": (
                    mean_seconds_per_stochastic_pass
                ),
                "estimated_forward_seconds_for_mc_n": (
                    mean_seconds_per_stochastic_pass * n
                ),
                "uncertainty_type": (
                    "approximate_epistemic_only"
                ),
            })

    return rows


def evaluate_one_rate(
    rate_string: str,
    validation_data: torch.Tensor,
    root: Path,
    device,
    batch_size: int,
    inference_seed: int,
):
    dropout_rate = float(rate_string)

    run_root = root / f"dropout_rate_screen_p{rate_string}"
    checkpoint = (
        run_root / "models" / "bpinn_dropout.pth"
    )
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    model = PlacementPINNNet(
        LAYER_MAT_PSI,
        dropout_rate=dropout_rate,
        dropout_placement="all",
    ).to(device)

    model.load_state_dict(
        safe_load_state(checkpoint, device),
        strict=True,
    )

    n_val = len(validation_data)

    # --------------------------------------------------------------
    # Deterministic dropout-OFF evaluation.
    # --------------------------------------------------------------
    det_sq = {
        var: 0.0 for var in VARIABLES
    }
    det_abs = {
        var: 0.0 for var in VARIABLES
    }

    deterministic_forward_seconds = 0.0

    for start in range(0, n_val, batch_size):
        batch = validation_data[
            start:start + batch_size
        ]
        target = batch[:, 3:6].cpu().numpy()

        t0 = time.perf_counter()
        pred = predict_batch(
            model=model,
            batch=batch,
            device=device,
            stochastic=False,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        deterministic_forward_seconds += (
            time.perf_counter() - t0
        )

        err = pred - target
        for j, var in enumerate(VARIABLES):
            det_sq[var] += float(
                np.sum(
                    err[:, j] ** 2,
                    dtype=np.float64,
                )
            )
            det_abs[var] += float(
                np.sum(
                    np.abs(err[:, j]),
                    dtype=np.float64,
                )
            )

    deterministic_rows = []
    for var in VARIABLES:
        deterministic_rows.append({
            "dropout_rate_string": rate_string,
            "dropout_rate": dropout_rate,
            "dropout_placement": "all",
            "inference_mode": "dropout_off",
            "variable": var,
            "validation_points": n_val,
            "rmse": math.sqrt(
                det_sq[var] / float(n_val)
            ),
            "mae": det_abs[var] / float(n_val),
            "forward_seconds": (
                deterministic_forward_seconds
            ),
        })

    # --------------------------------------------------------------
    # Stochastic evaluation.
    # One common 100-sample draw is used; N=5/10/20/50 are nested prefixes.
    # --------------------------------------------------------------
    torch.manual_seed(inference_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(inference_seed)

    acc = init_accumulator()

    single_sq = {
        var: 0.0 for var in VARIABLES
    }
    single_abs = {
        var: 0.0 for var in VARIABLES
    }

    stochastic_forward_seconds = 0.0

    for start in range(0, n_val, batch_size):
        batch = validation_data[
            start:start + batch_size
        ]
        target = batch[:, 3:6].cpu().numpy()

        samples: List[np.ndarray] = []

        for m in range(100):
            t0 = time.perf_counter()
            pred = predict_batch(
                model=model,
                batch=batch,
                device=device,
                stochastic=True,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            stochastic_forward_seconds += (
                time.perf_counter() - t0
            )
            samples.append(pred.astype(np.float32))

        samples_all = np.stack(
            samples,
            axis=0,
        )

        # Reproducible single stochastic subnetwork = first sample.
        single = samples_all[0]
        single_err = single - target

        for j, var in enumerate(VARIABLES):
            single_sq[var] += float(
                np.sum(
                    single_err[:, j] ** 2,
                    dtype=np.float64,
                )
            )
            single_abs[var] += float(
                np.sum(
                    np.abs(single_err[:, j]),
                    dtype=np.float64,
                )
            )

        update_mc_metrics(
            acc=acc,
            samples_all=samples_all,
            target=target,
        )

    single_rows = []
    mean_stochastic_pass_seconds = (
        stochastic_forward_seconds / 100.0
    )

    for var in VARIABLES:
        single_rows.append({
            "dropout_rate_string": rate_string,
            "dropout_rate": dropout_rate,
            "dropout_placement": "all",
            "inference_mode": "single_stochastic_pass",
            "variable": var,
            "validation_points": n_val,
            "rmse": math.sqrt(
                single_sq[var] / float(n_val)
            ),
            "mae": (
                single_abs[var] / float(n_val)
            ),
            "forward_seconds_estimated": (
                mean_stochastic_pass_seconds
            ),
            "inference_seed": inference_seed,
        })

    mc_rows = finalize_mc_metrics(
        acc=acc,
        rate_string=rate_string,
        dropout_rate=dropout_rate,
        stochastic_forward_seconds=stochastic_forward_seconds,
    )

    return (
        deterministic_rows,
        single_rows,
        mc_rows,
    )


def make_compact_summary(
    deterministic_df: pd.DataFrame,
    single_df: pd.DataFrame,
    mc_df: pd.DataFrame,
):
    rows = []

    for rate in RATE_STRINGS:
        det = deterministic_df[
            deterministic_df[
                "dropout_rate_string"
            ] == rate
        ]
        single = single_df[
            single_df[
                "dropout_rate_string"
            ] == rate
        ]
        mc100 = mc_df[
            (mc_df["dropout_rate_string"] == rate)
            & (mc_df["mc_samples"] == 100)
        ]

        def get(frame, var, col):
            return float(
                frame.loc[
                    frame["variable"] == var,
                    col,
                ].iloc[0]
            )

        rows.append({
            "dropout_rate": float(rate),
            "dropout_placement": "all",
            "dropout_off_u_rmse": get(
                det, "u", "rmse"
            ),
            "dropout_off_v_rmse": get(
                det, "v", "rmse"
            ),
            "dropout_off_uv_rmse_mean": 0.5 * (
                get(det, "u", "rmse")
                + get(det, "v", "rmse")
            ),
            "single_u_rmse": get(
                single, "u", "rmse"
            ),
            "single_v_rmse": get(
                single, "v", "rmse"
            ),
            "mc100_u_rmse": get(
                mc100, "u", "mc_mean_rmse"
            ),
            "mc100_v_rmse": get(
                mc100, "v", "mc_mean_rmse"
            ),
            "mc100_uv_rmse_mean": 0.5 * (
                get(mc100, "u", "mc_mean_rmse")
                + get(mc100, "v", "mc_mean_rmse")
            ),
            "mc100_u_picp95": get(
                mc100, "u", "picp_95"
            ),
            "mc100_v_picp95": get(
                mc100, "v", "picp_95"
            ),
            "mc100_u_mpiw95": get(
                mc100, "u", "mpiw_95"
            ),
            "mc100_v_mpiw95": get(
                mc100, "v", "mpiw_95"
            ),
            "mc100_u_calibration_error": get(
                mc100,
                "u",
                "mean_absolute_calibration_error",
            ),
            "mc100_v_calibration_error": get(
                mc100,
                "v",
                "mean_absolute_calibration_error",
            ),
            "mc100_u_crps": get(
                mc100, "u", "empirical_crps"
            ),
            "mc100_v_crps": get(
                mc100, "v", "empirical_crps"
            ),
            "mc100_u_pearson_error_uncertainty": get(
                mc100,
                "u",
                "pearson_abs_error_vs_std",
            ),
            "mc100_v_pearson_error_uncertainty": get(
                mc100,
                "v",
                "pearson_abs_error_vs_std",
            ),
            "mc100_u_spearman_error_uncertainty": get(
                mc100,
                "u",
                "spearman_abs_error_vs_std",
            ),
            "mc100_v_spearman_error_uncertainty": get(
                mc100,
                "v",
                "spearman_abs_error_vs_std",
            ),
            "mc100_forward_seconds": get(
                mc100,
                "u",
                "mc100_stochastic_forward_seconds",
            ),
        })

    return pd.DataFrame(rows)


def main():
    args = parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    protocol_root = Path(args.protocol_root)
    validation_data, protocol_summary = (
        load_validation_only(
            data_path=Path(args.data_path),
            protocol_root=protocol_root,
        )
    )

    output = Path(args.output_dir)
    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = get_device()
    root = Path(args.root)

    print("=" * 96)
    print("Cylinder dropout-rate MC validation evaluation")
    print(f"Validation points: {len(validation_data)}")
    print("Held-out test set: NOT LOADED")
    print(f"Device: {device}")
    print(
        "MC counts:",
        MC_COUNTS,
        "(nested prefixes of the same 100 stochastic draws)",
    )
    print(
        "Uncertainty interpretation: approximate epistemic only"
    )
    print("=" * 96)

    deterministic_rows = []
    single_rows = []
    mc_rows = []

    for rate in RATE_STRINGS:
        print(
            f"\nEvaluating dropout p={rate}, placement=all ..."
        )

        det, single, mc = evaluate_one_rate(
            rate_string=rate,
            validation_data=validation_data,
            root=root,
            device=device,
            batch_size=args.batch_size,
            inference_seed=args.inference_seed,
        )

        deterministic_rows.extend(det)
        single_rows.extend(single)
        mc_rows.extend(mc)

    deterministic_df = pd.DataFrame(
        deterministic_rows
    )
    single_df = pd.DataFrame(
        single_rows
    )
    mc_df = pd.DataFrame(
        mc_rows
    )

    deterministic_df.to_csv(
        output / "dropout_off_validation_metrics.csv",
        index=False,
    )
    single_df.to_csv(
        output / "single_stochastic_validation_metrics.csv",
        index=False,
    )
    mc_df.to_csv(
        output / "mc_convergence_calibration_metrics.csv",
        index=False,
    )

    compact = make_compact_summary(
        deterministic_df,
        single_df,
        mc_df,
    )
    compact.to_csv(
        output / "dropout_rate_mc100_compact_summary.csv",
        index=False,
    )

    readme = f"""Cylinder dropout-rate MC validation evaluation

Protocol:
    {protocol_root}

Validation points:
    {len(validation_data)}

Held-out test:
    NOT LOADED.

Dropout rates:
    {RATE_STRINGS}

Dropout placement:
    all

MC sample counts:
    {MC_COUNTS}

MC-count convergence uses nested prefixes of the same 100 stochastic draws.

Uncertainty interpretation:
    approximate epistemic uncertainty only.
    No aleatoric observation-noise model is included.

Calibration:
    empirical central interval coverage is measured at nominal 50%, 80%, 90%,
    and 95% levels. Mean absolute calibration error is the mean absolute
    difference between empirical and nominal coverage over those four levels.

Proper scoring rule:
    empirical-ensemble CRPS.

Error-uncertainty association:
    Pearson and Spearman correlations between pointwise absolute error of the
    MC predictive mean and MC predictive standard deviation.

Inference timing:
    MC100 forward time contains stochastic forward passes only. Per-N inference
    time in the long-format CSV is estimated from the mean time per stochastic
    pass in that same MC100 run.
"""
    (output / "README_MC_VALIDATION.txt").write_text(
        readme,
        encoding="utf-8",
    )

    show_cols = [
        "dropout_rate",
        "dropout_off_uv_rmse_mean",
        "single_u_rmse",
        "single_v_rmse",
        "mc100_uv_rmse_mean",
        "mc100_u_picp95",
        "mc100_v_picp95",
        "mc100_u_mpiw95",
        "mc100_v_mpiw95",
        "mc100_u_calibration_error",
        "mc100_v_calibration_error",
        "mc100_u_crps",
        "mc100_v_crps",
        "mc100_u_spearman_error_uncertainty",
        "mc100_v_spearman_error_uncertainty",
    ]

    print(
        "\n=== Dropout-rate MC100 validation summary ==="
    )
    print(
        compact[show_cols].to_string(
            index=False
        )
    )

    print(
        "\nSaved:",
        (
            output
            / "dropout_rate_mc100_compact_summary.csv"
        ).resolve(),
    )
    print("\nHeld-out test set was NOT loaded.")


if __name__ == "__main__":
    main()
