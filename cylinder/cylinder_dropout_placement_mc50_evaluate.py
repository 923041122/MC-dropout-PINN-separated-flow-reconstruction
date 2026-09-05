
"""Validation-only MC50 evaluator for cylinder dropout-placement screening.

This script does NOT train models and does NOT load the held-out test set.

Compared placements at fixed p=0.002:
    input, middle, output, alternating, all

For every placement it reports:
- dropout-OFF deterministic prediction;
- one reproducible stochastic prediction;
- MC50 predictive mean;
- approximate epistemic standard deviation;
- empirical 50/80/90/95% interval coverage and width;
- mean absolute calibration error;
- empirical ensemble CRPS;
- Pearson/Spearman error-uncertainty correlation;
- stochastic inference time.

Uncertainty interpretation:
    approximate epistemic only.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

from benchmark_config import LAYER_MAT_PSI
from benchmark_tools import get_device
from pinn_model import read_2D_data
from pinn_model_dropout_ablation import PlacementPINNNet

from scipy.stats import pearsonr, spearmanr

P = 0.002
MC_N = 50
VARIABLES = ("u", "v", "p")
NOMINAL_COVERAGES = (0.50, 0.80, 0.90, 0.95)

RUNS = {
    "input": "dropout_placement_screen_input",
    "middle": "dropout_placement_screen_middle",
    "output": "dropout_placement_screen_output",
    "alternating": "dropout_placement_screen_alternating",
    "all": "dropout_rate_screen_p0.002",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", default="./2d_cylinder_Re3900_100x100_kw_sst.mat")
    p.add_argument("--protocol-root", default="./results_cylinder_protocol_v1_1/protocol")
    p.add_argument("--root", default=".")
    p.add_argument("--output-dir", default="./dropout_placement_mc50_validation")
    p.add_argument("--batch-size", type=int, default=5000)
    p.add_argument("--inference-seed", type=int, default=9091)
    return p.parse_args()


def safe_load_state(path: Path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def load_validation_only(data_path: Path, protocol_root: Path):
    idx_path = protocol_root / "protocol_indices.npz"
    summary_path = protocol_root / "protocol_summary.csv"
    if not idx_path.exists():
        raise FileNotFoundError(idx_path)
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    protocol_npz = np.load(idx_path)
    if "validation_indices" not in protocol_npz.files:
        raise KeyError("validation_indices not found.")
    val_idx = np.asarray(protocol_npz["validation_indices"], dtype=np.int64)

    summary = pd.read_csv(summary_path).iloc[0]
    x, y, t, u, v, p, _ = read_2D_data(str(data_path))
    full = torch.cat([x, y, t, u, v, p], dim=1).float()
    val = full[torch.from_numpy(val_idx)]

    if len(val) != int(summary["validation_points"]):
        raise RuntimeError("Validation point count does not match protocol.")
    return val


def predict(model, batch, device, stochastic):
    x = batch[:, 0:1].to(device)
    y = batch[:, 1:2].to(device)
    t = batch[:, 2:3].to(device)

    u, v, p = model.predict_fields_safe(
        x, y, t, create_graph=False, train_mode=bool(stochastic)
    )

    return np.stack(
        [
            u.detach().cpu().numpy().reshape(-1),
            v.detach().cpu().numpy().reshape(-1),
            p.detach().cpu().numpy().reshape(-1),
        ],
        axis=-1,
    )


def empirical_crps(samples, target):
    m = samples.shape[0]
    term1 = np.mean(np.abs(samples - target[None, :]), axis=0)
    ordered = np.sort(samples, axis=0)
    coeff = (2 * np.arange(1, m + 1, dtype=np.float64) - m - 1)[:, None]
    pair_term = np.sum(coeff * ordered, axis=0) / float(m * m)
    return term1 - pair_term


def evaluate_placement(
    placement,
    run_root,
    validation_data,
    device,
    batch_size,
    inference_seed,
):
    checkpoint = run_root / "models" / "bpinn_dropout.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    model = PlacementPINNNet(
        LAYER_MAT_PSI,
        dropout_rate=P,
        dropout_placement=placement,
    ).to(device)

    model.load_state_dict(safe_load_state(checkpoint, device), strict=True)

    n_val = len(validation_data)

    det_sq = np.zeros(3, dtype=np.float64)
    det_abs = np.zeros(3, dtype=np.float64)
    deterministic_seconds = 0.0

    for start in range(0, n_val, batch_size):
        batch = validation_data[start:start + batch_size]
        target = batch[:, 3:6].numpy()
        t0 = time.perf_counter()
        pred = predict(model, batch, device, stochastic=False)
        if device.type == "cuda":
            torch.cuda.synchronize()
        deterministic_seconds += time.perf_counter() - t0

        err = pred - target
        det_sq += np.sum(err**2, axis=0, dtype=np.float64)
        det_abs += np.sum(np.abs(err), axis=0, dtype=np.float64)

    torch.manual_seed(inference_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(inference_seed)

    single_sq = np.zeros(3, dtype=np.float64)
    single_abs = np.zeros(3, dtype=np.float64)
    mc_sq = np.zeros(3, dtype=np.float64)
    mc_abs = np.zeros(3, dtype=np.float64)
    std_sum = np.zeros(3, dtype=np.float64)
    crps_sum = np.zeros(3, dtype=np.float64)

    coverage_counts = {
        nominal: np.zeros(3, dtype=np.int64)
        for nominal in NOMINAL_COVERAGES
    }
    width_sums = {
        nominal: np.zeros(3, dtype=np.float64)
        for nominal in NOMINAL_COVERAGES
    }

    abs_err_vectors = {var: [] for var in VARIABLES}
    std_vectors = {var: [] for var in VARIABLES}
    stochastic_seconds = 0.0

    for start in range(0, n_val, batch_size):
        batch = validation_data[start:start + batch_size]
        target = batch[:, 3:6].numpy()

        samples: List[np.ndarray] = []
        for _ in range(MC_N):
            t0 = time.perf_counter()
            pred = predict(model, batch, device, stochastic=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            stochastic_seconds += time.perf_counter() - t0
            samples.append(pred.astype(np.float32))

        samples = np.stack(samples, axis=0)

        single_err = samples[0] - target
        single_sq += np.sum(single_err**2, axis=0, dtype=np.float64)
        single_abs += np.sum(np.abs(single_err), axis=0, dtype=np.float64)

        mean = np.mean(samples, axis=0)
        std = np.std(samples, axis=0, ddof=1)
        err = mean - target
        abs_err = np.abs(err)

        mc_sq += np.sum(err**2, axis=0, dtype=np.float64)
        mc_abs += np.sum(abs_err, axis=0, dtype=np.float64)
        std_sum += np.sum(std, axis=0, dtype=np.float64)

        for j in range(3):
            crps_sum[j] += float(
                np.sum(empirical_crps(samples[:, :, j], target[:, j]), dtype=np.float64)
            )

        for nominal in NOMINAL_COVERAGES:
            alpha = 1.0 - nominal
            lo = np.quantile(samples, alpha / 2.0, axis=0)
            hi = np.quantile(samples, 1.0 - alpha / 2.0, axis=0)
            covered = (target >= lo) & (target <= hi)

            coverage_counts[nominal] += np.sum(covered, axis=0)
            width_sums[nominal] += np.sum(hi - lo, axis=0, dtype=np.float64)

        for j, var in enumerate(VARIABLES):
            abs_err_vectors[var].append(abs_err[:, j].astype(np.float32))
            std_vectors[var].append(std[:, j].astype(np.float32))

    rows = []

    for j, var in enumerate(VARIABLES):
        abs_err_all = np.concatenate(abs_err_vectors[var]).astype(np.float64)
        std_all = np.concatenate(std_vectors[var]).astype(np.float64)

        pearson = float(pearsonr(abs_err_all, std_all).statistic)
        spearman = float(spearmanr(abs_err_all, std_all).statistic)

        coverages = {
            nominal: coverage_counts[nominal][j] / float(n_val)
            for nominal in NOMINAL_COVERAGES
        }
        widths = {
            nominal: width_sums[nominal][j] / float(n_val)
            for nominal in NOMINAL_COVERAGES
        }

        calibration_error = float(
            np.mean([abs(coverages[c] - c) for c in NOMINAL_COVERAGES])
        )

        rows.append({
            "dropout_placement": placement,
            "dropout_rate": P,
            "mc_samples": MC_N,
            "variable": var,
            "validation_points": n_val,
            "dropout_off_rmse": math.sqrt(det_sq[j] / float(n_val)),
            "dropout_off_mae": det_abs[j] / float(n_val),
            "single_stochastic_rmse": math.sqrt(single_sq[j] / float(n_val)),
            "single_stochastic_mae": single_abs[j] / float(n_val),
            "mc50_mean_rmse": math.sqrt(mc_sq[j] / float(n_val)),
            "mc50_mean_mae": mc_abs[j] / float(n_val),
            "mean_epistemic_std": std_sum[j] / float(n_val),
            "picp_50": coverages[0.50],
            "mpiw_50": widths[0.50],
            "picp_80": coverages[0.80],
            "mpiw_80": widths[0.80],
            "picp_90": coverages[0.90],
            "mpiw_90": widths[0.90],
            "picp_95": coverages[0.95],
            "mpiw_95": widths[0.95],
            "mean_absolute_calibration_error": calibration_error,
            "empirical_crps": crps_sum[j] / float(n_val),
            "pearson_abs_error_vs_std": pearson,
            "spearman_abs_error_vs_std": spearman,
            "deterministic_forward_seconds": deterministic_seconds,
            "mc50_stochastic_forward_seconds": stochastic_seconds,
            "mean_seconds_per_stochastic_pass": stochastic_seconds / MC_N,
            "uncertainty_type": "approximate_epistemic_only",
        })

    return rows


def compact_summary(long_df):
    rows = []

    for placement in RUNS:
        s = long_df[long_df["dropout_placement"] == placement]

        def get(var, col):
            return float(
                s.loc[s["variable"] == var, col].iloc[0]
            )

        rows.append({
            "dropout_placement": placement,
            "dropout_rate": P,
            "dropout_off_uv_rmse_mean": 0.5 * (
                get("u", "dropout_off_rmse") + get("v", "dropout_off_rmse")
            ),
            "single_uv_rmse_mean": 0.5 * (
                get("u", "single_stochastic_rmse") + get("v", "single_stochastic_rmse")
            ),
            "mc50_uv_rmse_mean": 0.5 * (
                get("u", "mc50_mean_rmse") + get("v", "mc50_mean_rmse")
            ),
            "mc50_p_rmse": get("p", "mc50_mean_rmse"),
            "u_picp95": get("u", "picp_95"),
            "v_picp95": get("v", "picp_95"),
            "u_mpiw95": get("u", "mpiw_95"),
            "v_mpiw95": get("v", "mpiw_95"),
            "u_calibration_error": get("u", "mean_absolute_calibration_error"),
            "v_calibration_error": get("v", "mean_absolute_calibration_error"),
            "u_crps": get("u", "empirical_crps"),
            "v_crps": get("v", "empirical_crps"),
            "u_spearman_error_uncertainty": get(
                "u", "spearman_abs_error_vs_std"
            ),
            "v_spearman_error_uncertainty": get(
                "v", "spearman_abs_error_vs_std"
            ),
            "mc50_forward_seconds": get(
                "u", "mc50_stochastic_forward_seconds"
            ),
        })

    return pd.DataFrame(rows).sort_values("mc50_uv_rmse_mean")


def main():
    args = parse_args()

    validation_data = load_validation_only(
        Path(args.data_path),
        Path(args.protocol_root),
    )

    device = get_device()
    root = Path(args.root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    print("=" * 96)
    print("Cylinder dropout-placement MC50 validation evaluation")
    print(f"Dropout rate: {P}")
    print(f"MC samples: {MC_N}")
    print(f"Validation points: {len(validation_data)}")
    print("Held-out test set: NOT LOADED")
    print(f"Device: {device}")
    print("=" * 96)

    rows = []

    for placement, run_name in RUNS.items():
        print(f"\nEvaluating placement={placement}, p={P} ...")
        rows.extend(
            evaluate_placement(
                placement=placement,
                run_root=root / run_name,
                validation_data=validation_data,
                device=device,
                batch_size=args.batch_size,
                inference_seed=args.inference_seed,
            )
        )

    long_df = pd.DataFrame(rows)
    long_df.to_csv(
        output / "dropout_placement_mc50_metrics_long.csv",
        index=False,
    )

    compact = compact_summary(long_df)
    compact.to_csv(
        output / "dropout_placement_mc50_compact_summary.csv",
        index=False,
    )

    show_cols = [
        "dropout_placement",
        "dropout_off_uv_rmse_mean",
        "single_uv_rmse_mean",
        "mc50_uv_rmse_mean",
        "mc50_p_rmse",
        "u_picp95",
        "v_picp95",
        "u_mpiw95",
        "v_mpiw95",
        "u_calibration_error",
        "v_calibration_error",
        "u_crps",
        "v_crps",
        "u_spearman_error_uncertainty",
        "v_spearman_error_uncertainty",
    ]

    print("\n=== Dropout-placement MC50 validation summary ===")
    print(compact[show_cols].to_string(index=False))

    (output / "README_PLACEMENT_MC50.txt").write_text(
        """Cylinder dropout-placement MC50 validation evaluation

Fixed:
- supervised ratio = 2%
- dropout probability = 0.002
- physics-loss weight = 0.1
- MC samples = 50
- common validation set
- held-out test set NOT loaded

Compared placements:
- input
- middle
- output
- alternating
- all

Uncertainty is approximate epistemic uncertainty only.
""",
        encoding="utf-8",
    )

    print(
        "\nSaved:",
        (output / "dropout_placement_mc50_compact_summary.csv").resolve(),
    )
    print("\nHeld-out test set was NOT loaded.")


if __name__ == "__main__":
    main()
