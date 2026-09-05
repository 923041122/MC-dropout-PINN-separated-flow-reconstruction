
"""Final held-out evaluation for the frozen cylinder MC-dropout model.

This is the FIRST stage that loads the held-out test partition.

Frozen before test access:
    dropout rate      = 0.002
    dropout placement = all hidden layers
    MC sample count   = 50
    interval scale factors = loaded from validation calibration CSV

No hyperparameter tuning is performed in this script.

Reported:
- dropout-OFF deterministic accuracy;
- single stochastic accuracy;
- MC50 predictive-mean accuracy;
- raw and validation-calibrated interval coverage/width;
- empirical CRPS;
- Pearson/Spearman error-uncertainty correlation;
- pressure raw and per-time gauge-aligned error/coverage;
- inference timing;
- machine-readable pointwise predictions.

Uncertainty is approximate epistemic uncertainty only.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

from benchmark_config import LAYER_MAT_PSI
from benchmark_tools import get_device
from pinn_model import read_2D_data
from pinn_model_dropout_ablation import PlacementPINNNet


VARIABLES = ("u", "v", "p")
NOMINALS = (0.50, 0.80, 0.90, 0.95)
MC_N = 50
DROPOUT_RATE = 0.002
DROPOUT_PLACEMENT = "all"


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
        "--checkpoint",
        default="./cylinder_bpinn_final_p0002_all/models/bpinn_dropout.pth",
    )
    p.add_argument(
        "--calibration-factors",
        default=(
            "./cylinder_bpinn_final_p0002_all/"
            "validation_mc50_calibration/"
            "validation_fitted_interval_scale_factors.csv"
        ),
    )
    p.add_argument(
        "--output-dir",
        default="./cylinder_bpinn_final_p0002_all/heldout_mc50_final",
    )
    p.add_argument("--batch-size", type=int, default=5000)
    p.add_argument("--inference-seed", type=int, default=9091)
    return p.parse_args()


def safe_load_state(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def load_test_only(data_path, protocol_root):
    idx_path = protocol_root / "protocol_indices.npz"
    summary_path = protocol_root / "protocol_summary.csv"

    if not idx_path.exists():
        raise FileNotFoundError(idx_path)
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    protocol_npz = np.load(idx_path)

    if "test_indices" not in protocol_npz.files:
        raise KeyError("test_indices not found in protocol_indices.npz")

    test_idx = np.asarray(
        protocol_npz["test_indices"],
        dtype=np.int64,
    )

    summary = pd.read_csv(summary_path).iloc[0]

    x, y, t, u, v, p, _ = read_2D_data(str(data_path))
    full = torch.cat([x, y, t, u, v, p], dim=1).float()
    test = full[torch.from_numpy(test_idx)]

    expected = int(summary["heldout_test_points"])
    if len(test) != expected:
        raise RuntimeError(
            f"Test count mismatch: got {len(test)}, expected {expected}"
        )

    return test, test_idx


def predict(model, batch, device, stochastic):
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


def empirical_crps(samples, target):
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


def main():
    args = parse_args()

    test_data, test_indices = load_test_only(
        Path(args.data_path),
        Path(args.protocol_root),
    )

    factor_df = pd.read_csv(args.calibration_factors)
    factors = {
        row["variable"]: float(row["interval_scale_factor"])
        for _, row in factor_df.iterrows()
    }

    required_vars = set(VARIABLES)
    if set(factors) != required_vars:
        raise RuntimeError(
            f"Calibration factor variables are {set(factors)}, "
            f"expected {required_vars}"
        )

    device = get_device()

    model = PlacementPINNNet(
        LAYER_MAT_PSI,
        dropout_rate=DROPOUT_RATE,
        dropout_placement=DROPOUT_PLACEMENT,
    ).to(device)

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    model.load_state_dict(
        safe_load_state(checkpoint, device),
        strict=True,
    )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    n_test = len(test_data)

    print("=" * 104)
    print("FINAL cylinder held-out MC50 evaluation")
    print(f"Held-out test points: {n_test}")
    print("Hyperparameter tuning after this point: NONE")
    print(f"Checkpoint: {checkpoint}")
    print(f"Dropout rate: {DROPOUT_RATE}")
    print(f"Dropout placement: {DROPOUT_PLACEMENT}")
    print(f"MC samples: {MC_N}")
    print(f"Calibration factors: {factors}")
    print(f"Device: {device}")
    print("Uncertainty type: approximate epistemic only")
    print("=" * 104)

    det_sq = np.zeros(3, dtype=np.float64)
    det_abs = np.zeros(3, dtype=np.float64)

    single_sq = np.zeros(3, dtype=np.float64)
    single_abs = np.zeros(3, dtype=np.float64)

    mc_sq = np.zeros(3, dtype=np.float64)
    mc_abs = np.zeros(3, dtype=np.float64)

    crps_sum = np.zeros(3, dtype=np.float64)

    deterministic_seconds = 0.0
    stochastic_seconds = 0.0

    targets_all = []
    coords_all = []
    det_all = []
    single_all = []
    means_all = []
    stds_all = []

    raw_los = {n: [] for n in NOMINALS}
    raw_his = {n: [] for n in NOMINALS}

    torch.manual_seed(args.inference_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.inference_seed)

    for start in range(0, n_test, args.batch_size):
        batch = test_data[start:start + args.batch_size]

        coords = batch[:, 0:3].numpy().astype(np.float64)
        target = batch[:, 3:6].numpy().astype(np.float64)

        t0 = time.perf_counter()
        det_pred = predict(
            model,
            batch,
            device,
            stochastic=False,
        ).astype(np.float64)
        if device.type == "cuda":
            torch.cuda.synchronize()
        deterministic_seconds += (
            time.perf_counter() - t0
        )

        det_err = det_pred - target
        det_sq += np.sum(
            det_err**2,
            axis=0,
            dtype=np.float64,
        )
        det_abs += np.sum(
            np.abs(det_err),
            axis=0,
            dtype=np.float64,
        )

        samples = []

        for _ in range(MC_N):
            t0 = time.perf_counter()

            pred = predict(
                model,
                batch,
                device,
                stochastic=True,
            )

            if device.type == "cuda":
                torch.cuda.synchronize()

            stochastic_seconds += (
                time.perf_counter() - t0
            )

            samples.append(
                pred.astype(np.float32)
            )

        samples = np.stack(
            samples,
            axis=0,
        )

        single = samples[0].astype(np.float64)

        single_err = single - target
        single_sq += np.sum(
            single_err**2,
            axis=0,
            dtype=np.float64,
        )
        single_abs += np.sum(
            np.abs(single_err),
            axis=0,
            dtype=np.float64,
        )

        mc_mean = np.mean(
            samples,
            axis=0,
            dtype=np.float64,
        )

        mc_std = np.std(
            samples,
            axis=0,
            ddof=1,
            dtype=np.float64,
        )

        mc_err = mc_mean - target
        mc_sq += np.sum(
            mc_err**2,
            axis=0,
            dtype=np.float64,
        )
        mc_abs += np.sum(
            np.abs(mc_err),
            axis=0,
            dtype=np.float64,
        )

        for j in range(3):
            crps_sum[j] += float(
                np.sum(
                    empirical_crps(
                        samples[:, :, j].astype(np.float64),
                        target[:, j],
                    ),
                    dtype=np.float64,
                )
            )

        targets_all.append(target)
        coords_all.append(coords)
        det_all.append(det_pred)
        single_all.append(single)
        means_all.append(mc_mean)
        stds_all.append(mc_std)

        for nominal in NOMINALS:
            alpha = 1.0 - nominal

            lo = np.quantile(
                samples,
                alpha / 2.0,
                axis=0,
            ).astype(np.float64)

            hi = np.quantile(
                samples,
                1.0 - alpha / 2.0,
                axis=0,
            ).astype(np.float64)

            raw_los[nominal].append(lo)
            raw_his[nominal].append(hi)

    target = np.concatenate(targets_all, axis=0)
    coords = np.concatenate(coords_all, axis=0)
    det = np.concatenate(det_all, axis=0)
    single = np.concatenate(single_all, axis=0)
    mean = np.concatenate(means_all, axis=0)
    std = np.concatenate(stds_all, axis=0)

    raw_lo = {
        n: np.concatenate(raw_los[n], axis=0)
        for n in NOMINALS
    }

    raw_hi = {
        n: np.concatenate(raw_his[n], axis=0)
        for n in NOMINALS
    }

    summary_rows = []

    for j, var in enumerate(VARIABLES):
        abs_err = np.abs(
            mean[:, j] - target[:, j]
        )

        pearson = float(
            pearsonr(
                abs_err,
                std[:, j],
            ).statistic
        )

        spearman = float(
            spearmanr(
                abs_err,
                std[:, j],
            ).statistic
        )

        row = {
            "variable": var,
            "test_points": n_test,
            "dropout_off_rmse": math.sqrt(
                det_sq[j] / float(n_test)
            ),
            "dropout_off_mae": (
                det_abs[j] / float(n_test)
            ),
            "single_stochastic_rmse": math.sqrt(
                single_sq[j] / float(n_test)
            ),
            "single_stochastic_mae": (
                single_abs[j] / float(n_test)
            ),
            "mc50_mean_rmse": math.sqrt(
                mc_sq[j] / float(n_test)
            ),
            "mc50_mean_mae": (
                mc_abs[j] / float(n_test)
            ),
            "mean_epistemic_std": float(
                np.mean(std[:, j])
            ),
            "empirical_crps": (
                crps_sum[j] / float(n_test)
            ),
            "pearson_abs_error_vs_std": pearson,
            "spearman_abs_error_vs_std": spearman,
            "validation_frozen_scale_factor": factors[var],
        }

        raw_cal_errors = []
        calibrated_cal_errors = []

        for nominal in NOMINALS:
            lo = raw_lo[nominal][:, j]
            hi = raw_hi[nominal][:, j]

            raw_covered = (
                (target[:, j] >= lo)
                & (target[:, j] <= hi)
            )

            raw_picp = float(
                np.mean(raw_covered)
            )

            raw_mpiw = float(
                np.mean(hi - lo)
            )

            scale = factors[var]

            cal_lo = (
                mean[:, j]
                - scale * (
                    mean[:, j] - lo
                )
            )

            cal_hi = (
                mean[:, j]
                + scale * (
                    hi - mean[:, j]
                )
            )

            cal_picp = float(
                np.mean(
                    (target[:, j] >= cal_lo)
                    & (target[:, j] <= cal_hi)
                )
            )

            cal_mpiw = float(
                np.mean(
                    cal_hi - cal_lo
                )
            )

            row[f"raw_picp_{int(nominal*100)}"] = raw_picp
            row[f"raw_mpiw_{int(nominal*100)}"] = raw_mpiw
            row[f"calibrated_picp_{int(nominal*100)}"] = cal_picp
            row[f"calibrated_mpiw_{int(nominal*100)}"] = cal_mpiw

            raw_cal_errors.append(
                abs(raw_picp - nominal)
            )

            calibrated_cal_errors.append(
                abs(cal_picp - nominal)
            )

        row[
            "raw_mean_absolute_calibration_error"
        ] = float(
            np.mean(raw_cal_errors)
        )

        row[
            "calibrated_mean_absolute_calibration_error"
        ] = float(
            np.mean(calibrated_cal_errors)
        )

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    # --------------------------------------------------------------
    # Pressure gauge-invariant evaluation.
    # A single additive offset is fitted independently at each test time
    # using the held-out reference pressure. The same offset is applied to
    # all uncertainty bounds, preserving interval widths.
    # --------------------------------------------------------------
    times = coords[:, 2]
    unique_times = np.unique(times)

    p_target = target[:, 2]
    p_mean = mean[:, 2].copy()
    p_det = det[:, 2].copy()
    p_single = single[:, 2].copy()

    p_mean_aligned = p_mean.copy()
    p_det_aligned = p_det.copy()
    p_single_aligned = p_single.copy()

    p_cal95_lo = (
        mean[:, 2]
        - factors["p"] * (
            mean[:, 2]
            - raw_lo[0.95][:, 2]
        )
    )

    p_cal95_hi = (
        mean[:, 2]
        + factors["p"] * (
            raw_hi[0.95][:, 2]
            - mean[:, 2]
        )
    )

    p_cal95_lo_aligned = p_cal95_lo.copy()
    p_cal95_hi_aligned = p_cal95_hi.copy()

    gauge_rows = []

    for tv in unique_times:
        mask = np.isclose(
            times,
            tv,
            rtol=0.0,
            atol=1e-7,
        )

        # Gauge-invariant least-squares constant for each prediction mode.
        off_mean = float(
            np.mean(
                p_target[mask] - p_mean[mask]
            )
        )

        off_det = float(
            np.mean(
                p_target[mask] - p_det[mask]
            )
        )

        off_single = float(
            np.mean(
                p_target[mask] - p_single[mask]
            )
        )

        p_mean_aligned[mask] += off_mean
        p_det_aligned[mask] += off_det
        p_single_aligned[mask] += off_single

        p_cal95_lo_aligned[mask] += off_mean
        p_cal95_hi_aligned[mask] += off_mean

        gauge_rows.append({
            "time": tv,
            "test_points_at_time": int(
                np.sum(mask)
            ),
            "mc50_pressure_offset": off_mean,
            "dropout_off_pressure_offset": off_det,
            "single_pressure_offset": off_single,
        })

    p_gauge_summary = pd.DataFrame([{
        "variable": "p_gauge_aligned_per_time",
        "test_points": n_test,
        "dropout_off_rmse": float(
            np.sqrt(
                np.mean(
                    (
                        p_det_aligned
                        - p_target
                    ) ** 2
                )
            )
        ),
        "single_stochastic_rmse": float(
            np.sqrt(
                np.mean(
                    (
                        p_single_aligned
                        - p_target
                    ) ** 2
                )
            )
        ),
        "mc50_mean_rmse": float(
            np.sqrt(
                np.mean(
                    (
                        p_mean_aligned
                        - p_target
                    ) ** 2
                )
            )
        ),
        "mc50_mean_mae": float(
            np.mean(
                np.abs(
                    p_mean_aligned
                    - p_target
                )
            )
        ),
        "calibrated_picp_95": float(
            np.mean(
                (
                    p_target
                    >= p_cal95_lo_aligned
                )
                & (
                    p_target
                    <= p_cal95_hi_aligned
                )
            )
        ),
        "calibrated_mpiw_95": float(
            np.mean(
                p_cal95_hi_aligned
                - p_cal95_lo_aligned
            )
        ),
    }])

    summary.to_csv(
        output / "final_heldout_mc50_summary.csv",
        index=False,
    )

    p_gauge_summary.to_csv(
        output / "final_pressure_gauge_aligned_summary.csv",
        index=False,
    )

    pd.DataFrame(gauge_rows).to_csv(
        output / "pressure_gauge_offsets_by_time.csv",
        index=False,
    )

    # Save pointwise outputs sufficient for later figures/region analyses.
    pointwise = pd.DataFrame({
        "global_index": test_indices,
        "x": coords[:, 0],
        "y": coords[:, 1],
        "t": coords[:, 2],
        "u_ref": target[:, 0],
        "v_ref": target[:, 1],
        "p_ref": target[:, 2],
        "u_dropout_off": det[:, 0],
        "v_dropout_off": det[:, 1],
        "p_dropout_off": det[:, 2],
        "u_single_stochastic": single[:, 0],
        "v_single_stochastic": single[:, 1],
        "p_single_stochastic": single[:, 2],
        "u_mc50_mean": mean[:, 0],
        "v_mc50_mean": mean[:, 1],
        "p_mc50_mean": mean[:, 2],
        "u_mc50_std": std[:, 0],
        "v_mc50_std": std[:, 1],
        "p_mc50_std": std[:, 2],
        "p_mc50_mean_gauge_aligned": p_mean_aligned,
    })

    for j, var in enumerate(VARIABLES):
        lo = raw_lo[0.95][:, j]
        hi = raw_hi[0.95][:, j]
        scale = factors[var]

        pointwise[f"{var}_raw95_lo"] = lo
        pointwise[f"{var}_raw95_hi"] = hi
        pointwise[f"{var}_cal95_lo"] = (
            mean[:, j]
            - scale * (
                mean[:, j] - lo
            )
        )
        pointwise[f"{var}_cal95_hi"] = (
            mean[:, j]
            + scale * (
                hi - mean[:, j]
            )
        )

    pointwise[
        "p_cal95_lo_gauge_aligned"
    ] = p_cal95_lo_aligned

    pointwise[
        "p_cal95_hi_gauge_aligned"
    ] = p_cal95_hi_aligned

    pointwise.to_csv(
        output / "final_heldout_pointwise_predictions.csv",
        index=False,
    )

    cost = pd.DataFrame([{
        "evaluation_grid": "fixed held-out spatial blocks over all 100 time snapshots",
        "evaluation_point_count": n_test,
        "dropout_off_evaluation_seconds": deterministic_seconds,
        "mc50_stochastic_forward_seconds": stochastic_seconds,
        "mean_seconds_per_stochastic_pass": (
            stochastic_seconds / MC_N
        ),
        "mc_samples": MC_N,
        "dropout_rate": DROPOUT_RATE,
        "dropout_placement": DROPOUT_PLACEMENT,
    }])

    cost.to_csv(
        output / "final_heldout_inference_cost.csv",
        index=False,
    )

    print("\n=== FINAL held-out MC50 metrics ===")
    print(
        summary[
            [
                "variable",
                "dropout_off_rmse",
                "single_stochastic_rmse",
                "mc50_mean_rmse",
                "mean_epistemic_std",
                "raw_picp_95",
                "calibrated_picp_95",
                "calibrated_mpiw_95",
                "raw_mean_absolute_calibration_error",
                "calibrated_mean_absolute_calibration_error",
                "empirical_crps",
                "spearman_abs_error_vs_std",
            ]
        ].to_string(index=False)
    )

    print("\n=== FINAL gauge-invariant pressure evaluation ===")
    print(
        p_gauge_summary.to_string(
            index=False
        )
    )

    print("\n=== Inference cost ===")
    print(
        cost.to_string(
            index=False
        )
    )

    print(f"\nSaved: {output.resolve()}")
    print(
        "FINAL TEST COMPLETE. "
        "No hyperparameter changes should be made using these results."
    )


if __name__ == "__main__":
    main()
