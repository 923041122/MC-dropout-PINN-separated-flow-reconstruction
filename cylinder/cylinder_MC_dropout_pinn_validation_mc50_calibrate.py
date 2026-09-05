
from __future__ import annotations
import argparse, math, time
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
    p.add_argument("--data-path", default="./2d_cylinder_Re3900_100x100_kw_sst.mat")
    p.add_argument("--protocol-root", default="./results_cylinder_protocol_v1_1/protocol")
    p.add_argument("--checkpoint", default="./cylinder_bpinn_final_p0002_all/models/bpinn_dropout.pth")
    p.add_argument("--output-dir", default="./cylinder_bpinn_final_p0002_all/validation_mc50_calibration")
    p.add_argument("--batch-size", type=int, default=5000)
    p.add_argument("--inference-seed", type=int, default=9091)
    p.add_argument("--scale-min", type=float, default=0.50)
    p.add_argument("--scale-max", type=float, default=8.00)
    p.add_argument("--scale-step", type=float, default=0.005)
    return p.parse_args()

def safe_load_state(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)

def load_validation_only(data_path, protocol_root):
    idx_path = protocol_root / "protocol_indices.npz"
    summary_path = protocol_root / "protocol_summary.csv"
    protocol_npz = np.load(idx_path)
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

def required_scale_for_coverage(y, mean, lo, hi):
    eps = 1e-12
    left_width = np.maximum(mean - lo, eps)
    right_width = np.maximum(hi - mean, eps)
    req = np.empty_like(y, dtype=np.float64)
    left = y < mean
    req[left] = (mean[left] - y[left]) / left_width[left]
    req[~left] = (y[~left] - mean[~left]) / right_width[~left]
    return np.maximum(req, 0.0)

def fit_scale_factor(required_by_nominal, scale_min, scale_max, scale_step):
    scales = np.arange(scale_min, scale_max + 0.5 * scale_step, scale_step)
    cov_cols = []
    sorted_req = {}
    nominals = list(required_by_nominal.keys())
    for nominal in nominals:
        sr = np.sort(required_by_nominal[nominal])
        sorted_req[nominal] = sr
        cov_cols.append(np.searchsorted(sr, scales, side="right") / float(len(sr)))
    coverage_matrix = np.stack(cov_cols, axis=1)
    target = np.array(nominals, dtype=float)[None, :]
    cal_err = np.mean(np.abs(coverage_matrix - target), axis=1)
    best_idx = int(np.argmin(cal_err))
    best_scale = float(scales[best_idx])
    fitted = {
        nominal: float(np.searchsorted(sorted_req[nominal], best_scale, side="right") / float(len(sorted_req[nominal])))
        for nominal in nominals
    }
    return best_scale, float(cal_err[best_idx]), fitted

def main():
    args = parse_args()
    validation_data = load_validation_only(Path(args.data_path), Path(args.protocol_root))
    device = get_device()

    model = PlacementPINNNet(
        LAYER_MAT_PSI,
        dropout_rate=DROPOUT_RATE,
        dropout_placement=DROPOUT_PLACEMENT,
    ).to(device)

    checkpoint = Path(args.checkpoint)
    model.load_state_dict(safe_load_state(checkpoint, device), strict=True)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    n_val = len(validation_data)
    print("=" * 96)
    print("Final cylinder B-PINN validation MC50 + calibration")
    print(f"Validation points: {n_val}")
    print("Held-out test set: NOT LOADED")
    print(f"Dropout rate: {DROPOUT_RATE}")
    print(f"Dropout placement: {DROPOUT_PLACEMENT}")
    print(f"MC samples: {MC_N}")
    print("=" * 96)

    det_sq = np.zeros(3); det_abs = np.zeros(3)
    single_sq = np.zeros(3); single_abs = np.zeros(3)
    mc_sq = np.zeros(3); mc_abs = np.zeros(3)
    std_sum = np.zeros(3); crps_sum = np.zeros(3)

    targets_all, means_all, stds_all = [], [], []
    interval_los = {n: [] for n in NOMINALS}
    interval_his = {n: [] for n in NOMINALS}

    torch.manual_seed(args.inference_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.inference_seed)

    for start in range(0, n_val, args.batch_size):
        batch = validation_data[start:start + args.batch_size]
        target = batch[:, 3:6].numpy().astype(np.float64)

        det_pred = predict(model, batch, device, stochastic=False).astype(np.float64)
        det_err = det_pred - target
        det_sq += np.sum(det_err**2, axis=0)
        det_abs += np.sum(np.abs(det_err), axis=0)

        samples = []
        for _ in range(MC_N):
            pred = predict(model, batch, device, stochastic=True)
            samples.append(pred.astype(np.float32))
        samples = np.stack(samples, axis=0)

        single = samples[0].astype(np.float64)
        single_err = single - target
        single_sq += np.sum(single_err**2, axis=0)
        single_abs += np.sum(np.abs(single_err), axis=0)

        mc_mean = np.mean(samples, axis=0, dtype=np.float64)
        mc_std = np.std(samples, axis=0, ddof=1, dtype=np.float64)

        mc_err = mc_mean - target
        mc_sq += np.sum(mc_err**2, axis=0)
        mc_abs += np.sum(np.abs(mc_err), axis=0)
        std_sum += np.sum(mc_std, axis=0)

        for j in range(3):
            crps_sum[j] += float(np.sum(empirical_crps(samples[:, :, j].astype(np.float64), target[:, j])))

        targets_all.append(target)
        means_all.append(mc_mean)
        stds_all.append(mc_std)

        for nominal in NOMINALS:
            alpha = 1.0 - nominal
            lo = np.quantile(samples, alpha / 2.0, axis=0).astype(np.float64)
            hi = np.quantile(samples, 1.0 - alpha / 2.0, axis=0).astype(np.float64)
            interval_los[nominal].append(lo)
            interval_his[nominal].append(hi)

    target_all = np.concatenate(targets_all, axis=0)
    mean_all = np.concatenate(means_all, axis=0)
    std_all = np.concatenate(stds_all, axis=0)
    lo_all = {n: np.concatenate(interval_los[n], axis=0) for n in NOMINALS}
    hi_all = {n: np.concatenate(interval_his[n], axis=0) for n in NOMINALS}

    raw_rows, factor_rows, calibrated_rows = [], [], []

    for j, var in enumerate(VARIABLES):
        abs_err = np.abs(mean_all[:, j] - target_all[:, j])
        unc = std_all[:, j]
        pearson = float(pearsonr(abs_err, unc).statistic)
        spearman = float(spearmanr(abs_err, unc).statistic)

        raw_cov, raw_width, required = {}, {}, {}
        for nominal in NOMINALS:
            lo = lo_all[nominal][:, j]
            hi = hi_all[nominal][:, j]
            y = target_all[:, j]
            mu = mean_all[:, j]
            raw_cov[nominal] = float(np.mean((y >= lo) & (y <= hi)))
            raw_width[nominal] = float(np.mean(hi - lo))
            required[nominal] = required_scale_for_coverage(y, mu, lo, hi)

        raw_cal_error = float(np.mean([abs(raw_cov[n] - n) for n in NOMINALS]))
        best_scale, fitted_error, fitted_cov = fit_scale_factor(
            required, args.scale_min, args.scale_max, args.scale_step
        )

        raw_rows.append({
            "variable": var,
            "validation_points": n_val,
            "dropout_off_rmse": math.sqrt(det_sq[j] / n_val),
            "single_stochastic_rmse": math.sqrt(single_sq[j] / n_val),
            "mc50_mean_rmse": math.sqrt(mc_sq[j] / n_val),
            "mean_epistemic_std": std_sum[j] / n_val,
            "raw_picp_95": raw_cov[0.95],
            "raw_mpiw_95": raw_width[0.95],
            "raw_mean_absolute_calibration_error": raw_cal_error,
            "empirical_crps": crps_sum[j] / n_val,
            "pearson_abs_error_vs_std": pearson,
            "spearman_abs_error_vs_std": spearman,
        })

        factor_rows.append({
            "variable": var,
            "interval_scale_factor": best_scale,
            "validation_calibration_error_after_scaling": fitted_error,
            "uncertainty_type": "approximate_epistemic_only",
        })

        calibrated_rows.append({
            "variable": var,
            "interval_scale_factor": best_scale,
            "calibrated_picp_50": fitted_cov[0.50],
            "calibrated_picp_80": fitted_cov[0.80],
            "calibrated_picp_90": fitted_cov[0.90],
            "calibrated_picp_95": fitted_cov[0.95],
            "calibrated_mpiw_95": best_scale * raw_width[0.95],
            "calibrated_mean_absolute_calibration_error": fitted_error,
            "spearman_abs_error_vs_std_unchanged": spearman,
        })

    raw_df = pd.DataFrame(raw_rows)
    factor_df = pd.DataFrame(factor_rows)
    calibrated_df = pd.DataFrame(calibrated_rows)

    raw_df.to_csv(output / "final_bpinn_validation_mc50_raw_metrics.csv", index=False)
    factor_df.to_csv(output / "validation_fitted_interval_scale_factors.csv", index=False)
    calibrated_df.to_csv(output / "final_bpinn_validation_calibrated_intervals.csv", index=False)

    print("\n=== Final B-PINN validation: raw MC50 metrics ===")
    print(raw_df.to_string(index=False))
    print("\n=== Validation-fitted interval scale factors ===")
    print(factor_df.to_string(index=False))
    print("\n=== Calibrated validation intervals ===")
    print(calibrated_df.to_string(index=False))
    print(f"\nSaved: {output.resolve()}")
    print("Held-out test set was NOT loaded.")

if __name__ == "__main__":
    main()
