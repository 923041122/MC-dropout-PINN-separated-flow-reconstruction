"""
Final held-out evaluator for the frozen NASA-hump observation-only NN.

Evaluation-only safeguards
--------------------------
- Loads the fixed v1.2 protocol manifests.
- Uses only rows explicitly labeled "test".
- Performs no training, calibration, or hyperparameter selection.
- Evaluates held-out strict-interior u/v reconstruction.
- Evaluates held-out wall Cp using Cp = cp_scale * p.
- Reports both raw Cp error and a gauge-aligned diagnostic.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hump_data_only_train import DirectUVPNet


def parse_args():
    p = argparse.ArgumentParser(
        description="Final held-out evaluation of NASA hump observation-only NN."
    )
    p.add_argument(
        "--protocol-root",
        default="./hump_results_revision_r02/protocol",
    )
    p.add_argument(
        "--checkpoint",
        default="./baseline_suite/nasa_hump/data_only/formal/models/data_only_nn.pth",
    )
    p.add_argument(
        "--results-root",
        default="./baseline_suite/nasa_hump/data_only/formal",
    )
    p.add_argument("--batch-size", type=int, default=20000)
    p.add_argument("--cp-scale", type=float, default=2.0)
    p.add_argument("--hidden-layers", type=int, default=10)
    p.add_argument("--hidden-width", type=int, default=100)
    return p.parse_args()


def safe_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def metric_dict(pred, truth):
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    truth = np.asarray(truth, dtype=np.float64).reshape(-1)
    mask = np.isfinite(pred) & np.isfinite(truth)
    pred = pred[mask]
    truth = truth[mask]
    if truth.size == 0:
        raise RuntimeError("No finite evaluation points.")
    err = pred - truth
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    rel = float(np.linalg.norm(err) / max(np.linalg.norm(truth), 1e-12))
    span = float(np.max(truth) - np.min(truth))
    return {
        "n_points": int(truth.size),
        "mae": mae,
        "rmse": rmse,
        "relative_l2": rel,
        "nrmse_range": float(rmse / max(span, 1e-12)),
        "max_abs_error": float(np.max(np.abs(err))),
    }


@torch.no_grad()
def predict(model, xy_np, device, batch_size):
    out_u, out_v, out_p = [], [], []
    start = time.perf_counter()
    for i in range(0, len(xy_np), batch_size):
        xy = torch.tensor(
            xy_np[i:i + batch_size],
            dtype=torch.float32,
            device=device,
        )
        u, v, p = model(xy)
        out_u.append(u.cpu().numpy().reshape(-1))
        out_v.append(v.cpu().numpy().reshape(-1))
        out_p.append(p.cpu().numpy().reshape(-1))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return (
        np.concatenate(out_u),
        np.concatenate(out_v),
        np.concatenate(out_p),
        elapsed,
    )


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    protocol_root = Path(args.protocol_root)
    checkpoint = Path(args.checkpoint)
    results_root = Path(args.results_root)
    output_dir = results_root / "heldout_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    uv_path = protocol_root / "interior_velocity_split_manifest.csv"
    cp_path = protocol_root / "cp_split_manifest.csv"
    summary_path = protocol_root / "protocol_summary.csv"

    for path in (uv_path, cp_path, summary_path, checkpoint):
        if not path.exists():
            raise FileNotFoundError(f"Missing required artifact: {path}")

    uv = pd.read_csv(uv_path)
    cp = pd.read_csv(cp_path)
    summary = pd.read_csv(summary_path).iloc[0]

    required_uv = {"original_index", "x", "y", "u", "v", "split"}
    required_cp = {"original_index", "x", "y", "cp", "split"}
    if not required_uv.issubset(uv.columns):
        raise KeyError(f"Velocity manifest missing: {sorted(required_uv - set(uv.columns))}")
    if not required_cp.issubset(cp.columns):
        raise KeyError(f"Cp manifest missing: {sorted(required_cp - set(cp.columns))}")

    # Protocol integrity checks.
    expected = {
        "interior_train_points": 95,
        "interior_validation_points": 672,
        "interior_test_points": 987,
        "cp_train_points": 9,
        "cp_validation_points": 72,
        "cp_test_points": 96,
    }
    for key, value in expected.items():
        if key in summary.index and int(summary[key]) != value:
            raise RuntimeError(
                f"Protocol mismatch for {key}: found {int(summary[key])}, expected {value}"
            )

    uv_counts = uv["split"].astype(str).value_counts().to_dict()
    cp_counts = cp["split"].astype(str).value_counts().to_dict()
    if int(uv_counts.get("train", 0)) != 95:
        raise RuntimeError("Velocity training split is not the frozen 95-point split.")
    if int(uv_counts.get("validation", 0)) != 672:
        raise RuntimeError("Velocity validation split is not the frozen 672-point split.")
    if int(uv_counts.get("test", 0)) != 987:
        raise RuntimeError("Velocity held-out split is not the frozen 987-point split.")
    if int(cp_counts.get("train", 0)) != 9:
        raise RuntimeError("Cp training split is not the frozen 9-point split.")
    if int(cp_counts.get("validation", 0)) != 72:
        raise RuntimeError("Cp validation split is not the frozen 72-point split.")
    if int(cp_counts.get("test", 0)) != 96:
        raise RuntimeError("Cp held-out split is not the frozen 96-point split.")

    uv_test = uv[uv["split"].astype(str) == "test"].copy()
    cp_test = cp[cp["split"].astype(str) == "test"].copy()

    model = DirectUVPNet(
        hidden_layers=args.hidden_layers,
        hidden_width=args.hidden_width,
    ).to(device)
    model.load_state_dict(safe_load(checkpoint, device), strict=True)
    model.eval()

    # Strict-interior held-out velocity test.
    uv_xy = uv_test[["x", "y"]].to_numpy(dtype=np.float32)
    pu, pv, pp_uv, uv_seconds = predict(
        model, uv_xy, device, args.batch_size
    )

    velocity_rows = []
    for variable, pred in (("u", pu), ("v", pv)):
        row = {
            "method": "data_only_nn",
            "label": "Observation-only NN",
            "evaluation_set": "fixed_strict_interior_heldout_test",
            "variable": variable,
            **metric_dict(pred, uv_test[variable].to_numpy(dtype=float)),
        }
        velocity_rows.append(row)

    velocity_metrics = pd.DataFrame(velocity_rows)
    velocity_metrics.to_csv(
        output_dir / "heldout_global_uv_errors.csv", index=False
    )

    uv_pointwise = pd.DataFrame({
        "original_index": uv_test["original_index"].to_numpy(dtype=int),
        "x": uv_test["x"].to_numpy(dtype=float),
        "y": uv_test["y"].to_numpy(dtype=float),
        "split": "test",
        "u_ref": uv_test["u"].to_numpy(dtype=float),
        "u_pred": pu,
        "u_error": pu - uv_test["u"].to_numpy(dtype=float),
        "u_abs_error": np.abs(pu - uv_test["u"].to_numpy(dtype=float)),
        "v_ref": uv_test["v"].to_numpy(dtype=float),
        "v_pred": pv,
        "v_error": pv - uv_test["v"].to_numpy(dtype=float),
        "v_abs_error": np.abs(pv - uv_test["v"].to_numpy(dtype=float)),
        "p_pred_unreferenced_at_velocity_test": pp_uv,
    })
    uv_pointwise.to_csv(
        output_dir / "heldout_velocity_pointwise_predictions.csv", index=False
    )

    # Held-out wall-Cp test. Raw is the primary result because sparse Cp training
    # observations already constrain the pressure level. Gauge-aligned is diagnostic.
    cp_xy = cp_test[["x", "y"]].to_numpy(dtype=np.float32)
    _, _, p_cp, cp_seconds = predict(model, cp_xy, device, args.batch_size)
    cp_raw = float(args.cp_scale) * p_cp
    cp_truth = cp_test["cp"].to_numpy(dtype=float)
    gauge_shift = float(np.mean(cp_truth - cp_raw))
    cp_aligned = cp_raw + gauge_shift

    cp_rows = []
    for treatment, values, shift in (
        ("raw", cp_raw, 0.0),
        ("gauge_aligned_diagnostic", cp_aligned, gauge_shift),
    ):
        cp_rows.append({
            "method": "data_only_nn",
            "label": "Observation-only NN",
            "evaluation_set": "fixed_heldout_cp_test",
            "variable": "Cp_wall",
            "gauge_treatment": treatment,
            "cp_scale": float(args.cp_scale),
            "constant_offset_added": float(shift),
            **metric_dict(values, cp_truth),
        })

    cp_metrics = pd.DataFrame(cp_rows)
    cp_metrics.to_csv(
        output_dir / "heldout_cp_errors_raw_and_gauge_aligned.csv", index=False
    )

    cp_pointwise = cp_test.copy()
    cp_pointwise["cp_pred_raw"] = cp_raw
    cp_pointwise["cp_error_raw"] = cp_raw - cp_truth
    cp_pointwise["cp_pred_gauge_aligned"] = cp_aligned
    cp_pointwise["cp_error_gauge_aligned"] = cp_aligned - cp_truth
    cp_pointwise.to_csv(
        output_dir / "heldout_cp_pointwise_predictions.csv", index=False
    )

    run_summary = pd.DataFrame([{
        "protocol_version": str(summary.get("protocol_version", "unknown")),
        "method": "data_only_nn",
        "model_definition": "direct_(x,y)_to_(u,v,p)_MLP",
        "checkpoint": str(checkpoint),
        "heldout_velocity_points": len(uv_test),
        "heldout_cp_points": len(cp_test),
        "training_or_selection_performed_in_this_script": False,
        "heldout_labels_used_for_training_or_selection": False,
        "cp_scale": float(args.cp_scale),
        "cp_gauge_shift_diagnostic": gauge_shift,
        "velocity_inference_seconds": uv_seconds,
        "cp_inference_seconds": cp_seconds,
        "device": str(device),
    }])
    run_summary.to_csv(output_dir / "heldout_run_summary.csv", index=False)

    print("=" * 90)
    print("NASA DATA-ONLY FINAL HELD-OUT EVALUATION COMPLETE")
    print(f"Protocol version: {summary.get('protocol_version', 'unknown')}")
    print(f"Velocity held-out points: {len(uv_test)}")
    print(f"Cp held-out points: {len(cp_test)}")
    print(f"Checkpoint: {checkpoint}")
    print("\nVelocity metrics:")
    print(velocity_metrics.to_string(index=False))
    print("\nCp metrics:")
    print(cp_metrics.to_string(index=False))
    print(f"\nSaved to: {output_dir}")
    print("No training, calibration, or hyperparameter selection was performed.")
    print("=" * 90)


if __name__ == "__main__":
    main()
