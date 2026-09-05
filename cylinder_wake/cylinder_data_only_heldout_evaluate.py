"""
Final held-out evaluator for the frozen cylinder Data-only NN.

This script is evaluation-only:
- reads the fixed protocol test_indices only after the model configuration is frozen;
- loads the frozen direct (x,y,t)->(u,v,p) checkpoint;
- performs no training, calibration, or hyperparameter selection;
- reports u/v errors and both raw and gauge-aligned pressure errors;
- writes global, time-resolved, and pointwise held-out outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from pinn_model import read_2D_data
from cylinder_data_only_train import DirectUVPNet


def parse_args():
    p = argparse.ArgumentParser(description="Final held-out evaluation of cylinder Data-only NN.")
    p.add_argument("--data-path", default="./2d_cylinder_Re3900_100x100_kw_sst.mat")
    p.add_argument("--protocol-root", default="./results_cylinder_protocol_v1_1/protocol")
    p.add_argument(
        "--checkpoint",
        default="./baseline_suite/cylinder/data_only/formal/models/data_only_nn.pth",
    )
    p.add_argument(
        "--results-root",
        default="./baseline_suite/cylinder/data_only/formal",
    )
    p.add_argument("--batch-size", type=int, default=20000)
    p.add_argument("--hidden-layers", type=int, default=10)
    p.add_argument("--hidden-width", type=int, default=100)
    return p.parse_args()


def load_state(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def metrics(pred: np.ndarray, ref: np.ndarray):
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    ref = np.asarray(ref, dtype=np.float64).reshape(-1)
    err = pred - ref
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    rel_l2 = float(np.linalg.norm(err) / max(np.linalg.norm(ref), 1e-12))
    span = float(np.max(ref) - np.min(ref))
    nrmse_range = float(rmse / max(span, 1e-12))
    return {
        "mae": mae,
        "rmse": rmse,
        "relative_l2": rel_l2,
        "nrmse_range": nrmse_range,
        "max_abs_error": float(np.max(np.abs(err))),
    }


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    protocol_root = Path(args.protocol_root)
    idx_path = protocol_root / "protocol_indices.npz"
    summary_path = protocol_root / "protocol_summary.csv"
    checkpoint = Path(args.checkpoint)
    output_dir = Path(args.results_root) / "heldout_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in (idx_path, summary_path, checkpoint, Path(args.data_path)):
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    arrays = np.load(idx_path)
    required_keys = {"validation_indices", "test_indices", "train_2pct"}
    missing = required_keys.difference(arrays.files)
    if missing:
        raise KeyError(f"Protocol missing required keys: {sorted(missing)}")

    train_idx = np.asarray(arrays["train_2pct"], dtype=np.int64)
    val_idx = np.asarray(arrays["validation_indices"], dtype=np.int64)
    test_idx = np.asarray(arrays["test_indices"], dtype=np.int64)

    if np.intersect1d(train_idx, val_idx).size:
        raise RuntimeError("Protocol error: train overlaps validation.")
    if np.intersect1d(train_idx, test_idx).size:
        raise RuntimeError("Protocol error: train overlaps held-out test.")
    if np.intersect1d(val_idx, test_idx).size:
        raise RuntimeError("Protocol error: validation overlaps held-out test.")

    protocol_summary = pd.read_csv(summary_path).iloc[0]
    if "flat_index_order" in protocol_summary.index:
        order = str(protocol_summary["flat_index_order"])
        if "time_major" not in order:
            raise RuntimeError(f"Unexpected flattening convention: {order}")

    x, y, t, u, v, p, _ = read_2D_data(str(args.data_path))
    full = torch.cat([x, y, t, u, v, p], dim=1).float()

    if len(full) != int(protocol_summary["total_reference_points"]):
        raise RuntimeError(
            f"Reference size mismatch: data={len(full)}, "
            f"protocol={int(protocol_summary['total_reference_points'])}"
        )

    # The held-out labels are intentionally constructed only here, after freeze.
    test = full[torch.from_numpy(test_idx)]

    model = DirectUVPNet(
        hidden_layers=args.hidden_layers,
        hidden_width=args.hidden_width,
    ).to(device)
    model.load_state_dict(load_state(checkpoint, device), strict=True)
    model.eval()

    pred_u, pred_v, pred_p = [], [], []
    start = time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(test), args.batch_size):
            batch = test[i:i + args.batch_size]
            xb = batch[:, 0:1].to(device)
            yb = batch[:, 1:2].to(device)
            tb = batch[:, 2:3].to(device)
            uh, vh, ph = model(xb, yb, tb)
            pred_u.append(uh.cpu().numpy().reshape(-1))
            pred_v.append(vh.cpu().numpy().reshape(-1))
            pred_p.append(ph.cpu().numpy().reshape(-1))
    inference_seconds = time.perf_counter() - start

    pu = np.concatenate(pred_u)
    pv = np.concatenate(pred_v)
    pp = np.concatenate(pred_p)

    arr = test.numpy()
    xx, yy, tt = arr[:, 0], arr[:, 1], arr[:, 2]
    ru, rv, rp = arr[:, 3], arr[:, 4], arr[:, 5]

    # Deterministic pressure-gauge alignment for reporting only.
    # No model parameter is changed.
    pressure_gauge_shift = float(np.mean(rp.astype(np.float64) - pp.astype(np.float64)))
    pp_aligned = pp.astype(np.float64) + pressure_gauge_shift

    rows = []
    for field, pred, ref, variant in [
        ("u", pu, ru, "raw"),
        ("v", pv, rv, "raw"),
        ("p", pp, rp, "raw"),
        ("p", pp_aligned, rp, "gauge_aligned"),
    ]:
        row = {
            "method": "data_only_nn",
            "evaluation_set": "fixed_heldout_test",
            "field": field,
            "pressure_variant": variant if field == "p" else "not_applicable",
            "n_points": len(test_idx),
            **metrics(pred, ref),
        }
        rows.append(row)

    global_df = pd.DataFrame(rows)
    global_df.to_csv(output_dir / "heldout_global_metrics.csv", index=False)

    pointwise = pd.DataFrame({
        "flat_index": test_idx,
        "x": xx,
        "y": yy,
        "t": tt,
        "u_ref": ru,
        "u_pred": pu,
        "u_error": pu - ru,
        "u_abs_error": np.abs(pu - ru),
        "v_ref": rv,
        "v_pred": pv,
        "v_error": pv - rv,
        "v_abs_error": np.abs(pv - rv),
        "p_ref": rp,
        "p_pred_raw": pp,
        "p_error_raw": pp - rp,
        "p_pred_gauge_aligned": pp_aligned,
        "p_error_gauge_aligned": pp_aligned - rp,
    })
    pointwise.to_csv(output_dir / "heldout_pointwise_predictions.csv", index=False)

    time_rows = []
    for time_value, g in pointwise.groupby("t", sort=True):
        for field, pred_col, ref_col, variant in [
            ("u", "u_pred", "u_ref", "raw"),
            ("v", "v_pred", "v_ref", "raw"),
            ("p", "p_pred_raw", "p_ref", "raw"),
            ("p", "p_pred_gauge_aligned", "p_ref", "gauge_aligned"),
        ]:
            time_rows.append({
                "method": "data_only_nn",
                "evaluation_set": "fixed_heldout_test",
                "t": float(time_value),
                "field": field,
                "pressure_variant": variant if field == "p" else "not_applicable",
                "n_points": len(g),
                **metrics(g[pred_col].to_numpy(), g[ref_col].to_numpy()),
            })
    pd.DataFrame(time_rows).to_csv(
        output_dir / "heldout_time_resolved_metrics.csv", index=False
    )

    run_summary = pd.DataFrame([{
        "protocol_version": str(protocol_summary.get("protocol_version", "unknown")),
        "method": "data_only_nn",
        "model_definition": "direct_(x,y,t)_to_(u,v,p)_MLP",
        "checkpoint": str(checkpoint),
        "heldout_points": len(test_idx),
        "training_or_selection_performed_in_this_script": False,
        "pressure_gauge_shift": pressure_gauge_shift,
        "inference_seconds": inference_seconds,
        "batch_size": args.batch_size,
        "device": str(device),
    }])
    run_summary.to_csv(output_dir / "heldout_run_summary.csv", index=False)

    print("=" * 88)
    print("FINAL HELD-OUT EVALUATION COMPLETE")
    print(f"Protocol version: {protocol_summary.get('protocol_version', 'unknown')}")
    print(f"Held-out points: {len(test_idx)}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Pressure gauge shift (reporting only): {pressure_gauge_shift:.9g}")
    print(f"Inference time: {inference_seconds:.3f} s")
    print("\nGlobal metrics:")
    print(global_df.to_string(index=False))
    print(f"\nSaved to: {output_dir}")
    print("No training, calibration, or hyperparameter selection was performed.")
    print("=" * 88)


if __name__ == "__main__":
    main()
