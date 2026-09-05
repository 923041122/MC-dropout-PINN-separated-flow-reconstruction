
"""Final held-out evaluation for the selected cylinder model.

This script is intended to be run ONLY AFTER all hyperparameters have been fixed.
It reads the 200,000 held-out space-time points from the corrected protocol and
evaluates the selected checkpoint exactly once.

Outputs
-------
<results-root>/heldout_evaluation/
    heldout_global_metrics.csv
    heldout_time_resolved_metrics.csv
    heldout_pointwise_predictions.csv
    README_HELDOUT.txt

Metrics
-------
For u, v, p:
    MAE
    RMSE
    normalized RMSE (RMSE / reference range)
    relative L2
    maximum absolute error

For pressure, an additional gauge-invariant evaluation is reported by removing
the least-squares constant offset independently at each time snapshot:
    p_aligned = p_pred + mean(p_ref - p_pred) at that snapshot

This pressure alignment is used only for evaluation, never for training or
hyperparameter selection.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from benchmark_config import LAYER_MAT_PSI, METHODS
from benchmark_tools import build_model, get_device, safe_load_state
from pinn_model import read_2D_data


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
        default="./cylinder_weight_formal_0.1/models/standard_pinn.pth",
    )
    p.add_argument(
        "--results-root",
        default="./cylinder_final_lambda_0.1",
    )
    p.add_argument("--batch-size", type=int, default=20000)
    return p.parse_args()


def metric_dict(pred: np.ndarray, ref: np.ndarray) -> Dict[str, float]:
    pred = np.asarray(pred, dtype=float).reshape(-1)
    ref = np.asarray(ref, dtype=float).reshape(-1)

    err = pred - ref
    abs_err = np.abs(err)

    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(abs_err))
    max_abs = float(np.max(abs_err))

    ref_range = float(np.max(ref) - np.min(ref))
    nrmse = float(rmse / ref_range) if ref_range > 0 else np.nan

    denom = float(np.linalg.norm(ref))
    relative_l2 = float(np.linalg.norm(err) / max(denom, 1e-12))

    return {
        "mae": mae,
        "rmse": rmse,
        "nrmse_by_reference_range": nrmse,
        "relative_l2": relative_l2,
        "max_absolute_error": max_abs,
    }


def load_full_reference(data_path: Path):
    x, y, t, u, v, p, _ = read_2D_data(str(data_path))

    arrays = {
        "x": x.detach().cpu().numpy().reshape(-1),
        "y": y.detach().cpu().numpy().reshape(-1),
        "t": t.detach().cpu().numpy().reshape(-1),
        "u": u.detach().cpu().numpy().reshape(-1),
        "v": v.detach().cpu().numpy().reshape(-1),
        "p": p.detach().cpu().numpy().reshape(-1),
    }
    return arrays


def predict_selected_model(
    model,
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    device: torch.device,
    batch_size: int,
):
    pu: List[np.ndarray] = []
    pv: List[np.ndarray] = []
    pp: List[np.ndarray] = []

    model.eval()

    for start in range(0, len(x), batch_size):
        end = min(start + batch_size, len(x))

        xb = torch.tensor(
            x[start:end, None],
            dtype=torch.float32,
            device=device,
        )
        yb = torch.tensor(
            y[start:end, None],
            dtype=torch.float32,
            device=device,
        )
        tb = torch.tensor(
            t[start:end, None],
            dtype=torch.float32,
            device=device,
        )

        u_pred, v_pred, p_pred = model.predict_fields_safe(
            xb,
            yb,
            tb,
            create_graph=False,
            train_mode=False,
        )

        pu.append(u_pred.detach().cpu().numpy().reshape(-1))
        pv.append(v_pred.detach().cpu().numpy().reshape(-1))
        pp.append(p_pred.detach().cpu().numpy().reshape(-1))

    return (
        np.concatenate(pu),
        np.concatenate(pv),
        np.concatenate(pp),
    )


def pressure_align_per_time(
    p_pred: np.ndarray,
    p_ref: np.ndarray,
    time_index: np.ndarray,
):
    p_aligned = np.empty_like(p_pred, dtype=float)
    offsets = {}

    for ti in np.unique(time_index):
        mask = time_index == ti
        offset = float(np.mean(p_ref[mask] - p_pred[mask]))
        p_aligned[mask] = p_pred[mask] + offset
        offsets[int(ti)] = offset

    return p_aligned, offsets


def main():
    args = parse_args()

    protocol_root = Path(args.protocol_root)
    idx_path = protocol_root / "protocol_indices.npz"
    summary_path = protocol_root / "protocol_summary.csv"

    if not idx_path.exists():
        raise FileNotFoundError(idx_path)
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    protocol = pd.read_csv(summary_path).iloc[0]

    if "flat_index_order" in protocol.index:
        order = str(protocol["flat_index_order"])
        if "time_major" not in order:
            raise RuntimeError(
                f"Unexpected protocol flattening convention: {order}"
            )

    indices = np.load(idx_path)
    test_idx = np.asarray(indices["test_indices"], dtype=np.int64)

    expected_test = int(protocol["heldout_test_points"])
    if len(test_idx) != expected_test:
        raise RuntimeError(
            f"Protocol says {expected_test} test points but NPZ contains {len(test_idx)}."
        )

    full = load_full_reference(Path(args.data_path))
    n_total = len(full["x"])

    if int(protocol["total_reference_points"]) != n_total:
        raise RuntimeError(
            f"Reference size {n_total} does not match protocol."
        )

    # Decode time-major indexing:
    # flat_index = time_index * n_spatial + spatial_index
    nx = int(protocol["nx"])
    ny = int(protocol["ny"])
    n_spatial = nx * ny

    time_index = test_idx // n_spatial
    spatial_index = test_idx % n_spatial

    x = full["x"][test_idx]
    y = full["y"][test_idx]
    t = full["t"][test_idx]
    u_ref = full["u"][test_idx]
    v_ref = full["v"][test_idx]
    p_ref = full["p"][test_idx]

    cfg = dict(METHODS["standard_pinn"])
    model = build_model(cfg, LAYER_MAT_PSI)

    device = get_device()
    model = model.to(device)

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    model.load_state_dict(
        safe_load_state(checkpoint, device),
        strict=True,
    )
    model.eval()

    print("=" * 88)
    print("FINAL cylinder held-out evaluation")
    print(f"Checkpoint: {checkpoint}")
    print(f"Held-out points: {len(test_idx)}")
    print(f"Unique held-out time snapshots: {len(np.unique(time_index))}")
    print("This test set is used for reporting only; no hyperparameter tuning follows.")
    print("=" * 88)

    u_pred, v_pred, p_pred = predict_selected_model(
        model=model,
        x=x,
        y=y,
        t=t,
        device=device,
        batch_size=args.batch_size,
    )

    # Gauge-invariant pressure alignment, independently per time snapshot.
    p_aligned, pressure_offsets = pressure_align_per_time(
        p_pred=p_pred,
        p_ref=p_ref,
        time_index=time_index,
    )

    output = Path(args.results_root) / "heldout_evaluation"
    output.mkdir(parents=True, exist_ok=True)

    global_rows = []

    for name, pred, ref in [
        ("u", u_pred, u_ref),
        ("v", v_pred, v_ref),
        ("p_raw", p_pred, p_ref),
        ("p_gauge_aligned_per_time", p_aligned, p_ref),
    ]:
        global_rows.append({
            "variable": name,
            "point_count": len(ref),
            **metric_dict(pred, ref),
        })

    global_df = pd.DataFrame(global_rows)
    global_df.to_csv(
        output / "heldout_global_metrics.csv",
        index=False,
    )

    # Complete-time metrics: the spatial held-out mask is identical at every time.
    time_rows = []

    for ti in sorted(np.unique(time_index)):
        mask = time_index == ti
        time_value = float(np.mean(t[mask]))

        row_base = {
            "time_index": int(ti),
            "time_value": time_value,
            "heldout_points": int(np.sum(mask)),
            "pressure_alignment_offset": pressure_offsets[int(ti)],
        }

        for variable, pred, ref in [
            ("u", u_pred[mask], u_ref[mask]),
            ("v", v_pred[mask], v_ref[mask]),
            ("p_raw", p_pred[mask], p_ref[mask]),
            ("p_gauge_aligned", p_aligned[mask], p_ref[mask]),
        ]:
            metrics = metric_dict(pred, ref)
            time_rows.append({
                **row_base,
                "variable": variable,
                **metrics,
            })

    time_df = pd.DataFrame(time_rows)
    time_df.to_csv(
        output / "heldout_time_resolved_metrics.csv",
        index=False,
    )

    pointwise = pd.DataFrame({
        "flat_index": test_idx,
        "spatial_index": spatial_index,
        "time_index": time_index,
        "x": x,
        "y": y,
        "t": t,
        "u_ref": u_ref,
        "u_pred": u_pred,
        "u_abs_error": np.abs(u_pred - u_ref),
        "v_ref": v_ref,
        "v_pred": v_pred,
        "v_abs_error": np.abs(v_pred - v_ref),
        "p_ref": p_ref,
        "p_pred_raw": p_pred,
        "p_pred_gauge_aligned": p_aligned,
        "p_abs_error_raw": np.abs(p_pred - p_ref),
        "p_abs_error_gauge_aligned": np.abs(p_aligned - p_ref),
    })
    pointwise.to_csv(
        output / "heldout_pointwise_predictions.csv",
        index=False,
    )

    readme = """Final cylinder held-out evaluation

The held-out test partition was reserved during all training and physics-weight
selection steps. It is evaluated here only after the final physics-loss weight
has been fixed.

Pressure:
- p_raw reports the direct network/reference comparison.
- p_gauge_aligned_per_time removes one least-squares additive constant from each
  time snapshot. This is a gauge-invariant diagnostic only; no aligned pressure
  value was used for training or hyperparameter selection.

Files:
- heldout_global_metrics.csv
- heldout_time_resolved_metrics.csv
- heldout_pointwise_predictions.csv
"""
    (output / "README_HELDOUT.txt").write_text(
        readme,
        encoding="utf-8",
    )

    print("\n=== FINAL held-out global metrics ===")
    print(global_df.to_string(index=False))

    print("\nTime-resolved metrics saved for all time snapshots.")
    print(f"Saved to: {output.resolve()}")


if __name__ == "__main__":
    main()
