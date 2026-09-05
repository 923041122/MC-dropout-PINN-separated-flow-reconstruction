
"""Formal independent evaluation of the three cylinder physics-weight candidates.

This script DOES NOT train models and DOES NOT access the held-out test set.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from benchmark_config import LAYER_MAT_PSI, METHODS, REYNOLDS
from benchmark_tools import build_model, get_device, safe_load_state

WEIGHTS = ["0", "0.1", "1"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--protocol-root", default="./results_cylinder_protocol_v1_1/protocol")
    p.add_argument("--root", default=".")
    p.add_argument("--run-prefix", default="cylinder_weight_formal_")
    p.add_argument("--output-dir", default="./cylinder_physics_weight_formal_tradeoff")
    p.add_argument("--batch-size", type=int, default=1000)
    p.add_argument("--reynolds", type=float, default=REYNOLDS)
    return p.parse_args()


def residual_batch(model, xyt, device, reynolds):
    x = torch.tensor(xyt[:, 0:1], dtype=torch.float32, device=device, requires_grad=True)
    y = torch.tensor(xyt[:, 1:2], dtype=torch.float32, device=device, requires_grad=True)
    t = torch.tensor(xyt[:, 2:3], dtype=torch.float32, device=device, requires_grad=True)

    model.eval()
    u, v, p = model.predict_fields(x, y, t, create_graph=True)

    u_t = torch.autograd.grad(u.sum(), t, create_graph=True, retain_graph=True)[0]
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True, retain_graph=True)[0]
    u_y = torch.autograd.grad(u.sum(), y, create_graph=True, retain_graph=True)[0]

    v_t = torch.autograd.grad(v.sum(), t, create_graph=True, retain_graph=True)[0]
    v_x = torch.autograd.grad(v.sum(), x, create_graph=True, retain_graph=True)[0]
    v_y = torch.autograd.grad(v.sum(), y, create_graph=True, retain_graph=True)[0]

    p_x = torch.autograd.grad(p.sum(), x, create_graph=True, retain_graph=True)[0]
    p_y = torch.autograd.grad(p.sum(), y, create_graph=True, retain_graph=True)[0]

    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=False, retain_graph=True)[0]
    u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=False, retain_graph=True)[0]
    v_xx = torch.autograd.grad(v_x.sum(), x, create_graph=False, retain_graph=True)[0]
    v_yy = torch.autograd.grad(v_y.sum(), y, create_graph=False, retain_graph=False)[0]

    fx = u_t + u * u_x + v * u_y + p_x - (u_xx + u_yy) / float(reynolds)
    fy = v_t + u * v_x + v * v_y + p_y - (v_xx + v_yy) / float(reynolds)

    return (
        fx.detach().cpu().numpy().reshape(-1),
        fy.detach().cpu().numpy().reshape(-1),
    )


def evaluate_model(model, points, device, reynolds, batch_size) -> Dict[str, float]:
    fx_all: List[np.ndarray] = []
    fy_all: List[np.ndarray] = []

    for start in range(0, len(points), batch_size):
        fx, fy = residual_batch(
            model,
            points[start:start + batch_size],
            device,
            reynolds,
        )
        fx_all.append(fx)
        fy_all.append(fy)

    fx = np.concatenate(fx_all)
    fy = np.concatenate(fy_all)
    mag = np.sqrt(fx**2 + fy**2)

    return {
        "physics_fx_mae": float(np.mean(np.abs(fx))),
        "physics_fy_mae": float(np.mean(np.abs(fy))),
        "physics_fx_rmse": float(np.sqrt(np.mean(fx**2))),
        "physics_fy_rmse": float(np.sqrt(np.mean(fy**2))),
        "physics_vector_rmse": float(np.sqrt(np.mean(fx**2 + fy**2))),
        "physics_magnitude_mean": float(np.mean(mag)),
        "physics_magnitude_median": float(np.median(mag)),
        "physics_magnitude_p95": float(np.quantile(mag, 0.95)),
        "physics_magnitude_max": float(np.max(mag)),
        "physics_evaluation_points": int(len(points)),
    }


def read_training_summary(run_root: Path) -> Dict[str, float]:
    path = run_root / "training_logs" / "standard_pinn_training_summary.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    row = pd.read_csv(path).iloc[0]
    fields = [
        "val_u_rmse",
        "val_v_rmse",
        "val_p_rmse",
        "val_uv_rmse_mean",
        "final_total_loss",
        "final_data_loss",
        "final_equation_loss",
        "training_time_seconds",
        "supervised_ratio",
        "supervised_points",
        "validation_points",
        "equation_points",
        "batch_size_data",
        "batch_size_equation",
        "backbone_parameters",
        "additional_trainable_parameters",
        "total_optimized_parameters",
        "heldout_test_accessed_during_training",
    ]

    result = {}
    for field in fields:
        if field in row.index:
            value = row[field]
            if field == "heldout_test_accessed_during_training":
                result[field] = bool(value)
            else:
                try:
                    result[field] = float(value)
                except (TypeError, ValueError):
                    result[field] = value
    return result


def main():
    args = parse_args()

    protocol_root = Path(args.protocol_root)
    physics_points_path = protocol_root / "independent_physics_evaluation_points.csv"
    if not physics_points_path.exists():
        raise FileNotFoundError(physics_points_path)

    physics_df = pd.read_csv(physics_points_path)
    points = physics_df[["x", "y", "t"]].to_numpy(dtype=np.float32)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    root = Path(args.root)
    device = get_device()
    standard_cfg = dict(METHODS["standard_pinn"])

    rows = []

    for weight in WEIGHTS:
        run_root = root / f"{args.run_prefix}{weight}"
        checkpoint = run_root / "models" / "standard_pinn.pth"

        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

        model = build_model(standard_cfg, LAYER_MAT_PSI).to(device)
        state = safe_load_state(checkpoint, device)
        model.load_state_dict(state, strict=True)
        model.eval()

        print(
            f"Evaluating formal lambda_f={weight} on "
            f"{len(points)} common independent physics points..."
        )

        train_info = read_training_summary(run_root)

        row = {
            "equation_loss_weight": weight,
            **train_info,
            **evaluate_model(
                model=model,
                points=points,
                device=device,
                reynolds=args.reynolds,
                batch_size=args.batch_size,
            ),
        }
        rows.append(row)

    summary = pd.DataFrame(rows)

    summary.to_csv(
        output / "cylinder_physics_weight_formal_tradeoff_summary.csv",
        index=False,
    )

    residual_cols = [
        "equation_loss_weight",
        "physics_fx_mae",
        "physics_fy_mae",
        "physics_fx_rmse",
        "physics_fy_rmse",
        "physics_vector_rmse",
        "physics_magnitude_mean",
        "physics_magnitude_median",
        "physics_magnitude_p95",
        "physics_magnitude_max",
        "physics_evaluation_points",
    ]
    summary[residual_cols].to_csv(
        output / "independent_physics_residual_metrics.csv",
        index=False,
    )

    display_cols = [
        "equation_loss_weight",
        "val_u_rmse",
        "val_v_rmse",
        "val_p_rmse",
        "val_uv_rmse_mean",
        "physics_vector_rmse",
        "physics_magnitude_mean",
        "physics_magnitude_p95",
        "training_time_seconds",
    ]

    print("\n=== FORMAL Cylinder validation-physics trade-off ===")
    print(summary[display_cols].to_string(index=False))
    print(
        "\nSaved:",
        (output / "cylinder_physics_weight_formal_tradeoff_summary.csv").resolve(),
    )
    print("\nHeld-out test set was NOT accessed by this script.")


if __name__ == "__main__":
    main()
