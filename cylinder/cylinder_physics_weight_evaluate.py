
"""Independent physics-residual evaluation for cylinder physics-weight screening.

No model is trained by this script.

It loads the five Standard-PINN checkpoints:
    cylinder_weight_0/
    cylinder_weight_0.1/
    cylinder_weight_1/
    cylinder_weight_10/
    cylinder_weight_100/

and evaluates all of them on the SAME independent physics-evaluation points
created by cylinder_protocol_v1.1.

The residual is exactly the simplified residual used during training:
    fx = u_t + u*u_x + v*u_y + p_x - (1/Re)*(u_xx + u_yy)
    fy = v_t + u*v_x + v*v_y + p_y - (1/Re)*(v_xx + v_yy)

Outputs:
    cylinder_physics_weight_tradeoff/
        independent_physics_residual_metrics.csv
        cylinder_physics_weight_tradeoff_summary.csv
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

WEIGHTS = ["0", "0.1", "1", "10", "100"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--protocol-root",
        default="./results_cylinder_protocol_v1_1/protocol",
    )
    p.add_argument("--root", default=".")
    p.add_argument(
        "--output-dir",
        default="./cylinder_physics_weight_tradeoff",
    )
    p.add_argument("--batch-size", type=int, default=1000)
    p.add_argument("--reynolds", type=float, default=REYNOLDS)
    return p.parse_args()


def residual_batch(
    model: torch.nn.Module,
    xyt: np.ndarray,
    device: torch.device,
    reynolds: float,
):
    x = torch.tensor(
        xyt[:, 0:1],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    y = torch.tensor(
        xyt[:, 1:2],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    t = torch.tensor(
        xyt[:, 2:3],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )

    model.eval()

    u, v, p = model.predict_fields(
        x, y, t, create_graph=True
    )

    u_t = torch.autograd.grad(
        u.sum(), t, create_graph=True, retain_graph=True
    )[0]
    u_x = torch.autograd.grad(
        u.sum(), x, create_graph=True, retain_graph=True
    )[0]
    u_y = torch.autograd.grad(
        u.sum(), y, create_graph=True, retain_graph=True
    )[0]

    v_t = torch.autograd.grad(
        v.sum(), t, create_graph=True, retain_graph=True
    )[0]
    v_x = torch.autograd.grad(
        v.sum(), x, create_graph=True, retain_graph=True
    )[0]
    v_y = torch.autograd.grad(
        v.sum(), y, create_graph=True, retain_graph=True
    )[0]

    p_x = torch.autograd.grad(
        p.sum(), x, create_graph=True, retain_graph=True
    )[0]
    p_y = torch.autograd.grad(
        p.sum(), y, create_graph=True, retain_graph=True
    )[0]

    u_xx = torch.autograd.grad(
        u_x.sum(), x, create_graph=False, retain_graph=True
    )[0]
    u_yy = torch.autograd.grad(
        u_y.sum(), y, create_graph=False, retain_graph=True
    )[0]
    v_xx = torch.autograd.grad(
        v_x.sum(), x, create_graph=False, retain_graph=True
    )[0]
    v_yy = torch.autograd.grad(
        v_y.sum(), y, create_graph=False, retain_graph=False
    )[0]

    fx = (
        u_t
        + u * u_x
        + v * u_y
        + p_x
        - (1.0 / float(reynolds)) * (u_xx + u_yy)
    )
    fy = (
        v_t
        + u * v_x
        + v * v_y
        + p_y
        - (1.0 / float(reynolds)) * (v_xx + v_yy)
    )

    return (
        fx.detach().cpu().numpy().reshape(-1),
        fy.detach().cpu().numpy().reshape(-1),
    )


def evaluate_model(
    model,
    points,
    device,
    reynolds,
    batch_size,
) -> Dict[str, float]:
    fx_all: List[np.ndarray] = []
    fy_all: List[np.ndarray] = []

    for start in range(0, len(points), batch_size):
        batch = points[start:start + batch_size]
        fx, fy = residual_batch(
            model=model,
            xyt=batch,
            device=device,
            reynolds=reynolds,
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
        "physics_vector_rmse": float(
            np.sqrt(np.mean(fx**2 + fy**2))
        ),
        "physics_magnitude_mean": float(np.mean(mag)),
        "physics_magnitude_median": float(np.median(mag)),
        "physics_magnitude_p95": float(
            np.quantile(mag, 0.95)
        ),
        "physics_magnitude_max": float(np.max(mag)),
        "evaluation_points": int(len(points)),
    }


def read_training_summary(run_root: Path) -> Dict[str, float]:
    path = (
        run_root
        / "training_logs"
        / "standard_pinn_training_summary.csv"
    )
    if not path.exists():
        raise FileNotFoundError(path)

    row = pd.read_csv(path).iloc[0]

    keys = [
        "val_u_rmse",
        "val_v_rmse",
        "val_p_rmse",
        "val_uv_rmse_mean",
        "final_data_loss",
        "final_equation_loss",
        "training_time_seconds",
        "supervised_ratio",
        "supervised_points",
        "equation_points",
    ]

    out = {}
    for key in keys:
        if key in row.index:
            out[key] = float(row[key])
    return out


def main():
    args = parse_args()

    protocol_root = Path(args.protocol_root)
    eval_path = (
        protocol_root
        / "independent_physics_evaluation_points.csv"
    )
    if not eval_path.exists():
        raise FileNotFoundError(eval_path)

    df = pd.read_csv(eval_path)
    required = ["x", "y", "t"]
    if not all(c in df.columns for c in required):
        raise ValueError(
            f"{eval_path} must contain columns {required}"
        )

    points = df[required].to_numpy(dtype=np.float32)

    if len(points) != 20000:
        print(
            f"Warning: independent physics set has "
            f"{len(points)} points, not 20000."
        )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    device = get_device()
    root = Path(args.root)
    rows = []

    standard_cfg = dict(METHODS["standard_pinn"])

    for weight in WEIGHTS:
        run_root = root / f"cylinder_weight_{weight}"
        checkpoint = (
            run_root / "models" / "standard_pinn.pth"
        )

        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Missing checkpoint: {checkpoint}"
            )

        model = build_model(
            standard_cfg,
            LAYER_MAT_PSI,
        ).to(device)

        state = safe_load_state(checkpoint, device)
        model.load_state_dict(state, strict=True)
        model.eval()

        print(
            f"Evaluating lambda_f={weight} on "
            f"{len(points)} common independent physics points..."
        )

        row = {
            "equation_loss_weight": weight,
            **read_training_summary(run_root),
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
        "evaluation_points",
    ]
    summary[residual_cols].to_csv(
        output / "independent_physics_residual_metrics.csv",
        index=False,
    )

    summary.to_csv(
        output
        / "cylinder_physics_weight_tradeoff_summary.csv",
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
    ]

    print(
        "\n=== Cylinder independent accuracy-physics trade-off ==="
    )
    print(summary[display_cols].to_string(index=False))
    print(
        "\nSaved:",
        (
            output
            / "cylinder_physics_weight_tradeoff_summary.csv"
        ).resolve(),
    )


if __name__ == "__main__":
    main()
