
"""Evaluate the three formal NASA hump physics-weight candidates."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch

from benchmark_tools import build_model, get_device, safe_load_state
from hump_protocol import sample_fluid_collocation_points
from hump_train import LAYER_MAT_PSI, read_les_meanfield_tec

WEIGHTS = ["0", "1e-4", "1e-3"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=".")
    p.add_argument("--root", default=".")
    p.add_argument("--run-prefix", default="physics_weight_formal_")
    p.add_argument("--output-dir", default="./physics_weight_formal_tradeoff")
    p.add_argument("--n-points", type=int, default=20000)
    p.add_argument("--seed", type=int, default=9091)
    p.add_argument("--batch-size", type=int, default=1000)
    p.add_argument("--reynolds", type=float, default=935892.0)
    return p.parse_args()


def residual_batch(model, xy, device, reynolds):
    x = torch.tensor(xy[:, 0:1], dtype=torch.float32, device=device, requires_grad=True)
    y = torch.tensor(xy[:, 1:2], dtype=torch.float32, device=device, requires_grad=True)

    model.eval()
    out = model.forward(x, y)
    psi = out[:, 0:1]
    p = out[:, 1:2]

    u = torch.autograd.grad(psi.sum(), y, create_graph=True, retain_graph=True)[0]
    v = -torch.autograd.grad(psi.sum(), x, create_graph=True, retain_graph=True)[0]

    u_x = torch.autograd.grad(u.sum(), x, create_graph=True, retain_graph=True)[0]
    u_y = torch.autograd.grad(u.sum(), y, create_graph=True, retain_graph=True)[0]
    v_x = torch.autograd.grad(v.sum(), x, create_graph=True, retain_graph=True)[0]
    v_y = torch.autograd.grad(v.sum(), y, create_graph=True, retain_graph=True)[0]
    p_x = torch.autograd.grad(p.sum(), x, create_graph=True, retain_graph=True)[0]
    p_y = torch.autograd.grad(p.sum(), y, create_graph=True, retain_graph=True)[0]

    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=False, retain_graph=True)[0]
    u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=False, retain_graph=True)[0]
    v_xx = torch.autograd.grad(v_x.sum(), x, create_graph=False, retain_graph=True)[0]
    v_yy = torch.autograd.grad(v_y.sum(), y, create_graph=False, retain_graph=False)[0]

    fx = u * u_x + v * u_y + p_x - (u_xx + u_yy) / float(reynolds)
    fy = u * v_x + v * v_y + p_y - (v_xx + v_yy) / float(reynolds)

    return fx.detach().cpu().numpy().ravel(), fy.detach().cpu().numpy().ravel()


def evaluate_model(model, points, device, reynolds, batch_size):
    fx_all, fy_all = [], []
    for start in range(0, len(points), batch_size):
        fx, fy = residual_batch(model, points[start:start + batch_size], device, reynolds)
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
        "evaluation_points": int(len(points)),
    }


def read_heldout_errors(run_root: Path) -> Dict[str, float]:
    path = run_root / "heldout_evaluation" / "heldout_global_uv_errors.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}; run hump_heldout_evaluate.py first."
        )
    df = pd.read_csv(path)
    result = {}
    for q in ("u", "v"):
        row = df[df["variable"] == q].iloc[0]
        result[f"{q}_RMSE"] = float(row["rmse"])
        result[f"{q}_relative_L2"] = float(row["relative_l2"])
        result[f"{q}_MAE"] = float(row["mae"])
    return result


def read_training_summary(run_root: Path):
    path = run_root / "training_logs" / "standard_pinn_training_summary.csv"
    df = pd.read_csv(path)
    row = df.iloc[0]
    result = {}
    mapping = {
        "final_total_loss": "final_total_loss",
        "final_uv_loss": "final_data_loss",
        "final_equation_loss": "training_final_physics_loss",
        "final_cp_loss": "final_cp_loss",
        "final_wall_loss": "final_wall_loss",
        "final_subbc_loss": "final_subbc_loss",
    }
    for src, dst in mapping.items():
        if src in row.index:
            result[dst] = float(row[src])
    return result


def main():
    args = parse_args()
    data_path = Path(args.data_dir) / "LES_meanfield_nasahump2009_tec.dat"
    meanfield = read_les_meanfield_tec(data_path)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    points = sample_fluid_collocation_points(
        meanfield, n_points=args.n_points, seed=args.seed
    )
    pd.DataFrame(points, columns=["x", "y"]).to_csv(
        output / "independent_physics_evaluation_points.csv", index=False
    )

    root = Path(args.root)
    device = get_device()
    rows = []

    for weight in WEIGHTS:
        run_root = root / f"{args.run_prefix}{weight}"
        checkpoint = run_root / "models" / "standard_pinn.pth"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

        model = build_model(
            {"model_type": "psi", "dropout_rate": 0.0}, LAYER_MAT_PSI
        ).to(device)
        model.load_state_dict(safe_load_state(checkpoint, device))

        print(
            f"Evaluating formal lambda_f={weight} "
            f"on {len(points)} common physics points..."
        )

        row = {
            "equation_loss_weight": weight,
            **read_heldout_errors(run_root),
            **evaluate_model(
                model, points, device, args.reynolds, args.batch_size
            ),
            **read_training_summary(run_root),
        }
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(
        output / "physics_weight_formal_tradeoff_summary.csv", index=False
    )

    display_cols = [
        "equation_loss_weight",
        "u_RMSE",
        "v_RMSE",
        "u_relative_L2",
        "v_relative_L2",
        "physics_vector_rmse",
        "physics_magnitude_mean",
        "physics_magnitude_p95",
    ]

    print("\n=== Formal independent accuracy-physics trade-off ===")
    print(summary[display_cols].to_string(index=False))
    print(
        "\nSaved:",
        (output / "physics_weight_formal_tradeoff_summary.csv").resolve(),
    )


if __name__ == "__main__":
    main()
