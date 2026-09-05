
"""Independent physics-residual evaluation for the NASA hump physics-weight sweep.

This script DOES NOT train any model.

It:
1. builds one common geometry-aware evaluation set;
2. loads the five Standard-PINN checkpoints trained with different equation-loss weights;
3. evaluates the exact simplified PINN momentum residual on the same points;
4. combines those residual metrics with the already-computed held-out u/v errors;
5. writes one traceable trade-off table.

Expected existing folders:
    physics_weight_0/
    physics_weight_1e-6/
    physics_weight_1e-5/
    physics_weight_1e-4/
    physics_weight_1e-3/

Outputs:
    physics_weight_tradeoff/
        independent_physics_evaluation_points.csv
        independent_physics_residual_metrics.csv
        physics_weight_tradeoff_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from benchmark_tools import build_model, get_device, safe_load_state
from hump_protocol import sample_fluid_collocation_points
from hump_train import (
    LAYER_MAT_PSI,
    read_les_meanfield_tec,
)

WEIGHTS = ["0", "1e-6", "1e-5", "1e-4", "1e-3"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate common independent simplified-physics residuals for the NASA hump weight sweep."
    )
    p.add_argument("--data-dir", default=".")
    p.add_argument("--root", default=".")
    p.add_argument("--output-dir", default="./physics_weight_tradeoff")
    p.add_argument("--n-points", type=int, default=20000)
    p.add_argument("--seed", type=int, default=9091)
    p.add_argument("--batch-size", type=int, default=1000)
    p.add_argument("--reynolds", type=float, default=935892.0)
    return p.parse_args()


def residual_batch(model, xy: np.ndarray, device: torch.device, reynolds: float):
    x = torch.tensor(
        xy[:, 0:1], dtype=torch.float32, device=device, requires_grad=True
    )
    y = torch.tensor(
        xy[:, 1:2], dtype=torch.float32, device=device, requires_grad=True
    )

    model.eval()

    out = model.forward(x, y)
    psi = out[:, 0:1]
    p = out[:, 1:2]

    u = torch.autograd.grad(
        psi.sum(), y, create_graph=True, retain_graph=True
    )[0]
    v = -torch.autograd.grad(
        psi.sum(), x, create_graph=True, retain_graph=True
    )[0]

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

    fx = u * u_x + v * u_y + p_x - (1.0 / float(reynolds)) * (u_xx + u_yy)
    fy = u * v_x + v * v_y + p_y - (1.0 / float(reynolds)) * (v_xx + v_yy)

    return (
        fx.detach().cpu().numpy().reshape(-1),
        fy.detach().cpu().numpy().reshape(-1),
    )


def evaluate_model(
    model,
    points: np.ndarray,
    device: torch.device,
    reynolds: float,
    batch_size: int,
) -> Dict[str, float]:
    all_fx: List[np.ndarray] = []
    all_fy: List[np.ndarray] = []

    for start in range(0, len(points), batch_size):
        batch = points[start:start + batch_size]
        fx, fy = residual_batch(model, batch, device, reynolds)
        all_fx.append(fx)
        all_fy.append(fy)

    fx = np.concatenate(all_fx)
    fy = np.concatenate(all_fy)
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
            f"Missing held-out error file: {path}. Run hump_heldout_evaluate.py first."
        )

    df = pd.read_csv(path)
    out = {}
    for q in ("u", "v"):
        row = df[df["variable"] == q]
        if row.empty:
            raise RuntimeError(f"No {q} row in {path}")
        row = row.iloc[0]
        out[f"{q}_RMSE"] = float(row["rmse"])
        out[f"{q}_relative_L2"] = float(row["relative_l2"])
        if "mae" in row.index:
            out[f"{q}_MAE"] = float(row["mae"])
    return out


def read_training_summary(run_root: Path) -> Dict[str, float]:
    path = run_root / "training_logs" / "standard_pinn_training_summary.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    row = df.iloc[0]
    result = {}
    for source, dest in [
        ("final_total_loss", "final_total_loss"),
        ("final_uv_loss", "final_data_loss"),
        ("final_equation_loss", "training_final_physics_loss"),
        ("final_wall_loss", "final_wall_loss"),
        ("final_subbc_loss", "final_subbc_loss"),
    ]:
        if source in row.index:
            result[dest] = float(row[source])
    return result


def main() -> None:
    args = parse_args()

    data_path = Path(args.data_dir) / "LES_meanfield_nasahump2009_tec.dat"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing {data_path}")

    root = Path(args.root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    meanfield = read_les_meanfield_tec(data_path)

    # One common evaluation set, intentionally distinct from the training-collocation seed.
    points = sample_fluid_collocation_points(
        meanfield,
        n_points=args.n_points,
        seed=args.seed,
    )
    pd.DataFrame(points, columns=["x", "y"]).to_csv(
        output / "independent_physics_evaluation_points.csv", index=False
    )

    device = get_device()
    rows = []

    for weight in WEIGHTS:
        run_root = root / f"physics_weight_{weight}"
        checkpoint = run_root / "models" / "standard_pinn.pth"

        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

        model = build_model(
            {"model_type": "psi", "dropout_rate": 0.0},
            LAYER_MAT_PSI,
        ).to(device)
        state = safe_load_state(checkpoint, device)
        model.load_state_dict(state)

        print(f"Evaluating lambda_f={weight} on {len(points)} common physics points...")

        residual = evaluate_model(
            model=model,
            points=points,
            device=device,
            reynolds=args.reynolds,
            batch_size=args.batch_size,
        )
        accuracy = read_heldout_errors(run_root)
        training = read_training_summary(run_root)

        row = {
            "equation_loss_weight": weight,
            **accuracy,
            **residual,
            **training,
        }
        rows.append(row)

    residual_df = pd.DataFrame([
        {
            "equation_loss_weight": r["equation_loss_weight"],
            **{k: v for k, v in r.items() if k.startswith("physics_") or k == "evaluation_points"},
        }
        for r in rows
    ])
    residual_df.to_csv(
        output / "independent_physics_residual_metrics.csv", index=False
    )

    summary = pd.DataFrame(rows)
    summary.to_csv(
        output / "physics_weight_tradeoff_summary.csv", index=False
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

    print("\n=== Independent accuracy-physics trade-off ===")
    print(summary[display_cols].to_string(index=False))
    print(f"\nSaved: {(output / 'physics_weight_tradeoff_summary.csv').resolve()}")


if __name__ == "__main__":
    main()
