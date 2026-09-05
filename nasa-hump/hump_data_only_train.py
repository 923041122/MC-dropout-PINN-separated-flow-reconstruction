"""
Observation-only direct neural-network baseline for the steady NASA wall-mounted hump.

Direct MLP: (x, y) -> (u, v, p)

Included observations:
- same 95 strict-interior velocity training points;
- same LES velocity observations on left/right/top truncated-subdomain interfaces;
- same sparse wall-Cp training observations.

Excluded physics constraints:
- no interior momentum/continuity residual;
- no streamfunction hard incompressibility constraint;
- no physical no-slip hump-wall condition.

Validation partitions are used for diagnostics; held-out test labels are not used in
training, validation-based model selection, or loss construction.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Observation-only NASA hump baseline.")
    p.add_argument("--protocol-root", default="./hump_results_revision_r02/protocol")
    p.add_argument("--results-root", default="./baseline_suite/nasa_hump/data_only/formal")
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--min-learning-rate", type=float, default=1e-6)
    p.add_argument("--data-loss-weight", type=float, default=10.0)
    p.add_argument("--cp-loss-weight", type=float, default=1.0)
    p.add_argument("--subbc-loss-weight", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--diagnostic-interval", type=int, default=50)
    p.add_argument("--hidden-layers", type=int, default=10)
    p.add_argument("--hidden-width", type=int, default=100)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DirectUVPNet(nn.Module):
    def __init__(self, hidden_layers: int = 10, hidden_width: int = 100):
        super().__init__()
        dims = [2] + [hidden_width] * hidden_layers + [3]
        layers = []
        for i in range(len(dims) - 2):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.Tanh()]
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, xy: torch.Tensor):
        out = self.net(xy)
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]


def require_columns(df: pd.DataFrame, columns, name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"{name} missing columns: {missing}")


def load_protocol(protocol_root: Path):
    uv_path = protocol_root / "interior_velocity_split_manifest.csv"
    cp_path = protocol_root / "cp_split_manifest.csv"
    subbc_path = protocol_root / "subdomain_boundary_points.csv"
    wall_path = protocol_root / "hump_wall_points.csv"
    summary_path = protocol_root / "protocol_summary.csv"

    for path in (uv_path, cp_path, subbc_path, wall_path, summary_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing protocol artifact: {path}")

    uv = pd.read_csv(uv_path)
    cp = pd.read_csv(cp_path)
    subbc = pd.read_csv(subbc_path)
    wall = pd.read_csv(wall_path)
    summary = pd.read_csv(summary_path).iloc[0]

    require_columns(uv, ["x", "y", "u", "v", "split"], "interior manifest")
    require_columns(cp, ["x", "y", "p_target", "split"], "Cp manifest")
    require_columns(subbc, ["x", "y", "u", "v"], "subdomain boundary")

    uv_train = uv[uv["split"].astype(str) == "train"].copy()
    uv_val = uv[uv["split"].astype(str) == "validation"].copy()
    n_uv_test = int((uv["split"].astype(str) == "test").sum())

    cp_train = cp[cp["split"].astype(str) == "train"].copy()
    cp_val = cp[cp["split"].astype(str) == "validation"].copy()
    n_cp_test = int((cp["split"].astype(str) == "test").sum())

    expected = dict(uv_train=95, uv_val=672, uv_test=987,
                    cp_train=9, cp_val=72, cp_test=96,
                    subbc=249, wall=207)
    found = dict(uv_train=len(uv_train), uv_val=len(uv_val), uv_test=n_uv_test,
                 cp_train=len(cp_train), cp_val=len(cp_val), cp_test=n_cp_test,
                 subbc=len(subbc), wall=len(wall))
    for key, value in expected.items():
        if found[key] != value:
            raise RuntimeError(f"Protocol mismatch {key}: found {found[key]}, expected {value}")

    return uv_train, uv_val, cp_train, cp_val, subbc, summary, found


def tensor_uv(df: pd.DataFrame, device: torch.device):
    xy = torch.tensor(df[["x", "y"]].to_numpy(np.float32), device=device)
    u = torch.tensor(df[["u"]].to_numpy(np.float32), device=device)
    v = torch.tensor(df[["v"]].to_numpy(np.float32), device=device)
    return xy, u, v


def tensor_cp(df: pd.DataFrame, device: torch.device):
    xy = torch.tensor(df[["x", "y"]].to_numpy(np.float32), device=device)
    p = torch.tensor(df[["p_target"]].to_numpy(np.float32), device=device)
    return xy, p


@torch.no_grad()
def evaluate_validation(model, uv_val, cp_val, device) -> Dict[str, float]:
    was_training = model.training
    model.eval()

    xy, u_ref, v_ref = tensor_uv(uv_val, device)
    u_hat, v_hat, _ = model(xy)

    xy_cp, p_ref = tensor_cp(cp_val, device)
    _, _, p_hat = model(xy_cp)

    def m(pred, ref, prefix):
        err = pred - ref
        rmse = float(torch.sqrt(torch.mean(err**2)).cpu())
        mae = float(torch.mean(torch.abs(err)).cpu())
        denom = float(torch.linalg.vector_norm(ref).cpu())
        rel = float(torch.linalg.vector_norm(err).cpu()) / max(denom, 1e-12)
        return {f"val_{prefix}_rmse": rmse,
                f"val_{prefix}_mae": mae,
                f"val_{prefix}_relative_l2": rel}

    out = {}
    out.update(m(u_hat, u_ref, "u"))
    out.update(m(v_hat, v_ref, "v"))
    out.update(m(p_hat, p_ref, "p_from_cp"))
    out["val_uv_rmse_mean"] = 0.5 * (out["val_u_rmse"] + out["val_v_rmse"])
    model.train(mode=was_training)
    return out


def train(args):
    set_seed(args.seed)
    device = get_device()
    protocol_root = Path(args.protocol_root)
    results_root = Path(args.results_root)
    models_dir = results_root / "models"
    logs_dir = results_root / "training_logs"
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    uv_train, uv_val, cp_train, cp_val, subbc, summary, counts = load_protocol(protocol_root)

    model = DirectUVPNet(args.hidden_layers, args.hidden_width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs), eta_min=args.min_learning_rate
    )

    xy_uv, u_uv, v_uv = tensor_uv(uv_train, device)
    xy_cp, p_cp = tensor_cp(cp_train, device)
    xy_bc, u_bc, v_bc = tensor_uv(subbc, device)
    mse = nn.MSELoss()

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 92)
    print("NASA hump observation-only direct baseline")
    print("Model: direct (x,y) -> (u,v,p) MLP")
    print(f"Device: {device}")
    print(f"Protocol: {protocol_root}")
    print(f"Protocol version: {summary.get('protocol_version', 'unknown')}")
    print(f"Interior train/validation: {counts['uv_train']} / {counts['uv_val']}")
    print("Interior test: 987 RESERVED AND NOT USED FOR TRAINING OR MODEL SELECTION")
    print(f"Cp train/validation: {counts['cp_train']} / {counts['cp_val']}")
    print("Cp test: 96 RESERVED AND NOT USED FOR TRAINING OR MODEL SELECTION")
    print(f"Observed subdomain velocity points: {counts['subbc']}")
    print(f"Physical wall points: {counts['wall']} (NOT USED)")
    print("Interior physics residual: DISABLED")
    print("Streamfunction hard incompressibility: DISABLED")
    print("Physical no-slip wall condition: DISABLED")
    print("Observed subdomain velocities: ENABLED")
    print("Sparse Cp observations: ENABLED")
    print(f"Backbone parameters: {params}")
    print("Additional trainable parameters: 0")
    print(f"Total optimized parameters: {params}")
    print(f"Loss weights: uv={args.data_loss_weight}, cp={args.cp_loss_weight}, subbc={args.subbc_loss_weight}")
    print(f"LR: {args.learning_rate} -> {args.min_learning_rate}; epochs={args.epochs}; seed={args.seed}")
    print("=" * 92)

    rows, val_rows = [], []
    start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        u_hat, v_hat, _ = model(xy_uv)
        loss_uv = mse(u_hat, u_uv) + mse(v_hat, v_uv)

        _, _, p_hat = model(xy_cp)
        loss_cp = mse(p_hat, p_cp)

        u_hat_bc, v_hat_bc, _ = model(xy_bc)
        loss_bc = mse(u_hat_bc, u_bc) + mse(v_hat_bc, v_bc)

        total = (args.data_loss_weight * loss_uv
                 + args.cp_loss_weight * loss_cp
                 + args.subbc_loss_weight * loss_bc)

        if not torch.isfinite(total):
            raise FloatingPointError(f"Non-finite loss at epoch {epoch}")

        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        scheduler.step()

        elapsed = time.perf_counter() - start
        row = {
            "method": "data_only_nn",
            "epoch": epoch,
            "total_loss": float(total.detach().cpu()),
            "uv_loss": float(loss_uv.detach().cpu()),
            "cp_pressure_loss": float(loss_cp.detach().cpu()),
            "subbc_observation_loss": float(loss_bc.detach().cpu()),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_seconds": elapsed,
            "backbone_parameters": params,
            "additional_trainable_parameters": 0,
            "total_optimized_parameters": params,
            "seed": args.seed,
        }
        rows.append(row)

        do_val = epoch == 1 or epoch == args.epochs or (
            args.diagnostic_interval > 0 and epoch % args.diagnostic_interval == 0
        )
        text = ""
        if do_val:
            val = evaluate_validation(model, uv_val, cp_val, device)
            val_rows.append({"method": "data_only_nn", "epoch": epoch, "seed": args.seed, **val})
            text = (f" | val_u={val['val_u_rmse']:.3e}"
                    f" val_v={val['val_v_rmse']:.3e}"
                    f" val_p={val['val_p_from_cp_rmse']:.3e}")

        torch.save(model.state_dict(), models_dir / "data_only_nn.pth")
        pd.DataFrame(rows).to_csv(logs_dir / "data_only_nn_training_log.csv", index=False)
        if val_rows:
            pd.DataFrame(val_rows).to_csv(logs_dir / "data_only_nn_validation_log.csv", index=False)

        print(f"[data_only_nn] epoch {epoch:05d}/{args.epochs}"
              f" | total={row['total_loss']:.4e}"
              f" | uv={row['uv_loss']:.4e}"
              f" | cp={row['cp_pressure_loss']:.4e}"
              f" | subbc={row['subbc_observation_loss']:.4e}"
              f" | lr={row['learning_rate']:.2e}"
              f" | time={elapsed:.1f}s{text}")

    final_val = evaluate_validation(model, uv_val, cp_val, device)
    total_time = time.perf_counter() - start
    final = rows[-1]
    summary_row = {
        "protocol_version": str(summary.get("protocol_version", "unknown")),
        "method": "data_only_nn",
        "label": "Observation-only NN",
        "model_definition": "direct_(x,y)_to_(u,v,p)_MLP",
        "interior_physics_residual": False,
        "streamfunction_hard_incompressibility": False,
        "physical_no_slip_wall_condition": False,
        "observed_subdomain_velocity_constraints": True,
        "sparse_cp_observations": True,
        "checkpoint": str(models_dir / "data_only_nn.pth"),
        "epochs": args.epochs,
        "interior_supervised_velocity_points": counts["uv_train"],
        "interior_validation_velocity_points": counts["uv_val"],
        "heldout_velocity_test_used_for_training_or_selection": False,
        "cp_training_points": counts["cp_train"],
        "cp_validation_points": counts["cp_val"],
        "cp_test_used_for_training_or_selection": False,
        "subdomain_observation_points": counts["subbc"],
        "physical_wall_points_not_used": counts["wall"],
        "learning_rate_initial": args.learning_rate,
        "learning_rate_minimum": args.min_learning_rate,
        "data_loss_weight": args.data_loss_weight,
        "cp_loss_weight": args.cp_loss_weight,
        "subbc_loss_weight": args.subbc_loss_weight,
        "hidden_layers": args.hidden_layers,
        "hidden_width": args.hidden_width,
        "backbone_parameters": params,
        "additional_trainable_parameters": 0,
        "total_optimized_parameters": params,
        "training_time_seconds": total_time,
        "final_total_loss": final["total_loss"],
        "final_uv_loss": final["uv_loss"],
        "final_cp_pressure_loss": final["cp_pressure_loss"],
        "final_subbc_observation_loss": final["subbc_observation_loss"],
        "seed": args.seed,
        **final_val,
    }
    pd.DataFrame([summary_row]).to_csv(
        logs_dir / "data_only_nn_training_summary.csv", index=False
    )
    print("\nTraining complete.")
    print(f"Training summary: {logs_dir / 'data_only_nn_training_summary.csv'}")
    print("Velocity and Cp held-out test labels were not used for training or model selection.")


if __name__ == "__main__":
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    train(args)
