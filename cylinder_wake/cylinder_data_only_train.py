"""
Direct data-only neural-network baseline for the Re=3900 transient cylinder wake.

Purpose
-------
This script provides the reviewer-requested data-only baseline. It intentionally
does NOT use an interior physics residual and does NOT enforce incompressibility
through a streamfunction. The network maps (x, y, t) directly to (u, v, p).

Protocol safeguards
-------------------
1. Uses the same fixed sparse-data protocol_indices.npz as cylinder_train_v2.py.
2. Uses the same 2% supervised locations when --supervised-ratio 0.02 is selected.
3. Uses the same fixed validation partition.
4. Held-out test labels are never constructed during training/model selection.
5. Uses the same nominal optimizer-update budget as the PINN trainer by matching
   steps_per_epoch to the fixed collocation-count/batch-size schedule, although
   collocation coordinates do not enter the data-only loss.
6. Reports backbone and total optimized parameter counts explicitly.

Formal held-out testing must be performed only after the configuration is frozen.
"""

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pinn_model import read_2D_data

try:
    from learning_schdule import ChainedScheduler
except ImportError:
    from learning_schedule import ChainedScheduler


ALLOWED_RATIOS = (0.01, 0.02, 0.03, 0.04, 0.05)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Direct (x,y,t)->(u,v,p) data-only baseline for the cylinder wake."
    )
    p.add_argument("--data-path", default="./2d_cylinder_Re3900_100x100_kw_sst.mat")
    p.add_argument("--protocol-root", default="./results_cylinder_protocol_v1_1/protocol")
    p.add_argument("--results-root", default="./baseline_suite/cylinder/data_only/formal")
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--supervised-ratio", type=float, default=0.02)
    p.add_argument("--n-equation-points", type=int, default=100000,
                   help="Used only to match the PINN optimizer-update budget; no physics residual is evaluated.")
    p.add_argument("--batch-size-data", type=int, default=512)
    p.add_argument("--batch-size-equation", type=int, default=2048,
                   help="Used only to match the PINN steps-per-epoch budget.")
    p.add_argument("--validation-batch-size", type=int, default=20000)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--data-loss-weight", type=float, default=10.0)
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


def ratio_to_key(ratio: float) -> Tuple[int, str]:
    pct = int(round(float(ratio) * 100.0))
    if pct not in (1, 2, 3, 4, 5) or not np.isclose(ratio, pct / 100.0):
        raise ValueError("--supervised-ratio must be one of 0.01, 0.02, 0.03, 0.04, 0.05")
    return pct, f"train_{pct}pct"


def load_protocol(protocol_root: Path, ratio: float, n_equation_points: int):
    idx_path = protocol_root / "protocol_indices.npz"
    eq_path = protocol_root / "collocation_points.csv"
    summary_path = protocol_root / "protocol_summary.csv"

    for path in (idx_path, eq_path, summary_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing protocol artifact: {path}")

    pct, train_key = ratio_to_key(ratio)
    arrays = np.load(idx_path)

    if train_key not in arrays.files:
        raise KeyError(f"Protocol has no key {train_key}. Available: {arrays.files}")

    train_idx = np.asarray(arrays[train_key], dtype=np.int64)
    val_idx = np.asarray(arrays["validation_indices"], dtype=np.int64)
    test_idx = np.asarray(arrays["test_indices"], dtype=np.int64)

    if np.intersect1d(train_idx, val_idx).size:
        raise RuntimeError("Supervised training set overlaps validation set.")
    if np.intersect1d(train_idx, test_idx).size:
        raise RuntimeError("Supervised training set overlaps held-out test set.")
    if np.intersect1d(val_idx, test_idx).size:
        raise RuntimeError("Validation set overlaps held-out test set.")

    summary = pd.read_csv(summary_path).iloc[0]
    n_total = int(summary["total_reference_points"])
    expected = int(round(n_total * ratio))
    if len(train_idx) != expected:
        raise RuntimeError(f"{pct}% training set has {len(train_idx)} points; expected {expected}.")

    if "flat_index_order" in summary.index:
        order = str(summary["flat_index_order"])
        if "time_major" not in order:
            raise RuntimeError(f"Unexpected protocol flattening convention: {order}")

    eq_df = pd.read_csv(eq_path, usecols=["x", "y", "t"])
    if n_equation_points < 1 or n_equation_points > len(eq_df):
        raise ValueError(
            f"Requested {n_equation_points} nominal equation points; protocol stores {len(eq_df)}."
        )

    return train_idx, val_idx, test_idx, summary


def load_reference_stack(data_path: Path) -> torch.Tensor:
    x, y, t, u, v, p, _ = read_2D_data(str(data_path))
    return torch.cat([x, y, t, u, v, p], dim=1).float()


class DirectUVPNet(nn.Module):
    """Direct data-only MLP: (x,y,t) -> (u,v,p)."""

    def __init__(self, hidden_layers: int = 10, hidden_width: int = 100):
        super().__init__()
        dims = [3] + [hidden_width] * hidden_layers + [3]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.base = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.base:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor):
        out = self.base(torch.cat([x, y, t], dim=1))
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]


def batch_slice_with_wrap(tensor: torch.Tensor, start: int, batch_size: int) -> torch.Tensor:
    n = len(tensor)
    if n == 0:
        raise ValueError("Cannot batch an empty tensor.")
    start = start % n
    end = start + batch_size
    if end <= n:
        return tensor[start:end]
    return torch.cat([tensor[start:n], tensor[0:end - n]], dim=0)


@torch.no_grad()
def evaluate_validation(
    model: nn.Module,
    validation_data: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> Dict[str, float]:
    was_training = model.training
    model.eval()

    pred_u, pred_v, pred_p = [], [], []
    ref_u, ref_v, ref_p = [], [], []

    for start in range(0, len(validation_data), batch_size):
        batch = validation_data[start:start + batch_size]
        x = batch[:, 0:1].to(device)
        y = batch[:, 1:2].to(device)
        t = batch[:, 2:3].to(device)
        u_hat, v_hat, p_hat = model(x, y, t)

        pred_u.append(u_hat.cpu())
        pred_v.append(v_hat.cpu())
        pred_p.append(p_hat.cpu())
        ref_u.append(batch[:, 3:4].cpu())
        ref_v.append(batch[:, 4:5].cpu())
        ref_p.append(batch[:, 5:6].cpu())

    pu = torch.cat(pred_u).numpy().reshape(-1)
    pv = torch.cat(pred_v).numpy().reshape(-1)
    pp = torch.cat(pred_p).numpy().reshape(-1)
    ru = torch.cat(ref_u).numpy().reshape(-1)
    rv = torch.cat(ref_v).numpy().reshape(-1)
    rp = torch.cat(ref_p).numpy().reshape(-1)

    def metrics(pred, ref, prefix):
        err = pred - ref
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae = float(np.mean(np.abs(err)))
        denom = float(np.linalg.norm(ref))
        rel_l2 = float(np.linalg.norm(err) / max(denom, 1e-12))
        return {
            f"val_{prefix}_rmse": rmse,
            f"val_{prefix}_mae": mae,
            f"val_{prefix}_relative_l2": rel_l2,
        }

    out = {}
    out.update(metrics(pu, ru, "u"))
    out.update(metrics(pv, rv, "v"))
    out.update(metrics(pp, rp, "p"))
    out["val_uv_rmse_mean"] = 0.5 * (out["val_u_rmse"] + out["val_v_rmse"])

    model.train(mode=was_training)
    return out


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = get_device()

    data_path = Path(args.data_path)
    protocol_root = Path(args.protocol_root)
    results_root = Path(args.results_root)
    models_dir = results_root / "models"
    logs_dir = results_root / "training_logs"
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    train_idx, val_idx, test_idx, protocol_summary = load_protocol(
        protocol_root, args.supervised_ratio, args.n_equation_points
    )
    full_data = load_reference_stack(data_path)

    if int(full_data.shape[0]) != int(protocol_summary["total_reference_points"]):
        raise RuntimeError("Reference-data size does not match the protocol.")

    train_data = full_data[torch.from_numpy(train_idx)]
    validation_data = full_data[torch.from_numpy(val_idx)]

    # Held-out labels are intentionally not constructed.
    del test_idx

    model = DirectUVPNet(
        hidden_layers=args.hidden_layers,
        hidden_width=args.hidden_width,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ChainedScheduler(
        optimizer,
        T_0=50,
        T_mul=2,
        eta_min=0.0,
        gamma=0.9,
        max_lr=args.learning_rate,
        warmup_steps=2,
    )

    backbone_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_data = len(train_data)

    # Match the nominal optimizer-update budget of cylinder_train_v2.py.
    steps_per_epoch = max(
        math.ceil(n_data / args.batch_size_data),
        math.ceil(args.n_equation_points / args.batch_size_equation),
    )

    print("=" * 92)
    print("Cylinder direct data-only baseline")
    print("Model: direct (x,y,t) -> (u,v,p) MLP")
    print(f"Device: {device}")
    print(f"Protocol: {protocol_root}")
    print(f"Supervised ratio: {args.supervised_ratio:.2%}")
    print(f"Supervised points: {n_data}")
    print(f"Validation points: {len(validation_data)}")
    print("Held-out test points: RESERVED AND NOT ACCESSED DURING TRAINING")
    print("Interior physics residual: DISABLED")
    print("Streamfunction hard incompressibility constraint: DISABLED")
    print(f"Nominal equation-point count used only for update-budget matching: {args.n_equation_points}")
    print(f"Data batch size: {args.batch_size_data}")
    print(f"Nominal equation batch size: {args.batch_size_equation}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Backbone parameters: {backbone_params}")
    print("Additional trainable parameters: 0")
    print(f"Total optimized parameters: {backbone_params}")
    print(f"Data loss weight: {args.data_loss_weight}")
    print(f"Initial learning rate: {args.learning_rate}")
    print(f"Seed: {args.seed}")
    print("=" * 92)

    rows = []
    validation_rows = []
    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        data_perm = torch.randperm(n_data)
        data_epoch = train_data[data_perm]

        total_vals = []
        data_vals = []

        for step in range(steps_per_epoch):
            batch = batch_slice_with_wrap(
                data_epoch, step * args.batch_size_data, args.batch_size_data
            ).to(device)

            x = batch[:, 0:1]
            y = batch[:, 1:2]
            t = batch[:, 2:3]
            u = batch[:, 3:4]
            v = batch[:, 4:5]
            p = batch[:, 5:6]

            optimizer.zero_grad(set_to_none=True)
            u_hat, v_hat, p_hat = model(x, y, t)
            data_loss = (
                torch.mean((u_hat - u) ** 2)
                + torch.mean((v_hat - v) ** 2)
                + torch.mean((p_hat - p) ** 2)
            )
            total_loss = float(args.data_loss_weight) * data_loss

            if not torch.isfinite(total_loss):
                raise FloatingPointError(
                    f"Non-finite loss at epoch={epoch}, step={step+1}: "
                    f"total={float(total_loss.detach().cpu())}, "
                    f"data={float(data_loss.detach().cpu())}"
                )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            total_vals.append(float(total_loss.detach().cpu()))
            data_vals.append(float(data_loss.detach().cpu()))

        scheduler.step()
        elapsed = time.perf_counter() - start_time

        row = {
            "method": "data_only_nn",
            "epoch": epoch,
            "supervised_ratio": args.supervised_ratio,
            "supervised_points": n_data,
            "validation_points": len(validation_data),
            "interior_physics_residual": False,
            "streamfunction_constraint": False,
            "nominal_equation_points_for_update_budget": args.n_equation_points,
            "batch_size_data": args.batch_size_data,
            "nominal_batch_size_equation_for_update_budget": args.batch_size_equation,
            "steps_per_epoch": steps_per_epoch,
            "total_loss": float(np.mean(total_vals)),
            "data_loss": float(np.mean(data_vals)),
            "weighted_data_loss": float(args.data_loss_weight * np.mean(data_vals)),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_seconds": elapsed,
            "backbone_parameters": backbone_params,
            "additional_trainable_parameters": 0,
            "total_optimized_parameters": backbone_params,
            "seed": args.seed,
        }
        rows.append(row)

        should_validate = (
            epoch == args.epochs
            or (args.diagnostic_interval > 0 and epoch % args.diagnostic_interval == 0)
        )

        val_text = ""
        if should_validate:
            val = evaluate_validation(
                model, validation_data, device, args.validation_batch_size
            )
            val_row = {
                "method": "data_only_nn",
                "epoch": epoch,
                "supervised_ratio": args.supervised_ratio,
                "seed": args.seed,
                **val,
            }
            validation_rows.append(val_row)
            val_text = (
                f" | val_u={val['val_u_rmse']:.3e}"
                f" val_v={val['val_v_rmse']:.3e}"
                f" val_p={val['val_p_rmse']:.3e}"
            )

        torch.save(model.state_dict(), models_dir / "data_only_nn.pth")
        pd.DataFrame(rows).to_csv(
            logs_dir / "data_only_nn_training_log.csv", index=False
        )
        if validation_rows:
            pd.DataFrame(validation_rows).to_csv(
                logs_dir / "data_only_nn_validation_log.csv", index=False
            )

        print(
            f"[data_only_nn] epoch {epoch:05d}/{args.epochs}"
            f" | total={row['total_loss']:.4e}"
            f" | data={row['data_loss']:.4e}"
            f" | lr={row['learning_rate']:.2e}"
            f" | time={elapsed:.1f}s"
            f"{val_text}"
        )

    final_val = evaluate_validation(
        model, validation_data, device, args.validation_batch_size
    )
    total_time = time.perf_counter() - start_time

    summary = {
        "protocol_version": str(protocol_summary["protocol_version"]),
        "method": "data_only_nn",
        "label": "Data-only NN",
        "model_definition": "direct_(x,y,t)_to_(u,v,p)_MLP",
        "interior_physics_residual": False,
        "streamfunction_hard_incompressibility": False,
        "update_budget_matched_to_pinn": True,
        "checkpoint": str(models_dir / "data_only_nn.pth"),
        "epochs": args.epochs,
        "supervised_ratio": args.supervised_ratio,
        "supervised_points": n_data,
        "validation_points": len(validation_data),
        "heldout_test_accessed_during_training": False,
        "nominal_equation_points_for_update_budget": args.n_equation_points,
        "batch_size_data": args.batch_size_data,
        "nominal_equation_batch_size_for_update_budget": args.batch_size_equation,
        "steps_per_epoch": steps_per_epoch,
        "learning_rate_initial": args.learning_rate,
        "data_loss_weight": args.data_loss_weight,
        "hidden_layers": args.hidden_layers,
        "hidden_width": args.hidden_width,
        "backbone_parameters": backbone_params,
        "additional_trainable_parameters": 0,
        "total_optimized_parameters": backbone_params,
        "training_time_seconds": total_time,
        "final_total_loss": rows[-1]["total_loss"],
        "final_data_loss": rows[-1]["data_loss"],
        "seed": args.seed,
        **final_val,
    }

    pd.DataFrame([summary]).to_csv(
        logs_dir / "data_only_nn_training_summary.csv", index=False
    )

    print("\nTraining complete.")
    print(f"Checkpoint: {models_dir / 'data_only_nn.pth'}")
    print(f"Training summary: {logs_dir / 'data_only_nn_training_summary.csv'}")
    print("Held-out test set was not accessed.")


def main():
    args = parse_args()
    if not any(np.isclose(args.supervised_ratio, r) for r in ALLOWED_RATIOS):
        raise ValueError(f"--supervised-ratio must be one of {ALLOWED_RATIOS}")
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.hidden_layers < 1 or args.hidden_width < 1:
        raise ValueError("hidden-layers and hidden-width must be >= 1")
    train(args)


if __name__ == "__main__":
    main()
