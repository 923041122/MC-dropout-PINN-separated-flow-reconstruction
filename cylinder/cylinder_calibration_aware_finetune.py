from __future__ import annotations

"""
Calibration-aware fine-tuning pilot for the frozen Re=3900 cylinder MC-dropout PINN.

Purpose
-------
This script adds a small, explicit distribution-level training term to the existing
MC-dropout physics-constrained PINN without changing the network architecture.

The base checkpoint is frozen BEFORE this pilot. Fine-tuning uses only:
    - the existing nested supervised TRAINING set,
    - the existing fixed collocation points,
    - the existing fixed VALIDATION set for deterministic diagnostics only.

The held-out TEST set is never loaded or used by this script.

Method
------
Base objective:
    L_base = w_data * L_data + w_eq * L_eq

Calibration-aware extension:
    L_total = L_base + lambda_uq * L_CRPS

L_CRPS is an empirical sample-based CRPS computed from a small number of stochastic
dropout forward passes on a supervised mini-batch. The three variables (u, v, p)
are normalized by TRAINING-set target standard deviations before CRPS is computed,
so one variable does not dominate purely because of scale.

Important design control
------------------------
Run an otherwise identical lambda_uq = 0 branch. This isolates the effect of the
proper-scoring regularizer from the effect of simply continuing training longer.

Recommended pilot:
    lambda_uq in {0, 0.01, 0.1}
    epochs = 150
    learning rate = 1e-4
    minimum learning rate = 1e-5
    uq_mc_samples = 4
    uq_batch_size = 256

The short fine-tuning stage intentionally uses a monotone cosine decay rather than
the original warm-restart scheduler. Restarting the learning rate during a short
fine-tune can move the model far from the frozen checkpoint and confound the pilot.

Final model selection MUST use validation metrics only. Do not open the held-out
test set until lambda_uq and all other pilot settings are frozen.
"""

import argparse
import math
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch

from benchmark_config import LAYER_MAT_PSI, REYNOLDS
from benchmark_tools import count_parameters, get_device, set_seed
from cylinder_train_v2 import (
    ALLOWED_RATIOS,
    batch_slice_with_wrap,
    evaluate_validation,
    load_protocol,
    load_reference_stack,
)
from pinn_model_dropout_ablation import PlacementPINNNet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Calibration-aware fine-tuning pilot for the frozen cylinder MC-dropout PINN."
    )

    p.add_argument(
        "--data-path",
        default="./2d_cylinder_Re3900_100x100_kw_sst.mat",
    )
    p.add_argument(
        "--protocol-root",
        default="./results_cylinder_protocol_v1_1/protocol",
    )
    p.add_argument(
        "--base-checkpoint",
        default="./cylinder_bpinn_final_p0002_all/models/bpinn_dropout.pth",
    )
    p.add_argument("--results-root", required=True)

    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--supervised-ratio", type=float, default=0.02)
    p.add_argument("--n-equation-points", type=int, default=100000)

    p.add_argument("--batch-size-data", type=int, default=2048)
    p.add_argument("--batch-size-equation", type=int, default=2048)
    p.add_argument("--uq-batch-size", type=int, default=256)
    p.add_argument("--validation-batch-size", type=int, default=20000)

    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--min-learning-rate", type=float, default=1e-5)

    p.add_argument("--data-loss-weight", type=float, default=10.0)
    p.add_argument("--equation-loss-weight", type=float, default=0.1)
    p.add_argument("--uq-loss-weight", type=float, required=True)
    p.add_argument("--uq-mc-samples", type=int, default=4)

    p.add_argument("--dropout-rate", type=float, default=0.002)
    p.add_argument(
        "--dropout-placement",
        choices=("none", "input", "middle", "output", "alternating", "all"),
        default="all",
    )

    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--diagnostic-interval", type=int, default=25)
    return p.parse_args()


def safe_load_state(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def validate_args(args: argparse.Namespace) -> None:
    if not any(np.isclose(args.supervised_ratio, r) for r in ALLOWED_RATIOS):
        raise ValueError(f"--supervised-ratio must be one of {ALLOWED_RATIOS}")
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.batch_size_data < 1 or args.batch_size_equation < 1:
        raise ValueError("Data/equation batch sizes must be >= 1")
    if args.uq_batch_size < 1:
        raise ValueError("--uq-batch-size must be >= 1")
    if args.validation_batch_size < 1:
        raise ValueError("--validation-batch-size must be >= 1")
    if args.uq_mc_samples < 2 and args.uq_loss_weight > 0.0:
        raise ValueError("--uq-mc-samples must be >= 2 when UQ loss is enabled")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be > 0")
    if not (0.0 <= args.min_learning_rate <= args.learning_rate):
        raise ValueError("--min-learning-rate must be in [0, learning-rate]")
    if args.uq_loss_weight < 0.0:
        raise ValueError("--uq-loss-weight must be >= 0")
    if not (0.0 <= args.dropout_rate < 1.0):
        raise ValueError("--dropout-rate must satisfy 0 <= p < 1")


def component_scales_from_training(train_data: torch.Tensor) -> torch.Tensor:
    """
    Return frozen u/v/p scales computed ONLY from training targets.

    Using training-set standard deviations keeps the CRPS term dimensionless
    and prevents p or one velocity component from dominating only because of
    numerical scale.
    """
    targets = train_data[:, 3:6].float()
    scales = targets.std(dim=0, unbiased=False)
    floor = torch.tensor(1e-6, dtype=scales.dtype)
    return torch.maximum(scales, floor)


def stochastic_prediction(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """
    One stochastic dropout prediction with create_graph=True.

    Shape returned: [batch, 3] for u, v, p.
    """
    u_pred, v_pred, p_pred = model.predict_fields(
        x,
        y,
        t,
        create_graph=True,
    )
    return torch.cat([u_pred, v_pred, p_pred], dim=1)


def empirical_crps_loss(
    model: torch.nn.Module,
    batch: torch.Tensor,
    device: torch.device,
    target_scales: torch.Tensor,
    mc_samples: int,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Differentiable empirical CRPS from stochastic MC-dropout samples.

    Empirical CRPS:
        E|X - y| - 0.5 E|X - X'|

    The calculation is performed in training-set-standardized u/v/p units.
    torch.abs() is sub-differentiable and works directly with autograd.
    """
    x = batch[:, 0:1].to(device).detach().clone().requires_grad_(True)
    y = batch[:, 1:2].to(device).detach().clone().requires_grad_(True)
    t = batch[:, 2:3].to(device).detach().clone().requires_grad_(True)
    target = batch[:, 3:6].to(device)

    samples = []
    for _ in range(mc_samples):
        samples.append(stochastic_prediction(model, x, y, t))

    stack = torch.stack(samples, dim=0)  # [M, B, 3]

    scales = target_scales.to(device).view(1, 1, 3)
    sample_norm = stack / scales
    target_norm = target.view(1, target.shape[0], 3) / scales

    term1 = torch.abs(sample_norm - target_norm).mean(dim=0)  # [B, 3]

    # Pairwise sample distance, including diagonal terms (which are zero).
    pairwise = torch.abs(
        sample_norm[:, None, :, :] - sample_norm[None, :, :, :]
    ).mean(dim=(0, 1))  # [B, 3]

    crps_point_var = term1 - 0.5 * pairwise
    crps_by_var = crps_point_var.mean(dim=0)
    crps_total = crps_by_var.mean()

    details = {
        "crps_u": crps_by_var[0],
        "crps_v": crps_by_var[1],
        "crps_p": crps_by_var[2],
    }
    return crps_total, details


def main() -> None:
    args = parse_args()
    validate_args(args)

    set_seed(args.seed)
    device = get_device()

    results_root = Path(args.results_root)
    models_dir = results_root / "models"
    logs_dir = results_root / "training_logs"
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    train_idx, val_idx, test_idx, eq_np, protocol_summary = load_protocol(
        protocol_root=Path(args.protocol_root),
        ratio=args.supervised_ratio,
        n_equation_points=args.n_equation_points,
    )

    full_data = load_reference_stack(Path(args.data_path))

    if len(full_data) != int(protocol_summary["total_reference_points"]):
        raise RuntimeError("Reference-data size does not match protocol.")

    train_data = full_data[torch.from_numpy(train_idx)]
    validation_data = full_data[torch.from_numpy(val_idx)]

    heldout_count = len(test_idx)
    del test_idx

    equation_points = torch.tensor(eq_np, dtype=torch.float32)

    target_scales = component_scales_from_training(train_data)

    model = PlacementPINNNet(
        LAYER_MAT_PSI,
        dropout_rate=args.dropout_rate,
        dropout_placement=args.dropout_placement,
    ).to(device)

    base_checkpoint = Path(args.base_checkpoint)
    if not base_checkpoint.exists():
        raise FileNotFoundError(base_checkpoint)

    model.load_state_dict(
        safe_load_state(base_checkpoint, device),
        strict=True,
    )

    checkpoint = models_dir / "bpinn_calibration_aware.pth"

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.0,
    )

    # Short fine-tuning should decay monotonically rather than restart.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
        eta_min=args.min_learning_rate,
    )

    n_data = len(train_data)
    n_eq = len(equation_points)
    steps_per_epoch = max(
        math.ceil(n_data / args.batch_size_data),
        math.ceil(n_eq / args.batch_size_equation),
    )

    backbone_params = int(count_parameters(model))
    cfg = model.dropout_configuration()

    print("=" * 104)
    print("Cylinder calibration-aware MC-dropout fine-tuning pilot")
    print(f"Device: {device}")
    print(f"Base checkpoint: {base_checkpoint}")
    print(f"Output checkpoint: {checkpoint}")
    print(f"Protocol: {args.protocol_root}")
    print(f"Supervised ratio: {args.supervised_ratio:.2%}")
    print(f"Supervised points: {n_data}")
    print(f"Validation points: {len(validation_data)}")
    print(f"Held-out test points: {heldout_count} (RESERVED AND NOT ACCESSED)")
    print(f"Equation points: {n_eq}")
    print(f"Dropout rate: {args.dropout_rate}")
    print(f"Dropout placement: {args.dropout_placement}")
    print(f"Dropout hidden layers: {cfg['dropout_hidden_layer_indices']}")
    print(f"Backbone parameters: {backbone_params}")
    print(f"Data loss weight: {args.data_loss_weight}")
    print(f"Equation loss weight: {args.equation_loss_weight}")
    print(f"UQ/CRPS loss weight: {args.uq_loss_weight}")
    print(f"UQ MC samples during training: {args.uq_mc_samples}")
    print(f"UQ batch size: {args.uq_batch_size}")
    print(f"Learning rate: {args.learning_rate} -> {args.min_learning_rate}")
    print(
        "Training-only target scales (u,v,p): "
        + ", ".join(f"{float(x):.6g}" for x in target_scales)
    )
    print("=" * 104)

    rows = []
    validation_rows = []
    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()

        data_perm = torch.randperm(n_data)
        eq_perm = torch.randperm(n_eq)
        data_epoch = train_data[data_perm]
        eq_epoch = equation_points[eq_perm]

        total_vals = []
        data_vals = []
        eq_vals = []
        uq_vals = []
        crps_u_vals = []
        crps_v_vals = []
        crps_p_vals = []

        for step in range(steps_per_epoch):
            db = batch_slice_with_wrap(
                data_epoch,
                step * args.batch_size_data,
                args.batch_size_data,
            )
            eb = batch_slice_with_wrap(
                eq_epoch,
                step * args.batch_size_equation,
                args.batch_size_equation,
            )

            x = db[:, 0:1].to(device).detach().clone().requires_grad_(True)
            y = db[:, 1:2].to(device).detach().clone().requires_grad_(True)
            t = db[:, 2:3].to(device).detach().clone().requires_grad_(True)
            u = db[:, 3:4].to(device)
            v = db[:, 4:5].to(device)
            p = db[:, 5:6].to(device)

            xe = eb[:, 0:1].to(device).detach().clone().requires_grad_(True)
            ye = eb[:, 1:2].to(device).detach().clone().requires_grad_(True)
            te = eb[:, 2:3].to(device).detach().clone().requires_grad_(True)

            optimizer.zero_grad(set_to_none=True)

            data_loss = model.data_mse_psi(x, y, t, u, v, p)

            if float(args.equation_loss_weight) == 0.0:
                equation_loss = torch.zeros((), device=device)
            else:
                equation_loss = model.equation_mse_dimensionless_psi(
                    xe, ye, te, Re=REYNOLDS
                )

            if args.uq_loss_weight > 0.0:
                uq_n = min(args.uq_batch_size, len(db))
                uq_batch = db[:uq_n]
                uq_loss, uq_details = empirical_crps_loss(
                    model=model,
                    batch=uq_batch,
                    device=device,
                    target_scales=target_scales,
                    mc_samples=args.uq_mc_samples,
                )
            else:
                uq_loss = torch.zeros((), device=device)
                uq_details = {
                    "crps_u": torch.zeros((), device=device),
                    "crps_v": torch.zeros((), device=device),
                    "crps_p": torch.zeros((), device=device),
                }

            total_loss = (
                float(args.data_loss_weight) * data_loss
                + float(args.equation_loss_weight) * equation_loss
                + float(args.uq_loss_weight) * uq_loss
            )

            if not torch.isfinite(total_loss):
                raise FloatingPointError(
                    f"Non-finite total loss at epoch={epoch}, step={step+1}"
                )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            total_vals.append(float(total_loss.detach().cpu()))
            data_vals.append(float(data_loss.detach().cpu()))
            eq_vals.append(float(equation_loss.detach().cpu()))
            uq_vals.append(float(uq_loss.detach().cpu()))
            crps_u_vals.append(float(uq_details["crps_u"].detach().cpu()))
            crps_v_vals.append(float(uq_details["crps_v"].detach().cpu()))
            crps_p_vals.append(float(uq_details["crps_p"].detach().cpu()))

        scheduler.step()

        elapsed = time.perf_counter() - start_time

        row = {
            "epoch": epoch,
            "supervised_ratio": args.supervised_ratio,
            "supervised_points": n_data,
            "validation_points": len(validation_data),
            "heldout_test_points_reserved": heldout_count,
            "heldout_test_accessed_during_training": False,
            "equation_points": n_eq,
            "batch_size_data": args.batch_size_data,
            "batch_size_equation": args.batch_size_equation,
            "uq_batch_size": args.uq_batch_size,
            "steps_per_epoch": steps_per_epoch,
            "dropout_rate": args.dropout_rate,
            "dropout_placement": args.dropout_placement,
            "uq_mc_samples": args.uq_mc_samples,
            "data_loss_weight": args.data_loss_weight,
            "equation_loss_weight": args.equation_loss_weight,
            "uq_loss_weight": args.uq_loss_weight,
            "total_loss": float(np.mean(total_vals)),
            "data_loss": float(np.mean(data_vals)),
            "equation_loss": float(np.mean(eq_vals)),
            "uq_crps_loss": float(np.mean(uq_vals)),
            "uq_crps_u": float(np.mean(crps_u_vals)),
            "uq_crps_v": float(np.mean(crps_v_vals)),
            "uq_crps_p": float(np.mean(crps_p_vals)),
            "weighted_data_loss": float(args.data_loss_weight * np.mean(data_vals)),
            "weighted_equation_loss": float(
                args.equation_loss_weight * np.mean(eq_vals)
            ),
            "weighted_uq_loss": float(args.uq_loss_weight * np.mean(uq_vals)),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_seconds": elapsed,
            "training_scale_u": float(target_scales[0]),
            "training_scale_v": float(target_scales[1]),
            "training_scale_p": float(target_scales[2]),
            "backbone_parameters": backbone_params,
            "additional_trainable_parameters": 0,
            "total_optimized_parameters": backbone_params,
            "seed": args.seed,
        }
        rows.append(row)

        should_validate = (
            epoch == args.epochs
            or (
                args.diagnostic_interval > 0
                and epoch % args.diagnostic_interval == 0
            )
        )

        val_text = ""
        if should_validate:
            # Deterministic/dropout-off validation only.
            # MC50 UQ validation is deliberately done later by the existing
            # cylinder_bpinn_final_validation_mc50_calibrate.py script.
            val = evaluate_validation(
                model=model,
                validation_data=validation_data,
                device=device,
                batch_size=args.validation_batch_size,
            )
            validation_rows.append({
                "epoch": epoch,
                "uq_loss_weight": args.uq_loss_weight,
                "inference_mode": "dropout_off",
                **val,
            })
            val_text = (
                f" | val_u={val['val_u_rmse']:.3e}"
                f" val_v={val['val_v_rmse']:.3e}"
                f" val_p={val['val_p_rmse']:.3e}"
            )

        torch.save(model.state_dict(), checkpoint)
        pd.DataFrame(rows).to_csv(
            logs_dir / "calibration_aware_training_log.csv",
            index=False,
        )
        if validation_rows:
            pd.DataFrame(validation_rows).to_csv(
                logs_dir / "calibration_aware_validation_log.csv",
                index=False,
            )

        print(
            f"[lambda_uq={args.uq_loss_weight:g}] "
            f"epoch {epoch:04d}/{args.epochs}"
            f" | total={row['total_loss']:.4e}"
            f" data={row['data_loss']:.4e}"
            f" eq={row['equation_loss']:.4e}"
            f" uq={row['uq_crps_loss']:.4e}"
            f" lr={row['learning_rate']:.2e}"
            f" time={elapsed:.1f}s"
            f"{val_text}"
        )

    final_val = evaluate_validation(
        model=model,
        validation_data=validation_data,
        device=device,
        batch_size=args.validation_batch_size,
    )

    total_time = time.perf_counter() - start_time

    summary = {
        "protocol_version": str(protocol_summary["protocol_version"]),
        "method": "crps_regularized_mc_dropout_finetune",
        "base_checkpoint": str(base_checkpoint),
        "checkpoint": str(checkpoint),
        "epochs": args.epochs,
        "supervised_ratio": args.supervised_ratio,
        "supervised_points": n_data,
        "validation_points": len(validation_data),
        "heldout_test_points_reserved": heldout_count,
        "heldout_test_accessed_during_training": False,
        "equation_points": n_eq,
        "batch_size_data": args.batch_size_data,
        "batch_size_equation": args.batch_size_equation,
        "uq_batch_size": args.uq_batch_size,
        "uq_mc_samples": args.uq_mc_samples,
        "dropout_rate": args.dropout_rate,
        "dropout_placement": args.dropout_placement,
        "dropout_hidden_layer_indices": ";".join(
            str(i) for i in cfg["dropout_hidden_layer_indices"]
        ),
        "data_loss_weight": args.data_loss_weight,
        "equation_loss_weight": args.equation_loss_weight,
        "uq_loss_weight": args.uq_loss_weight,
        "learning_rate_initial": args.learning_rate,
        "learning_rate_minimum": args.min_learning_rate,
        "scheduler": "CosineAnnealingLR_no_restarts",
        "training_scale_u": float(target_scales[0]),
        "training_scale_v": float(target_scales[1]),
        "training_scale_p": float(target_scales[2]),
        "backbone_parameters": backbone_params,
        "additional_trainable_parameters": 0,
        "total_optimized_parameters": backbone_params,
        "training_time_seconds": total_time,
        "final_total_loss": rows[-1]["total_loss"],
        "final_data_loss": rows[-1]["data_loss"],
        "final_equation_loss": rows[-1]["equation_loss"],
        "final_uq_crps_loss": rows[-1]["uq_crps_loss"],
        "seed": args.seed,
        **final_val,
    }

    pd.DataFrame([summary]).to_csv(
        logs_dir / "calibration_aware_training_summary.csv",
        index=False,
    )

    print("\nCalibration-aware fine-tuning pilot complete.")
    print(f"Checkpoint: {checkpoint}")
    print(
        "Summary:",
        logs_dir / "calibration_aware_training_summary.csv",
    )
    print("Held-out test set was NOT accessed.")
    print(
        "NEXT: run cylinder_bpinn_final_validation_mc50_calibrate.py on this "
        "checkpoint. Do not run held-out evaluation until lambda_uq is frozen."
    )


if __name__ == "__main__":
    main()
