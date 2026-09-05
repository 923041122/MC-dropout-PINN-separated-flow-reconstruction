
"""Train independently retrained MC-dropout cylinder PINNs.

This trainer uses the corrected cylinder sparse-data protocol and keeps the
held-out test set sealed. It is intended for Reviewer #8 dropout-rate and
dropout-placement ablations.

Key controls
------------
--dropout-rate:
    Dropout probability used DURING TRAINING. Each principal rate must be
    trained independently; do not change the rate only at inference time and
    present that as a retraining ablation.

--dropout-placement:
    none / input / middle / output / alternating / all

All configurations:
- use the same nested supervised set;
- use the same validation partition;
- use the same fixed collocation points;
- use the same optimizer/schedule/epochs/seed;
- have the same trainable backbone parameter count.

The held-out test set is not accessed.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from benchmark_config import DEFAULT_TRAINING, LAYER_MAT_PSI, REYNOLDS
from benchmark_tools import count_parameters, get_device, set_seed
from cylinder_train_v2 import (
    ALLOWED_RATIOS,
    batch_slice_with_wrap,
    compute_total_loss,
    evaluate_validation,
    load_protocol,
    load_reference_stack,
)
from pinn_model_dropout_ablation import (
    PlacementPINNNet,
    VALID_DROPOUT_PLACEMENTS,
)

try:
    from learning_schdule import ChainedScheduler
except ImportError:
    from learning_schedule import ChainedScheduler


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Corrected-protocol cylinder MC-dropout training with explicit "
            "rate and placement controls."
        )
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
        "--results-root",
        required=True,
    )

    p.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_TRAINING["epochs"],
    )
    p.add_argument(
        "--supervised-ratio",
        type=float,
        default=0.02,
    )
    p.add_argument(
        "--n-equation-points",
        type=int,
        default=100000,
    )
    p.add_argument(
        "--batch-size-data",
        type=int,
        default=2048,
    )
    p.add_argument(
        "--batch-size-equation",
        type=int,
        default=2048,
    )
    p.add_argument(
        "--validation-batch-size",
        type=int,
        default=20000,
    )
    p.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_TRAINING["learning_rate"],
    )
    p.add_argument(
        "--data-loss-weight",
        type=float,
        default=10.0,
    )
    p.add_argument(
        "--equation-loss-weight",
        type=float,
        default=0.1,
    )

    p.add_argument(
        "--dropout-rate",
        type=float,
        required=True,
    )
    p.add_argument(
        "--dropout-placement",
        choices=VALID_DROPOUT_PLACEMENTS,
        required=True,
    )

    p.add_argument("--seed", type=int, default=2025)
    p.add_argument(
        "--diagnostic-interval",
        type=int,
        default=50,
    )
    p.add_argument("--resume", action="store_true")

    return p.parse_args()


def safe_load_checkpoint(model, checkpoint, device):
    try:
        state = torch.load(
            checkpoint,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        state = torch.load(
            checkpoint,
            map_location=device,
        )

    model.load_state_dict(state, strict=True)


def train(args):
    if not any(
        np.isclose(args.supervised_ratio, r)
        for r in ALLOWED_RATIOS
    ):
        raise ValueError(
            f"--supervised-ratio must be one of {ALLOWED_RATIOS}"
        )

    if not (0.0 <= args.dropout_rate < 1.0):
        raise ValueError("--dropout-rate must satisfy 0 <= p < 1.")

    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")

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

    if len(full_data) != int(
        protocol_summary["total_reference_points"]
    ):
        raise RuntimeError(
            "Reference-data size does not match protocol."
        )

    train_data = full_data[
        torch.from_numpy(train_idx)
    ]
    validation_data = full_data[
        torch.from_numpy(val_idx)
    ]

    # Explicitly discard held-out indices so this trainer cannot use labels.
    heldout_count = len(test_idx)
    del test_idx

    equation_points = torch.tensor(
        eq_np,
        dtype=torch.float32,
    )

    model = PlacementPINNNet(
        LAYER_MAT_PSI,
        dropout_rate=args.dropout_rate,
        dropout_placement=args.dropout_placement,
    ).to(device)

    checkpoint = models_dir / "bpinn_dropout.pth"

    if args.resume and checkpoint.exists():
        safe_load_checkpoint(
            model=model,
            checkpoint=checkpoint,
            device=device,
        )
        print(f"[resume] loaded: {checkpoint}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.0,
    )

    scheduler = ChainedScheduler(
        optimizer,
        T_0=DEFAULT_TRAINING["scheduler_T0"],
        T_mul=DEFAULT_TRAINING["scheduler_Tmul"],
        eta_min=0.0,
        gamma=DEFAULT_TRAINING["decay_rate"],
        max_lr=args.learning_rate,
        warmup_steps=DEFAULT_TRAINING["warmup_steps"],
    )

    n_data = len(train_data)
    n_eq = len(equation_points)

    steps_per_epoch = max(
        math.ceil(n_data / args.batch_size_data),
        math.ceil(n_eq / args.batch_size_equation),
    )

    backbone_params = int(count_parameters(model))
    additional_params = 0
    total_optimized_params = backbone_params

    cfg = model.dropout_configuration()

    print("=" * 96)
    print("Cylinder dropout-ablation training")
    print(f"Device: {device}")
    print(f"Protocol: {args.protocol_root}")
    print(f"Supervised ratio: {args.supervised_ratio:.2%}")
    print(f"Supervised points: {n_data}")
    print(f"Validation points: {len(validation_data)}")
    print(
        f"Held-out test points: {heldout_count} "
        "(RESERVED AND NOT ACCESSED)"
    )
    print(f"Equation points: {n_eq}")
    print(f"Data batch size: {args.batch_size_data}")
    print(f"Equation batch size: {args.batch_size_equation}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Backbone parameters: {backbone_params}")
    print(f"Additional trainable parameters: {additional_params}")
    print(f"Total optimized parameters: {total_optimized_params}")
    print(f"Dropout rate: {args.dropout_rate}")
    print(f"Dropout placement: {args.dropout_placement}")
    print(
        "Dropout hidden-layer indices: "
        f"{cfg['dropout_hidden_layer_indices']}"
    )
    print(f"Hidden-layer count: {cfg['hidden_layer_count']}")
    print(f"Data loss weight: {args.data_loss_weight}")
    print(f"Equation loss weight: {args.equation_loss_weight}")
    print(f"Seed: {args.seed}")
    print("=" * 96)

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

            x = (
                db[:, 0:1]
                .to(device)
                .detach()
                .clone()
                .requires_grad_(True)
            )
            y = (
                db[:, 1:2]
                .to(device)
                .detach()
                .clone()
                .requires_grad_(True)
            )
            t = (
                db[:, 2:3]
                .to(device)
                .detach()
                .clone()
                .requires_grad_(True)
            )

            u = db[:, 3:4].to(device)
            v = db[:, 4:5].to(device)
            p = db[:, 5:6].to(device)

            xe = (
                eb[:, 0:1]
                .to(device)
                .detach()
                .clone()
                .requires_grad_(True)
            )
            ye = (
                eb[:, 1:2]
                .to(device)
                .detach()
                .clone()
                .requires_grad_(True)
            )
            te = (
                eb[:, 2:3]
                .to(device)
                .detach()
                .clone()
                .requires_grad_(True)
            )

            optimizer.zero_grad(set_to_none=True)

            data_loss = model.data_mse_psi(
                x, y, t, u, v, p
            )

            equation_loss = (
                model.equation_mse_dimensionless_psi(
                    xe, ye, te, Re=REYNOLDS
                )
            )

            total_loss = compute_total_loss(
                data_loss=data_loss,
                equation_loss=equation_loss,
                adaptive_loss=None,
                data_loss_weight=args.data_loss_weight,
                equation_loss_weight=args.equation_loss_weight,
            )

            if not torch.isfinite(total_loss):
                raise FloatingPointError(
                    f"Non-finite loss at epoch={epoch}, step={step+1}"
                )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=10.0,
            )
            optimizer.step()

            total_vals.append(
                float(total_loss.detach().cpu())
            )
            data_vals.append(
                float(data_loss.detach().cpu())
            )
            eq_vals.append(
                float(equation_loss.detach().cpu())
            )

        scheduler.step()

        elapsed = time.perf_counter() - start_time

        row = {
            "epoch": epoch,
            "supervised_ratio": args.supervised_ratio,
            "supervised_points": n_data,
            "validation_points": len(validation_data),
            "equation_points": n_eq,
            "dropout_rate": args.dropout_rate,
            "dropout_placement": args.dropout_placement,
            "dropout_hidden_layer_indices": ";".join(
                str(i)
                for i in cfg["dropout_hidden_layer_indices"]
            ),
            "hidden_layer_count": cfg["hidden_layer_count"],
            "batch_size_data": args.batch_size_data,
            "batch_size_equation": args.batch_size_equation,
            "steps_per_epoch": steps_per_epoch,
            "total_loss": float(np.mean(total_vals)),
            "data_loss": float(np.mean(data_vals)),
            "equation_loss": float(np.mean(eq_vals)),
            "weighted_data_loss": float(
                args.data_loss_weight
                * np.mean(data_vals)
            ),
            "weighted_equation_loss": float(
                args.equation_loss_weight
                * np.mean(eq_vals)
            ),
            "learning_rate": float(
                optimizer.param_groups[0]["lr"]
            ),
            "elapsed_seconds": elapsed,
            "backbone_parameters": backbone_params,
            "additional_trainable_parameters": additional_params,
            "total_optimized_parameters": total_optimized_params,
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
            # Dropout OFF here: this measures the deterministic inference of the
            # dropout-regularized checkpoint for model-selection purposes.
            val = evaluate_validation(
                model=model,
                validation_data=validation_data,
                device=device,
                batch_size=args.validation_batch_size,
            )

            validation_rows.append({
                "epoch": epoch,
                "supervised_ratio": args.supervised_ratio,
                "dropout_rate": args.dropout_rate,
                "dropout_placement": args.dropout_placement,
                "equation_loss_weight": args.equation_loss_weight,
                "seed": args.seed,
                "inference_mode": "dropout_off",
                **val,
            })

            val_text = (
                f" | val_u={val['val_u_rmse']:.3e}"
                f" val_v={val['val_v_rmse']:.3e}"
                f" val_p={val['val_p_rmse']:.3e}"
            )

        torch.save(
            model.state_dict(),
            checkpoint,
        )

        pd.DataFrame(rows).to_csv(
            logs_dir / "bpinn_dropout_training_log.csv",
            index=False,
        )

        if validation_rows:
            pd.DataFrame(validation_rows).to_csv(
                logs_dir / "bpinn_dropout_validation_log.csv",
                index=False,
            )

        print(
            f"[dropout p={args.dropout_rate:g}, "
            f"{args.dropout_placement}] "
            f"epoch {epoch:05d}/{args.epochs}"
            f" | total={row['total_loss']:.4e}"
            f" | data={row['data_loss']:.4e}"
            f" | eq={row['equation_loss']:.4e}"
            f" | lr={row['learning_rate']:.2e}"
            f" | time={elapsed:.1f}s"
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
        "protocol_version": str(
            protocol_summary["protocol_version"]
        ),
        "method": "bpinn_dropout",
        "label": "MC-dropout physics-constrained PINN",
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
        "steps_per_epoch": steps_per_epoch,
        "learning_rate_initial": args.learning_rate,
        "data_loss_weight": args.data_loss_weight,
        "equation_loss_weight": args.equation_loss_weight,
        "dropout_rate": args.dropout_rate,
        "dropout_placement": args.dropout_placement,
        "dropout_hidden_layer_indices": ";".join(
            str(i)
            for i in cfg["dropout_hidden_layer_indices"]
        ),
        "hidden_layer_count": cfg["hidden_layer_count"],
        "backbone_parameters": backbone_params,
        "additional_trainable_parameters": additional_params,
        "total_optimized_parameters": total_optimized_params,
        "training_time_seconds": total_time,
        "final_total_loss": rows[-1]["total_loss"],
        "final_data_loss": rows[-1]["data_loss"],
        "final_equation_loss": rows[-1]["equation_loss"],
        "validation_inference_mode": "dropout_off",
        "seed": args.seed,
        **final_val,
    }

    pd.DataFrame([summary]).to_csv(
        logs_dir / "bpinn_dropout_training_summary.csv",
        index=False,
    )

    print("\nDropout-ablation training complete.")
    print(f"Checkpoint: {checkpoint}")
    print(
        "Summary:",
        logs_dir / "bpinn_dropout_training_summary.csv",
    )
    print("Held-out test set was not accessed.")


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
