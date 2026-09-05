"""Train benchmark PINN baselines for Re=3900 turbulent cylinder flow."""

import argparse
import time

import numpy as np
import pandas as pd
import torch

from benchmark_config import (
    DATA_PATH,
    DEFAULT_TRAINING,
    LAYER_MAT_PSI,
    METHODS,
    MODEL_ROOT,
    REYNOLDS,
    RESULT_ROOT,
)
from benchmark_tools import build_model, count_parameters, get_device, set_seed
from learning_schdule import ChainedScheduler
from pinn_model import (
    generate_eqp_rect,
    print_field_statistics,
    read_2D_data,
    shuffle_data,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train one or all benchmark PINN baselines."
    )

    parser.add_argument(
        "--method",
        default="all",
        choices=["all"] + list(METHODS.keys()),
        help="Choose one baseline to train, or use 'all' to train all configured methods.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_TRAINING["epochs"],
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--ratio",
        type=float,
        default=DEFAULT_TRAINING["ratio"],
        help="Mini-batch sampling ratio used for data points and equation points.",
    )

    parser.add_argument(
        "--n-equation-points",
        type=int,
        default=DEFAULT_TRAINING["n_equation_points"],
        help="Number of collocation points for the Navier-Stokes residual.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_TRAINING["learning_rate"],
        help="Initial learning rate.",
    )

    parser.add_argument(
        "--data-loss-weight",
        type=float,
        default=10.0,
        help="Weight for supervised data loss.",
    )

    parser.add_argument(
        "--equation-loss-weight",
        type=float,
        default=1.0,
        help="Weight for Navier-Stokes equation residual loss.",
    )

    parser.add_argument(
        "--diagnostic-interval",
        type=int,
        default=50,
        help="Print diagnostic field statistics every N epochs. Set to 0 to disable.",
    )

    parser.add_argument(
        "--disable-diagnostics",
        action="store_true",
        help="Disable field statistics diagnostics.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Random seed for reproducibility.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from an existing checkpoint if available.",
    )

    return parser.parse_args()


def _base_layer_index(key):
    parts = key.split(".")

    if len(parts) < 3:
        return -1

    try:
        return int(parts[1])
    except ValueError:
        return -1


def _load_linear_layers_by_order(model, source_state_dict):
    """Load Linear layer weights by order, ignoring Sequential index shifts from Dropout."""

    target_state_dict = model.state_dict()

    source_weight_keys = [
        key
        for key in source_state_dict.keys()
        if key.startswith("base.") and key.endswith(".weight")
    ]

    target_weight_keys = [
        key
        for key in target_state_dict.keys()
        if key.startswith("base.") and key.endswith(".weight")
    ]

    source_weight_keys = sorted(source_weight_keys, key=_base_layer_index)
    target_weight_keys = sorted(target_weight_keys, key=_base_layer_index)

    if len(source_weight_keys) != len(target_weight_keys):
        raise RuntimeError(
            "Cannot map checkpoint by Linear-layer order: "
            f"source linear layers={len(source_weight_keys)}, "
            f"target linear layers={len(target_weight_keys)}"
        )

    loaded_layers = 0

    for source_weight_key, target_weight_key in zip(source_weight_keys, target_weight_keys):
        source_bias_key = source_weight_key.replace(".weight", ".bias")
        target_bias_key = target_weight_key.replace(".weight", ".bias")

        if source_weight_key not in source_state_dict:
            continue
        if source_bias_key not in source_state_dict:
            continue
        if target_weight_key not in target_state_dict:
            continue
        if target_bias_key not in target_state_dict:
            continue

        source_weight = source_state_dict[source_weight_key]
        source_bias = source_state_dict[source_bias_key]

        target_weight = target_state_dict[target_weight_key]
        target_bias = target_state_dict[target_bias_key]

        if source_weight.shape != target_weight.shape:
            raise RuntimeError(
                f"Weight shape mismatch: {source_weight_key} {source_weight.shape} "
                f"-> {target_weight_key} {target_weight.shape}"
            )

        if source_bias.shape != target_bias.shape:
            raise RuntimeError(
                f"Bias shape mismatch: {source_bias_key} {source_bias.shape} "
                f"-> {target_bias_key} {target_bias.shape}"
            )

        target_state_dict[target_weight_key] = source_weight
        target_state_dict[target_bias_key] = source_bias
        loaded_layers += 1

    model.load_state_dict(target_state_dict, strict=True)

    return loaded_layers


def load_checkpoint_if_needed(model, checkpoint, device, resume):
    """Resume training if requested.

    For normal debugging, use resume=False to make sure the model trains from scratch.
    """

    if not resume:
        return

    if not checkpoint.exists():
        print(f"[resume] checkpoint not found, train from scratch: {checkpoint}")
        return

    try:
        state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint, map_location=device)

    try:
        model.load_state_dict(state_dict)
        print(f"[resume] loaded checkpoint strictly: {checkpoint}")
    except RuntimeError as exc:
        print("[resume] strict checkpoint loading failed.")
        print("[resume] trying order-based Linear-layer transfer instead.")
        print(f"[resume] original error: {exc}")

        loaded_layers = _load_linear_layers_by_order(
            model=model,
            source_state_dict=state_dict,
        )

        print(
            f"[resume] loaded {loaded_layers} Linear layers by order from: {checkpoint}"
        )


def build_optimizer(model, cfg, adaptive_loss, learning_rate):
    trainable_params = list(model.parameters())

    if adaptive_loss is not None:
        trainable_params.append(adaptive_loss)

    optimizer = torch.optim.Adam(
        trainable_params,
        lr=learning_rate,
        weight_decay=cfg.get("weight_decay", 0.0),
    )

    return optimizer


def compute_total_loss(
    data_loss,
    equation_loss,
    adaptive_loss,
    data_loss_weight=1.0,
    equation_loss_weight=1.0,
):
    """Compute weighted total loss.

    Debug recommendation for B-PINN:
        data_loss_weight = 10.0
        equation_loss_weight = 1.0

    If adaptive_loss is used:
        L = exp(-s1) * w_data * L_data
          + exp(-s2) * w_eq   * L_eq
          + s1 + s2
    """

    weighted_data_loss = data_loss_weight * data_loss
    weighted_equation_loss = equation_loss_weight * equation_loss

    if adaptive_loss is None:
        return weighted_data_loss + weighted_equation_loss

    return (
        torch.exp(-adaptive_loss[0]) * weighted_data_loss
        + torch.exp(-adaptive_loss[1]) * weighted_equation_loss
        + adaptive_loss.sum()
    )


def _make_grad_tensor(batch, col, device):
    """Create a tensor that can be used for autograd derivatives."""

    return batch[:, col:col + 1].to(device).detach().clone().requires_grad_(True)


def _make_plain_tensor(batch, col, device):
    """Create a normal target tensor."""

    return batch[:, col:col + 1].to(device).detach().clone()


def _run_training_diagnostic(model, method_name, diagnostic_batch, device, epoch):
    """Print predicted/reference field statistics on a small diagnostic batch."""

    model.eval()

    x_diag = diagnostic_batch[:, 0:1].to(device)
    y_diag = diagnostic_batch[:, 1:2].to(device)
    t_diag = diagnostic_batch[:, 2:3].to(device)

    u_ref = diagnostic_batch[:, 3:4].to(device)
    v_ref = diagnostic_batch[:, 4:5].to(device)
    p_ref = diagnostic_batch[:, 5:6].to(device)

    u_pred, v_pred, p_pred = model.predict_fields_safe(
        x_diag,
        y_diag,
        t_diag,
        create_graph=False,
        train_mode=False,
    )

    print_field_statistics(
        name=f"{method_name} diagnostic epoch {epoch}",
        u_pred=u_pred,
        v_pred=v_pred,
        p_pred=p_pred,
        u_ref=u_ref,
        v_ref=v_ref,
        p_ref=p_ref,
    )

    model.train()


def train_method(method_name, cfg, args, x_random, eqa_points, device):
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)

    metrics_dir = RESULT_ROOT / "training_logs"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    model = build_model(cfg, LAYER_MAT_PSI).to(device)
    checkpoint = cfg["checkpoint"]
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    load_checkpoint_if_needed(model, checkpoint, device, args.resume)

    adaptive_loss = None

    if cfg.get("adaptive_loss", False):
        adaptive_loss = torch.nn.Parameter(torch.zeros(2, device=device))

    learning_rate = getattr(args, "learning_rate", DEFAULT_TRAINING["learning_rate"])
    data_loss_weight = getattr(args, "data_loss_weight", 1.0)
    equation_loss_weight = getattr(args, "equation_loss_weight", 1.0)
    diagnostic_interval = getattr(args, "diagnostic_interval", 50)
    disable_diagnostics = getattr(args, "disable_diagnostics", False)

    optimizer = build_optimizer(
        model=model,
        cfg=cfg,
        adaptive_loss=adaptive_loss,
        learning_rate=learning_rate,
    )

    scheduler = ChainedScheduler(
        optimizer,
        T_0=DEFAULT_TRAINING["scheduler_T0"],
        T_mul=DEFAULT_TRAINING["scheduler_Tmul"],
        eta_min=0.0,
        gamma=DEFAULT_TRAINING["decay_rate"],
        max_lr=learning_rate,
        warmup_steps=DEFAULT_TRAINING["warmup_steps"],
    )

    batch_size_data = max(1, int(args.ratio * x_random.shape[0]))
    batch_size_eqa = max(1, int(args.ratio * eqa_points.shape[0]))

    inner_iter = int(np.ceil(x_random.size(0) / batch_size_data))

    # Fixed small diagnostic batch.
    diagnostic_size = min(4096, x_random.shape[0])
    diagnostic_batch = x_random[:diagnostic_size].clone()

    print("=" * 80)
    print(f"Training method: {method_name}")
    print(f"Label: {cfg.get('label', method_name)}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Model type: {cfg.get('model_type', 'psi')}")
    print(f"Dropout rate: {cfg.get('dropout_rate', 0.0)}")
    print(f"Weight decay: {cfg.get('weight_decay', 0.0)}")
    print(f"Adaptive loss: {cfg.get('adaptive_loss', False)}")
    print(f"Parameters: {count_parameters(model)}")
    print(f"Learning rate: {learning_rate}")
    print(f"Data loss weight: {data_loss_weight}")
    print(f"Equation loss weight: {equation_loss_weight}")
    print(f"Data batch size: {batch_size_data}")
    print(f"Equation batch size: {batch_size_eqa}")
    print(f"Inner iterations per epoch: {inner_iter}")
    print(f"Diagnostic interval: {diagnostic_interval}")
    print(f"Diagnostics disabled: {disable_diagnostics}")
    print("=" * 80)

    rows = []
    start_time = time.perf_counter()

    if not disable_diagnostics:
        print(f"[{method_name}] Initial diagnostic before training:")
        _run_training_diagnostic(
            model=model,
            method_name=method_name,
            diagnostic_batch=diagnostic_batch,
            device=device,
            epoch=0,
        )

    for epoch in range(args.epochs):
        model.train()

        # Shuffle supervised data every epoch.
        permutation_data = torch.randperm(x_random.shape[0])
        x_epoch = x_random[permutation_data]

        # Shuffle equation points every epoch.
        permutation_eqa = torch.randperm(eqa_points.shape[0])
        eqa_epoch = eqa_points[permutation_eqa]

        epoch_total_loss = []
        epoch_data_loss = []
        epoch_equation_loss = []
        epoch_weighted_data_loss = []
        epoch_weighted_equation_loss = []

        for batch_iter in range(inner_iter):
            data_start = batch_iter * batch_size_data
            data_end = min((batch_iter + 1) * batch_size_data, x_epoch.shape[0])

            eqa_start = batch_iter * batch_size_eqa
            eqa_end = min((batch_iter + 1) * batch_size_eqa, eqa_epoch.shape[0])

            data_batch = x_epoch[data_start:data_end]
            equation_batch = eqa_epoch[eqa_start:eqa_end]

            if data_batch.numel() == 0:
                continue

            if equation_batch.numel() == 0:
                # Wrap around if equation points are exhausted.
                equation_batch = eqa_epoch[0:batch_size_eqa]

            x_train = _make_grad_tensor(data_batch, 0, device)
            y_train = _make_grad_tensor(data_batch, 1, device)
            t_train = _make_grad_tensor(data_batch, 2, device)

            u_train = _make_plain_tensor(data_batch, 3, device)
            v_train = _make_plain_tensor(data_batch, 4, device)
            p_train = _make_plain_tensor(data_batch, 5, device)

            x_eqa = _make_grad_tensor(equation_batch, 0, device)
            y_eqa = _make_grad_tensor(equation_batch, 1, device)
            t_eqa = _make_grad_tensor(equation_batch, 2, device)

            optimizer.zero_grad(set_to_none=True)

            data_loss = model.data_mse_psi(
                x_train,
                y_train,
                t_train,
                u_train,
                v_train,
                p_train,
            )

            equation_loss = model.equation_mse_dimensionless_psi(
                x_eqa,
                y_eqa,
                t_eqa,
                Re=REYNOLDS,
            )

            total_loss = compute_total_loss(
                data_loss=data_loss,
                equation_loss=equation_loss,
                adaptive_loss=adaptive_loss,
                data_loss_weight=data_loss_weight,
                equation_loss_weight=equation_loss_weight,
            )

            if not torch.isfinite(total_loss):
                raise FloatingPointError(
                    f"[{method_name}] Non-finite loss detected at "
                    f"epoch={epoch + 1}, batch={batch_iter + 1}. "
                    f"total_loss={total_loss.item()}, "
                    f"data_loss={data_loss.item()}, "
                    f"equation_loss={equation_loss.item()}"
                )

            total_loss.backward()

            # Mild gradient clipping helps avoid occasional instability.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

            optimizer.step()

            epoch_total_loss.append(float(total_loss.detach().cpu()))
            epoch_data_loss.append(float(data_loss.detach().cpu()))
            epoch_equation_loss.append(float(equation_loss.detach().cpu()))
            epoch_weighted_data_loss.append(
                float((data_loss_weight * data_loss).detach().cpu())
            )
            epoch_weighted_equation_loss.append(
                float((equation_loss_weight * equation_loss).detach().cpu())
            )

            del (
                x_train,
                y_train,
                t_train,
                u_train,
                v_train,
                p_train,
                x_eqa,
                y_eqa,
                t_eqa,
                data_loss,
                equation_loss,
                total_loss,
            )

        scheduler.step()

        mean_total_loss = float(np.mean(epoch_total_loss))
        mean_data_loss = float(np.mean(epoch_data_loss))
        mean_equation_loss = float(np.mean(epoch_equation_loss))
        mean_weighted_data_loss = float(np.mean(epoch_weighted_data_loss))
        mean_weighted_equation_loss = float(np.mean(epoch_weighted_equation_loss))

        elapsed_seconds = time.perf_counter() - start_time

        row = {
            "method": method_name,
            "epoch": epoch + 1,
            "total_loss": mean_total_loss,
            "data_loss": mean_data_loss,
            "equation_loss": mean_equation_loss,
            "weighted_data_loss": mean_weighted_data_loss,
            "weighted_equation_loss": mean_weighted_equation_loss,
            "data_loss_weight": data_loss_weight,
            "equation_loss_weight": equation_loss_weight,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "parameters": count_parameters(model),
            "training_time_seconds": elapsed_seconds,
        }

        if adaptive_loss is not None:
            row["adaptive_log_var_data"] = float(adaptive_loss[0].detach().cpu())
            row["adaptive_log_var_equation"] = float(adaptive_loss[1].detach().cpu())
            row["adaptive_weight_data"] = float(
                torch.exp(-adaptive_loss[0]).detach().cpu()
            )
            row["adaptive_weight_equation"] = float(
                torch.exp(-adaptive_loss[1]).detach().cpu()
            )

        rows.append(row)

        # Keep saving model.state_dict() to remain compatible with your evaluation code.
        torch.save(model.state_dict(), checkpoint)

        log_path = metrics_dir / f"{method_name}_training_log.csv"
        pd.DataFrame(rows).to_csv(log_path, index=False)

        print(
            f"[{method_name}] "
            f"epoch {epoch + 1}/{args.epochs} | "
            f"total={mean_total_loss:.6e} | "
            f"data={mean_data_loss:.6e} | "
            f"eq={mean_equation_loss:.6e} | "
            f"w_data={mean_weighted_data_loss:.6e} | "
            f"w_eq={mean_weighted_equation_loss:.6e} | "
            f"lr={optimizer.param_groups[0]['lr']:.3e} | "
            f"time={elapsed_seconds:.2f}s"
        )

        if (
            not disable_diagnostics
            and diagnostic_interval is not None
            and diagnostic_interval > 0
            and ((epoch + 1) % diagnostic_interval == 0 or (epoch + 1) == args.epochs)
        ):
            _run_training_diagnostic(
                model=model,
                method_name=method_name,
                diagnostic_batch=diagnostic_batch,
                device=device,
                epoch=epoch + 1,
            )

    total_training_time = time.perf_counter() - start_time

    final_log_path = metrics_dir / f"{method_name}_training_log.csv"
    pd.DataFrame(rows).to_csv(final_log_path, index=False)

    summary_path = metrics_dir / f"{method_name}_training_summary.csv"

    summary = pd.DataFrame(
        [
            {
                "method": method_name,
                "label": cfg.get("label", method_name),
                "checkpoint": str(checkpoint),
                "epochs": args.epochs,
                "parameters": count_parameters(model),
                "training_time_seconds": total_training_time,
                "final_total_loss": rows[-1]["total_loss"],
                "final_data_loss": rows[-1]["data_loss"],
                "final_equation_loss": rows[-1]["equation_loss"],
                "final_weighted_data_loss": rows[-1]["weighted_data_loss"],
                "final_weighted_equation_loss": rows[-1]["weighted_equation_loss"],
                "data_loss_weight": data_loss_weight,
                "equation_loss_weight": equation_loss_weight,
                "dropout_rate": cfg.get("dropout_rate", 0.0),
                "weight_decay": cfg.get("weight_decay", 0.0),
                "adaptive_loss": cfg.get("adaptive_loss", False),
                "model_type": cfg.get("model_type", "psi"),
            }
        ]
    )

    summary.to_csv(summary_path, index=False)

    print(f"[{method_name}] finished.")
    print(f"[{method_name}] model saved to: {checkpoint}")
    print(f"[{method_name}] training log saved to: {final_log_path}")
    print(f"[{method_name}] training summary saved to: {summary_path}")
    print(f"[{method_name}] total training time: {total_training_time:.2f}s")


def prepare_training_data(args):
    x, y, t, u, v, p, feature_mat = read_2D_data(str(DATA_PATH))

    x_random = shuffle_data(x, y, t, u, v, p)

    lb = feature_mat[1, 0:3].numpy()
    ub = feature_mat[0, 0:3].numpy()

    eqa_points = generate_eqp_rect(
        lb,
        ub,
        dimension=3,
        points=args.n_equation_points,
    )

    return x_random, eqa_points


def main():
    args = parse_args()

    device = get_device()
    set_seed(args.seed)

    print("=" * 80)
    print("benchmark_train.py")
    print(f"Using device: {device}")
    print(f"Data path: {DATA_PATH}")
    print(f"Reynolds number: {REYNOLDS}")
    print(f"Selected method: {args.method}")
    print(f"Epochs: {args.epochs}")
    print(f"Equation points: {args.n_equation_points}")
    print(f"Sampling ratio: {args.ratio}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Data loss weight: {args.data_loss_weight}")
    print(f"Equation loss weight: {args.equation_loss_weight}")
    print("=" * 80)

    x_random, eqa_points = prepare_training_data(args)

    if args.method == "all":
        selected_methods = list(METHODS.keys())
    else:
        selected_methods = [args.method]

    for method_name in selected_methods:
        train_method(
            method_name=method_name,
            cfg=METHODS[method_name],
            args=args,
            x_random=x_random,
            eqa_points=eqa_points,
            device=device,
        )


if __name__ == "__main__":
    main()
