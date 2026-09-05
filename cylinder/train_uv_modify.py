"""Legacy-compatible training entry for Re=3900 cylinder-flow PINN benchmarks.

This script keeps the original train_uv_modify.py entry point, but delegates
the actual multi-baseline training to benchmark_train.py.

For B-PINN debugging, recommended command:

python train_uv_modify.py \
--method bpinn_dropout \
--epochs 300 \
--ratio 0.02 \
--n-equation-points 100000 \
--data-loss-weight 10.0 \
--equation-loss-weight 1.0 \
--diagnostic-interval 50
"""

import argparse
from pathlib import Path

from benchmark_config import (
    DATA_PATH,
    DEFAULT_EVAL,
    DEFAULT_TRAINING,
    LAYER_MAT_PSI,
    METHODS,
    REYNOLDS,
)
from benchmark_tools import get_device, set_seed
from benchmark_train import prepare_training_data, train_method


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Re=3900 cylinder-flow PINN/B-PINN models."
    )

    parser.add_argument(
        "--method",
        default="all",
        choices=["all"] + list(METHODS.keys()),
        help="Choose one method to train, or use 'all' to train all configured baselines.",
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
        help="Mini-batch sampling ratio for supervised data and equation points.",
    )

    parser.add_argument(
        "--n-equation-points",
        type=int,
        default=DEFAULT_TRAINING["n_equation_points"],
        help="Number of collocation points used for Navier-Stokes residual training.",
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
        help=(
            "Weight for supervised data loss. "
            "For B-PINN debugging, use 10.0 or larger to avoid collapse."
        ),
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
        help=(
            "Print field statistics every N epochs. "
            "Set to 0 to disable periodic diagnostics."
        ),
    )

    parser.add_argument(
        "--disable-diagnostics",
        action="store_true",
        help="Disable diagnostic prediction statistics during training.",
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

    parser.add_argument(
        "--post-uq",
        action="store_true",
        help="Run quick MC-dropout uncertainty plots after training.",
    )

    parser.add_argument(
        "--post-uq-method",
        default="bpinn_dropout",
        choices=list(METHODS.keys()),
        help="Method used for optional post-training uncertainty plotting.",
    )

    parser.add_argument(
        "--post-uq-time-indices",
        nargs="+",
        type=int,
        default=DEFAULT_EVAL["time_indices"],
        help="Time indices used by optional quick uncertainty plotting.",
    )

    parser.add_argument(
        "--post-uq-samples",
        type=int,
        default=DEFAULT_EVAL["mc_samples"],
        help="MC-dropout sample number used by optional quick uncertainty plotting.",
    )

    parser.add_argument(
        "--post-uq-save-dir",
        default="./uncertainty_plots_selected_times",
        help="Save directory for optional quick uncertainty plots.",
    )

    return parser.parse_args()


def selected_method_names(method):
    if method == "all":
        return list(METHODS.keys())

    return [method]


def validate_post_uq_method(method_name):
    method_cfg = METHODS[method_name]

    if method_cfg.get("uncertainty") != "dropout":
        print(
            f"[post-uq] Warning: method '{method_name}' is not marked as a "
            "dropout uncertainty model in benchmark_config.METHODS."
        )

    checkpoint = Path(method_cfg["checkpoint"])

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"[post-uq] Missing checkpoint for {method_name}: {checkpoint}. "
            "Please train the model before running post-training uncertainty plots."
        )

    return checkpoint


def run_post_training_uncertainty(
    method_name,
    time_indices,
    samples,
    save_dir,
):
    """Optional quick uncertainty plotting after training.

    Full reviewer-response calibration should be performed by benchmark_evaluate.py
    and uncertainty_ablation.py. This function only keeps a convenient quick plot
    entry for the original train_uv_modify.py workflow.
    """

    from bayesian_uncertainty_plot import run_uncertainty_demo

    method_cfg = METHODS[method_name]
    checkpoint = validate_post_uq_method(method_name)

    dropout_rate = method_cfg.get("dropout_rate", 0.0)

    print("=" * 80)
    print("[post-uq] Running quick uncertainty plotting")
    print(f"[post-uq] Method: {method_name}")
    print(f"[post-uq] Model path: {checkpoint}")
    print(f"[post-uq] Data path: {DATA_PATH}")
    print(f"[post-uq] Dropout rate: {dropout_rate}")
    print(f"[post-uq] Time indices: {time_indices}")
    print(f"[post-uq] MC samples: {samples}")
    print(f"[post-uq] Save dir: {save_dir}")
    print("=" * 80)

    run_uncertainty_demo(
        data_path=str(DATA_PATH),
        model_path=str(checkpoint),
        layer_mat=tuple(LAYER_MAT_PSI),
        dropout_rate=dropout_rate,
        time_indices=time_indices,
        samples=samples,
        save_dir=save_dir,
    )


def main():
    args = parse_args()

    device = get_device()
    set_seed(args.seed)

    print("=" * 80)
    print("train_uv_modify.py")
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
    print(f"Diagnostic interval: {args.diagnostic_interval}")
    print(f"Diagnostics disabled: {args.disable_diagnostics}")
    print("=" * 80)

    x_random, eqa_points = prepare_training_data(args)

    for method_name in selected_method_names(args.method):
        train_method(
            method_name=method_name,
            cfg=METHODS[method_name],
            args=args,
            x_random=x_random,
            eqa_points=eqa_points,
            device=device,
        )

    if args.post_uq:
        run_post_training_uncertainty(
            method_name=args.post_uq_method,
            time_indices=args.post_uq_time_indices,
            samples=args.post_uq_samples,
            save_dir=args.post_uq_save_dir,
        )

    print("=" * 80)
    print("Training workflow finished.")
    print("For full reviewer-response evaluation, run:")
    print(
        "python benchmark_evaluate.py "
        "--methods standard_pinn weight_decay_pinn adaptive_weight_pinn "
        "fourier_feature_pinn bpinn_dropout "
        "--time-indices 0 25 50 75 99 "
        "--mc-samples 30"
    )
    print("For dropout-rate and MC-sample ablation, run:")
    print(
        "python uncertainty_ablation.py "
        "--time-indices 0 25 50 75 99 "
        "--dropout-rates 0.005 0.01 0.02 0.05 "
        "--mc-samples 5 10 20 30 50"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()

