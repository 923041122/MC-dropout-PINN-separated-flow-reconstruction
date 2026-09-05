"""Shared benchmark configuration for Re=3900 cylinder-flow PINN experiments.

This configuration is used for the reviewer-response benchmark experiments.

The benchmark includes:
    1. Standard PINN
    2. Weight-decay PINN
    3. Adaptive-weight PINN
    4. B-PINN with MC-dropout uncertainty estimation

The Fourier-feature PINN configuration is kept for reproducibility, but it is
excluded from the default formal comparison because it did not achieve stable
convergence under the current problem setting.
"""

from pathlib import Path


DATA_PATH = Path("./2d_cylinder_Re3900_100x100_kw_sst.mat")

RESULT_ROOT = Path("./benchmark_results")
MODEL_ROOT = RESULT_ROOT / "models"
EVAL_ROOT = RESULT_ROOT / "evaluation"

# Psi-p network:
# input : x, y, t
# output: psi, p
# velocity:
#   u = d psi / d y
#   v = - d psi / d x
LAYER_MAT_PSI = [3, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 2]

REYNOLDS = 3900.0


METHODS = {
    "standard_pinn": {
        "label": "Standard PINN",
        "model_type": "psi",
        "dropout_rate": 0.0,
        "weight_decay": 0.0,
        "adaptive_loss": False,
        "checkpoint": MODEL_ROOT / "standard_pinn.pth",
    },

    "weight_decay_pinn": {
        "label": "Weight-decay PINN",
        "model_type": "psi",
        "dropout_rate": 0.0,
        "weight_decay": 1e-5,
        "adaptive_loss": False,
        "checkpoint": MODEL_ROOT / "weight_decay_pinn.pth",
    },

    "adaptive_weight_pinn": {
        "label": "Adaptive-weight PINN",
        "model_type": "psi",
        "dropout_rate": 0.0,
        "weight_decay": 0.0,
        "adaptive_loss": True,
        "checkpoint": MODEL_ROOT / "adaptive_weight_pinn.pth",
    },

    # Kept for reproducibility, but excluded from the default formal comparison.
    "fourier_feature_pinn": {
        "label": "Fourier-feature PINN",
        "model_type": "fourier_psi",
        "dropout_rate": 0.0,
        "weight_decay": 0.0,
        "adaptive_loss": False,
        "fourier_features": 128,
        "fourier_sigma": 1.0,
        "checkpoint": MODEL_ROOT / "fourier_feature_pinn.pth",
    },

    "bpinn_dropout": {
        "label": "B-PINN",
        "model_type": "psi",
        "dropout_rate": 0.002,
        "weight_decay": 0.0,
        "adaptive_loss": False,
        "checkpoint": MODEL_ROOT / "bpinn_dropout.pth",
        "uncertainty": "dropout",
    },
}


# Formal methods used in the manuscript tables and figures.
ACTIVE_METHODS = [
    "standard_pinn",
    "weight_decay_pinn",
    "adaptive_weight_pinn",
    "bpinn_dropout",
]


# Methods attempted but not included in the formal quantitative comparison.
EXCLUDED_METHODS = {
    "fourier_feature_pinn": (
        "The Fourier-feature PINN was tested but did not achieve stable "
        "convergence under the same training budget and problem setting."
    ),
}


DEFAULT_TRAINING = {
    # For formal training, use a sufficiently large value.
    # If the current B-PINN accuracy is not competitive, increase this to
    # 3000-10000 and retrain.
    "epochs": 2000,

    "learning_rate": 1e-3,
    "decay_rate": 0.9,

    # Supervised data ratio.
    "ratio": 0.02,

    # PDE collocation points.
    "n_equation_points": 100_000,

    "scheduler_T0": 50,
    "scheduler_Tmul": 2,
    "warmup_steps": 2,
}


DEFAULT_EVAL = {
    # Multiple representative time snapshots for reviewer-response figures.
    "time_indices": [0, 25, 50, 75, 99],

    "batch_size": 20000,

    # For formal uncertainty calibration, 50 or 100 is more stable than 30.
    "mc_samples": 50,

    # Nominal coverages for calibration curves.
    # If mc_samples is small, 0.99 may be unstable; it is retained here because
    # the formal setting uses mc_samples >= 50.
    "interval_levels": [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99],

    # Dropout-rate ablation.
    # These values are used for inference-time dropout sensitivity unless
    # separate checkpoints are trained for each dropout rate.
    "dropout_rates": [0.002, 0.005, 0.01, 0.02, 0.05],

    # MC-sample-number ablation.
    "mc_sample_grid": [5, 10, 20, 30, 50, 100],
}
