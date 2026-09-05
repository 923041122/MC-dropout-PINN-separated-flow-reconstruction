"""
hump_validation.py

Validation/post-processing script for the NASA 2D wall-mounted hump separated-flow case.
It is intentionally separated from the Re=3900 cylinder-flow scripts, but follows the same
reviewer-response logic used in benchmark_evaluate.py:

1. Read NASA hump LES/experiment data.
2. Load trained PINN baselines and B-PINN checkpoints.
3. Evaluate full-grid reference-conditioned diagnostics; use hump_heldout_evaluate.py for independent test metrics.
4. Plot pointwise absolute-error maps with consistent color scales.
5. Compute local errors in separated-flow regions.
6. Compare velocity profiles and wall pressure coefficient Cp.
7. Report inference time and parameter count.
8. For B-PINN/MC-dropout, compute PICP, MPIW, calibration curves,
   error-uncertainty correlation, and uncertainty maps.
9. Optional dropout-rate / MC-sample ablation for B-PINN.

Place this file in the same project folder as:
    pinn_model.py
    benchmark_tools.py

Expected data files by default:
    ./hump_data/LES_meanfield_nasahump2009_tec.dat
    ./hump_data/LES_statistics_profiles_nasahump2009.dat
    ./hump_data/LES_cp_nasahump2009.dat
    ./hump_data/noflow_cp.exp.dat
    ./hump_data/noflow_cf.exp.dat
    ./hump_data/noflow_vel_and_turb.exp.dat

Expected model checkpoints by default:
    ./hump_results/models/standard_pinn.pth
    ./hump_results/models/weight_decay_pinn.pth
    ./hump_results/models/adaptive_weight_pinn.pth
    ./hump_results/models/bpinn_dropout.pth

Example:
    python hump_validation.py --data-dir ./hump_data --models-root ./hump_results/models

If checkpoints are not found, the script skips model-dependent parts and still writes
reference/experimental plots. Use --require-models if missing checkpoints should be fatal.
"""

from __future__ import annotations

import argparse
import math
import re
import time
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
import torch

try:
    from benchmark_tools import (
        build_model,
        count_parameters,
        get_device,
        predict_uvp_numpy,
        safe_load_state,
    )
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Cannot import benchmark_tools.py. Put hump_validation.py in the same "
        "folder as your existing pinn_model.py and benchmark_tools.py."
    ) from exc


# -----------------------------------------------------------------------------
# Module 1. Hump-case configuration
# -----------------------------------------------------------------------------

RE_HUMP = 935_892.0

# Same psi-p network setting as the cylinder benchmark:
# steady input: x, y; output: psi, p; u=dpsi/dy, v=-dpsi/dx.
LAYER_MAT_PSI = [2, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 2]

HUMP_METHODS = {
    "standard_pinn": {
        "label": "Standard PINN",
        "model_type": "psi",
        "dropout_rate": 0.0,
        "weight_decay": 0.0,
        "adaptive_loss": False,
        "checkpoint_name": "standard_pinn.pth",
    },
    "weight_decay_pinn": {
        "label": "Weight-decay PINN",
        "model_type": "psi",
        "dropout_rate": 0.0,
        "weight_decay": 1e-5,
        "adaptive_loss": False,
        "checkpoint_name": "weight_decay_pinn.pth",
    },
    "adaptive_weight_pinn": {
        "label": "Adaptive-weight PINN",
        "model_type": "psi",
        "dropout_rate": 0.0,
        "weight_decay": 0.0,
        "adaptive_loss": True,
        "checkpoint_name": "adaptive_weight_pinn.pth",
    },
    "bpinn_dropout": {
        "label": "B-PINN",
        "model_type": "psi",
        "dropout_rate": 0.002,
        "weight_decay": 0.0,
        "adaptive_loss": False,
        "checkpoint_name": "bpinn_dropout.pth",
        "uncertainty": "dropout",
    },

    "fourier_feature_pinn": {
        "label": "Fourier-feature PINN",
        "model_type": "fourier_psi",
        "dropout_rate": 0.0,
        "weight_decay": 0.0,
        "adaptive_loss": False,
        "fourier_features": 128,
        "fourier_sigma": 1.0,
        "checkpoint_name": "fourier_feature_pinn.pth",
    },
}


# ---------------------------------------------------------------------
# Reviewer-requested SIREN baseline: evaluation registration
# Training/protocol are unchanged.  This entry only allows the frozen
# SIREN checkpoint to be loaded by held-out/validation evaluators.
# ---------------------------------------------------------------------
HUMP_METHODS["siren_pinn"] = {
    **HUMP_METHODS["standard_pinn"],
    "label": "SIREN PINN",
    "model_type": "siren_psi",
    "checkpoint_name": "siren_pinn.pth",
    "dropout_rate": 0.0,
    "first_omega_0": 30.0,
    "hidden_omega_0": 1.0,
    "uncertainty": None,
}

ACTIVE_HUMP_METHODS = [
    "standard_pinn",
    "weight_decay_pinn",
    "adaptive_weight_pinn",
    "bpinn_dropout",
]

FIELD_QUANTITIES = ["u", "v"]
UQ_LEVELS = [0.50, 0.68, 0.80, 0.90, 0.95, 0.99]


# -----------------------------------------------------------------------------
# Module 2. General utility functions
# -----------------------------------------------------------------------------

FLOAT_PATTERN = re.compile(
    r"[-+]?\d*\.\d+(?:[Ee][-+]?\d+)?|[-+]?\d+(?:[Ee][-+]?\d+)?"
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def extract_floats(line: str) -> List[float]:
    return [float(x) for x in FLOAT_PATTERN.findall(line)]


def read_numeric_rows(path: Path, min_cols: int = 1) -> pd.DataFrame:
    """Read loose Tecplot-like ASCII files by keeping only numeric rows.

    This works for files such as:
        noflow_cp.exp.dat
        noflow_cf.exp.dat
        noflow_vel_and_turb.exp.dat
        LES_cp_nasahump2009.dat
    """
    rows: List[List[float]] = []

    with path.open("r", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue
            low = line.lower()
            if low.startswith("variables") or low.startswith("zone"):
                continue
            if not (line[0].isdigit() or line[0] in "+-."):
                continue
            values = extract_floats(line)
            if len(values) >= min_cols:
                rows.append(values)

    if not rows:
        raise ValueError(f"No numeric rows were found in {path}")

    max_cols = max(len(row) for row in rows)
    padded = [row + [np.nan] * (max_cols - len(row)) for row in rows]
    return pd.DataFrame(padded)


def metric_dict(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    mask = np.isfinite(pred) & np.isfinite(true)

    if mask.sum() == 0:
        return {
            "mae": np.nan,
            "rmse": np.nan,
            "max_abs_error": np.nan,
            "relative_l2": np.nan,
            "points": 0,
        }

    err = pred[mask] - true[mask]
    abs_err = np.abs(err)

    return {
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "max_abs_error": float(abs_err.max()),
        "relative_l2": float(np.linalg.norm(err.ravel()) / (np.linalg.norm(true[mask].ravel()) + 1e-12)),
        "points": int(mask.sum()),
    }


def load_optional_training_time(method: str, logs_root: Path) -> float:
    """Read training time from an optional log/summary file.

    This keeps the validation script independent from your training script. If one of
    the listed files exists and contains a likely time column, it is used. Otherwise NaN
    is returned.
    """
    candidates = [
        logs_root / f"{method}_training_summary.csv",
        logs_root / f"{method}_train_summary.csv",
        logs_root / f"{method}_training_log.csv",
        logs_root / f"{method}.csv",
    ]
    possible_columns = [
        "training_time_seconds",
        "total_training_time_seconds",
        "elapsed_time_seconds",
        "elapsed_seconds",
        "wall_time_seconds",
    ]

    for path in candidates:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        for col in possible_columns:
            if col in df.columns and len(df[col].dropna()) > 0:
                return float(df[col].dropna().iloc[-1])

    return float("nan")


# -----------------------------------------------------------------------------
# Module 3. NASA hump data readers
# -----------------------------------------------------------------------------


def parse_variable_names(text: str) -> List[str]:
    m = re.search(r"variables\s*=\s*(.*)", text, flags=re.IGNORECASE)
    if not m:
        return []
    return re.findall(r'"([^"]+)"', m.group(1))


def read_les_meanfield_tec(path: Path) -> Dict[str, object]:
    """Read LES_meanfield_nasahump2009_tec.dat.

    The file is Tecplot ordered BLOCK format with I=207, J=23 and 11 variables.
    Arrays are returned with shape (J, I). Flattened coordinates preserve the same
    node ordering and are suitable for direct model evaluation.
    """
    text = path.read_text(errors="ignore")

    dim_match = re.search(r"I\s*=\s*(\d+)\s*,\s*J\s*=\s*(\d+)", text)
    if not dim_match:
        raise ValueError(f"Cannot find I,J dimensions in {path}")

    n_i = int(dim_match.group(1))
    n_j = int(dim_match.group(2))

    variables = parse_variable_names(text)
    if not variables:
        variables = [
            "x/c",
            "y/c",
            "u/U_in",
            "v/U_in",
            "w/U_in",
            "uu/U_in^2",
            "vv/U_in^2",
            "ww/U_in^2",
            "uv/U_in^2",
            "uw/U_in^2",
            "vw/U_in^2",
        ]

    # Numerical data start after the DT line.
    dt_pos = text.lower().find("dt=")
    if dt_pos < 0:
        raise ValueError(f"Cannot find DT line in {path}")
    start = text.find("\n", dt_pos) + 1

    values = np.array(extract_floats(text[start:]), dtype=float)
    expected = len(variables) * n_i * n_j
    if values.size != expected:
        raise ValueError(
            f"Unexpected number of values in {path}: got {values.size}, expected {expected}."
        )

    block = values.reshape(len(variables), n_j, n_i)

    arrays = {
        "x": block[0],
        "y": block[1],
        "u": block[2],
        "v": block[3],
        "w": block[4],
        "uu": block[5],
        "vv": block[6],
        "ww": block[7],
        "uv": block[8],
        "uw": block[9],
        "vw": block[10],
    }

    arrays["tke"] = 0.5 * (arrays["uu"] + arrays["vv"] + arrays["ww"])

    flat = pd.DataFrame({key: np.asarray(val).reshape(-1) for key, val in arrays.items()})

    return {
        "path": path,
        "I": n_i,
        "J": n_j,
        "variables": variables,
        "arrays": arrays,
        "flat": flat,
    }


def read_les_profiles(path: Path) -> pd.DataFrame:
    """Read LES_statistics_profiles_nasahump2009.dat.

    Output columns:
        x, y, u, v, uu, vv, ww, uv, vw, uw, tke
    """
    rows: List[List[float]] = []
    current_x: Optional[float] = None

    zone_pattern = re.compile(
        r"zone\s*,?\s*t\s*=\s*\"x/c\s*=\s*([0-9.+\-Ee]+)\"",
        flags=re.IGNORECASE,
    )

    with path.open("r", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            zone_match = zone_pattern.search(line)
            if zone_match:
                current_x = float(zone_match.group(1))
                continue

            if current_x is None:
                continue
            if not line or line.startswith("#") or line.lower().startswith("variables"):
                continue
            if not (line[0].isdigit() or line[0] in "+-."):
                continue

            values = extract_floats(line)
            if len(values) >= 10:
                rows.append([current_x] + values[:10])

    if not rows:
        raise ValueError(f"No LES profile rows were found in {path}")

    columns = ["x", "y", "u", "v", "uu", "vv", "ww", "uv", "vw", "uw", "tke"]
    return pd.DataFrame(rows, columns=columns)


def read_exp_profiles(path: Path) -> pd.DataFrame:
    """Read noflow_vel_and_turb.exp.dat.

    Output columns:
        x, y, u, v, uu, vv, uv
    """
    df = read_numeric_rows(path, min_cols=7)
    df = df.iloc[:, :7]
    df.columns = ["x", "y", "u", "v", "uu", "vv", "uv"]
    return df


def read_cp_file(path: Path) -> pd.DataFrame:
    df = read_numeric_rows(path, min_cols=2).iloc[:, :2]
    df.columns = ["x", "cp"]
    return df


def read_cf_exp(path: Path) -> pd.DataFrame:
    df = read_numeric_rows(path, min_cols=3).iloc[:, :3]
    df.columns = ["x", "cf", "cf_uncertainty"]
    return df


def load_hump_data(data_dir: Path) -> Dict[str, object]:
    data = {
        "meanfield": read_les_meanfield_tec(data_dir / "LES_meanfield_nasahump2009_tec.dat"),
        "les_profiles": read_les_profiles(data_dir / "LES_statistics_profiles_nasahump2009.dat"),
        "les_cp": read_cp_file(data_dir / "LES_cp_nasahump2009.dat"),
        "exp_cp": read_cp_file(data_dir / "noflow_cp.exp.dat"),
        "exp_cf": read_cf_exp(data_dir / "noflow_cf.exp.dat"),
        "exp_profiles": read_exp_profiles(data_dir / "noflow_vel_and_turb.exp.dat"),
    }
    return data


# -----------------------------------------------------------------------------
# Module 4. Hump local-region masks and reference diagnostics
# -----------------------------------------------------------------------------


def make_hump_region_masks(meanfield: Dict[str, object]) -> Dict[str, np.ndarray]:
    """Separated-flow local regions for the NASA hump mean-field grid.

    These are practical masks for reviewer-response local error tables. They use
    the LES mean-field grid and the LES reference velocity to identify recirculation.

    Returned masks are flattened arrays with length J*I.
    """
    arrays = meanfield["arrays"]
    x = arrays["x"]
    y = arrays["y"]
    u = arrays["u"]
    n_j, n_i = x.shape

    j_index = np.repeat(np.arange(n_j)[:, None], n_i, axis=1)

    near_wall = j_index <= max(2, int(round(0.18 * (n_j - 1))))
    lower_shear = (j_index > 1) & (j_index <= max(6, int(round(0.45 * (n_j - 1)))))
    upper_recovery = j_index >= max(8, int(round(0.55 * (n_j - 1))))

    masks_2d = {
        "separation_region": (x >= 0.65) & (x <= 0.90) & near_wall,
        "recirculation_region": (x >= 0.65) & (x <= 1.15) & near_wall & (u < 0.0),
        "shear_layer": (x >= 0.65) & (x <= 1.20) & lower_shear,
        "reattachment_region": (x >= 1.00) & (x <= 1.20) & near_wall,
        "downstream_recovery_region": (x > 1.20) & (x <= 1.58) & (near_wall | upper_recovery),
        "whole_les_window": np.isfinite(x) & np.isfinite(y),
    }

    return {name: mask.reshape(-1) for name, mask in masks_2d.items()}


def save_reference_summary(data: Dict[str, object], save_dir: Path) -> None:
    ensure_dir(save_dir)

    meanfield = data["meanfield"]
    arrays = meanfield["arrays"]
    summary = {
        "I": meanfield["I"],
        "J": meanfield["J"],
        "points": int(meanfield["I"] * meanfield["J"]),
        "x_min": float(np.min(arrays["x"])),
        "x_max": float(np.max(arrays["x"])),
        "y_min": float(np.min(arrays["y"])),
        "y_max": float(np.max(arrays["y"])),
        "u_min": float(np.min(arrays["u"])),
        "u_max": float(np.max(arrays["u"])),
        "v_min": float(np.min(arrays["v"])),
        "v_max": float(np.max(arrays["v"])),
        "re_hump": RE_HUMP,
    }
    pd.DataFrame([summary]).to_csv(save_dir / "hump_reference_summary.csv", index=False)


# -----------------------------------------------------------------------------
# Module 5. Model loading and prediction
# -----------------------------------------------------------------------------


def method_config_with_checkpoint(method: str, models_root: Path) -> Dict[str, object]:
    cfg = dict(HUMP_METHODS[method])
    cfg["checkpoint"] = models_root / cfg["checkpoint_name"]
    return cfg


def load_hump_models(
    methods: Iterable[str],
    models_root: Path,
    device: torch.device,
    require_models: bool = False,
) -> Dict[str, torch.nn.Module]:
    loaded: Dict[str, torch.nn.Module] = {}

    for method in methods:
        if method not in HUMP_METHODS:
            raise KeyError(f"Unknown method: {method}. Available: {list(HUMP_METHODS)}")

        cfg = method_config_with_checkpoint(method, models_root)
        checkpoint = Path(cfg["checkpoint"])

        if not checkpoint.exists():
            msg = f"Checkpoint not found for {method}: {checkpoint}"
            if require_models:
                raise FileNotFoundError(msg)
            warnings.warn(msg + " -- this method will be skipped.")
            continue

        model = build_model(cfg, LAYER_MAT_PSI).to(device)
        state = safe_load_state(checkpoint, device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            warnings.warn(f"{method}: missing keys while loading checkpoint: {missing[:5]}...")
        if unexpected:
            warnings.warn(f"{method}: unexpected keys while loading checkpoint: {unexpected[:5]}...")
        model.eval()
        loaded[method] = model

    return loaded


def predict_on_points(
    model: torch.nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    batch_size: int,
    train_mode: bool = False,
) -> Tuple[Dict[str, np.ndarray], float]:
    """Predict steady hump fields from spatial coordinates only."""
    x_np = np.asarray(x, dtype=np.float32).reshape(-1, 1)
    y_np = np.asarray(y, dtype=np.float32).reshape(-1, 1)

    preds, elapsed = predict_uvp_numpy(
        model,
        x_np,
        y_np,
        device,
        batch_size=batch_size,
        eval_mode=not train_mode,
    )
    return {k: v.reshape(-1) for k, v in preds.items()}, elapsed


def predict_all_models_on_meanfield(
    models: Dict[str, torch.nn.Module],
    meanfield: Dict[str, object],
    device: torch.device,
    batch_size: int,
    logs_root: Path,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], pd.DataFrame]:
    arrays = meanfield["arrays"]
    shape = arrays["x"].shape
    x_flat = arrays["x"].reshape(-1, 1)
    y_flat = arrays["y"].reshape(-1, 1)

    predictions: Dict[str, Dict[str, np.ndarray]] = {}
    timing_rows: List[Dict[str, object]] = []

    for method, model in models.items():
        preds, elapsed = predict_on_points(
            model=model,
            x=x_flat,
            y=y_flat,
            device=device,
            batch_size=batch_size,
            train_mode=False,
        )
        predictions[method] = {
            "u": preds["u"].reshape(shape),
            "v": preds["v"].reshape(shape),
            "p": preds["p"].reshape(shape),
        }
        timing_rows.append(
            {
                "method": method,
                "label": HUMP_METHODS[method]["label"],
                "inference_time_seconds": float(elapsed),
                "points": int(x_flat.shape[0]),
                "parameters": int(count_parameters(model)),
                "training_time_seconds": load_optional_training_time(method, logs_root),
            }
        )

    return predictions, pd.DataFrame(timing_rows)


# -----------------------------------------------------------------------------
# Module 6. Field/global/local/profile/Cp metrics
# -----------------------------------------------------------------------------


def compute_field_error_tables(
    predictions: Dict[str, Dict[str, np.ndarray]],
    meanfield: Dict[str, object],
    masks: Dict[str, np.ndarray],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    arrays = meanfield["arrays"]
    global_rows: List[Dict[str, object]] = []
    local_rows: List[Dict[str, object]] = []

    for method, pred in predictions.items():
        for quantity in FIELD_QUANTITIES:
            true = arrays[quantity]
            metrics = metric_dict(pred[quantity], true)
            metrics.update(
                {
                    "method": method,
                    "label": HUMP_METHODS[method]["label"],
                    "variable": quantity,
                    "reference": "LES_meanfield",
                }
            )
            global_rows.append(metrics)

            abs_error_flat = np.abs(pred[quantity].reshape(-1) - true.reshape(-1))
            signed_error_flat = pred[quantity].reshape(-1) - true.reshape(-1)

            for region, mask in masks.items():
                valid = mask & np.isfinite(abs_error_flat)
                values_abs = abs_error_flat[valid]
                values_signed = signed_error_flat[valid]
                if values_abs.size == 0:
                    local_rows.append(
                        {
                            "method": method,
                            "label": HUMP_METHODS[method]["label"],
                            "variable": quantity,
                            "region": region,
                            "local_mae": np.nan,
                            "local_rmse": np.nan,
                            "local_max_abs_error": np.nan,
                            "points": 0,
                        }
                    )
                else:
                    local_rows.append(
                        {
                            "method": method,
                            "label": HUMP_METHODS[method]["label"],
                            "variable": quantity,
                            "region": region,
                            "local_mae": float(values_abs.mean()),
                            "local_rmse": float(np.sqrt(np.mean(values_signed ** 2))),
                            "local_max_abs_error": float(values_abs.max()),
                            "points": int(values_abs.size),
                        }
                    )

    return pd.DataFrame(global_rows), pd.DataFrame(local_rows)


def compute_profile_metrics(
    models: Dict[str, torch.nn.Module],
    les_profiles: pd.DataFrame,
    exp_profiles: pd.DataFrame,
    device: torch.device,
    batch_size: int,
    save_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    ensure_dir(save_dir)

    les_rows: List[Dict[str, object]] = []
    exp_rows: List[Dict[str, object]] = []
    les_pred_tables: Dict[str, pd.DataFrame] = {}
    exp_pred_tables: Dict[str, pd.DataFrame] = {}

    for method, model in models.items():
        pred_les, _ = predict_on_points(
            model,
            les_profiles["x"].values,
            les_profiles["y"].values,
            device,
            batch_size,
        )
        les_out = les_profiles.copy()
        les_out["u_pred"] = pred_les["u"]
        les_out["v_pred"] = pred_les["v"]
        les_out["p_pred"] = pred_les["p"]
        les_out.to_csv(save_dir / f"{method}_profiles_on_les_points.csv", index=False)
        les_pred_tables[method] = les_out

        for quantity in FIELD_QUANTITIES:
            m = metric_dict(les_out[f"{quantity}_pred"].values, les_out[quantity].values)
            m.update({"method": method, "label": HUMP_METHODS[method]["label"], "variable": quantity, "reference": "LES_profiles"})
            les_rows.append(m)

        pred_exp, _ = predict_on_points(
            model,
            exp_profiles["x"].values,
            exp_profiles["y"].values,
            device,
            batch_size,
        )
        exp_out = exp_profiles.copy()
        exp_out["u_pred"] = pred_exp["u"]
        exp_out["v_pred"] = pred_exp["v"]
        exp_out["p_pred"] = pred_exp["p"]
        exp_out.to_csv(save_dir / f"{method}_profiles_on_exp_points.csv", index=False)
        exp_pred_tables[method] = exp_out

        for quantity in FIELD_QUANTITIES:
            m = metric_dict(exp_out[f"{quantity}_pred"].values, exp_out[quantity].values)
            m.update({"method": method, "label": HUMP_METHODS[method]["label"], "variable": quantity, "reference": "experiment_profiles"})
            exp_rows.append(m)

    return pd.DataFrame(les_rows), pd.DataFrame(exp_rows), les_pred_tables, exp_pred_tables


def surface_y_from_meanfield(meanfield: Dict[str, object], x_query: np.ndarray) -> np.ndarray:
    """Infer near-wall/surface y from the first grid line of the LES mean-field file.

    This is enough for pressure/Cp diagnostics inside the limited LES window.
    If you later use the official grid file, replace this with the exact wall geometry.
    """
    arrays = meanfield["arrays"]
    x0 = arrays["x"][0, :]
    y0 = arrays["y"][0, :]
    order = np.argsort(x0)
    return np.interp(np.asarray(x_query, dtype=float), x0[order], y0[order])


def compute_cp_comparison(
    models: Dict[str, torch.nn.Module],
    meanfield: Dict[str, object],
    les_cp: pd.DataFrame,
    exp_cp: pd.DataFrame,
    device: torch.device,
    batch_size: int,
    pressure_to_cp_scale: float,
    align_cp_offset: bool,
    save_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Report both raw and best-constant-offset gauge-aligned wall-Cp errors."""
    ensure_dir(save_dir)

    x_min = float(np.min(meanfield["arrays"]["x"]))
    x_max = float(np.max(meanfield["arrays"]["x"]))
    les_window = les_cp[(les_cp["x"] >= x_min) & (les_cp["x"] <= x_max)].copy()

    if len(les_window) < 3:
        warnings.warn(
            "Too few LES Cp points inside the mean-field x-window; Cp comparison skipped."
        )
        return pd.DataFrame(), pd.DataFrame()

    y_wall = surface_y_from_meanfield(meanfield, les_window["x"].values)
    cp_pred_tables: List[pd.DataFrame] = []
    metric_rows: List[Dict[str, object]] = []

    for method, model in models.items():
        pred, _ = predict_on_points(
            model,
            les_window["x"].values,
            y_wall,
            device,
            batch_size,
        )
        cp_raw = float(pressure_to_cp_scale) * pred["p"]
        offset = float(np.nanmean(les_window["cp"].values - cp_raw))
        cp_aligned = cp_raw + offset

        out = les_window.copy()
        out["y_surface_inferred"] = y_wall
        out["cp_pred_raw"] = cp_raw
        out["cp_pred_gauge_aligned"] = cp_aligned
        out["cp_pred"] = cp_aligned if align_cp_offset else cp_raw
        out["pressure_to_cp_scale"] = float(pressure_to_cp_scale)
        out["cp_offset_for_gauge_alignment"] = float(offset)
        out["plot_uses_gauge_alignment"] = bool(align_cp_offset)
        out["method"] = method
        out["label"] = HUMP_METHODS[method]["label"]
        cp_pred_tables.append(out)

        for treatment, values, used_offset in [
            ("raw", cp_raw, 0.0),
            ("gauge_aligned", cp_aligned, offset),
        ]:
            m = metric_dict(values, les_window["cp"].values)
            m.update(
                {
                    "method": method,
                    "label": HUMP_METHODS[method]["label"],
                    "variable": "Cp_wall",
                    "reference": "LES_cp_limited_window",
                    "gauge_treatment": treatment,
                    "pressure_to_cp_scale": float(pressure_to_cp_scale),
                    "cp_offset_added": float(used_offset),
                    "x_min_used": float(les_window["x"].min()),
                    "x_max_used": float(les_window["x"].max()),
                }
            )
            metric_rows.append(m)

    cp_pred_df = (
        pd.concat(cp_pred_tables, ignore_index=True)
        if cp_pred_tables else pd.DataFrame()
    )
    cp_metrics_df = pd.DataFrame(metric_rows)

    if not cp_pred_df.empty:
        cp_pred_df.to_csv(
            save_dir / "model_cp_predictions_raw_and_aligned.csv", index=False
        )
        cp_metrics_df.to_csv(
            save_dir / "cp_metrics_raw_and_gauge_aligned.csv", index=False
        )

    les_cp.to_csv(save_dir / "les_cp_reference.csv", index=False)
    exp_cp.to_csv(save_dir / "exp_cp_reference.csv", index=False)
    return cp_pred_df, cp_metrics_df


def compute_cf_proxy_from_grid(
    arrays: Dict[str, np.ndarray],
    u_field: np.ndarray,
    v_field: np.ndarray,
    reynolds: float,
) -> pd.DataFrame:
    """Compute a near-wall shear proxy from the first two grid lines.

    This is not a replacement for exact wall Cf unless the first grid line is the wall
    and the second line is wall-normal. It is still useful as a diagnostic that follows
    the same post-processing for LES and model fields.
    """
    x = arrays["x"]
    y = arrays["y"]

    xb = x[0, :]
    yb = y[0, :]
    x1 = x[1, :]
    y1 = y[1, :]

    # Tangent on the bottom grid line.
    tx = np.gradient(xb)
    ty = np.gradient(yb)
    tnorm = np.sqrt(tx ** 2 + ty ** 2) + 1e-12
    tx = tx / tnorm
    ty = ty / tnorm

    ut0 = u_field[0, :] * tx + v_field[0, :] * ty
    ut1 = u_field[1, :] * tx + v_field[1, :] * ty

    dn = np.sqrt((x1 - xb) ** 2 + (y1 - yb) ** 2) + 1e-12
    dut_dn = (ut1 - ut0) / dn
    cf_proxy = 2.0 * dut_dn / reynolds

    return pd.DataFrame(
        {
            "x": xb,
            "y_bottom_line": yb,
            "cf_proxy": cf_proxy,
            "dut_dn_proxy": dut_dn,
        }
    )


# -----------------------------------------------------------------------------
# Module 7. UQ calibration and ablation
# -----------------------------------------------------------------------------


def mc_dropout_stats(
    model: torch.nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    samples: int,
    batch_size: int,
) -> Dict[str, Dict[str, object]]:
    if samples < 2:
        raise ValueError("MC dropout requires at least 2 samples.")

    x_np = np.asarray(x, dtype=np.float32).reshape(-1, 1)
    y_np = np.asarray(y, dtype=np.float32).reshape(-1, 1)

    was_training = model.training
    model.train()
    stacks = {"u": [], "v": [], "p": []}

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(samples):
        preds, _ = predict_uvp_numpy(
            model,
            x_np,
            y_np,
            device,
            batch_size=batch_size,
            eval_mode=False,
        )
        for quantity in stacks:
            stacks[quantity].append(preds[quantity].reshape(-1))

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    model.train(mode=was_training)

    out: Dict[str, Dict[str, object]] = {}
    for quantity, values in stacks.items():
        arr = np.stack(values, axis=0)
        out[quantity] = {
            "samples": arr,
            "mean": arr.mean(axis=0),
            "std": arr.std(axis=0, ddof=0),
            "time": float(elapsed),
        }
    return out


def uq_metric_rows(
    method: str,
    quantity: str,
    truth: np.ndarray,
    samples: np.ndarray,
    levels: Iterable[float],
    mc_samples: int,
    mc_inference_time_seconds: float,
    region_name: str = "field",
) -> List[Dict[str, object]]:
    truth_flat = np.asarray(truth, dtype=float).reshape(-1)
    sample_arr = np.asarray(samples, dtype=float)
    sample_arr = sample_arr.reshape(sample_arr.shape[0], -1)

    mean = sample_arr.mean(axis=0)
    std = sample_arr.std(axis=0, ddof=0)
    error = np.abs(mean - truth_flat)

    mask = np.isfinite(truth_flat) & np.isfinite(mean) & np.isfinite(std)
    truth_flat = truth_flat[mask]
    sample_arr = sample_arr[:, mask]
    mean = mean[mask]
    std = std[mask]
    error = error[mask]

    if truth_flat.size == 0:
        return []

    if np.std(std) > 0:
        corr = float(np.corrcoef(error.ravel(), std.ravel())[0, 1])
    else:
        corr = float("nan")

    rows: List[Dict[str, object]] = []
    for level in levels:
        level = float(level)
        lower = np.quantile(sample_arr, (1.0 - level) / 2.0, axis=0)
        upper = np.quantile(sample_arr, (1.0 + level) / 2.0, axis=0)
        covered = (truth_flat >= lower) & (truth_flat <= upper)
        empirical = float(covered.mean())
        mpiw = float((upper - lower).mean())

        rows.append(
            {
                "method": method,
                "label": HUMP_METHODS[method]["label"],
                "variable": quantity,
                "region_name": region_name,
                "nominal_coverage": level,
                "empirical_coverage": empirical,
                "mean_prediction_interval_width": mpiw,
                "calibration_error": abs(empirical - level),
                "error_uncertainty_correlation": corr,
                "mc_samples": int(mc_samples),
                "mc_inference_time_seconds": float(mc_inference_time_seconds),
                "interval_method": "mc_quantile",
                "points": int(truth_flat.size),
            }
        )

    return rows


# -----------------------------------------------------------------------------
# Module 8. Plotting functions
# -----------------------------------------------------------------------------


def make_triangulation(x: np.ndarray, y: np.ndarray, z: Optional[np.ndarray] = None) -> Tuple[mtri.Triangulation, np.ndarray]:
    x_flat = np.asarray(x, dtype=float).reshape(-1)
    y_flat = np.asarray(y, dtype=float).reshape(-1)
    mask = np.isfinite(x_flat) & np.isfinite(y_flat)
    if z is not None:
        mask &= np.isfinite(np.asarray(z).reshape(-1))
    tri = mtri.Triangulation(x_flat[mask], y_flat[mask])
    return tri, mask


def plot_field_reference_maps(meanfield: Dict[str, object], save_dir: Path) -> None:
    ensure_dir(save_dir)
    arrays = meanfield["arrays"]
    x = arrays["x"]
    y = arrays["y"]

    for quantity in ["u", "v", "tke"]:
        z = arrays[quantity]
        tri, mask = make_triangulation(x, y, z)
        z_flat = z.reshape(-1)[mask]
        levels = np.linspace(float(np.nanmin(z_flat)), float(np.nanmax(z_flat)), 40)

        fig, ax = plt.subplots(figsize=(7, 3.2))
        im = ax.tricontourf(tri, z_flat, levels=levels, cmap="jet")
        ax.set_xlabel("x/C")
        ax.set_ylabel("y/C")
        ax.set_title(f"LES reference {quantity}")
        fig.colorbar(im, ax=ax, label=quantity)
        fig.tight_layout()
        fig.savefig(save_dir / f"reference_{quantity}.png", dpi=300)
        fig.savefig(save_dir / f"reference_{quantity}.pdf", dpi=300)
        plt.close(fig)


def plot_absolute_error_maps(
    predictions: Dict[str, Dict[str, np.ndarray]],
    meanfield: Dict[str, object],
    save_dir: Path,
) -> None:
    ensure_dir(save_dir)
    arrays = meanfield["arrays"]
    x = arrays["x"]
    y = arrays["y"]

    method_list = list(predictions.keys())
    if not method_list:
        return

    for quantity in FIELD_QUANTITIES:
        error_maps = {
            method: np.abs(predictions[method][quantity] - arrays[quantity])
            for method in method_list
        }
        vmax = max(float(np.nanpercentile(err, 99)) for err in error_maps.values())
        vmax = max(vmax, 1e-12)
        levels = np.linspace(0.0, vmax, 40)

        fig, axes = plt.subplots(
            1,
            len(method_list),
            figsize=(5.2 * len(method_list), 3.6),
            squeeze=False,
        )

        im = None
        for ax, method in zip(axes.ravel(), method_list):
            err = error_maps[method]
            tri, mask = make_triangulation(x, y, err)
            im = ax.tricontourf(tri, err.reshape(-1)[mask], levels=levels, cmap="magma")
            ax.set_xlabel("x/C")
            ax.set_ylabel("y/C")
            ax.set_title(HUMP_METHODS[method]["label"])

        fig.suptitle(f"Pointwise absolute error of {quantity} on NASA hump LES mean field")
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, label="Absolute error")
        fig.savefig(save_dir / f"{quantity}_absolute_error_map.png", dpi=300, bbox_inches="tight")
        fig.savefig(save_dir / f"{quantity}_absolute_error_map.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_uncertainty_maps(
    method: str,
    uq_stats: Dict[str, Dict[str, object]],
    meanfield: Dict[str, object],
    save_dir: Path,
) -> None:
    ensure_dir(save_dir)
    arrays = meanfield["arrays"]
    shape = arrays["x"].shape
    x = arrays["x"]
    y = arrays["y"]

    for quantity in FIELD_QUANTITIES:
        truth = arrays[quantity]
        mean = np.asarray(uq_stats[quantity]["mean"]).reshape(shape)
        std = np.asarray(uq_stats[quantity]["std"]).reshape(shape)
        abs_err = np.abs(mean - truth)

        panels = [
            (truth, f"LES {quantity}", "jet", None, None),
            (mean, f"B-PINN mean {quantity}", "jet", None, None),
            (abs_err, "absolute error", "magma", 0.0, np.nanpercentile(abs_err, 99)),
            (std, "predictive std", "magma", 0.0, np.nanpercentile(std, 99)),
        ]

        field_min = min(float(np.nanmin(truth)), float(np.nanmin(mean)))
        field_max = max(float(np.nanmax(truth)), float(np.nanmax(mean)))
        panels[0] = (truth, panels[0][1], panels[0][2], field_min, field_max)
        panels[1] = (mean, panels[1][1], panels[1][2], field_min, field_max)

        fig, axes = plt.subplots(1, 4, figsize=(18, 3.6))
        for ax, (z, title, cmap, vmin, vmax) in zip(axes, panels):
            tri, mask = make_triangulation(x, y, z)
            z_flat = z.reshape(-1)[mask]
            if vmin is None:
                vmin = float(np.nanmin(z_flat))
            if vmax is None:
                vmax = float(np.nanmax(z_flat))
            vmax = max(float(vmax), float(vmin) + 1e-12)
            levels = np.linspace(float(vmin), float(vmax), 40)
            im = ax.tricontourf(tri, z_flat, levels=levels, cmap=cmap)
            ax.set_xlabel("x/C")
            ax.set_ylabel("y/C")
            ax.set_title(title)
            fig.colorbar(im, ax=ax, fraction=0.046)

        fig.suptitle(f"{HUMP_METHODS[method]['label']} uncertainty/error maps on NASA hump")
        fig.tight_layout()
        fig.savefig(save_dir / f"{method}_{quantity}_uncertainty_error_map.png", dpi=300, bbox_inches="tight")
        fig.savefig(save_dir / f"{method}_{quantity}_uncertainty_error_map.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_calibration_curves(uq_df: pd.DataFrame, save_dir: Path) -> None:
    ensure_dir(save_dir)
    if uq_df.empty:
        return

    for quantity in sorted(uq_df["variable"].unique()):
        fig, ax = plt.subplots(figsize=(5, 4))
        sub = uq_df[uq_df["variable"] == quantity]
        for method in sorted(sub["method"].unique()):
            mdf = sub[sub["method"] == method]
            curve = mdf.groupby("nominal_coverage")["empirical_coverage"].mean().sort_index()
            ax.plot(curve.index, curve.values, marker="o", linewidth=2, label=HUMP_METHODS[method]["label"])
        ax.plot([0.5, 1.0], [0.5, 1.0], "k--", linewidth=1, label="ideal")
        ax.set_xlabel("Nominal coverage")
        ax.set_ylabel("Empirical coverage")
        ax.set_xlim(0.5, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.set_title(f"Calibration curve for {quantity}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(save_dir / f"calibration_curve_{quantity}.png", dpi=300)
        fig.savefig(save_dir / f"calibration_curve_{quantity}.pdf", dpi=300)
        plt.close(fig)


def plot_cp_curves(
    les_cp: pd.DataFrame,
    exp_cp: pd.DataFrame,
    cp_pred_df: pd.DataFrame,
    save_dir: Path,
) -> None:
    ensure_dir(save_dir)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(les_cp["x"], les_cp["cp"], linewidth=2, label="LES Cp")
    ax.scatter(exp_cp["x"], exp_cp["cp"], s=15, marker="o", label="Experiment Cp")

    if cp_pred_df is not None and not cp_pred_df.empty:
        for method in cp_pred_df["method"].unique():
            mdf = cp_pred_df[cp_pred_df["method"] == method]
            ax.plot(mdf["x"], mdf["cp_pred"], linewidth=1.8, label=f"{HUMP_METHODS[method]['label']} Cp")

    ax.set_xlabel("x/C")
    ax.set_ylabel("Cp")
    ax.set_title("Surface pressure coefficient comparison")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_dir / "cp_comparison.png", dpi=300)
    fig.savefig(save_dir / "cp_comparison.pdf", dpi=300)
    plt.close(fig)


def plot_cf_proxy(
    cf_exp: pd.DataFrame,
    cf_proxy_tables: Dict[str, pd.DataFrame],
    save_dir: Path,
) -> None:
    ensure_dir(save_dir)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(
        cf_exp["x"],
        cf_exp["cf"],
        yerr=cf_exp["cf_uncertainty"],
        fmt="o",
        markersize=3,
        linewidth=1,
        capsize=2,
        label="Experiment Cf",
    )

    for label, df in cf_proxy_tables.items():
        ax.plot(df["x"], df["cf_proxy"], linewidth=1.8, label=label)

    ax.axhline(0.0, color="k", linewidth=0.8)
    ax.set_xlabel("x/C")
    ax.set_ylabel("Cf or near-wall shear proxy")
    ax.set_title("Cf reference and near-wall shear proxy")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_dir / "cf_reference_and_proxy.png", dpi=300)
    fig.savefig(save_dir / "cf_reference_and_proxy.pdf", dpi=300)
    plt.close(fig)


def plot_profile_comparisons(
    les_profiles: pd.DataFrame,
    exp_profiles: pd.DataFrame,
    les_pred_tables: Dict[str, pd.DataFrame],
    save_dir: Path,
    selected_x: Iterable[float],
) -> None:
    ensure_dir(save_dir)
    selected_x = list(selected_x)

    for quantity in FIELD_QUANTITIES:
        for x0 in selected_x:
            les_sub = les_profiles[np.isclose(les_profiles["x"], x0, atol=1e-8)].copy()
            if les_sub.empty:
                continue
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(les_sub[quantity], les_sub["y"], linewidth=2, label="LES profile")

            exp_sub = exp_profiles[np.abs(exp_profiles["x"] - x0) < 0.01].copy()
            if not exp_sub.empty:
                ax.scatter(exp_sub[quantity], exp_sub["y"], s=12, marker="o", label="Experiment profile")

            for method, table in les_pred_tables.items():
                sub = table[np.isclose(table["x"], x0, atol=1e-8)].copy()
                if sub.empty:
                    continue
                ax.plot(sub[f"{quantity}_pred"], sub["y"], linewidth=1.6, label=HUMP_METHODS[method]["label"])

            ax.set_xlabel(f"{quantity}/Uinf")
            ax.set_ylabel("y/C")
            ax.set_title(f"{quantity} profile at x/C={x0:g}")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(save_dir / f"profile_{quantity}_x_{x0:g}.png", dpi=300)
            fig.savefig(save_dir / f"profile_{quantity}_x_{x0:g}.pdf", dpi=300)
            plt.close(fig)


# -----------------------------------------------------------------------------
# Module 9. Optional dropout-rate / MC-sample ablation
# -----------------------------------------------------------------------------


def run_hump_uq_ablation(
    bpinn_checkpoint: Path,
    meanfield: Dict[str, object],
    device: torch.device,
    batch_size: int,
    dropout_rates: Iterable[float],
    mc_sample_grid: Iterable[int],
    save_dir: Path,
) -> None:
    ensure_dir(save_dir)

    if not bpinn_checkpoint.exists():
        warnings.warn(f"B-PINN checkpoint for ablation not found: {bpinn_checkpoint}. Ablation skipped.")
        return

    arrays = meanfield["arrays"]
    x_flat = arrays["x"].reshape(-1)
    y_flat = arrays["y"].reshape(-1)

    rows: List[Dict[str, object]] = []

    for dropout_rate in dropout_rates:
        cfg = dict(HUMP_METHODS["bpinn_dropout"])
        cfg["dropout_rate"] = float(dropout_rate)
        cfg["checkpoint"] = bpinn_checkpoint
        model = build_model(cfg, LAYER_MAT_PSI).to(device)
        model.load_state_dict(safe_load_state(bpinn_checkpoint, device), strict=False)
        model.eval()

        for mc_samples in mc_sample_grid:
            stats = mc_dropout_stats(
                model=model,
                x=x_flat,
                y=y_flat,
                device=device,
                samples=int(mc_samples),
                batch_size=batch_size,
            )
            for quantity in FIELD_QUANTITIES:
                q_rows = uq_metric_rows(
                    method="bpinn_dropout",
                    quantity=quantity,
                    truth=arrays[quantity].reshape(-1),
                    samples=stats[quantity]["samples"],
                    levels=[0.95],
                    mc_samples=int(mc_samples),
                    mc_inference_time_seconds=float(stats[quantity]["time"]),
                    region_name="field",
                )
                for row in q_rows:
                    row["dropout_rate"] = float(dropout_rate)
                    rows.append(row)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not rows:
        return

    df = pd.DataFrame(rows)
    df.to_csv(save_dir / "hump_dropout_mc_ablation.csv", index=False)

    summary = (
        df.groupby(["dropout_rate", "mc_samples", "variable"])
        .agg(
            empirical_coverage_mean=("empirical_coverage", "mean"),
            empirical_coverage_std=("empirical_coverage", "std"),
            mpiw_mean=("mean_prediction_interval_width", "mean"),
            mpiw_std=("mean_prediction_interval_width", "std"),
            calibration_error_mean=("calibration_error", "mean"),
            error_uncertainty_correlation_mean=("error_uncertainty_correlation", "mean"),
            mc_inference_time_seconds_mean=("mc_inference_time_seconds", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(save_dir / "hump_dropout_mc_ablation_summary.csv", index=False)

    for metric, ylabel, fname in [
        ("empirical_coverage", "Empirical 95% coverage", "coverage"),
        ("mean_prediction_interval_width", "Mean prediction interval width", "interval_width"),
        ("error_uncertainty_correlation", "Error-uncertainty correlation", "correlation"),
        ("calibration_error", "|Coverage - 0.95|", "calibration_error"),
    ]:
        for quantity in FIELD_QUANTITIES:
            fig, ax = plt.subplots(figsize=(6, 4))
            sub = df[df["variable"] == quantity]
            for dr in sorted(sub["dropout_rate"].unique()):
                ss = sub[sub["dropout_rate"] == dr]
                curve = ss.groupby("mc_samples")[metric].mean().sort_index()
                ax.plot(curve.index, curve.values, marker="o", linewidth=2, label=f"dropout={dr}")
            if metric == "empirical_coverage":
                ax.axhline(0.95, color="k", linestyle="--", linewidth=1, label="ideal 95%")
            ax.set_xlabel("MC samples")
            ax.set_ylabel(ylabel)
            ax.set_title(f"Hump {quantity}: {ylabel}")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(save_dir / f"{fname}_ablation_{quantity}.png", dpi=300)
            fig.savefig(save_dir / f"{fname}_ablation_{quantity}.pdf", dpi=300)
            plt.close(fig)


# -----------------------------------------------------------------------------
# Module 10. Main orchestration
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NASA hump validation for B-PINN/PINN baselines.")

    parser.add_argument("--data-dir", type=str, default="./hump_data", help="Directory containing NASA hump .dat files.")
    parser.add_argument("--results-root", type=str, default="./hump_results", help="Root directory for all outputs.")
    parser.add_argument("--models-root", type=str, default="./hump_results/models", help="Directory containing trained hump checkpoints.")
    parser.add_argument("--logs-root", type=str, default="./hump_results/training_logs", help="Optional training-log directory.")

    parser.add_argument("--methods", nargs="+", default=ACTIVE_HUMP_METHODS, help="Methods to evaluate.")
    parser.add_argument("--batch-size", type=int, default=20000, help="Inference batch size.")
    parser.add_argument("--mc-samples", type=int, default=30, help="MC-dropout samples for B-PINN UQ.")
    parser.add_argument("--skip-uq", action="store_true", help="Skip B-PINN uncertainty calibration.")
    parser.add_argument("--require-models", action="store_true", help="Raise an error if any requested checkpoint is missing.")

    parser.add_argument(
        "--pressure-to-cp-scale",
        type=float,
        default=2.0,
        help="Scale used to convert model pressure output to Cp. Use 2.0 if p is nondimensionalized by rho*U^2.",
    )
    parser.add_argument(
        "--no-align-cp-offset",
        action="store_true",
        help="Disable constant offset alignment when comparing predicted pressure/Cp with LES Cp.",
    )

    parser.add_argument("--run-ablation", action="store_true", help="Run B-PINN dropout-rate/MC-sample ablation.")
    parser.add_argument("--ablation-dropout-rates", nargs="+", type=float, default=[0.001, 0.002, 0.005, 0.01])
    parser.add_argument("--ablation-mc-samples", nargs="+", type=int, default=[10, 20, 30, 50])

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir)
    results_root = Path(args.results_root)
    models_root = Path(args.models_root)
    logs_root = Path(args.logs_root)

    eval_root = results_root / "hump_evaluation"
    figures_root = eval_root / "figures"
    tables_root = eval_root / "tables"
    ensure_dir(figures_root)
    ensure_dir(tables_root)

    device = get_device()
    print(f"Using device: {device}")
    print(f"Reading NASA hump data from: {data_dir.resolve()}")

    data = load_hump_data(data_dir)
    save_reference_summary(data, tables_root)
    plot_field_reference_maps(data["meanfield"], figures_root / "reference_maps")

    models = load_hump_models(args.methods, models_root, device, require_models=args.require_models)
    print(f"Loaded methods: {list(models.keys())}")

    # Always save raw reference plots for Cp and Cf.
    plot_cp_curves(data["les_cp"], data["exp_cp"], pd.DataFrame(), figures_root / "cp")
    plot_cf_proxy(data["exp_cf"], {}, figures_root / "cf")

    if not models:
        print("No checkpoints were loaded. Reference plots were created, but model validation was skipped.")
        print(f"Outputs saved to: {eval_root.resolve()}")
        return

    meanfield = data["meanfield"]
    arrays = meanfield["arrays"]
    masks = make_hump_region_masks(meanfield)

    predictions, timing_df = predict_all_models_on_meanfield(
        models=models,
        meanfield=meanfield,
        device=device,
        batch_size=args.batch_size,
        logs_root=logs_root,
    )
    timing_df.to_csv(tables_root / "hump_timing_and_parameters.csv", index=False)

    global_df, local_df = compute_field_error_tables(predictions, meanfield, masks)
    global_df.to_csv(tables_root / "hump_fullgrid_reference_conditioned_uv_errors.csv", index=False)
    local_df.to_csv(tables_root / "hump_local_region_errors.csv", index=False)

    plot_absolute_error_maps(predictions, meanfield, figures_root / "absolute_error_maps")

    les_prof_metrics, exp_prof_metrics, les_pred_tables, exp_pred_tables = compute_profile_metrics(
        models=models,
        les_profiles=data["les_profiles"],
        exp_profiles=data["exp_profiles"],
        device=device,
        batch_size=args.batch_size,
        save_dir=tables_root / "profile_predictions",
    )
    les_prof_metrics.to_csv(tables_root / "hump_profile_errors_les.csv", index=False)
    exp_prof_metrics.to_csv(tables_root / "hump_profile_errors_exp.csv", index=False)

    plot_profile_comparisons(
        les_profiles=data["les_profiles"],
        exp_profiles=data["exp_profiles"],
        les_pred_tables=les_pred_tables,
        save_dir=figures_root / "profiles",
        selected_x=[0.65, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
    )

    cp_pred_df, cp_metrics_df = compute_cp_comparison(
        models=models,
        meanfield=meanfield,
        les_cp=data["les_cp"],
        exp_cp=data["exp_cp"],
        device=device,
        batch_size=args.batch_size,
        pressure_to_cp_scale=args.pressure_to_cp_scale,
        align_cp_offset=not args.no_align_cp_offset,
        save_dir=tables_root / "cp",
    )
    if not cp_metrics_df.empty:
        cp_metrics_df.to_csv(tables_root / "hump_cp_metrics.csv", index=False)
    plot_cp_curves(data["les_cp"], data["exp_cp"], cp_pred_df, figures_root / "cp")

    cf_proxy_tables: Dict[str, pd.DataFrame] = {}
    les_cf_proxy = compute_cf_proxy_from_grid(arrays, arrays["u"], arrays["v"], RE_HUMP)
    les_cf_proxy.to_csv(tables_root / "les_cf_proxy_from_meanfield_grid.csv", index=False)
    cf_proxy_tables["LES near-wall shear proxy"] = les_cf_proxy

    for method, pred in predictions.items():
        proxy = compute_cf_proxy_from_grid(arrays, pred["u"], pred["v"], RE_HUMP)
        proxy["method"] = method
        proxy["label"] = HUMP_METHODS[method]["label"]
        proxy.to_csv(tables_root / f"{method}_cf_proxy_from_meanfield_grid.csv", index=False)
        cf_proxy_tables[f"{HUMP_METHODS[method]['label']} proxy"] = proxy

    plot_cf_proxy(data["exp_cf"], cf_proxy_tables, figures_root / "cf")

    # UQ for B-PINN.
    uq_rows_all: List[Dict[str, object]] = []
    if not args.skip_uq:
        for method, model in models.items():
            if HUMP_METHODS[method].get("uncertainty") != "dropout":
                continue

            x_flat = arrays["x"].reshape(-1)
            y_flat = arrays["y"].reshape(-1)
            stats = mc_dropout_stats(
                model=model,
                x=x_flat,
                y=y_flat,
                device=device,
                samples=args.mc_samples,
                batch_size=args.batch_size,
            )

            plot_uncertainty_maps(method, stats, meanfield, figures_root / "uncertainty_maps")

            for quantity in FIELD_QUANTITIES:
                uq_rows_all.extend(
                    uq_metric_rows(
                        method=method,
                        quantity=quantity,
                        truth=arrays[quantity].reshape(-1),
                        samples=stats[quantity]["samples"],
                        levels=UQ_LEVELS,
                        mc_samples=args.mc_samples,
                        mc_inference_time_seconds=stats[quantity]["time"],
                        region_name="full_LES_grid_reference_conditioned",
                    )
                )

            # Optional wall Cp uncertainty, only where LES Cp and inferred surface y are available.
            if not cp_pred_df.empty:
                x_min = float(np.min(arrays["x"]))
                x_max = float(np.max(arrays["x"]))
                les_window = data["les_cp"][(data["les_cp"]["x"] >= x_min) & (data["les_cp"]["x"] <= x_max)].copy()
                if len(les_window) > 3:
                    y_wall = surface_y_from_meanfield(meanfield, les_window["x"].values)
                    cp_stats = mc_dropout_stats(
                        model=model,
                        x=les_window["x"].values,
                        y=y_wall,
                        device=device,
                        samples=args.mc_samples,
                        batch_size=args.batch_size,
                    )
                    cp_samples = args.pressure_to_cp_scale * cp_stats["p"]["samples"]
                    if not args.no_align_cp_offset:
                        cp_mean = cp_samples.mean(axis=0)
                        offset = float(np.nanmean(les_window["cp"].values - cp_mean))
                        cp_samples = cp_samples + offset
                    uq_rows_all.extend(
                        uq_metric_rows(
                            method=method,
                            quantity="Cp_wall",
                            truth=les_window["cp"].values,
                            samples=cp_samples,
                            levels=UQ_LEVELS,
                            mc_samples=args.mc_samples,
                            mc_inference_time_seconds=cp_stats["p"]["time"],
                            region_name="LES_cp_limited_window",
                        )
                    )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if uq_rows_all:
        uq_df = pd.DataFrame(uq_rows_all)
        uq_df.to_csv(tables_root / "hump_fullgrid_reference_conditioned_uq_metrics.csv", index=False)
        plot_calibration_curves(uq_df, figures_root / "calibration_curves")

    if args.run_ablation:
        bpinn_cfg = method_config_with_checkpoint("bpinn_dropout", models_root)
        run_hump_uq_ablation(
            bpinn_checkpoint=Path(bpinn_cfg["checkpoint"]),
            meanfield=meanfield,
            device=device,
            batch_size=args.batch_size,
            dropout_rates=args.ablation_dropout_rates,
            mc_sample_grid=args.ablation_mc_samples,
            save_dir=eval_root / "uq_ablation",
        )

    print("NASA hump validation finished.")
    print(f"Tables saved to:  {tables_root.resolve()}")
    print(f"Figures saved to: {figures_root.resolve()}")


if __name__ == "__main__":
    main()
