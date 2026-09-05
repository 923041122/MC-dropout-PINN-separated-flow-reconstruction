"""
hump_train.py

Training script for the NASA 2D wall-mounted hump separated-flow validation case.

This script is intentionally separated from the Re=3900 cylinder-flow training code, but
keeps the same reviewer-response baseline logic:
    1. Standard PINN
    2. Weight-decay PINN
    3. Adaptive-weight PINN
    4. B-PINN with MC-dropout

Network form for this steady case:
    input : x, y
    output: psi, p
    u = d psi / d y
    v = - d psi / d x

NASA hump data used by this trainer:
    LES_meanfield_nasahump2009_tec.dat     -> supervised u, v mean field
    LES_cp_nasahump2009.dat                -> optional wall Cp supervision for p
    noflow_cp.exp.dat                      -> optional experimental wall Cp supervision

Default output:
    ./hump_results/models/standard_pinn.pth
    ./hump_results/models/weight_decay_pinn.pth
    ./hump_results/models/adaptive_weight_pinn.pth
    ./hump_results/models/bpinn_dropout.pth
    ./hump_results/training_logs/*.csv

Quick test:
    python hump_train.py --data-dir ./hump_data --method standard_pinn --epochs 5 --n-equation-points 2000

Formal run:
    python hump_train.py --data-dir ./hump_data --method all --epochs 2000 --n-equation-points 50000

After training:
    python hump_validation.py --data-dir ./hump_data --models-root ./hump_results/models --results-root ./hump_results
"""

from __future__ import annotations

import argparse
import math
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    from benchmark_tools import build_model, count_parameters, get_device, safe_load_state
    from hump_protocol import (
        build_interior_velocity_split,
        build_cp_split,
        sample_fluid_collocation_points,
        diagnose_legacy_random_box,
        save_protocol_artifacts,
    )
except Exception as exc:
    raise RuntimeError(
        "Cannot import benchmark_tools.py. Put hump_train.py in the same folder as "
        "your existing pinn_model.py and benchmark_tools.py."
    ) from exc


RE_HUMP = 935_892.0

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
    "bpinn_dropout": {
        "label": "B-PINN",
        "model_type": "psi",
        "dropout_rate": 0.002,
        "weight_decay": 0.0,
        "adaptive_loss": False,
        "checkpoint_name": "bpinn_dropout.pth",
        "uncertainty": "dropout",
    },

    # --------------------------------------------------------
    # Reviewer-requested SIREN baseline
    #
    # Same:
    #   - steady (x,y) -> (psi,p) formulation
    #   - 10 x 100 backbone
    #   - supervised protocol
    #   - validation/test split
    #   - wall/subdomain observations
    #   - fluid-domain collocation
    #   - optimizer / LR schedule
    #   - training budget
    #
    # Changed:
    #   Tanh/Xavier -> sine/SIREN initialization
    # --------------------------------------------------------
    "siren_pinn": {
        "label": "SIREN PINN",
        "model_type": "siren_psi",
        "dropout_rate": 0.0,
        "weight_decay": 0.0,
        "adaptive_loss": False,
        "first_omega_0": 30.0,
        "hidden_omega_0": 1.0,
        "checkpoint_name": "siren_pinn.pth",
    },


    "resnet_pinn": {
        "label": "ResNet PINN",
        "model_type": "resnet_psi",
        "dropout_rate": 0.0,
        "weight_decay": 0.0,
        "adaptive_loss": False,
        "checkpoint_name": "resnet_pinn.pth",
    },

}

ACTIVE_HUMP_METHODS = [
    "standard_pinn",
    "weight_decay_pinn",
    "adaptive_weight_pinn",
    "bpinn_dropout",
]

FLOAT_PATTERN = re.compile(
    r"[-+]?\d*\.\d+(?:[Ee][-+]?\d+)?|[-+]?\d+(?:[Ee][-+]?\d+)?"
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def extract_floats(line: str) -> List[float]:
    return [float(x) for x in FLOAT_PATTERN.findall(line)]


def read_numeric_rows(path: Path, min_cols: int = 1) -> pd.DataFrame:
    rows: List[List[float]] = []

    with path.open("r", errors="ignore") as f:
        for raw in f:
            line = raw.strip()

            if not line or line.startswith("#"):
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


def parse_variable_names(text: str) -> List[str]:
    match = re.search(r"variables\s*=\s*(.*)", text, flags=re.IGNORECASE)

    if not match:
        return []

    return re.findall(r'"([^"]+)"', match.group(1))


def to_tensor(
    array: np.ndarray,
    device: torch.device,
    requires_grad: bool = False,
) -> torch.Tensor:
    tensor = torch.tensor(array, dtype=torch.float32, device=device)

    if requires_grad:
        tensor = tensor.detach().clone().requires_grad_(True)

    return tensor


def safe_load_checkpoint(
    model: torch.nn.Module,
    path: Path,
    device: torch.device,
) -> bool:
    if not path.exists():
        return False

    state = safe_load_state(path, device)

    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        model.load_state_dict(state, strict=False)

    return True


def read_les_meanfield_tec(path: Path) -> Dict[str, object]:
    """
    Read LES_meanfield_nasahump2009_tec.dat.

    The file is Tecplot ordered BLOCK format with I=207, J=23 and 11 variables.
    Arrays are returned with shape (J, I).
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

    flat = pd.DataFrame(
        {key: np.asarray(val).reshape(-1) for key, val in arrays.items()}
    )

    flat.insert(0, "original_index", np.arange(len(flat), dtype=int))
    flat["j_index"] = np.repeat(np.arange(n_j, dtype=int), n_i)
    flat["i_index"] = np.tile(np.arange(n_i, dtype=int), n_j)

    return {
        "path": path,
        "I": n_i,
        "J": n_j,
        "variables": variables,
        "arrays": arrays,
        "flat": flat,
    }


def read_cp_file(path: Path) -> pd.DataFrame:
    df = read_numeric_rows(path, min_cols=2).iloc[:, :2]
    df.columns = ["x", "cp"]

    return df


def build_wall_y_interpolator(meanfield: Dict[str, object]):
    """
    Return y_wall(x) based on the lowest LES row.
    """
    x_wall = np.asarray(meanfield["arrays"]["x"])[0, :]
    y_wall = np.asarray(meanfield["arrays"]["y"])[0, :]

    order = np.argsort(x_wall)
    x_wall = x_wall[order]
    y_wall = y_wall[order]

    def interp(x: np.ndarray) -> np.ndarray:
        return np.interp(x.reshape(-1), x_wall, y_wall).reshape(-1, 1)

    return interp


def build_supervised_uv_data(
    meanfield: Dict[str, object],
    supervised_ratio: float,
    seed: int,
) -> pd.DataFrame:
    """Legacy convenience helper; formal revision runs use load_training_data()."""
    df = meanfield["flat"][
        ["original_index", "i_index", "j_index", "x", "y", "u", "v"]
    ].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    if not (0.0 < supervised_ratio <= 1.0):
        raise ValueError(f"supervised_ratio must be in (0,1], got {supervised_ratio}")

    if supervised_ratio < 1.0:
        df = (
            df.sample(frac=supervised_ratio, random_state=seed)
            .sort_index()
            .reset_index(drop=True)
        )
    else:
        df = df.reset_index(drop=True)
    return df


def build_cp_supervision(
    data_dir: Path,
    meanfield: Dict[str, object],
    cp_source: str = "les",
    cp_scale: float = 2.0,
) -> pd.DataFrame:
    """
    Build wall-pressure supervision points.

    The network pressure p is interpreted as P/(rho U^2).
    The public wall data are Cp = P/(0.5 rho U^2), so Cp ~= cp_scale * p.
    Default cp_scale=2.0.
    """
    frames: List[pd.DataFrame] = []

    if cp_source in {"les", "both"}:
        path = data_dir / "LES_cp_nasahump2009.dat"

        if path.exists():
            tmp = read_cp_file(path)
            tmp["source"] = "les_cp"
            frames.append(tmp)

    if cp_source in {"exp", "both"}:
        path = data_dir / "noflow_cp.exp.dat"

        if path.exists():
            tmp = read_cp_file(path)
            tmp["source"] = "exp_cp"
            frames.append(tmp)

    if not frames:
        return pd.DataFrame(columns=["x", "y", "cp", "p_target", "source"])

    cp_df = pd.concat(frames, ignore_index=True)

    x_min = float(meanfield["flat"]["x"].min())
    x_max = float(meanfield["flat"]["x"].max())

    cp_df = cp_df[(cp_df["x"] >= x_min) & (cp_df["x"] <= x_max)].copy()

    if cp_df.empty:
        return pd.DataFrame(columns=["x", "y", "cp", "p_target", "source"])

    y_wall_fn = build_wall_y_interpolator(meanfield)

    x_arr = cp_df["x"].to_numpy(dtype=float).reshape(-1, 1)
    y_arr = y_wall_fn(x_arr)

    cp_df["y"] = y_arr.reshape(-1)
    cp_df["p_target"] = cp_df["cp"] / float(cp_scale)

    return cp_df[["x", "y", "cp", "p_target", "source"]].reset_index(drop=True)


def build_collocation_points(
    meanfield: Dict[str, object],
    n_points: int,
    seed: int,
    mode: str = "fluid_domain",
) -> np.ndarray:
    """Generate steady 2-D collocation points."""
    rng = np.random.default_rng(seed)
    flat = meanfield["flat"]

    if n_points <= 0:
        raise ValueError("n_points must be positive")

    if mode == "reference_points":
        idx = rng.integers(0, len(flat), size=n_points)
        return flat.iloc[idx][["x", "y"]].to_numpy(dtype=float)

    if mode == "fluid_domain":
        return sample_fluid_collocation_points(
            meanfield, n_points=n_points, seed=seed
        )

    if mode == "random_box_legacy":
        x = rng.uniform(float(flat["x"].min()), float(flat["x"].max()), size=(n_points, 1))
        y = rng.uniform(float(flat["y"].min()), float(flat["y"].max()), size=(n_points, 1))
        return np.concatenate([x, y], axis=1)

    raise ValueError(f"Unknown collocation mode: {mode}")


def load_training_data(args: argparse.Namespace) -> Dict[str, object]:
    """Build the complete v1.2 interior/boundary/collocation protocol."""
    data_dir = Path(args.data_dir)
    meanfield_path = data_dir / "LES_meanfield_nasahump2009_tec.dat"

    if not meanfield_path.exists():
        raise FileNotFoundError(
            f"Missing {meanfield_path}. Put the NASA hump .dat files in --data-dir."
        )

    meanfield = read_les_meanfield_tec(meanfield_path)

    uv_train_df, uv_val_df, uv_test_df, uv_manifest, boundaries = (
        build_interior_velocity_split(
            meanfield,
            supervised_ratio=args.supervised_ratio,
            validation_fraction=args.validation_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed,
            split_mode=args.split_mode,
            block_width=args.spatial_block_width,
        )
    )

    cp_all_df = build_cp_supervision(
        data_dir=data_dir,
        meanfield=meanfield,
        cp_source=args.cp_source,
        cp_scale=args.cp_scale,
    )
    cp_train_df, cp_val_df, cp_test_df, cp_manifest = build_cp_split(
        cp_all_df,
        supervised_ratio=args.supervised_ratio,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        split_mode=args.split_mode,
        block_width=args.cp_spatial_block_width,
    )

    eqa_np = build_collocation_points(
        meanfield=meanfield,
        n_points=args.n_equation_points,
        seed=args.seed + 123,
        mode=args.collocation_mode,
    )

    legacy_table, legacy_outside, legacy_summary = diagnose_legacy_random_box(
        meanfield,
        n_points=args.n_equation_points,
        seed=args.seed + 123,
    )

    protocol_root = Path(args.results_root) / "protocol"

    protocol_summary = {
        "protocol_version": "NASA_hump_revision_v1_2",
        "input_dimension": 2,
        "input_coordinates": "x,y",
        "full_les_points": int(len(boundaries["full"])),
        "strict_interior_points": int(len(boundaries["interior"])),
        "hump_wall_points": int(len(boundaries["wall"])),
        "left_subdomain_points": int(len(boundaries["left"])),
        "right_subdomain_points": int(len(boundaries["right"])),
        "top_subdomain_points": int(len(boundaries["top"])),
        "combined_subdomain_boundary_points": int(len(boundaries["subbc"])),
        "interior_supervised_ratio_of_full_les_grid": float(args.supervised_ratio),
        "interior_train_points": int(len(uv_train_df)),
        "interior_validation_points": int(len(uv_val_df)),
        "interior_test_points": int(len(uv_test_df)),
        "interior_train_pool_unused_points": int(
            (uv_manifest["split"] == "train_pool_unused").sum()
        ),
        "validation_fraction_requested": float(args.validation_fraction),
        "test_fraction_requested": float(args.test_fraction),
        "split_mode": str(args.split_mode),
        "spatial_block_width": int(args.spatial_block_width),
        "cp_train_points": int(len(cp_train_df)),
        "cp_validation_points": int(len(cp_val_df)),
        "cp_test_points": int(len(cp_test_df)),
        "valid_collocation_points": int(len(eqa_np)),
        "collocation_mode": str(args.collocation_mode),
        "wall_bc": "u=v=0 on physical hump wall",
        "subdomain_bc": (
            "LES u,v observations on left/right/top truncated interfaces"
            if not args.disable_subbc
            else "disabled"
        ),
        "pressure_gauge_training": (
            "Cp supervision fixes pressure level through p_target=Cp/cp_scale"
            if (args.cp_source != "none" and len(cp_train_df) > 0)
            else "no Cp pressure-level constraint"
        ),
        **legacy_summary,
    }

    save_protocol_artifacts(
        output_dir=protocol_root,
        meanfield=meanfield,
        interior_manifest=uv_manifest,
        cp_manifest=cp_manifest,
        boundaries=boundaries,
        collocation_points=eqa_np,
        legacy_table=legacy_table,
        legacy_outside=legacy_outside,
        summary=protocol_summary,
    )

    return {
        "meanfield": meanfield,
        "uv_df": uv_train_df,
        "uv_val_df": uv_val_df,
        "uv_test_df": uv_test_df,
        "uv_manifest": uv_manifest,
        "wall_df": boundaries["wall"],
        "subbc_df": boundaries["subbc"],
        "left_bc_df": boundaries["left"],
        "right_bc_df": boundaries["right"],
        "top_bc_df": boundaries["top"],
        "cp_df": cp_train_df,
        "cp_val_df": cp_val_df,
        "cp_test_df": cp_test_df,
        "cp_manifest": cp_manifest,
        "eqa_np": eqa_np,
        "legacy_collocation_summary": legacy_summary,
        "protocol_root": protocol_root,
    }


def predict_uvp(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
):
    return model.predict_fields(x, y, create_graph=True)


def supervised_uv_loss(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    u_true: torch.Tensor,
    v_true: torch.Tensor,
) -> torch.Tensor:
    u_pred, v_pred, _ = predict_uvp(model, x, y)
    mse = nn.MSELoss()
    return mse(u_pred, u_true) + mse(v_pred, v_true)


def no_slip_wall_loss(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Physical hump-wall condition u=v=0."""
    u_pred, v_pred, _ = predict_uvp(model, x, y)
    mse = nn.MSELoss()
    return mse(u_pred, torch.zeros_like(u_pred)) + mse(
        v_pred, torch.zeros_like(v_pred)
    )


def subdomain_boundary_loss(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    u_true: torch.Tensor,
    v_true: torch.Tensor,
) -> torch.Tensor:
    """LES u,v observations on truncated left/right/top reconstruction interfaces."""
    u_pred, v_pred, _ = predict_uvp(model, x, y)
    mse = nn.MSELoss()
    return mse(u_pred, u_true) + mse(v_pred, v_true)


def cp_loss(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    cp_true: torch.Tensor,
    cp_scale: float = 2.0,
) -> torch.Tensor:
    out = model.forward(x, y)
    p_pred = out[:, 1:2]
    cp_pred = float(cp_scale) * p_pred
    return nn.MSELoss()(cp_pred, cp_true)


def steady_ns_residual_loss(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    re: float,
) -> torch.Tensor:
    """Simplified steady incompressible momentum residual used as a soft regularizer."""
    u, v, p = predict_uvp(model, x, y)

    u_x = torch.autograd.grad(u.sum(), x, create_graph=True, retain_graph=True)[0]
    u_y = torch.autograd.grad(u.sum(), y, create_graph=True, retain_graph=True)[0]
    v_x = torch.autograd.grad(v.sum(), x, create_graph=True, retain_graph=True)[0]
    v_y = torch.autograd.grad(v.sum(), y, create_graph=True, retain_graph=True)[0]
    p_x = torch.autograd.grad(p.sum(), x, create_graph=True, retain_graph=True)[0]
    p_y = torch.autograd.grad(p.sum(), y, create_graph=True, retain_graph=True)[0]

    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True, retain_graph=True)[0]
    u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True, retain_graph=True)[0]
    v_xx = torch.autograd.grad(v_x.sum(), x, create_graph=True, retain_graph=True)[0]
    v_yy = torch.autograd.grad(v_y.sum(), y, create_graph=True, retain_graph=True)[0]

    fx = u * u_x + v * u_y + p_x - (1.0 / float(re)) * (u_xx + u_yy)
    fy = u * v_x + v * v_y + p_y - (1.0 / float(re)) * (v_xx + v_yy)

    mse = nn.MSELoss()
    return mse(fx, torch.zeros_like(fx)) + mse(fy, torch.zeros_like(fy))


def combine_losses(
    uv_loss: torch.Tensor,
    eq_loss: torch.Tensor,
    cp_loss_value: torch.Tensor,
    wall_loss_value: torch.Tensor,
    subbc_loss_value: torch.Tensor,
    adaptive_log_vars: Optional[torch.nn.Parameter],
    data_weight: float,
    equation_weight: float,
    cp_weight: float,
    wall_weight: float,
    subbc_weight: float,
) -> torch.Tensor:
    weighted_uv = float(data_weight) * uv_loss
    weighted_eq = float(equation_weight) * eq_loss
    weighted_cp = float(cp_weight) * cp_loss_value
    weighted_wall = float(wall_weight) * wall_loss_value
    weighted_subbc = float(subbc_weight) * subbc_loss_value

    # Keep the legacy global adaptive weighting mechanism limited to the three
    # terms it originally adjusted. Boundary terms are explicit fixed controls.
    if adaptive_log_vars is None:
        core = weighted_uv + weighted_eq + weighted_cp
    else:
        core = (
            torch.exp(-adaptive_log_vars[0]) * weighted_uv
            + torch.exp(-adaptive_log_vars[1]) * weighted_eq
            + torch.exp(-adaptive_log_vars[2]) * weighted_cp
            + adaptive_log_vars.sum()
        )

    return core + weighted_wall + weighted_subbc


def sample_rows(
    df: pd.DataFrame,
    batch_size: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if len(df) == 0:
        return df.copy()
    if batch_size >= len(df):
        return (
            df.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1)))
            .reset_index(drop=True)
        )
    idx = rng.choice(len(df), size=batch_size, replace=False)
    return df.iloc[idx].reset_index(drop=True)


def sample_array(
    arr: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if batch_size >= arr.shape[0]:
        return arr[rng.permutation(arr.shape[0])]
    idx = rng.choice(arr.shape[0], size=batch_size, replace=False)
    return arr[idx]


def make_uv_tensors(batch: pd.DataFrame, device: torch.device):
    x = to_tensor(batch[["x"]].to_numpy(), device, requires_grad=True)
    y = to_tensor(batch[["y"]].to_numpy(), device, requires_grad=True)
    u = to_tensor(batch[["u"]].to_numpy(), device, requires_grad=False)
    v = to_tensor(batch[["v"]].to_numpy(), device, requires_grad=False)
    return x, y, u, v


def make_target_uv_tensors(batch: pd.DataFrame, device: torch.device):
    x = to_tensor(batch[["x"]].to_numpy(), device, requires_grad=True)
    y = to_tensor(batch[["y"]].to_numpy(), device, requires_grad=True)
    u = to_tensor(batch[["u_target"]].to_numpy(), device, requires_grad=False)
    v = to_tensor(batch[["v_target"]].to_numpy(), device, requires_grad=False)
    return x, y, u, v


def make_xy_tensors(batch: pd.DataFrame, device: torch.device):
    x = to_tensor(batch[["x"]].to_numpy(), device, requires_grad=True)
    y = to_tensor(batch[["y"]].to_numpy(), device, requires_grad=True)
    return x, y


def make_cp_tensors(batch: pd.DataFrame, device: torch.device):
    x = to_tensor(batch[["x"]].to_numpy(), device, requires_grad=True)
    y = to_tensor(batch[["y"]].to_numpy(), device, requires_grad=True)
    cp = to_tensor(batch[["cp"]].to_numpy(), device, requires_grad=False)
    return x, y, cp


def make_eqa_tensors(batch: np.ndarray, device: torch.device):
    x = to_tensor(batch[:, 0:1], device, requires_grad=True)
    y = to_tensor(batch[:, 1:2], device, requires_grad=True)
    return x, y


def evaluate_uv_on_reference(
    model: torch.nn.Module,
    uv_df: pd.DataFrame,
    device: torch.device,
    max_points: int = 4096,
) -> Dict[str, float]:
    was_training = model.training
    model.eval()

    if len(uv_df) > max_points:
        eval_df = uv_df.sample(n=max_points, random_state=1234).reset_index(drop=True)
    else:
        eval_df = uv_df.reset_index(drop=True)

    x, y, u_ref, v_ref = make_uv_tensors(eval_df, device)

    with torch.enable_grad():
        u_pred, v_pred, _ = model.predict_fields(x, y, create_graph=False)

    u_err = u_pred.detach() - u_ref
    v_err = v_pred.detach() - v_ref

    out = {
        "diag_u_mae": float(torch.mean(torch.abs(u_err)).cpu()),
        "diag_v_mae": float(torch.mean(torch.abs(v_err)).cpu()),
        "diag_u_rmse": float(torch.sqrt(torch.mean(u_err ** 2)).cpu()),
        "diag_v_rmse": float(torch.sqrt(torch.mean(v_err ** 2)).cpu()),
    }

    if was_training:
        model.train()
    else:
        model.eval()

    return out


def method_config_with_checkpoint(method: str, models_root: Path) -> Dict[str, object]:
    cfg = dict(HUMP_METHODS[method])
    cfg["checkpoint"] = models_root / cfg["checkpoint_name"]

    return cfg


def build_optimizer(
    model: torch.nn.Module,
    cfg: Dict[str, object],
    adaptive_log_vars: Optional[torch.nn.Parameter],
    learning_rate: float,
) -> torch.optim.Optimizer:
    params = list(model.parameters())

    if adaptive_log_vars is not None:
        params.append(adaptive_log_vars)

    return torch.optim.Adam(
        params,
        lr=learning_rate,
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )


def train_method(
    method: str,
    args: argparse.Namespace,
    data: Dict[str, object],
    device: torch.device,
) -> Path:
    if method not in HUMP_METHODS:
        raise KeyError(f"Unknown method {method}. Available: {list(HUMP_METHODS)}")

    models_root = Path(args.results_root) / "models"
    logs_root = Path(args.results_root) / "training_logs"

    ensure_dir(models_root)
    ensure_dir(logs_root)

    cfg = method_config_with_checkpoint(method, models_root)
    checkpoint = Path(cfg["checkpoint"])

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    model = build_model(cfg, LAYER_MAT_PSI).to(device)

    if args.resume:
        loaded = safe_load_checkpoint(model, checkpoint, device)

        if loaded:
            print(f"[{method}] resumed from checkpoint: {checkpoint}")
        else:
            print(f"[{method}] resume requested, but checkpoint not found: {checkpoint}")

    adaptive_log_vars = None

    if bool(cfg.get("adaptive_loss", False)):
        adaptive_log_vars = torch.nn.Parameter(
            torch.zeros(3, dtype=torch.float32, device=device)
        )

    optimizer = build_optimizer(
        model=model,
        cfg=cfg,
        adaptive_log_vars=adaptive_log_vars,
        learning_rate=args.learning_rate,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
        eta_min=args.min_learning_rate,
    )

    uv_df: pd.DataFrame = data["uv_df"]
    uv_val_df: pd.DataFrame = data["uv_val_df"]
    uv_test_df: pd.DataFrame = data["uv_test_df"]
    wall_df: pd.DataFrame = data["wall_df"]
    subbc_df: pd.DataFrame = data["subbc_df"]
    cp_df: pd.DataFrame = data["cp_df"]
    cp_val_df: pd.DataFrame = data["cp_val_df"]
    cp_test_df: pd.DataFrame = data["cp_test_df"]
    eqa_np: np.ndarray = data["eqa_np"]

    use_cp = (args.cp_loss_weight > 0.0) and (not cp_df.empty)
    use_wall = (
        (not args.disable_wall_bc)
        and (args.wall_loss_weight > 0.0)
        and (not wall_df.empty)
    )
    use_subbc = (
        (not args.disable_subbc)
        and (args.subbc_loss_weight > 0.0)
        and (not subbc_df.empty)
    )

    if args.cp_loss_weight > 0.0 and cp_df.empty:
        print(
            f"[{method}] Cp loss weight > 0, but no Cp points found in the LES x-window. "
            "Cp loss is disabled."
        )

    print("=" * 88)
    print(f"Training NASA hump method: {method}")
    print(f"Label: {cfg.get('label', method)}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Device: {device}")
    print(f"Parameters: {count_parameters(model)}")
    print(f"Dropout rate: {cfg.get('dropout_rate', 0.0)}")
    print(f"Weight decay: {cfg.get('weight_decay', 0.0)}")
    print(f"Adaptive loss: {cfg.get('adaptive_loss', False)}")
    print(f"Interior supervised u/v train points: {len(uv_df)}")
    print(f"Interior validation u/v points: {len(uv_val_df)}")
    print(f"Interior held-out test u/v points: {len(uv_test_df)}")
    print(f"Physical hump-wall points: {len(wall_df)} | no-slip enabled: {use_wall}")
    print(f"Subdomain boundary observation points: {len(subbc_df)} | enabled: {use_subbc}")
    print(f"Cp train points: {len(cp_df)} | validation: {len(cp_val_df)} | test: {len(cp_test_df)} | enabled: {use_cp} | source: {args.cp_source}")
    print(f"Equation collocation points: {eqa_np.shape[0]} | mode: {args.collocation_mode}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch sizes: uv={args.batch_size_data}, eq={args.batch_size_equation}, cp={args.batch_size_cp}, wall={args.batch_size_wall}, subbc={args.batch_size_subbc}")
    print(
        f"Loss weights: data={args.data_loss_weight}, "
        f"equation={args.equation_loss_weight}, cp={args.cp_loss_weight}, "
        f"wall={args.wall_loss_weight if use_wall else 0.0}, "
        f"subbc={args.subbc_loss_weight if use_subbc else 0.0}"
    )
    print("=" * 88)

    rows: List[Dict[str, float]] = []
    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()

        n_inner = max(
            1,
            math.ceil(len(uv_df) / max(1, min(args.batch_size_data, len(uv_df)))),
        )

        epoch_rows: List[Dict[str, float]] = []

        for inner in range(n_inner):
            uv_batch = sample_rows(uv_df, args.batch_size_data, rng)
            eqa_batch = sample_array(eqa_np, args.batch_size_equation, rng)

            x_data, y_data, u_true, v_true = make_uv_tensors(uv_batch, device)
            x_eqa, y_eqa = make_eqa_tensors(eqa_batch, device)

            optimizer.zero_grad(set_to_none=True)

            loss_uv = supervised_uv_loss(
                model=model,
                x=x_data,
                y=y_data,
                u_true=u_true,
                v_true=v_true,
            )

            if args.equation_loss_weight > 0.0:
                loss_eq = steady_ns_residual_loss(
                    model=model,
                    x=x_eqa,
                    y=y_eqa,
                    re=args.reynolds,
                )
            else:
                loss_eq = torch.zeros((), dtype=torch.float32, device=device)

            if use_cp:
                cp_batch = sample_rows(cp_df, args.batch_size_cp, rng)
                x_cp, y_cp, cp_true = make_cp_tensors(cp_batch, device)
                loss_cp_value = cp_loss(
                    model=model,
                    x=x_cp,
                    y=y_cp,
                    cp_true=cp_true,
                    cp_scale=args.cp_scale,
                )
            else:
                loss_cp_value = torch.zeros((), dtype=torch.float32, device=device)

            if use_wall:
                wall_batch = sample_rows(wall_df, args.batch_size_wall, rng)
                x_wall, y_wall = make_xy_tensors(wall_batch, device)
                loss_wall_value = no_slip_wall_loss(
                    model=model,
                    x=x_wall,
                    y=y_wall,
                )
            else:
                loss_wall_value = torch.zeros((), dtype=torch.float32, device=device)

            if use_subbc:
                subbc_batch = sample_rows(subbc_df, args.batch_size_subbc, rng)
                x_subbc, y_subbc, u_subbc, v_subbc = make_target_uv_tensors(
                    subbc_batch, device
                )
                loss_subbc_value = subdomain_boundary_loss(
                    model=model,
                    x=x_subbc,
                    y=y_subbc,
                    u_true=u_subbc,
                    v_true=v_subbc,
                )
            else:
                loss_subbc_value = torch.zeros((), dtype=torch.float32, device=device)

            total_loss = combine_losses(
                uv_loss=loss_uv,
                eq_loss=loss_eq,
                cp_loss_value=loss_cp_value,
                wall_loss_value=loss_wall_value,
                subbc_loss_value=loss_subbc_value,
                adaptive_log_vars=adaptive_log_vars,
                data_weight=args.data_loss_weight,
                equation_weight=args.equation_loss_weight,
                cp_weight=args.cp_loss_weight if use_cp else 0.0,
                wall_weight=args.wall_loss_weight if use_wall else 0.0,
                subbc_weight=args.subbc_loss_weight if use_subbc else 0.0,
            )

            if not torch.isfinite(total_loss):
                raise FloatingPointError(
                    f"[{method}] non-finite loss at epoch={epoch}, inner={inner + 1}: "
                    f"total={total_loss.item()}, uv={loss_uv.item()}, "
                    f"eq={loss_eq.item()}, cp={loss_cp_value.item()}, "
                    f"wall={loss_wall_value.item()}, subbc={loss_subbc_value.item()}"
                )

            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

            optimizer.step()

            epoch_rows.append(
                {
                    "total_loss": float(total_loss.detach().cpu()),
                    "uv_loss": float(loss_uv.detach().cpu()),
                    "equation_loss": float(loss_eq.detach().cpu()),
                    "cp_loss": float(loss_cp_value.detach().cpu()),
                    "wall_loss": float(loss_wall_value.detach().cpu()),
                    "subbc_loss": float(loss_subbc_value.detach().cpu()),
                    "weighted_uv_loss": float(
                        (args.data_loss_weight * loss_uv).detach().cpu()
                    ),
                    "weighted_equation_loss": float(
                        (args.equation_loss_weight * loss_eq).detach().cpu()
                    ),
                    "weighted_cp_loss": float(
                        (
                            (args.cp_loss_weight if use_cp else 0.0)
                            * loss_cp_value
                        ).detach().cpu()
                    ),
                    "weighted_wall_loss": float(
                        (
                            (args.wall_loss_weight if use_wall else 0.0)
                            * loss_wall_value
                        ).detach().cpu()
                    ),
                    "weighted_subbc_loss": float(
                        (
                            (args.subbc_loss_weight if use_subbc else 0.0)
                            * loss_subbc_value
                        ).detach().cpu()
                    ),
                }
            )

            del x_data, y_data, u_true, v_true, x_eqa, y_eqa
            del loss_uv, loss_eq, loss_cp_value, loss_wall_value, loss_subbc_value, total_loss

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        scheduler.step()

        elapsed = time.perf_counter() - start_time
        epoch_df = pd.DataFrame(epoch_rows)
        diag: Dict[str, float] = {}

        if (
            (epoch % args.diagnostic_interval == 0)
            or (epoch == 1)
            or (epoch == args.epochs)
        ):
            diag = evaluate_uv_on_reference(
                model=model,
                uv_df=uv_val_df,
                device=device,
                max_points=args.diagnostic_points,
            )

        row: Dict[str, float] = {
            "method": method,
            "epoch": epoch,
            "total_loss": float(epoch_df["total_loss"].mean()),
            "uv_loss": float(epoch_df["uv_loss"].mean()),
            "equation_loss": float(epoch_df["equation_loss"].mean()),
            "cp_loss": float(epoch_df["cp_loss"].mean()),
            "wall_loss": float(epoch_df["wall_loss"].mean()),
            "subbc_loss": float(epoch_df["subbc_loss"].mean()),
            "weighted_uv_loss": float(epoch_df["weighted_uv_loss"].mean()),
            "weighted_equation_loss": float(epoch_df["weighted_equation_loss"].mean()),
            "weighted_cp_loss": float(epoch_df["weighted_cp_loss"].mean()),
            "weighted_wall_loss": float(epoch_df["weighted_wall_loss"].mean()),
            "weighted_subbc_loss": float(epoch_df["weighted_subbc_loss"].mean()),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "training_time_seconds": float(elapsed),
            "parameters": int(count_parameters(model)),
            "data_loss_weight": float(args.data_loss_weight),
            "equation_loss_weight": float(args.equation_loss_weight),
            "cp_loss_weight": float(args.cp_loss_weight if use_cp else 0.0),
            "wall_loss_weight": float(args.wall_loss_weight if use_wall else 0.0),
            "subbc_loss_weight": float(args.subbc_loss_weight if use_subbc else 0.0),
            "supervised_points": int(len(uv_df)),
            "validation_points": int(len(uv_val_df)),
            "heldout_test_points": int(len(uv_test_df)),
            "equation_points": int(eqa_np.shape[0]),
            "wall_points": int(len(wall_df)),
            "subbc_points": int(len(subbc_df)),
            "cp_points": int(len(cp_df)),
            "cp_validation_points": int(len(cp_val_df)),
            "cp_test_points": int(len(cp_test_df)),
            "reynolds": float(args.reynolds),
        }

        row.update(diag)

        if adaptive_log_vars is not None:
            row["adaptive_log_var_data"] = float(adaptive_log_vars[0].detach().cpu())
            row["adaptive_log_var_equation"] = float(adaptive_log_vars[1].detach().cpu())
            row["adaptive_log_var_cp"] = float(adaptive_log_vars[2].detach().cpu())
            row["adaptive_weight_data"] = float(
                torch.exp(-adaptive_log_vars[0]).detach().cpu()
            )
            row["adaptive_weight_equation"] = float(
                torch.exp(-adaptive_log_vars[1]).detach().cpu()
            )
            row["adaptive_weight_cp"] = float(
                torch.exp(-adaptive_log_vars[2]).detach().cpu()
            )

        rows.append(row)

        if epoch % args.save_interval == 0 or epoch == args.epochs:
            torch.save(model.state_dict(), checkpoint)

        log_path = logs_root / f"{method}_training_log.csv"
        pd.DataFrame(rows).to_csv(log_path, index=False)

        print(
            f"[{method}] epoch {epoch:05d}/{args.epochs} | "
            f"total={row['total_loss']:.4e} | "
            f"uv={row['uv_loss']:.4e} | "
            f"eq={row['equation_loss']:.4e} | "
            f"cp={row['cp_loss']:.4e} | "
            f"wall={row['wall_loss']:.4e} | "
            f"subbc={row['subbc_loss']:.4e} | "
            f"lr={row['learning_rate']:.2e} | "
            f"time={elapsed:.1f}s"
            + (
                f" | diag_u_mae={diag.get('diag_u_mae', np.nan):.4e} "
                f"diag_v_mae={diag.get('diag_v_mae', np.nan):.4e}"
                if diag
                else ""
            )
        )

    total_time = time.perf_counter() - start_time

    torch.save(model.state_dict(), checkpoint)

    summary = {
        "method": method,
        "label": cfg.get("label", method),
        "checkpoint": str(checkpoint),
        "epochs": int(args.epochs),
        "parameters": int(count_parameters(model)),
        "training_time_seconds": float(total_time),
        "final_total_loss": rows[-1]["total_loss"],
        "final_uv_loss": rows[-1]["uv_loss"],
        "final_equation_loss": rows[-1]["equation_loss"],
        "final_cp_loss": rows[-1]["cp_loss"],
        "final_wall_loss": rows[-1]["wall_loss"],
        "final_subbc_loss": rows[-1]["subbc_loss"],
        "input_dimension": 2,
        "input_coordinates": "x,y",
        "supervised_ratio": float(args.supervised_ratio),
        "supervised_points": int(len(uv_df)),
        "validation_points": int(len(uv_val_df)),
        "heldout_test_points": int(len(uv_test_df)),
        "equation_points": int(eqa_np.shape[0]),
        "wall_points": int(len(wall_df)),
        "subbc_points": int(len(subbc_df)),
        "cp_points": int(len(cp_df)),
        "cp_validation_points": int(len(cp_val_df)),
        "cp_test_points": int(len(cp_test_df)),
        "cp_source": str(args.cp_source),
        "cp_scale": float(args.cp_scale),
        "reynolds": float(args.reynolds),
        "data_loss_weight": float(args.data_loss_weight),
        "equation_loss_weight": float(args.equation_loss_weight),
        "cp_loss_weight": float(args.cp_loss_weight if use_cp else 0.0),
        "wall_loss_weight": float(args.wall_loss_weight if use_wall else 0.0),
        "subbc_loss_weight": float(args.subbc_loss_weight if use_subbc else 0.0),
        "wall_bc_enabled": bool(use_wall),
        "subdomain_boundary_observations_enabled": bool(use_subbc),
        "pressure_gauge_training": (
            "Cp reference fixes pressure level through p_target=Cp/cp_scale"
            if use_cp else "no explicit Cp pressure-level constraint"
        ),
        "legacy_below_wall_solid_points": int(
            data["legacy_collocation_summary"]["legacy_below_wall_solid_points"]
        ),
        "legacy_below_wall_solid_fraction": float(
            data["legacy_collocation_summary"]["legacy_below_wall_solid_fraction"]
        ),
    }

    for key in ["diag_u_mae", "diag_v_mae", "diag_u_rmse", "diag_v_rmse"]:
        if key in rows[-1]:
            summary[f"final_{key}"] = rows[-1][key]

    pd.DataFrame([summary]).to_csv(
        logs_root / f"{method}_training_summary.csv",
        index=False,
    )

    print(f"[{method}] training complete. Checkpoint saved to: {checkpoint}")
    print(f"[{method}] summary saved to: {logs_root / f'{method}_training_summary.csv'}")

    return checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PINN baselines for the NASA wall-mounted hump case."
    )

    parser.add_argument(
        "--data-dir",
        default="./hump_data",
        help="Directory containing NASA hump .dat files.",
    )

    parser.add_argument(
        "--results-root",
        default="./hump_results",
        help="Output root for models and training logs.",
    )

    parser.add_argument(
        "--method",
        default="all",
        choices=["all"] + list(HUMP_METHODS.keys()),
        help="Which method to train.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=2000,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Initial learning rate.",
    )

    parser.add_argument(
        "--min-learning-rate",
        type=float,
        default=1e-6,
        help="Cosine scheduler minimum learning rate.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Random seed.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing checkpoint if present.",
    )

    parser.add_argument(
        "--supervised-ratio",
        type=float,
        default=0.02,
        help=(
            "Sparse strict-interior velocity observation ratio expressed relative "
            "to the full LES mean-field grid. Same seed gives nested 1--5%% subsets."
        ),
    )

    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.15,
        help="Requested strict-interior validation fraction.",
    )

    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.20,
        help="Requested strict-interior held-out test fraction.",
    )

    parser.add_argument(
        "--split-mode",
        choices=["spatial_blocks", "random"],
        default="spatial_blocks",
        help="Use whole x-column blocks for leakage-resistant holdout.",
    )

    parser.add_argument(
        "--spatial-block-width",
        type=int,
        default=8,
        help="Structured-grid i-columns per interior holdout block.",
    )

    parser.add_argument(
        "--cp-spatial-block-width",
        type=int,
        default=12,
        help="Contiguous wall-Cp samples per holdout block.",
    )

    parser.add_argument(
        "--protocol-only",
        action="store_true",
        help="Generate all split/boundary/collocation artifacts and exit before training.",
    )

    parser.add_argument(
        "--n-equation-points",
        type=int,
        default=50000,
        help="Number of steady NS collocation points.",
    )

    parser.add_argument(
        "--collocation-mode",
        choices=["fluid_domain", "reference_points", "random_box_legacy"],
        default="fluid_domain",
        help=(
            "fluid_domain is the formal revision setting; random_box_legacy is "
            "retained only for reproduction/diagnostics."
        ),
    )

    parser.add_argument(
        "--batch-size-data",
        type=int,
        default=512,
        help="Batch size for supervised u/v data.",
    )

    parser.add_argument(
        "--batch-size-equation",
        type=int,
        default=2048,
        help="Batch size for equation residual points.",
    )

    parser.add_argument(
        "--batch-size-cp",
        type=int,
        default=512,
        help="Batch size for Cp supervision points.",
    )

    parser.add_argument(
        "--batch-size-wall",
        type=int,
        default=512,
        help="Batch size for physical hump-wall no-slip points.",
    )

    parser.add_argument(
        "--batch-size-subbc",
        type=int,
        default=512,
        help="Batch size for left/right/top subdomain boundary observations.",
    )

    parser.add_argument(
        "--data-loss-weight",
        type=float,
        default=10.0,
        help="Weight for supervised u/v data loss.",
    )

    parser.add_argument(
        "--equation-loss-weight",
        type=float,
        default=1e-4,
        help=(
            "Weight for steady NS residual. Default is small because the NASA hump "
            "reference is a turbulent LES mean field, not a laminar DNS snapshot."
        ),
    )

    parser.add_argument(
        "--cp-loss-weight",
        type=float,
        default=1.0,
        help="Weight for wall Cp supervision. Set 0 to disable pressure/Cp supervision.",
    )

    parser.add_argument(
        "--wall-loss-weight",
        type=float,
        default=1.0,
        help="Weight for physical hump-wall no-slip loss.",
    )

    parser.add_argument(
        "--subbc-loss-weight",
        type=float,
        default=1.0,
        help=(
            "Weight for LES u,v observations on truncated left/right/top "
            "reconstruction interfaces. This is not a full-domain inlet/outlet/top-wall BC."
        ),
    )

    parser.add_argument(
        "--disable-wall-bc",
        action="store_true",
        help="Disable the physical no-slip hump-wall loss for an ablation/control run.",
    )

    parser.add_argument(
        "--disable-subbc",
        action="store_true",
        help="Disable left/right/top subdomain boundary observations for an ablation/control run.",
    )

    parser.add_argument(
        "--cp-source",
        choices=["les", "exp", "both", "none"],
        default="les",
        help="Cp source used for pressure supervision.",
    )

    parser.add_argument(
        "--cp-scale",
        type=float,
        default=2.0,
        help="Cp = cp_scale * p_pred. Use 2.0 if p is nondimensionalized by rho*U^2.",
    )

    parser.add_argument(
        "--reynolds",
        type=float,
        default=RE_HUMP,
        help="Reynolds number used in residual.",
    )

    parser.add_argument(
        "--grad-clip",
        type=float,
        default=10.0,
        help="Gradient clipping max norm.",
    )

    parser.add_argument(
        "--save-interval",
        type=int,
        default=50,
        help="Checkpoint save interval in epochs.",
    )

    parser.add_argument(
        "--diagnostic-interval",
        type=int,
        default=50,
        help="Diagnostic print interval in epochs.",
    )

    parser.add_argument(
        "--diagnostic-points",
        type=int,
        default=4096,
        help="Max points for diagnostic MAE/RMSE.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.cp_source == "none":
        args.cp_loss_weight = 0.0

    device = get_device()
    set_seed(args.seed)

    data = load_training_data(args)

    meanfield = data["meanfield"]
    flat = meanfield["flat"]

    print("NASA hump data loaded.")
    print(f"Mean-field grid: I={meanfield['I']}, J={meanfield['J']}, points={len(flat)}")
    print(f"x range: [{flat['x'].min():.6f}, {flat['x'].max():.6f}]")
    print(f"y range: [{flat['y'].min():.6f}, {flat['y'].max():.6f}]")
    print(f"u range: [{flat['u'].min():.6f}, {flat['u'].max():.6f}]")
    print(f"v range: [{flat['v'].min():.6f}, {flat['v'].max():.6f}]")
    print(f"Protocol artifacts: {data['protocol_root'].resolve()}")
    print(
        "Legacy rectangular sampler below-wall solid fraction: "
        f"{data['legacy_collocation_summary']['legacy_below_wall_solid_fraction']:.6f}"
    )
    print(
        "Legacy rectangular sampler above local LES-window fraction: "
        f"{data['legacy_collocation_summary']['legacy_above_local_reference_window_fraction']:.6f}"
    )

    if args.protocol_only:
        print("Protocol-only mode complete; no model training was started.")
        return

    if args.method == "all":
        methods = ACTIVE_HUMP_METHODS
    else:
        methods = [args.method]

    checkpoints: List[Path] = []

    for method in methods:
        checkpoint = train_method(method, args, data, device)
        checkpoints.append(checkpoint)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nAll requested NASA hump training runs finished.")

    for ckpt in checkpoints:
        print(f"  {ckpt}")

    print("\nNext step:")

    print(
        f"  python hump_heldout_evaluate.py --data-dir {args.data_dir} "
        f"--models-root {Path(args.results_root) / 'models'} --results-root {args.results_root}"
    )
    print(
        f"  python hump_validation.py --data-dir {args.data_dir} "
        f"--models-root {Path(args.results_root) / 'models'} --results-root {args.results_root}"
    )


if __name__ == "__main__":
    main()
