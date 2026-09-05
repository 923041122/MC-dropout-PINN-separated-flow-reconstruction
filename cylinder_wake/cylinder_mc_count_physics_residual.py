#!/usr/bin/env python3
"""
cylinder_mc_count_physics_residual.py

Independent-grid physical-residual convergence study for the final Re=3900
MC-dropout cylinder PINN.

Purpose
-------
Load ONE frozen MC-dropout B-PINN checkpoint and evaluate, on the same fixed
independent physics grid:

    1. deterministic dropout-OFF prediction
    2. one stochastic dropout prediction
    3. MC predictive means for N = 5, 10, 20, 50, 100 by default

The stochastic sequence is nested:
    MC5 ⊂ MC10 ⊂ MC20 ⊂ MC50 ⊂ MC100

For each MC count, the script reports BOTH:

A) residual_of_mc_predictive_mean
   The residual of the MC-mean field. Because differentiation is linear, the
   required derivatives are averaged across stochastic passes first; the
   nonlinear convective products are then formed from those averaged fields.

B) stochastic_subnetwork_residual_statistics
   Residual statistics of the individual stochastic subnetworks, including the
   pointwise mean residual field and RMS residual over all stochastic passes.

The dimensionless residual is the same simplified incompressible momentum form
used in the cylinder PINN workflow:

    continuity = u_x + v_y

    f_x = u_t + u*u_x + v*u_y + p_x
          - (1/Re)*(u_xx + u_yy)

    f_y = v_t + u*v_x + v*v_y + p_y
          - (1/Re)*(v_xx + v_yy)

with
    u = d(psi)/dy
    v = -d(psi)/dx

Expected project files
----------------------
Place this script in the cylinder project directory containing:
    pinn_model.py
    pinn_model_dropout_ablation.py
    benchmark_config.py

The default independent grid is:
    ./results_cylinder_protocol_v1_1/protocol/
        independent_physics_evaluation_points.csv

The default checkpoint is:
    ./cylinder_bpinn_final_p0002_all/models/bpinn_dropout.pth

Outputs
-------
<output-dir>/
    mc_count_physics_residual_summary.csv
    mc_count_physics_residual_timing.csv
    mc_count_physics_residual_pointwise.csv
    mc_count_physics_residual_convergence.png
    mc_count_physics_residual_metadata.txt
"""

from __future__ import annotations

import argparse
import hashlib
import math
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from benchmark_config import LAYER_MAT_PSI, METHODS, REYNOLDS
from pinn_model_dropout_ablation import PlacementPINNNet


DERIVATIVE_KEYS = (
    "u", "v",
    "u_t", "u_x", "u_y", "u_xx", "u_yy",
    "v_t", "v_x", "v_y", "v_xx", "v_yy",
    "p_x", "p_y",
)


def parse_args() -> argparse.Namespace:
    default_dropout = float(METHODS["bpinn_dropout"].get("dropout_rate", 0.002))

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate MC-sample-count convergence of independent-grid "
            "continuity/momentum residuals for the final cylinder MC-dropout PINN."
        )
    )
    parser.add_argument(
        "--protocol-root",
        type=Path,
        default=Path("./results_cylinder_protocol_v1_1/protocol"),
        help="Protocol directory containing independent_physics_evaluation_points.csv.",
    )
    parser.add_argument(
        "--physics-points",
        type=Path,
        default=None,
        help=(
            "Optional direct path to the independent physics CSV. "
            "Overrides --protocol-root."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("./cylinder_bpinn_final_p0002_all/models/bpinn_dropout.pth"),
        help="Frozen final B-PINN state_dict checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./cylinder_mc_count_physics_residual"),
    )
    parser.add_argument(
        "--mc-counts",
        type=int,
        nargs="+",
        default=[5, 10, 20, 50, 100],
        help="Nested MC sample counts to summarize.",
    )
    parser.add_argument(
        "--n-physics-points",
        type=int,
        default=20000,
        help="Number of independent points to use. Use 0 to use the full CSV.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help=(
            "Point batch size. Second derivatives are memory intensive; "
            "reduce this value if CUDA runs out of memory."
        ),
    )
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--dropout-rate", type=float, default=default_dropout)
    parser.add_argument(
        "--dropout-placement",
        type=str,
        default="all",
        choices=["none", "input", "middle", "output", "alternating", "all"],
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto, cpu, cuda, cuda:0, ...",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Disable the small untimed autograd warm-up.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.mc_counts:
        raise ValueError("--mc-counts cannot be empty.")
    if any(n < 1 for n in args.mc_counts):
        raise ValueError("All --mc-counts values must be >= 1.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")
    if args.n_physics_points < 0:
        raise ValueError("--n-physics-points must be >= 0.")
    if not (0.0 < args.dropout_rate < 1.0):
        raise ValueError("--dropout-rate must satisfy 0 < p < 1.")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device}, but CUDA is not available.")
    return device


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def safe_torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def unwrap_state_dict(obj):
    if isinstance(obj, Mapping):
        for key in ("state_dict", "model_state_dict", "model"):
            candidate = obj.get(key)
            if isinstance(candidate, Mapping):
                return candidate
    return obj


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_model(
    checkpoint: Path,
    device: torch.device,
    dropout_rate: float,
    dropout_placement: str,
) -> PlacementPINNNet:
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model = PlacementPINNNet(
        layer_mat=LAYER_MAT_PSI,
        dropout_rate=dropout_rate,
        dropout_placement=dropout_placement,
    ).to(device)

    state = unwrap_state_dict(safe_torch_load(checkpoint, device))
    if not isinstance(state, Mapping):
        raise TypeError(
            f"Unsupported checkpoint object type: {type(state).__name__}. "
            "Expected a state_dict-like mapping."
        )

    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint does not strictly match PlacementPINNNet with "
            f"dropout_rate={dropout_rate} and "
            f"dropout_placement={dropout_placement!r}. "
            "Check that this is the final independently trained B-PINN checkpoint "
            "and that the placement/rate match training."
        ) from exc

    return model


def resolve_physics_path(args: argparse.Namespace) -> Path:
    if args.physics_points is not None:
        return args.physics_points
    return args.protocol_root / "independent_physics_evaluation_points.csv"


def _find_column(df: pd.DataFrame, candidates: Sequence[str], quantity: str) -> str:
    lower_to_original = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    raise KeyError(
        f"Cannot find {quantity} column. Available columns: {list(df.columns)}"
    )


def load_physics_points(
    path: Path,
    n_points: int,
    seed: int,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Independent physics CSV not found: {path}")

    raw = pd.read_csv(path)
    x_col = _find_column(raw, ["x", "x_star", "x_eq", "x_f"], "x")
    y_col = _find_column(raw, ["y", "y_star", "y_eq", "y_f"], "y")
    t_col = _find_column(raw, ["t", "t_star", "t_eq", "t_f", "time"], "t")

    points = raw[[x_col, y_col, t_col]].copy()
    points.columns = ["x", "y", "t"]
    points.insert(0, "source_row", raw.index.to_numpy(dtype=int))
    points = points.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    if points.empty:
        raise ValueError(f"No finite x/y/t rows found in {path}")

    if n_points > 0 and len(points) > n_points:
        rng = np.random.default_rng(seed)
        chosen = np.sort(rng.choice(len(points), size=n_points, replace=False))
        points = points.iloc[chosen].reset_index(drop=True)

    return points


def grad(output: torch.Tensor, input_: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        output,
        input_,
        grad_outputs=torch.ones_like(output),
        create_graph=True,
        retain_graph=True,
        allow_unused=False,
    )[0]


def derivative_bundle(
    model: torch.nn.Module,
    x_np: np.ndarray,
    y_np: np.ndarray,
    t_np: np.ndarray,
    device: torch.device,
    reynolds: float,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    One forward/autograd pass.

    Returns
    -------
    derivs:
        u, v and all derivative components needed to build the residual.
    residuals:
        continuity, fx, fy, magnitude.
    """
    x = torch.as_tensor(x_np, dtype=torch.float32, device=device).reshape(-1, 1)
    y = torch.as_tensor(y_np, dtype=torch.float32, device=device).reshape(-1, 1)
    t = torch.as_tensor(t_np, dtype=torch.float32, device=device).reshape(-1, 1)

    x = x.detach().clone().requires_grad_(True)
    y = y.detach().clone().requires_grad_(True)
    t = t.detach().clone().requires_grad_(True)

    out = model(x, y, t)
    if out.ndim != 2 or out.shape[1] < 2:
        raise ValueError(
            f"Model output must have at least two columns [psi, p], got {tuple(out.shape)}"
        )

    psi = out[:, 0:1]
    p = out[:, 1:2]

    u = grad(psi, y)
    v = -grad(psi, x)

    u_t = grad(u, t)
    u_x = grad(u, x)
    u_y = grad(u, y)
    u_xx = grad(u_x, x)
    u_yy = grad(u_y, y)

    v_t = grad(v, t)
    v_x = grad(v, x)
    v_y = grad(v, y)
    v_xx = grad(v_x, x)
    v_yy = grad(v_y, y)

    p_x = grad(p, x)
    p_y = grad(p, y)

    inv_re = 1.0 / float(reynolds)
    continuity = u_x + v_y
    fx = u_t + u * u_x + v * u_y + p_x - inv_re * (u_xx + u_yy)
    fy = v_t + u * v_x + v * v_y + p_y - inv_re * (v_xx + v_yy)
    magnitude = torch.sqrt(fx.square() + fy.square())

    tensors = {
        "u": u,
        "v": v,
        "u_t": u_t,
        "u_x": u_x,
        "u_y": u_y,
        "u_xx": u_xx,
        "u_yy": u_yy,
        "v_t": v_t,
        "v_x": v_x,
        "v_y": v_y,
        "v_xx": v_xx,
        "v_yy": v_yy,
        "p_x": p_x,
        "p_y": p_y,
    }
    residual_tensors = {
        "continuity": continuity,
        "fx": fx,
        "fy": fy,
        "magnitude": magnitude,
    }

    derivs = {
        k: value.detach().cpu().numpy().reshape(-1).astype(np.float64, copy=False)
        for k, value in tensors.items()
    }
    residuals = {
        k: value.detach().cpu().numpy().reshape(-1).astype(np.float64, copy=False)
        for k, value in residual_tensors.items()
    }

    del x, y, t, out, psi, p
    return derivs, residuals


def residual_from_mean_derivatives(
    mean: Mapping[str, np.ndarray],
    reynolds: float,
) -> Dict[str, np.ndarray]:
    inv_re = 1.0 / float(reynolds)
    continuity = mean["u_x"] + mean["v_y"]
    fx = (
        mean["u_t"]
        + mean["u"] * mean["u_x"]
        + mean["v"] * mean["u_y"]
        + mean["p_x"]
        - inv_re * (mean["u_xx"] + mean["u_yy"])
    )
    fy = (
        mean["v_t"]
        + mean["u"] * mean["v_x"]
        + mean["v"] * mean["v_y"]
        + mean["p_y"]
        - inv_re * (mean["v_xx"] + mean["v_yy"])
    )
    magnitude = np.sqrt(fx**2 + fy**2)
    return {
        "continuity": continuity,
        "fx": fx,
        "fy": fy,
        "magnitude": magnitude,
    }


def empty_accumulator(n: int) -> Dict[str, np.ndarray]:
    keys = list(DERIVATIVE_KEYS) + [
        "fx",
        "fy",
        "continuity",
        "magnitude",
        "fx_sq",
        "fy_sq",
        "continuity_sq",
        "magnitude_sq",
    ]
    return {key: np.zeros(n, dtype=np.float64) for key in keys}


def update_accumulator(
    acc: MutableMapping[str, np.ndarray],
    derivs: Mapping[str, np.ndarray],
    residuals: Mapping[str, np.ndarray],
) -> None:
    for key in DERIVATIVE_KEYS:
        acc[key] += derivs[key]

    acc["fx"] += residuals["fx"]
    acc["fy"] += residuals["fy"]
    acc["continuity"] += residuals["continuity"]
    acc["magnitude"] += residuals["magnitude"]

    acc["fx_sq"] += residuals["fx"] ** 2
    acc["fy_sq"] += residuals["fy"] ** 2
    acc["continuity_sq"] += residuals["continuity"] ** 2
    acc["magnitude_sq"] += residuals["magnitude"] ** 2


def metric_values(residuals: Mapping[str, np.ndarray]) -> Dict[str, float]:
    fx = np.asarray(residuals["fx"], dtype=np.float64)
    fy = np.asarray(residuals["fy"], dtype=np.float64)
    cont = np.asarray(residuals["continuity"], dtype=np.float64)
    mag = np.asarray(residuals["magnitude"], dtype=np.float64)

    return {
        "fx_rmse": float(np.sqrt(np.mean(fx**2))),
        "fy_rmse": float(np.sqrt(np.mean(fy**2))),
        "physics_vector_rmse": float(np.sqrt(np.mean(fx**2 + fy**2))),
        "physics_magnitude_mean": float(np.mean(mag)),
        "physics_magnitude_median": float(np.median(mag)),
        "physics_magnitude_p95": float(np.quantile(mag, 0.95)),
        "physics_magnitude_max": float(np.max(mag)),
        "continuity_rmse": float(np.sqrt(np.mean(cont**2))),
        "continuity_abs_mean": float(np.mean(np.abs(cont))),
        "continuity_abs_p95": float(np.quantile(np.abs(cont), 0.95)),
    }


def add_pointwise_columns(
    pointwise: pd.DataFrame,
    prefix: str,
    residuals: Mapping[str, np.ndarray],
) -> None:
    pointwise[f"{prefix}_continuity"] = residuals["continuity"]
    pointwise[f"{prefix}_fx"] = residuals["fx"]
    pointwise[f"{prefix}_fy"] = residuals["fy"]
    pointwise[f"{prefix}_momentum_magnitude"] = residuals["magnitude"]


def summary_row(
    mode: str,
    definition: str,
    mc_samples: int,
    residuals: Mapping[str, np.ndarray],
    elapsed_seconds: float,
    extra: Mapping[str, float] | None = None,
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "mode": mode,
        "definition": definition,
        "mc_samples": int(mc_samples),
        "elapsed_seconds": float(elapsed_seconds),
    }
    row.update(metric_values(residuals))
    if extra:
        row.update(extra)
    return row


def evaluate(
    model: PlacementPINNNet,
    points: pd.DataFrame,
    device: torch.device,
    mc_counts: Sequence[int],
    batch_size: int,
    seed: int,
    reynolds: float,
    do_warmup: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mc_counts = sorted(set(int(n) for n in mc_counts))
    max_mc = max(mc_counts)
    n_total = len(points)

    pointwise = points.copy()

    # Full-grid arrays assembled batch by batch.
    deterministic_full = {
        key: np.empty(n_total, dtype=np.float64)
        for key in ("continuity", "fx", "fy", "magnitude")
    }
    single_full = {
        key: np.empty(n_total, dtype=np.float64)
        for key in ("continuity", "fx", "fy", "magnitude")
    }

    mc_mean_full: Dict[int, Dict[str, np.ndarray]] = {
        n: {
            key: np.empty(n_total, dtype=np.float64)
            for key in ("continuity", "fx", "fy", "magnitude")
        }
        for n in mc_counts
    }
    mc_mean_subnet_field_full: Dict[int, Dict[str, np.ndarray]] = {
        n: {
            key: np.empty(n_total, dtype=np.float64)
            for key in ("continuity", "fx", "fy", "magnitude")
        }
        for n in mc_counts
    }
    mc_subnet_mag_mean_full = {
        n: np.empty(n_total, dtype=np.float64) for n in mc_counts
    }
    mc_subnet_mag_std_full = {
        n: np.empty(n_total, dtype=np.float64) for n in mc_counts
    }
    mc_nonlinearity_gap_full = {
        n: np.empty(n_total, dtype=np.float64) for n in mc_counts
    }

    deterministic_time = 0.0
    stochastic_pass_times = np.zeros(max_mc, dtype=np.float64)

    # Optional untimed warm-up. Reset RNG afterward so it cannot alter the formal
    # nested stochastic sequence.
    if do_warmup:
        warm_n = min(64, n_total)
        xw = points["x"].to_numpy(dtype=np.float32)[:warm_n]
        yw = points["y"].to_numpy(dtype=np.float32)[:warm_n]
        tw = points["t"].to_numpy(dtype=np.float32)[:warm_n]

        model.eval()
        _ = derivative_bundle(model, xw, yw, tw, device, reynolds)
        model.train()
        _ = derivative_bundle(model, xw, yw, tw, device, reynolds)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    set_seed(seed)

    x_all = points["x"].to_numpy(dtype=np.float32)
    y_all = points["y"].to_numpy(dtype=np.float32)
    t_all = points["t"].to_numpy(dtype=np.float32)

    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        x_batch = x_all[start:end]
        y_batch = y_all[start:end]
        t_batch = t_all[start:end]
        batch_n = end - start

        # 1) Deterministic dropout-OFF.
        model.eval()
        sync_if_cuda(device)
        tic = time.perf_counter()
        _, det_res = derivative_bundle(
            model, x_batch, y_batch, t_batch, device, reynolds
        )
        sync_if_cuda(device)
        deterministic_time += time.perf_counter() - tic

        for key in deterministic_full:
            deterministic_full[key][start:end] = det_res[key]

        # 2) Nested stochastic sequence.
        model.train()
        acc = empty_accumulator(batch_n)

        for pass_idx in range(1, max_mc + 1):
            sync_if_cuda(device)
            tic = time.perf_counter()
            derivs, res = derivative_bundle(
                model, x_batch, y_batch, t_batch, device, reynolds
            )
            sync_if_cuda(device)
            stochastic_pass_times[pass_idx - 1] += time.perf_counter() - tic

            update_accumulator(acc, derivs, res)

            if pass_idx == 1:
                for key in single_full:
                    single_full[key][start:end] = res[key]

            if pass_idx in mc_counts:
                n = float(pass_idx)
                mean_derivs = {key: acc[key] / n for key in DERIVATIVE_KEYS}
                mean_field_res = residual_from_mean_derivatives(mean_derivs, reynolds)

                mean_subnet_fx = acc["fx"] / n
                mean_subnet_fy = acc["fy"] / n
                mean_subnet_cont = acc["continuity"] / n
                mean_subnet_mag_field = np.sqrt(
                    mean_subnet_fx**2 + mean_subnet_fy**2
                )

                mean_subnet_field_res = {
                    "continuity": mean_subnet_cont,
                    "fx": mean_subnet_fx,
                    "fy": mean_subnet_fy,
                    "magnitude": mean_subnet_mag_field,
                }

                subnet_mag_mean = acc["magnitude"] / n
                subnet_mag_var = np.maximum(
                    acc["magnitude_sq"] / n - subnet_mag_mean**2,
                    0.0,
                )
                subnet_mag_std = np.sqrt(subnet_mag_var)

                # Difference between residual(mean prediction) and
                # mean(residual of stochastic subnetworks). For these equations,
                # this isolates the nonlinear averaging effect in the convective
                # terms.
                gap = np.sqrt(
                    (mean_field_res["fx"] - mean_subnet_fx) ** 2
                    + (mean_field_res["fy"] - mean_subnet_fy) ** 2
                )

                for key in mc_mean_full[pass_idx]:
                    mc_mean_full[pass_idx][key][start:end] = mean_field_res[key]
                    mc_mean_subnet_field_full[pass_idx][key][start:end] = (
                        mean_subnet_field_res[key]
                    )
                mc_subnet_mag_mean_full[pass_idx][start:end] = subnet_mag_mean
                mc_subnet_mag_std_full[pass_idx][start:end] = subnet_mag_std
                mc_nonlinearity_gap_full[pass_idx][start:end] = gap

            del derivs, res
            if device.type == "cuda" and pass_idx % 10 == 0:
                torch.cuda.empty_cache()

        print(
            f"Processed points {start + 1:,}-{end:,}/{n_total:,} "
            f"({100.0 * end / n_total:.1f}%)"
        )

    model.eval()

    # Assemble pointwise table.
    add_pointwise_columns(pointwise, "deterministic", deterministic_full)
    add_pointwise_columns(pointwise, "single_stochastic", single_full)

    for n in mc_counts:
        add_pointwise_columns(pointwise, f"mc{n}_mean", mc_mean_full[n])
        add_pointwise_columns(
            pointwise,
            f"mc{n}_mean_subnetwork_field",
            mc_mean_subnet_field_full[n],
        )
        pointwise[f"mc{n}_subnetwork_magnitude_mean"] = mc_subnet_mag_mean_full[n]
        pointwise[f"mc{n}_subnetwork_magnitude_std"] = mc_subnet_mag_std_full[n]
        pointwise[f"mc{n}_nonlinearity_gap"] = mc_nonlinearity_gap_full[n]

    # Summary table.
    rows: List[Dict[str, object]] = []
    rows.append(
        summary_row(
            mode="deterministic_dropout_off",
            definition="single_deterministic_field",
            mc_samples=0,
            residuals=deterministic_full,
            elapsed_seconds=deterministic_time,
        )
    )
    rows.append(
        summary_row(
            mode="single_stochastic",
            definition="single_stochastic_subnetwork",
            mc_samples=1,
            residuals=single_full,
            elapsed_seconds=float(stochastic_pass_times[0]),
        )
    )

    cumulative_times = np.cumsum(stochastic_pass_times)

    for n in mc_counts:
        # Residual of MC predictive mean.
        rows.append(
            summary_row(
                mode=f"mc{n}_predictive_mean",
                definition="residual_of_mc_predictive_mean",
                mc_samples=n,
                residuals=mc_mean_full[n],
                elapsed_seconds=float(cumulative_times[n - 1]),
                extra={
                    "nonlinearity_gap_mean": float(
                        np.mean(mc_nonlinearity_gap_full[n])
                    ),
                    "nonlinearity_gap_p95": float(
                        np.quantile(mc_nonlinearity_gap_full[n], 0.95)
                    ),
                },
            )
        )

        # Pointwise mean residual field across stochastic subnetworks.
        rows.append(
            summary_row(
                mode=f"mc{n}_mean_subnetwork_residual_field",
                definition="mean_residual_field_across_stochastic_subnetworks",
                mc_samples=n,
                residuals=mc_mean_subnet_field_full[n],
                elapsed_seconds=float(cumulative_times[n - 1]),
                extra={
                    "subnetwork_magnitude_mean_over_samples_and_points": float(
                        np.mean(mc_subnet_mag_mean_full[n])
                    ),
                    "subnetwork_magnitude_pointwise_mean_p95": float(
                        np.quantile(mc_subnet_mag_mean_full[n], 0.95)
                    ),
                    "subnetwork_magnitude_pointwise_std_mean": float(
                        np.mean(mc_subnet_mag_std_full[n])
                    ),
                    "subnetwork_vector_rms_over_samples_and_points": float(
                        math.sqrt(
                            np.mean(
                                # E_s[f_x^2 + f_y^2] reconstructed from
                                # pointwise residual moments is equivalent to
                                # RMS over all points and stochastic passes.
                                (
                                    (
                                        mc_subnet_mag_std_full[n] ** 2
                                        + mc_subnet_mag_mean_full[n] ** 2
                                    )
                                )
                            )
                        )
                    ),
                },
            )
        )

    summary = pd.DataFrame(rows)

    timing_rows = [
        {
            "mode": "deterministic_dropout_off",
            "mc_samples": 0,
            "elapsed_seconds": deterministic_time,
            "seconds_per_stochastic_pass_equivalent": np.nan,
        },
        {
            "mode": "single_stochastic",
            "mc_samples": 1,
            "elapsed_seconds": float(stochastic_pass_times[0]),
            "seconds_per_stochastic_pass_equivalent": float(
                stochastic_pass_times[0]
            ),
        },
    ]
    for n in mc_counts:
        timing_rows.append(
            {
                "mode": f"mc{n}",
                "mc_samples": n,
                "elapsed_seconds": float(cumulative_times[n - 1]),
                "seconds_per_stochastic_pass_equivalent": float(
                    cumulative_times[n - 1] / n
                ),
            }
        )

    timing = pd.DataFrame(timing_rows)
    return summary, timing, pointwise


def make_plot(summary: pd.DataFrame, output_path: Path) -> None:
    pred = summary[
        summary["definition"] == "residual_of_mc_predictive_mean"
    ].sort_values("mc_samples")

    subnet = summary[
        summary["definition"] == "mean_residual_field_across_stochastic_subnetworks"
    ].sort_values("mc_samples")

    det = summary[summary["mode"] == "deterministic_dropout_off"]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    if not pred.empty:
        ax.plot(
            pred["mc_samples"],
            pred["physics_vector_rmse"],
            marker="o",
            label="Residual of MC predictive mean",
        )

    if not subnet.empty:
        ax.plot(
            subnet["mc_samples"],
            subnet["physics_vector_rmse"],
            marker="s",
            label="Mean stochastic-subnetwork residual field",
        )

    if not det.empty:
        det_value = float(det.iloc[0]["physics_vector_rmse"])
        ax.axhline(
            det_value,
            linestyle="--",
            label="Deterministic dropout-OFF",
        )

    ax.set_xlabel("MC sample count")
    ax.set_ylabel("Independent momentum-residual vector RMSE")
    ax.set_title("Cylinder MC-count physical-residual convergence")
    ax.set_xticks(pred["mc_samples"].to_numpy(dtype=int) if not pred.empty else [])
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_metadata(
    path: Path,
    args: argparse.Namespace,
    physics_path: Path,
    checkpoint: Path,
    device: torch.device,
    points: pd.DataFrame,
    model: PlacementPINNNet,
) -> None:
    parameter_count = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    lines = [
        "Cylinder MC-count physical-residual evaluation",
        "==============================================",
        f"checkpoint={checkpoint.resolve()}",
        f"checkpoint_sha256={sha256_file(checkpoint)}",
        f"physics_points_csv={physics_path.resolve()}",
        f"points_used={len(points)}",
        f"reynolds={REYNOLDS}",
        f"layer_mat={LAYER_MAT_PSI}",
        f"trainable_parameters={parameter_count}",
        f"dropout_rate={args.dropout_rate}",
        f"dropout_placement={args.dropout_placement}",
        f"dropout_hidden_layer_indices={getattr(model, 'dropout_hidden_layer_indices', None)}",
        f"mc_counts={sorted(set(args.mc_counts))}",
        f"seed={args.seed}",
        f"batch_size={args.batch_size}",
        f"device={device}",
        "",
        "Residual definition:",
        "continuity = u_x + v_y",
        "fx = u_t + u*u_x + v*u_y + p_x - (1/Re)*(u_xx + u_yy)",
        "fy = v_t + u*v_x + v*v_y + p_y - (1/Re)*(v_xx + v_yy)",
        "",
        "MC counts are nested within one fixed stochastic sequence.",
        "The checkpoint is never retrained or modified.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)

    device = get_device(args.device)
    physics_path = resolve_physics_path(args)

    print(f"Device: {device}")
    print(f"Independent physics points: {physics_path}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dropout: p={args.dropout_rate}, placement={args.dropout_placement}")
    print(f"MC counts: {sorted(set(args.mc_counts))}")

    points = load_physics_points(
        physics_path,
        n_points=args.n_physics_points,
        seed=args.seed,
    )
    print(f"Physics points used: {len(points):,}")

    model = load_model(
        checkpoint=args.checkpoint,
        device=device,
        dropout_rate=args.dropout_rate,
        dropout_placement=args.dropout_placement,
    )
    print(
        "Trainable parameters: "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )
    print(
        "Dropout hidden-layer indices: "
        f"{getattr(model, 'dropout_hidden_layer_indices', None)}"
    )

    summary, timing, pointwise = evaluate(
        model=model,
        points=points,
        device=device,
        mc_counts=args.mc_counts,
        batch_size=args.batch_size,
        seed=args.seed,
        reynolds=REYNOLDS,
        do_warmup=not args.no_warmup,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = args.output_dir / "mc_count_physics_residual_summary.csv"
    timing_path = args.output_dir / "mc_count_physics_residual_timing.csv"
    pointwise_path = args.output_dir / "mc_count_physics_residual_pointwise.csv"
    plot_path = args.output_dir / "mc_count_physics_residual_convergence.png"
    metadata_path = args.output_dir / "mc_count_physics_residual_metadata.txt"

    summary.to_csv(summary_path, index=False)
    timing.to_csv(timing_path, index=False)
    pointwise.to_csv(pointwise_path, index=False)
    make_plot(summary, plot_path)
    write_metadata(
        metadata_path,
        args,
        physics_path,
        args.checkpoint,
        device,
        points,
        model,
    )

    display_cols = [
        "mode",
        "definition",
        "mc_samples",
        "fx_rmse",
        "fy_rmse",
        "physics_vector_rmse",
        "physics_magnitude_mean",
        "physics_magnitude_p95",
        "continuity_rmse",
        "elapsed_seconds",
    ]

    print("\n=== Cylinder MC-count physical-residual summary ===")
    print(summary[display_cols].to_string(index=False))

    print("\n=== Timing summary ===")
    print(timing.to_string(index=False))

    print("\nSaved:")
    for path in (
        summary_path,
        timing_path,
        pointwise_path,
        plot_path,
        metadata_path,
    ):
        print(f"  {path.resolve()}")


if __name__ == "__main__":
    main()
