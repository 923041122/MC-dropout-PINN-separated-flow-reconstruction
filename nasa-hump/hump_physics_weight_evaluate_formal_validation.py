"""Validation-only formal evaluator for NASA hump physics-loss-weight selection.

Purpose
-------
Compare the three formal Standard-PINN candidates

    lambda_f = 0, 1e-4, 1e-3

without reading held-out-test labels.

Accuracy is evaluated exclusively on the fixed validation subset stored by the
training protocol in each run's:

    protocol/interior_velocity_split_manifest.csv

Physics consistency is evaluated on one common, independently sampled,
geometry-aware set of fluid-domain points.

This script deliberately never reads ``heldout_evaluation`` outputs and never
uses rows labelled as held-out/test for hyperparameter selection.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch

from benchmark_tools import build_model, get_device, safe_load_state
from hump_protocol import sample_fluid_collocation_points
from hump_train import LAYER_MAT_PSI, read_les_meanfield_tec


WEIGHTS = ["0", "1e-4", "1e-3"]
DEFAULT_MANIFEST_RELPATH = Path("protocol") / "interior_velocity_split_manifest.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Validation-only accuracy/physics trade-off evaluation for the three "
            "formal NASA hump physics-loss-weight candidates."
        )
    )
    p.add_argument("--data-dir", default=".")
    p.add_argument("--root", default=".")
    p.add_argument("--run-prefix", default="physics_weight_formal_")
    p.add_argument(
        "--output-dir",
        default="./physics_weight_formal_validation_tradeoff",
    )
    p.add_argument(
        "--manifest-relpath",
        default=str(DEFAULT_MANIFEST_RELPATH),
        help=(
            "Manifest path relative to each candidate run root. "
            "Default: protocol/interior_velocity_split_manifest.csv"
        ),
    )
    p.add_argument(
        "--validation-label",
        default="validation",
        help="Split label to evaluate. Default: validation",
    )
    p.add_argument(
        "--expected-validation-points",
        type=int,
        default=672,
        help=(
            "Fail if the validation subset does not have this many points. "
            "Use 0 to disable the count check."
        ),
    )
    p.add_argument(
        "--n-points",
        type=int,
        default=20000,
        help="Number of common geometry-aware physics evaluation points.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=9091,
        help="Seed used only for the independent physics-evaluation points.",
    )
    p.add_argument("--batch-size", type=int, default=1000)
    p.add_argument("--reynolds", type=float, default=935892.0)
    return p.parse_args()


def _normalize_name(name: str) -> str:
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def _resolve_column(
    df: pd.DataFrame,
    aliases: Iterable[str],
    *,
    required: bool = True,
) -> str | None:
    normalized = {_normalize_name(col): col for col in df.columns}
    for alias in aliases:
        key = _normalize_name(alias)
        if key in normalized:
            return normalized[key]
    if required:
        raise KeyError(
            f"Could not find any of columns {list(aliases)} in manifest. "
            f"Available columns: {list(df.columns)}"
        )
    return None


def _canonicalize_validation_rows(
    df: pd.DataFrame,
    *,
    validation_label: str,
    expected_points: int,
    source_path: Path,
) -> pd.DataFrame:
    split_col = _resolve_column(df, ["split", "subset", "role", "partition"])
    x_col = _resolve_column(df, ["x", "x_coord", "x_coordinate"])
    y_col = _resolve_column(df, ["y", "y_coord", "y_coordinate"])
    u_col = _resolve_column(df, ["u", "u_ref", "u_true", "u_reference"])
    v_col = _resolve_column(df, ["v", "v_ref", "v_true", "v_reference"])
    index_col = _resolve_column(
        df,
        ["original_index", "point_index", "global_index", "index"],
        required=False,
    )

    split_norm = (
        df[split_col]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace("-", "_", regex=False)
        .str.replace(" ", "_", regex=False)
    )
    target = _normalize_name(validation_label)

    validation_mask = split_norm == target
    validation = df.loc[validation_mask].copy()

    if validation.empty:
        available = sorted(split_norm.dropna().unique().tolist())
        raise ValueError(
            f"No rows labelled {validation_label!r} in {source_path}. "
            f"Available split labels: {available}"
        )

    out = pd.DataFrame(
        {
            "x": pd.to_numeric(validation[x_col], errors="coerce"),
            "y": pd.to_numeric(validation[y_col], errors="coerce"),
            "u_ref": pd.to_numeric(validation[u_col], errors="coerce"),
            "v_ref": pd.to_numeric(validation[v_col], errors="coerce"),
        }
    )

    if index_col is not None:
        out.insert(
            0,
            "original_index",
            pd.to_numeric(validation[index_col], errors="coerce"),
        )

    before = len(out)
    out = out.replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(out) != before:
        raise ValueError(
            f"{source_path}: validation subset contains non-finite x/y/u/v "
            f"values ({before - len(out)} rows would be dropped). Formal "
            "selection aborts rather than silently changing the validation set."
        )

    if expected_points > 0 and len(out) != expected_points:
        raise ValueError(
            f"{source_path}: expected {expected_points} validation points, "
            f"found {len(out)}."
        )

    if "original_index" in out.columns:
        if out["original_index"].duplicated().any():
            raise ValueError(
                f"{source_path}: duplicate original_index values in validation subset."
            )
        out = out.sort_values("original_index").reset_index(drop=True)
    else:
        if out.duplicated(subset=["x", "y"]).any():
            raise ValueError(
                f"{source_path}: duplicate (x, y) coordinates in validation subset "
                "and no original_index is available for unambiguous alignment."
            )
        out = out.sort_values(["x", "y"]).reset_index(drop=True)

    return out


def read_validation_manifest(
    path: Path,
    *,
    validation_label: str,
    expected_points: int,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing protocol manifest: {path}\n"
            "The formal evaluator requires the exact training-stage manifest and "
            "will not reconstruct or guess the split."
        )

    df = pd.read_csv(path)
    return _canonicalize_validation_rows(
        df,
        validation_label=validation_label,
        expected_points=expected_points,
        source_path=path,
    )


def assert_same_validation_protocol(
    candidate_tables: List[Tuple[str, Path, pd.DataFrame]]
) -> pd.DataFrame:
    if not candidate_tables:
        raise ValueError("No candidate validation manifests were loaded.")

    ref_weight, ref_path, ref = candidate_tables[0]
    compare_cols = ["x", "y", "u_ref", "v_ref"]

    if "original_index" in ref.columns:
        compare_cols = ["original_index", *compare_cols]

    for weight, path, table in candidate_tables[1:]:
        if list(table.columns) != list(ref.columns):
            raise ValueError(
                "Candidate manifests do not expose the same canonical columns:\n"
                f"  reference lambda_f={ref_weight}: {list(ref.columns)}\n"
                f"  lambda_f={weight}: {list(table.columns)}"
            )

        if len(table) != len(ref):
            raise ValueError(
                "Candidate validation subsets differ in size:\n"
                f"  reference {ref_path}: {len(ref)}\n"
                f"  {path}: {len(table)}"
            )

        for col in compare_cols:
            a = ref[col].to_numpy()
            b = table[col].to_numpy()

            if col == "original_index":
                same = np.array_equal(a, b)
            else:
                same = np.allclose(
                    a.astype(float),
                    b.astype(float),
                    rtol=0.0,
                    atol=1e-12,
                    equal_nan=False,
                )

            if not same:
                raise ValueError(
                    "Formal candidates do not share an identical validation "
                    f"protocol. First mismatch detected in column {col!r} for "
                    f"lambda_f={weight}.\n"
                    f"Reference manifest: {ref_path}\n"
                    f"Candidate manifest: {path}"
                )

    return ref.copy()


def residual_batch(
    model: torch.nn.Module,
    xy: np.ndarray,
    device: torch.device,
    reynolds: float,
) -> Tuple[np.ndarray, np.ndarray]:
    x = torch.tensor(
        xy[:, 0:1],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    y = torch.tensor(
        xy[:, 1:2],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )

    model.eval()

    with torch.enable_grad():
        out = model.forward(x, y)
        psi = out[:, 0:1]
        p = out[:, 1:2]

        u = torch.autograd.grad(
            psi.sum(), y, create_graph=True, retain_graph=True
        )[0]
        v = -torch.autograd.grad(
            psi.sum(), x, create_graph=True, retain_graph=True
        )[0]

        u_x = torch.autograd.grad(
            u.sum(), x, create_graph=True, retain_graph=True
        )[0]
        u_y = torch.autograd.grad(
            u.sum(), y, create_graph=True, retain_graph=True
        )[0]
        v_x = torch.autograd.grad(
            v.sum(), x, create_graph=True, retain_graph=True
        )[0]
        v_y = torch.autograd.grad(
            v.sum(), y, create_graph=True, retain_graph=True
        )[0]
        p_x = torch.autograd.grad(
            p.sum(), x, create_graph=True, retain_graph=True
        )[0]
        p_y = torch.autograd.grad(
            p.sum(), y, create_graph=True, retain_graph=True
        )[0]

        u_xx = torch.autograd.grad(
            u_x.sum(), x, create_graph=False, retain_graph=True
        )[0]
        u_yy = torch.autograd.grad(
            u_y.sum(), y, create_graph=False, retain_graph=True
        )[0]
        v_xx = torch.autograd.grad(
            v_x.sum(), x, create_graph=False, retain_graph=True
        )[0]
        v_yy = torch.autograd.grad(
            v_y.sum(), y, create_graph=False, retain_graph=False
        )[0]

        fx = (
            u * u_x
            + v * u_y
            + p_x
            - (u_xx + u_yy) / float(reynolds)
        )
        fy = (
            u * v_x
            + v * v_y
            + p_y
            - (v_xx + v_yy) / float(reynolds)
        )

    return (
        fx.detach().cpu().numpy().ravel(),
        fy.detach().cpu().numpy().ravel(),
    )


def evaluate_physics(
    model: torch.nn.Module,
    points: np.ndarray,
    device: torch.device,
    reynolds: float,
    batch_size: int,
) -> Dict[str, float]:
    fx_all: List[np.ndarray] = []
    fy_all: List[np.ndarray] = []

    for start in range(0, len(points), batch_size):
        stop = min(start + batch_size, len(points))
        fx, fy = residual_batch(
            model,
            points[start:stop],
            device,
            reynolds,
        )
        fx_all.append(fx)
        fy_all.append(fy)

    fx = np.concatenate(fx_all)
    fy = np.concatenate(fy_all)
    mag = np.sqrt(fx**2 + fy**2)

    return {
        "physics_fx_mae": float(np.mean(np.abs(fx))),
        "physics_fy_mae": float(np.mean(np.abs(fy))),
        "physics_fx_rmse": float(np.sqrt(np.mean(fx**2))),
        "physics_fy_rmse": float(np.sqrt(np.mean(fy**2))),
        "physics_vector_rmse": float(
            np.sqrt(np.mean(fx**2 + fy**2))
        ),
        "physics_magnitude_mean": float(np.mean(mag)),
        "physics_magnitude_median": float(np.median(mag)),
        "physics_magnitude_p95": float(np.quantile(mag, 0.95)),
        "physics_magnitude_max": float(np.max(mag)),
        "physics_evaluation_points": int(len(points)),
    }


def predict_velocity_batch(
    model: torch.nn.Module,
    xy: np.ndarray,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    x = torch.tensor(
        xy[:, 0:1],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    y = torch.tensor(
        xy[:, 1:2],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )

    model.eval()

    with torch.enable_grad():
        out = model.forward(x, y)
        psi = out[:, 0:1]

        u = torch.autograd.grad(
            psi.sum(),
            y,
            create_graph=False,
            retain_graph=True,
        )[0]
        v = -torch.autograd.grad(
            psi.sum(),
            x,
            create_graph=False,
            retain_graph=False,
        )[0]

    return (
        u.detach().cpu().numpy().ravel(),
        v.detach().cpu().numpy().ravel(),
    )


def metric_dict(
    pred: np.ndarray,
    truth: np.ndarray,
    prefix: str,
) -> Dict[str, float]:
    pred = np.asarray(pred, dtype=float).reshape(-1)
    truth = np.asarray(truth, dtype=float).reshape(-1)

    if pred.shape != truth.shape:
        raise ValueError(
            f"{prefix}: prediction/reference shape mismatch "
            f"{pred.shape} vs {truth.shape}."
        )

    error = pred - truth
    denominator = np.linalg.norm(truth)

    return {
        f"{prefix}_MAE": float(np.mean(np.abs(error))),
        f"{prefix}_RMSE": float(np.sqrt(np.mean(error**2))),
        f"{prefix}_relative_L2": float(
            np.linalg.norm(error) / denominator
            if denominator > 0
            else np.nan
        ),
        f"{prefix}_max_abs_error": float(np.max(np.abs(error))),
    }


def evaluate_validation_accuracy(
    model: torch.nn.Module,
    validation: pd.DataFrame,
    device: torch.device,
    batch_size: int,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    xy = validation[["x", "y"]].to_numpy(dtype=np.float32)

    u_chunks: List[np.ndarray] = []
    v_chunks: List[np.ndarray] = []

    for start in range(0, len(validation), batch_size):
        stop = min(start + batch_size, len(validation))
        u_pred, v_pred = predict_velocity_batch(
            model,
            xy[start:stop],
            device,
        )
        u_chunks.append(u_pred)
        v_chunks.append(v_pred)

    u_pred = np.concatenate(u_chunks)
    v_pred = np.concatenate(v_chunks)
    u_ref = validation["u_ref"].to_numpy(dtype=float)
    v_ref = validation["v_ref"].to_numpy(dtype=float)

    du = u_pred - u_ref
    dv = v_pred - v_ref

    metrics = {
        **metric_dict(u_pred, u_ref, "u"),
        **metric_dict(v_pred, v_ref, "v"),
        "uv_vector_RMSE": float(
            np.sqrt(np.mean(du**2 + dv**2))
        ),
        "uv_mean_component_RMSE": float(
            0.5
            * (
                np.sqrt(np.mean(du**2))
                + np.sqrt(np.mean(dv**2))
            )
        ),
        "validation_points": int(len(validation)),
    }

    pred_table = validation.copy()
    pred_table["u_pred"] = u_pred
    pred_table["v_pred"] = v_pred
    pred_table["u_abs_error"] = np.abs(du)
    pred_table["v_abs_error"] = np.abs(dv)

    return metrics, pred_table


def read_training_summary(run_root: Path) -> Dict[str, float]:
    path = (
        run_root
        / "training_logs"
        / "standard_pinn_training_summary.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing training summary: {path}"
        )

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Training summary is empty: {path}")

    row = df.iloc[-1]
    result: Dict[str, float] = {}

    mapping = {
        "final_total_loss": "final_total_loss",
        "final_uv_loss": "final_data_loss",
        "final_equation_loss": "training_final_physics_loss",
        "final_cp_loss": "final_cp_loss",
        "final_wall_loss": "final_wall_loss",
        "final_subbc_loss": "final_subbc_loss",
        "training_time_seconds": "training_time_seconds",
        "total_training_time_seconds": "training_time_seconds",
    }

    for src, dst in mapping.items():
        if src in row.index and pd.notna(row[src]):
            result[dst] = float(row[src])

    return result


def load_model(
    checkpoint: Path,
    device: torch.device,
) -> torch.nn.Module:
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {checkpoint}"
        )

    model = build_model(
        {"model_type": "psi", "dropout_rate": 0.0},
        LAYER_MAT_PSI,
    ).to(device)

    state = safe_load_state(checkpoint, device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def safe_weight_slug(weight: str) -> str:
    return (
        str(weight)
        .replace("+", "")
        .replace("-", "m")
        .replace(".", "p")
    )


def main() -> None:
    args = parse_args()

    if args.n_points <= 0:
        raise ValueError("--n-points must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.expected_validation_points < 0:
        raise ValueError(
            "--expected-validation-points must be >= 0."
        )

    root = Path(args.root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    run_roots = {
        weight: root / f"{args.run_prefix}{weight}"
        for weight in WEIGHTS
    }

    # ------------------------------------------------------------------
    # 1. Load and cross-check the exact training-stage validation manifest
    #    for every formal candidate. This prevents split reconstruction and
    #    ensures all candidates are compared on the same validation points.
    # ------------------------------------------------------------------
    candidate_tables: List[Tuple[str, Path, pd.DataFrame]] = []
    protocol_rows: List[Dict[str, object]] = []

    for weight in WEIGHTS:
        run_root = run_roots[weight]
        manifest_path = run_root / Path(args.manifest_relpath)
        validation = read_validation_manifest(
            manifest_path,
            validation_label=args.validation_label,
            expected_points=args.expected_validation_points,
        )
        candidate_tables.append(
            (weight, manifest_path, validation)
        )
        protocol_rows.append(
            {
                "equation_loss_weight": weight,
                "run_root": str(run_root),
                "manifest_path": str(manifest_path),
                "validation_points": int(len(validation)),
            }
        )

    validation = assert_same_validation_protocol(
        candidate_tables
    )

    validation.to_csv(
        output / "validation_points_used.csv",
        index=False,
    )

    pd.DataFrame(protocol_rows).to_csv(
        output / "protocol_consistency_check.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 2. Build one common independent geometry-aware physics grid.
    # ------------------------------------------------------------------
    data_path = (
        Path(args.data_dir)
        / "LES_meanfield_nasahump2009_tec.dat"
    )
    if not data_path.exists():
        raise FileNotFoundError(
            f"Missing NASA hump mean-field file: {data_path}"
        )

    meanfield = read_les_meanfield_tec(data_path)
    physics_points = sample_fluid_collocation_points(
        meanfield,
        n_points=args.n_points,
        seed=args.seed,
    )
    physics_points = np.asarray(
        physics_points,
        dtype=float,
    )

    if physics_points.ndim != 2 or physics_points.shape[1] < 2:
        raise ValueError(
            "sample_fluid_collocation_points() must return an "
            f"(N, >=2) array, got {physics_points.shape}."
        )

    physics_xy = physics_points[:, :2]
    pd.DataFrame(
        physics_xy,
        columns=["x", "y"],
    ).to_csv(
        output
        / "independent_physics_evaluation_points.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 3. Evaluate all formal candidates.
    # ------------------------------------------------------------------
    device = get_device()
    rows: List[Dict[str, object]] = []

    print(f"Device: {device}")
    print(
        f"Validation-only accuracy evaluation: "
        f"{len(validation)} fixed points"
    )
    print(
        f"Independent geometry-aware physics evaluation: "
        f"{len(physics_xy)} common points"
    )
    print(
        "Held-out test labels are not read by this script."
    )

    for weight in WEIGHTS:
        run_root = run_roots[weight]
        checkpoint = (
            run_root
            / "models"
            / "standard_pinn.pth"
        )

        print(
            f"\nEvaluating formal lambda_f={weight}..."
        )

        model = load_model(
            checkpoint=checkpoint,
            device=device,
        )

        validation_metrics, pred_table = (
            evaluate_validation_accuracy(
                model=model,
                validation=validation,
                device=device,
                batch_size=args.batch_size,
            )
        )

        pred_table.insert(
            0,
            "equation_loss_weight",
            weight,
        )
        pred_table.to_csv(
            output
            / (
                "validation_predictions_lambda_"
                f"{safe_weight_slug(weight)}.csv"
            ),
            index=False,
        )

        physics_metrics = evaluate_physics(
            model=model,
            points=physics_xy,
            device=device,
            reynolds=args.reynolds,
            batch_size=args.batch_size,
        )

        row: Dict[str, object] = {
            "equation_loss_weight": weight,
            "equation_loss_weight_float": float(weight),
            "selection_split": args.validation_label,
            "heldout_test_used_for_selection": False,
            **validation_metrics,
            **physics_metrics,
            **read_training_summary(run_root),
        }
        rows.append(row)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = pd.DataFrame(rows)
    summary_path = (
        output
        / "physics_weight_formal_validation_tradeoff_summary.csv"
    )
    summary.to_csv(summary_path, index=False)

    display_cols = [
        "equation_loss_weight",
        "u_RMSE",
        "v_RMSE",
        "u_relative_L2",
        "v_relative_L2",
        "uv_vector_RMSE",
        "physics_vector_rmse",
        "physics_magnitude_mean",
        "physics_magnitude_p95",
    ]

    print(
        "\n=== Formal validation-only accuracy-physics trade-off ==="
    )
    print(
        summary[display_cols].to_string(index=False)
    )

    print("\nSaved:")
    for path in [
        output / "validation_points_used.csv",
        output / "protocol_consistency_check.csv",
        output / "independent_physics_evaluation_points.csv",
        summary_path,
    ]:
        print(path.resolve())

    print(
        "\nSelection rule reminder: choose/freeze lambda_f using "
        "validation accuracy plus the common independent physics residual. "
        "Do not use the held-out test for hyperparameter selection."
    )


if __name__ == "__main__":
    main()
