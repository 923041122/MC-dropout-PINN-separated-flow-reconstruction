
"""Traceable sampling and boundary protocol for the revised NASA hump benchmark.

Key design choices
------------------
1. The steady model uses spatial coordinates (x,y) only.
2. The bottom structured-grid row is treated as the physical hump wall and is
   assigned a no-slip constraint u=v=0.
3. The left, right, and top edges of the released LES mean-field window are NOT
   called the physical inlet, outlet, or upper wall of the full NASA benchmark.
   They are truncated-reconstruction-subdomain interfaces. When enabled, their
   LES u,v values are used as explicit subdomain-boundary observations.
4. Train/validation/test splitting is performed only on strict interior LES
   points, excluding all four edge sets.
5. The sparse supervised ratio is converted to a target count using the full
   4761-point LES mean-field table, so a 2% run still uses 95 interior supervised
   velocity points. Boundary observations are reported separately.
6. Physics collocation points are sampled above the physical hump wall and
   inside the local released LES reference window.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def clean_meanfield(meanfield: Dict[str, object]) -> pd.DataFrame:
    flat = meanfield["flat"].copy()
    required = ["x", "y", "u", "v"]
    missing = [c for c in required if c not in flat.columns]
    if missing:
        raise KeyError(f"meanfield['flat'] is missing required columns: {missing}")

    n_i = int(meanfield["I"])
    n_j = int(meanfield["J"])
    expected = n_i * n_j
    if len(flat) != expected:
        raise ValueError(f"Expected I*J={expected} rows, found {len(flat)}.")

    if "original_index" not in flat.columns:
        flat.insert(0, "original_index", np.arange(len(flat), dtype=int))
    if "j_index" not in flat.columns:
        flat["j_index"] = np.repeat(np.arange(n_j, dtype=int), n_i)
    if "i_index" not in flat.columns:
        flat["i_index"] = np.tile(np.arange(n_i, dtype=int), n_j)

    cols = ["original_index", "i_index", "j_index", "x", "y", "u", "v"]
    return (
        flat[cols]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .reset_index(drop=True)
    )


def build_boundary_sets(meanfield: Dict[str, object]) -> Dict[str, pd.DataFrame]:
    """Extract disjoint wall/left/right/top boundary sets and strict interior."""
    full = clean_meanfield(meanfield)
    n_i = int(meanfield["I"])
    n_j = int(meanfield["J"])

    wall = full[full["j_index"] == 0].copy()
    wall["boundary_name"] = "hump_wall"
    wall["boundary_role"] = "physical_wall"
    wall["bc_type"] = "no_slip"
    wall["u_target"] = 0.0
    wall["v_target"] = 0.0

    # Exclude wall and top corners from left/right so every boundary node is unique.
    left = full[
        (full["i_index"] == 0)
        & (full["j_index"] > 0)
        & (full["j_index"] < n_j - 1)
    ].copy()
    left["boundary_name"] = "left_subdomain"
    left["boundary_role"] = "truncated_subdomain_interface"
    left["bc_type"] = "LES_uv_observation"
    left["u_target"] = left["u"]
    left["v_target"] = left["v"]

    right = full[
        (full["i_index"] == n_i - 1)
        & (full["j_index"] > 0)
        & (full["j_index"] < n_j - 1)
    ].copy()
    right["boundary_name"] = "right_subdomain"
    right["boundary_role"] = "truncated_subdomain_interface"
    right["bc_type"] = "LES_uv_observation"
    right["u_target"] = right["u"]
    right["v_target"] = right["v"]

    top = full[full["j_index"] == n_j - 1].copy()
    top["boundary_name"] = "top_subdomain"
    top["boundary_role"] = "truncated_subdomain_interface"
    top["bc_type"] = "LES_uv_observation"
    top["u_target"] = top["u"]
    top["v_target"] = top["v"]

    interior = full[
        (full["i_index"] > 0)
        & (full["i_index"] < n_i - 1)
        & (full["j_index"] > 0)
        & (full["j_index"] < n_j - 1)
    ].copy()

    subbc = pd.concat([left, right, top], ignore_index=True)

    # Defensive uniqueness check.
    boundary_indices = pd.concat(
        [
            wall[["original_index"]],
            left[["original_index"]],
            right[["original_index"]],
            top[["original_index"]],
        ],
        ignore_index=True,
    )["original_index"]
    if boundary_indices.duplicated().any():
        raise RuntimeError("Boundary extraction produced duplicate original indices.")

    if set(boundary_indices).intersection(set(interior["original_index"])):
        raise RuntimeError("Interior and boundary sets overlap.")

    return {
        "full": full.reset_index(drop=True),
        "wall": wall.reset_index(drop=True),
        "left": left.reset_index(drop=True),
        "right": right.reset_index(drop=True),
        "top": top.reset_index(drop=True),
        "subbc": subbc.reset_index(drop=True),
        "interior": interior.reset_index(drop=True),
    }


def _assign_block_holdout(
    df: pd.DataFrame,
    *,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
    block_width: int,
) -> pd.Series:
    if block_width < 1:
        raise ValueError("block_width must be >= 1.")
    if validation_fraction < 0 or test_fraction < 0:
        raise ValueError("Holdout fractions must be non-negative.")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be < 1.")

    block_id = (df["i_index"].to_numpy(dtype=int) // int(block_width)).astype(int)
    unique_blocks = np.unique(block_id)

    rng = np.random.default_rng(seed)
    shuffled = unique_blocks.copy()
    rng.shuffle(shuffled)

    n_total = len(df)
    test_target = int(np.ceil(test_fraction * n_total))
    val_target = int(np.ceil(validation_fraction * n_total))

    cursor = 0
    test_blocks = []
    test_count = 0
    while cursor < len(shuffled) and test_count < test_target:
        b = int(shuffled[cursor])
        test_blocks.append(b)
        test_count += int(np.sum(block_id == b))
        cursor += 1

    val_blocks = []
    val_count = 0
    while cursor < len(shuffled) and val_count < val_target:
        b = int(shuffled[cursor])
        val_blocks.append(b)
        val_count += int(np.sum(block_id == b))
        cursor += 1

    split = np.full(n_total, "train_pool", dtype=object)
    split[np.isin(block_id, test_blocks)] = "test"
    split[np.isin(block_id, val_blocks)] = "validation"
    return pd.Series(split, index=df.index, dtype=object)


def _assign_random_holdout(
    df: pd.DataFrame,
    *,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> pd.Series:
    if validation_fraction < 0 or test_fraction < 0:
        raise ValueError("Holdout fractions must be non-negative.")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be < 1.")

    n = len(df)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_test = int(round(test_fraction * n))
    n_val = int(round(validation_fraction * n))

    split = np.full(n, "train_pool", dtype=object)
    split[order[:n_test]] = "test"
    split[order[n_test:n_test + n_val]] = "validation"
    return pd.Series(split, index=df.index, dtype=object)


def build_interior_velocity_split(
    meanfield: Dict[str, object],
    *,
    supervised_ratio: float,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
    split_mode: str = "spatial_blocks",
    block_width: int = 8,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Split strict interior points and select a nested sparse training subset."""
    if not (0.0 < supervised_ratio <= 1.0):
        raise ValueError("supervised_ratio must be in (0,1].")

    boundaries = build_boundary_sets(meanfield)
    interior = boundaries["interior"].copy()

    if split_mode == "spatial_blocks":
        split = _assign_block_holdout(
            interior,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            seed=seed,
            block_width=block_width,
        )
    elif split_mode == "random":
        split = _assign_random_holdout(
            interior,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown split_mode={split_mode!r}")

    interior["split"] = split
    interior["sampling_rank"] = np.nan

    pool_idx = interior.index[interior["split"] == "train_pool"].to_numpy()
    rng = np.random.default_rng(seed + 104729)
    ranked = pool_idx.copy()
    rng.shuffle(ranked)
    interior.loc[ranked, "sampling_rank"] = np.arange(1, len(ranked) + 1)

    # Preserve the reviewer's 1--5% convention relative to the complete LES table.
    n_full = len(boundaries["full"])
    n_target = max(1, int(round(supervised_ratio * n_full)))

    if n_target > len(ranked):
        raise ValueError(
            f"supervised_ratio={supervised_ratio} implies {n_target} training points, "
            f"but only {len(ranked)} strict-interior train-pool points are available."
        )

    chosen = ranked[:n_target]
    unused = ranked[n_target:]
    interior.loc[chosen, "split"] = "train"
    interior.loc[unused, "split"] = "train_pool_unused"

    train_df = interior[interior["split"] == "train"].sort_values("sampling_rank")
    val_df = interior[interior["split"] == "validation"].sort_values("original_index")
    test_df = interior[interior["split"] == "test"].sort_values("original_index")

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
        interior.sort_values("original_index").reset_index(drop=True),
        boundaries,
    )


def _split_cp_holdout(
    full: pd.DataFrame,
    *,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
    block_width: int,
) -> pd.Series:
    temp = full.copy()
    temp["i_index"] = np.arange(len(temp), dtype=int)
    return _assign_block_holdout(
        temp,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=seed,
        block_width=max(1, int(block_width)),
    )


def build_cp_split(
    cp_df: pd.DataFrame,
    *,
    supervised_ratio: float,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
    split_mode: str = "spatial_blocks",
    block_width: int = 12,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if cp_df.empty:
        empty = cp_df.copy()
        empty["split"] = pd.Series(dtype=object)
        empty["sampling_rank"] = pd.Series(dtype=float)
        return empty.copy(), empty.copy(), empty.copy(), empty.copy()

    full = (
        cp_df.copy()
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["x", "cp"])
        .sort_values(["source", "x"] if "source" in cp_df.columns else ["x"])
        .reset_index(drop=True)
    )
    full.insert(0, "original_index", np.arange(len(full), dtype=int))

    if split_mode == "spatial_blocks":
        split = _split_cp_holdout(
            full,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            seed=seed + 17,
            block_width=block_width,
        )
    elif split_mode == "random":
        split = _assign_random_holdout(
            full,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            seed=seed + 17,
        )
    else:
        raise ValueError(f"Unknown split_mode={split_mode!r}")

    full["split"] = split
    full["sampling_rank"] = np.nan
    pool_idx = full.index[full["split"] == "train_pool"].to_numpy()

    rng = np.random.default_rng(seed + 130363)
    ranked = pool_idx.copy()
    rng.shuffle(ranked)
    full.loc[ranked, "sampling_rank"] = np.arange(1, len(ranked) + 1)

    n_target = max(1, int(round(supervised_ratio * len(full))))
    n_target = min(n_target, len(ranked))
    full.loc[ranked[:n_target], "split"] = "train"
    full.loc[ranked[n_target:], "split"] = "train_pool_unused"

    return (
        full[full["split"] == "train"].sort_values("sampling_rank").reset_index(drop=True),
        full[full["split"] == "validation"].sort_values("original_index").reset_index(drop=True),
        full[full["split"] == "test"].sort_values("original_index").reset_index(drop=True),
        full.sort_values("original_index").reset_index(drop=True),
    )


def build_local_window_interpolators(meanfield: Dict[str, object]):
    arrays = meanfield["arrays"]

    xb = np.asarray(arrays["x"])[0, :].astype(float)
    yb = np.asarray(arrays["y"])[0, :].astype(float)
    xt = np.asarray(arrays["x"])[-1, :].astype(float)
    yt = np.asarray(arrays["y"])[-1, :].astype(float)

    ob = np.argsort(xb)
    ot = np.argsort(xt)
    xb, yb = xb[ob], yb[ob]
    xt, yt = xt[ot], yt[ot]

    x_min = max(float(xb.min()), float(xt.min()))
    x_max = min(float(xb.max()), float(xt.max()))
    if not x_min < x_max:
        raise ValueError("Bottom/top lines do not share a valid x-range.")

    def y_wall(x):
        arr = np.asarray(x, dtype=float).reshape(-1)
        return np.interp(arr, xb, yb).reshape(-1, 1)

    def y_local_top(x):
        arr = np.asarray(x, dtype=float).reshape(-1)
        return np.interp(arr, xt, yt).reshape(-1, 1)

    return y_wall, y_local_top, x_min, x_max


def sample_fluid_collocation_points(
    meanfield: Dict[str, object],
    *,
    n_points: int,
    seed: int,
    margin: float = 1e-6,
) -> np.ndarray:
    """Sample above the hump wall inside the released local LES window."""
    if n_points <= 0:
        raise ValueError("n_points must be positive.")

    y_wall, y_top, x_min, x_max = build_local_window_interpolators(meanfield)
    rng = np.random.default_rng(seed)

    x = rng.uniform(x_min, x_max, size=(n_points, 1))
    yl = y_wall(x) + float(margin)
    yu = y_top(x) - float(margin)

    bad = yu <= yl
    if np.any(bad):
        raise ValueError(
            f"{int(np.sum(bad))} sampled x locations have non-positive local window height."
        )

    y = yl + rng.random((n_points, 1)) * (yu - yl)
    return np.concatenate([x, y], axis=1)


def diagnose_legacy_random_box(
    meanfield: Dict[str, object],
    *,
    n_points: int,
    seed: int,
):
    """Audit the superseded global rectangular-box collocation sampler."""
    full = clean_meanfield(meanfield)
    rng = np.random.default_rng(seed)

    x = rng.uniform(float(full["x"].min()), float(full["x"].max()), size=(n_points, 1))
    y = rng.uniform(float(full["y"].min()), float(full["y"].max()), size=(n_points, 1))

    y_wall, y_local_top, _, _ = build_local_window_interpolators(meanfield)
    yl = y_wall(x)
    yu = y_local_top(x)

    below_wall = y < yl
    above_local = y > yu
    outside_local = below_wall | above_local

    table = pd.DataFrame({
        "x": x.reshape(-1),
        "y": y.reshape(-1),
        "y_wall": yl.reshape(-1),
        "y_local_reference_window_top": yu.reshape(-1),
        "is_below_wall_solid": below_wall.reshape(-1),
        "is_above_local_reference_window": above_local.reshape(-1),
        "is_inside_local_reference_window": (~outside_local).reshape(-1),
    })

    outside = table[outside_local.reshape(-1)].copy().reset_index(drop=True)

    summary = {
        "legacy_candidate_points": int(n_points),
        "legacy_below_wall_solid_points": int(below_wall.sum()),
        "legacy_below_wall_solid_fraction": float(below_wall.mean()),
        "legacy_above_local_reference_window_points": int(above_local.sum()),
        "legacy_above_local_reference_window_fraction": float(above_local.mean()),
        "legacy_outside_local_reference_window_points": int(outside_local.sum()),
        "legacy_outside_local_reference_window_fraction": float(outside_local.mean()),
    }
    return table, outside, summary


def save_protocol_artifacts(
    *,
    output_dir: Path,
    meanfield: Dict[str, object],
    interior_manifest: pd.DataFrame,
    cp_manifest: pd.DataFrame,
    boundaries: Dict[str, pd.DataFrame],
    collocation_points: np.ndarray,
    legacy_table: pd.DataFrame,
    legacy_outside: pd.DataFrame,
    summary: Dict[str, object],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    interior_manifest.to_csv(
        output_dir / "interior_velocity_split_manifest.csv", index=False
    )
    # Compatibility alias for scripts written against v1.1.
    interior_manifest.to_csv(
        output_dir / "velocity_split_manifest.csv", index=False
    )
    cp_manifest.to_csv(output_dir / "cp_split_manifest.csv", index=False)

    boundaries["wall"].to_csv(output_dir / "hump_wall_points.csv", index=False)
    boundaries["left"].to_csv(output_dir / "left_subdomain_boundary.csv", index=False)
    boundaries["right"].to_csv(output_dir / "right_subdomain_boundary.csv", index=False)
    boundaries["top"].to_csv(output_dir / "top_subdomain_boundary.csv", index=False)
    boundaries["subbc"].to_csv(output_dir / "subdomain_boundary_points.csv", index=False)

    pd.DataFrame(collocation_points, columns=["x", "y"]).to_csv(
        output_dir / "valid_fluid_collocation_points.csv", index=False
    )

    legacy_table.to_csv(
        output_dir / "legacy_random_box_diagnostic_all_points.csv", index=False
    )
    legacy_outside.to_csv(
        output_dir / "legacy_outside_local_reference_window_points.csv", index=False
    )
    pd.DataFrame([summary]).to_csv(output_dir / "protocol_summary.csv", index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Figure 1: revised protocol.
        fig, ax = plt.subplots(figsize=(9.4, 5.0))

        def _scatter(df, label, nmax=3000, marker="."):
            if df is None or len(df) == 0:
                return
            shown = (
                df.sample(n=nmax, random_state=1)
                if len(df) > nmax else df
            )
            ax.scatter(
                shown["x"], shown["y"],
                s=10, alpha=0.7, marker=marker, label=label
            )

        _scatter(interior_manifest[interior_manifest["split"] == "train"], "interior train")
        _scatter(interior_manifest[interior_manifest["split"] == "validation"], "interior validation", marker="x")
        _scatter(interior_manifest[interior_manifest["split"] == "test"], "interior held-out test", marker="+")
        _scatter(pd.DataFrame(collocation_points, columns=["x", "y"]), "physics collocation", nmax=3500)

        ax.plot(
            boundaries["wall"]["x"], boundaries["wall"]["y"],
            linewidth=2.0, label="physical no-slip hump wall"
        )
        _scatter(boundaries["left"], "left subdomain observations", marker="s")
        _scatter(boundaries["right"], "right subdomain observations", marker="s")
        _scatter(boundaries["top"], "top subdomain observations", marker="s")

        ax.set_xlabel("x/c")
        ax.set_ylabel("y/c")
        ax.set_title("Revised NASA hump reconstruction protocol")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / "protocol_boundary_map.png", dpi=250)
        plt.close(fig)

        # Figure 2: legacy sampler audit.
        fig, ax = plt.subplots(figsize=(9.4, 5.0))
        inside = legacy_table[legacy_table["is_inside_local_reference_window"]]
        below = legacy_table[legacy_table["is_below_wall_solid"]]
        above = legacy_table[legacy_table["is_above_local_reference_window"]]

        for df, label, marker in [
            (inside, "legacy points inside local LES window", "."),
            (below, "legacy below-wall solid points", "x"),
            (above, "legacy above local LES window", "+"),
        ]:
            shown = df.sample(n=min(3500, len(df)), random_state=2) if len(df) else df
            if len(shown):
                ax.scatter(shown["x"], shown["y"], s=10, alpha=0.7, marker=marker, label=label)

        ax.plot(boundaries["wall"]["x"], boundaries["wall"]["y"], linewidth=2.0, label="hump wall")
        ax.plot(boundaries["top"]["x"], boundaries["top"]["y"], linewidth=2.0, label="top of released LES window")
        ax.set_xlabel("x/c")
        ax.set_ylabel("y/c")
        ax.set_title("Audit of superseded rectangular collocation sampler")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / "legacy_sampler_audit.png", dpi=250)
        plt.close(fig)

    except Exception as exc:
        (output_dir / "protocol_plot_warning.txt").write_text(
            f"CSV artifacts were saved, but plotting failed: {exc}\n",
            encoding="utf-8",
        )
