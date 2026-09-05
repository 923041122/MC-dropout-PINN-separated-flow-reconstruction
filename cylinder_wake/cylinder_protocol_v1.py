
"""Reproducible sparse-data protocol v1.1 for the Re=3900 transient cylinder wake.

This script DOES NOT train a neural network.

Purpose
-------
The previous benchmark_train.py used --ratio to set the mini-batch size, while
all reference points were still traversed during each epoch. Therefore --ratio
was not a supervised-data fraction.

This protocol separates:
    1. supervised-data ratio;
    2. batch size;
    3. fixed validation/test partitions;
    4. collocation points;
    5. independent physics-evaluation points.

Protocol
--------
- Original data: structured 100 x 100 spatial grid x 100 times.
- The spatial grid is partitioned into a reproducible 10 x 10 block grid.
- 65 blocks: training pool.
- 15 blocks: validation.
- 20 blocks: held-out test.
- The same spatial partition is applied at every time snapshot, so validation
  and held-out testing cover the complete temporal evolution.
- Sparse supervised sets are nested and expressed relative to ALL 1,000,000
  reference space-time points:
      1% = 10,000
      2% = 20,000
      3% = 30,000
      4% = 40,000
      5% = 50,000
  with D_1% subset D_2% subset ... subset D_5%.
- Physics collocation points are generated independently in the rectangular
  wake reconstruction domain.
- A second independent point set is generated for post-training residual
  evaluation; it is not used in optimization.

Outputs
-------
<results-root>/protocol/
    protocol_summary.csv
    spatial_split_manifest.csv
    nested_supervised_5pct_manifest.csv
    protocol_indices.npz
    collocation_points.csv
    independent_physics_evaluation_points.csv
    spatial_split_map.png
    representative_2pct_points_t050.png
    README_PROTOCOL.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-path",
        default="./2d_cylinder_Re3900_100x100_kw_sst.mat",
    )
    p.add_argument(
        "--results-root",
        default="./results_cylinder_protocol_v1",
    )
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--block-nx", type=int, default=10)
    p.add_argument("--block-ny", type=int, default=10)
    p.add_argument("--validation-blocks", type=int, default=15)
    p.add_argument("--test-blocks", type=int, default=20)
    p.add_argument("--n-equation-points", type=int, default=100000)
    p.add_argument("--n-physics-eval-points", type=int, default=20000)
    p.add_argument("--collocation-seed", type=int, default=31415)
    p.add_argument("--physics-eval-seed", type=int, default=9091)
    p.add_argument("--representative-time-index", type=int, default=50)
    return p.parse_args()


def load_reference(path: Path):
    d = scipy.io.loadmat(path)

    required = ["X_star", "U_star", "P_star", "T_star"]
    missing = [k for k in required if k not in d]
    if missing:
        raise KeyError(f"MAT file missing required arrays: {missing}")

    X = np.asarray(d["X_star"], dtype=float)
    U = np.asarray(d["U_star"], dtype=float)
    P = np.asarray(d["P_star"], dtype=float)
    T = np.asarray(d["T_star"], dtype=float).reshape(-1)

    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError(f"Expected X_star (N,2), got {X.shape}")
    if U.ndim != 3 or U.shape[1] < 2:
        raise ValueError(f"Expected U_star (N,>=2,T), got {U.shape}")
    if P.shape != (X.shape[0], len(T)):
        raise ValueError(
            f"P_star shape {P.shape} inconsistent with X/T "
            f"({X.shape[0]}, {len(T)})"
        )

    x = np.unique(X[:, 0])
    y = np.unique(X[:, 1])
    nx = len(x)
    ny = len(y)
    nt = len(T)

    if nx * ny != X.shape[0]:
        raise ValueError(
            f"Expected complete rectangular spatial grid; nx*ny={nx*ny}, "
            f"N={X.shape[0]}"
        )

    mesh_x, mesh_y = np.meshgrid(x, y)
    if not (
        np.allclose(X[:, 0], mesh_x.reshape(-1), rtol=0.0, atol=1e-9)
        and np.allclose(X[:, 1], mesh_y.reshape(-1), rtol=0.0, atol=1e-9)
    ):
        raise ValueError(
            "X_star ordering is not the expected y-row/x-column order."
        )

    return {
        "X": X,
        "U": U,
        "P": P,
        "T": T,
        "x": x,
        "y": y,
        "nx": nx,
        "ny": ny,
        "nt": nt,
        "mesh_x": mesh_x,
        "mesh_y": mesh_y,
    }


def make_spatial_blocks(nx, ny, block_nx, block_ny):
    if nx % block_nx != 0 or ny % block_ny != 0:
        raise ValueError(
            f"Grid ({nx},{ny}) must be divisible by block grid "
            f"({block_nx},{block_ny})."
        )

    cells_x = nx // block_nx
    cells_y = ny // block_ny

    block_id = np.empty((ny, nx), dtype=int)

    for j in range(ny):
        by = j // cells_y
        for i in range(nx):
            bx = i // cells_x
            block_id[j, i] = by * block_nx + bx

    return block_id


def split_blocks(total_blocks, validation_blocks, test_blocks, seed):
    if validation_blocks + test_blocks >= total_blocks:
        raise ValueError("Validation + test blocks leave no training blocks.")

    rng = np.random.default_rng(seed)
    order = rng.permutation(total_blocks)

    test_ids = np.sort(order[:test_blocks])
    validation_ids = np.sort(
        order[test_blocks:test_blocks + validation_blocks]
    )
    train_ids = np.sort(
        order[test_blocks + validation_blocks:]
    )

    return train_ids, validation_ids, test_ids


def expand_spatial_indices_to_spacetime(spatial_indices, n_spatial, nt):
    # IMPORTANT: this matches pinn_model._read_xstar_ustar_format().
    #
    # read_2D_data() uses:
    #   x = tile(x_space, (n_time, 1))
    #   y = tile(y_space, (n_time, 1))
    #   t = repeat(T_star, n_space, axis=0)
    #   u/v/p = reshape(..., order="F")
    #
    # Therefore the flattened ordering is TIME-MAJOR:
    #   flat_index = time_index * n_spatial + spatial_index
    spatial_indices = np.asarray(spatial_indices, dtype=np.int64)
    times = np.arange(nt, dtype=np.int64)

    return (
        times[:, None] * n_spatial + spatial_indices[None, :]
    ).reshape(-1)


def random_rect_points(lb, ub, n, seed):
    rng = np.random.default_rng(seed)
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    return lb + (ub - lb) * rng.random((n, len(lb)))


def main():
    args = parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    out = Path(args.results_root) / "protocol"
    out.mkdir(parents=True, exist_ok=True)

    ref = load_reference(data_path)
    nx, ny, nt = ref["nx"], ref["ny"], ref["nt"]
    n_spatial = nx * ny
    n_total = n_spatial * nt

    block_id = make_spatial_blocks(
        nx, ny, args.block_nx, args.block_ny
    )
    total_blocks = args.block_nx * args.block_ny

    train_block_ids, val_block_ids, test_block_ids = split_blocks(
        total_blocks,
        validation_blocks=args.validation_blocks,
        test_blocks=args.test_blocks,
        seed=args.seed,
    )

    flat_blocks = block_id.reshape(-1)
    spatial_split = np.full(n_spatial, "train_pool", dtype=object)
    spatial_split[np.isin(flat_blocks, val_block_ids)] = "validation"
    spatial_split[np.isin(flat_blocks, test_block_ids)] = "test"

    train_spatial = np.where(spatial_split == "train_pool")[0]
    val_spatial = np.where(spatial_split == "validation")[0]
    test_spatial = np.where(spatial_split == "test")[0]

    train_pool_indices = expand_spatial_indices_to_spacetime(
        train_spatial, n_spatial, nt
    )
    validation_indices = expand_spatial_indices_to_spacetime(
        val_spatial, n_spatial, nt
    )
    test_indices = expand_spatial_indices_to_spacetime(
        test_spatial, n_spatial, nt
    )

    # Strict partition checks.
    assert len(np.intersect1d(train_pool_indices, validation_indices)) == 0
    assert len(np.intersect1d(train_pool_indices, test_indices)) == 0
    assert len(np.intersect1d(validation_indices, test_indices)) == 0
    assert (
        len(train_pool_indices)
        + len(validation_indices)
        + len(test_indices)
        == n_total
    )

    # Nested sparse training sets.
    rng = np.random.default_rng(args.seed + 1000)
    train_perm = rng.permutation(train_pool_indices)

    ratios = [1, 2, 3, 4, 5]
    train_sets = {}

    for pct in ratios:
        requested = int(round(n_total * pct / 100.0))
        if requested > len(train_pool_indices):
            raise RuntimeError(
                f"{pct}% requires {requested} points but train pool has "
                f"{len(train_pool_indices)}."
            )
        train_sets[pct] = np.sort(train_perm[:requested])

    for a, b in zip(ratios[:-1], ratios[1:]):
        if not np.all(np.isin(train_sets[a], train_sets[b])):
            raise RuntimeError(f"Nested-set assertion failed: {a}% not subset {b}%.")

    # Compact spatial split manifest.
    spatial_manifest = pd.DataFrame({
        "spatial_index": np.arange(n_spatial, dtype=int),
        "x": ref["X"][:, 0],
        "y": ref["X"][:, 1],
        "block_id": flat_blocks,
        "split": spatial_split,
    })
    spatial_manifest.to_csv(
        out / "spatial_split_manifest.csv", index=False
    )

    # One manifest for the maximum (5%) supervised set.
    # min_ratio_pct identifies the first ratio at which each point appears.
    five = train_sets[5]
    # Decode according to the same time-major ordering used by read_2D_data().
    spatial_idx = five % n_spatial
    time_idx = five // n_spatial

    min_ratio = np.full(len(five), 5, dtype=int)
    for pct in [4, 3, 2, 1]:
        min_ratio[np.isin(five, train_sets[pct])] = pct

    nested_manifest = pd.DataFrame({
        "flat_index": five,
        "spatial_index": spatial_idx,
        "time_index": time_idx,
        "x": ref["X"][spatial_idx, 0],
        "y": ref["X"][spatial_idx, 1],
        "t": ref["T"][time_idx],
        "min_ratio_pct": min_ratio,
    })
    nested_manifest.to_csv(
        out / "nested_supervised_5pct_manifest.csv", index=False
    )

    # Fixed equation and independent physics-evaluation points.
    lb = np.array(
        [ref["x"].min(), ref["y"].min(), ref["T"].min()],
        dtype=float,
    )
    ub = np.array(
        [ref["x"].max(), ref["y"].max(), ref["T"].max()],
        dtype=float,
    )

    collocation = random_rect_points(
        lb, ub, args.n_equation_points, args.collocation_seed
    )
    physics_eval = random_rect_points(
        lb, ub, args.n_physics_eval_points, args.physics_eval_seed
    )

    pd.DataFrame(
        collocation, columns=["x", "y", "t"]
    ).to_csv(out / "collocation_points.csv", index=False)

    pd.DataFrame(
        physics_eval, columns=["x", "y", "t"]
    ).to_csv(
        out / "independent_physics_evaluation_points.csv",
        index=False,
    )

    # Traceable compact binary indices.
    np.savez_compressed(
        out / "protocol_indices.npz",
        train_pool_indices=train_pool_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
        train_1pct=train_sets[1],
        train_2pct=train_sets[2],
        train_3pct=train_sets[3],
        train_4pct=train_sets[4],
        train_5pct=train_sets[5],
        train_block_ids=train_block_ids,
        validation_block_ids=val_block_ids,
        test_block_ids=test_block_ids,
    )

    # Summary.
    rows = [{
        "protocol_version": "cylinder_sparse_v1_1",
        "seed": args.seed,
        "nx": nx,
        "ny": ny,
        "nt": nt,
        "total_reference_points": n_total,
        "spatial_blocks_x": args.block_nx,
        "spatial_blocks_y": args.block_ny,
        "total_spatial_blocks": total_blocks,
        "train_blocks": len(train_block_ids),
        "validation_blocks": len(val_block_ids),
        "test_blocks": len(test_block_ids),
        "train_pool_points": len(train_pool_indices),
        "validation_points": len(validation_indices),
        "heldout_test_points": len(test_indices),
        "train_1pct_points": len(train_sets[1]),
        "train_2pct_points": len(train_sets[2]),
        "train_3pct_points": len(train_sets[3]),
        "train_4pct_points": len(train_sets[4]),
        "train_5pct_points": len(train_sets[5]),
        "nested_sampling": True,
        "ratio_denominator": "all_reference_space_time_points",
        "collocation_points": len(collocation),
        "collocation_seed": args.collocation_seed,
        "independent_physics_eval_points": len(physics_eval),
        "physics_eval_seed": args.physics_eval_seed,
        "split_mode": "fixed_spatial_blocks_applied_to_all_times",
        "flat_index_order": "time_major_flat=time_index*n_spatial+spatial_index",
    }]
    pd.DataFrame(rows).to_csv(
        out / "protocol_summary.csv", index=False
    )

    # Figure 1: spatial block partition.
    split_code = np.zeros((ny, nx), dtype=int)
    split_code[np.isin(block_id, val_block_ids)] = 1
    split_code[np.isin(block_id, test_block_ids)] = 2

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    mesh = ax.pcolormesh(
        ref["mesh_x"], ref["mesh_y"], split_code,
        shading="nearest",
    )
    cbar = fig.colorbar(mesh, ax=ax, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(
        ["Training pool", "Validation", "Held-out test"]
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        "Cylinder wake fixed spatial-block split applied to all time snapshots"
    )
    fig.tight_layout()
    fig.savefig(out / "spatial_split_map.png", dpi=250)
    plt.close(fig)

    # Figure 2: representative time snapshot with 2% supervised points.
    rep_t = int(args.representative_time_index)
    if rep_t < 0 or rep_t >= nt:
        raise ValueError(
            f"representative-time-index {rep_t} outside [0,{nt-1}]"
        )

    train2 = train_sets[2]
    # Time-major decoding:
    #   time_index = flat_index // n_spatial
    #   spatial_index = flat_index % n_spatial
    mask_t = (train2 // n_spatial) == rep_t
    train2_t = train2[mask_t]
    train2_spatial = train2_t % n_spatial

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.scatter(
        ref["X"][train_spatial, 0],
        ref["X"][train_spatial, 1],
        s=6, alpha=0.12,
        label="Training-pool spatial locations",
    )
    ax.scatter(
        ref["X"][val_spatial, 0],
        ref["X"][val_spatial, 1],
        s=7, alpha=0.25,
        label="Validation spatial blocks",
    )
    ax.scatter(
        ref["X"][test_spatial, 0],
        ref["X"][test_spatial, 1],
        s=7, alpha=0.25,
        label="Held-out test spatial blocks",
    )
    if len(train2_spatial):
        ax.scatter(
            ref["X"][train2_spatial, 0],
            ref["X"][train2_spatial, 1],
            s=24,
            label=f"2% supervised points at t-index {rep_t}",
        )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        f"Representative 2% supervised sampling at t={ref['T'][rep_t]:.6g}"
    )
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(
        out / f"representative_2pct_points_t{rep_t:03d}.png",
        dpi=250,
    )
    plt.close(fig)

    readme = f"""Cylinder sparse-data protocol v1.1

This protocol does NOT train any neural network.

Reference grid:
    nx={nx}, ny={ny}, nt={nt}, total={n_total}

Fixed spatial-block split:
    training blocks={len(train_block_ids)}
    validation blocks={len(val_block_ids)}
    held-out test blocks={len(test_block_ids)}

Because the same spatial split is applied to every time snapshot:
    train pool points={len(train_pool_indices)}
    validation points={len(validation_indices)}
    held-out test points={len(test_indices)}

Nested supervised data sets, relative to all {n_total} reference points:
    1%={len(train_sets[1])}
    2%={len(train_sets[2])}
    3%={len(train_sets[3])}
    4%={len(train_sets[4])}
    5%={len(train_sets[5])}

Nested property:
    D1 subset D2 subset D3 subset D4 subset D5 = TRUE

Collocation:
    points={len(collocation)}
    seed={args.collocation_seed}

Independent physics evaluation:
    points={len(physics_eval)}
    seed={args.physics_eval_seed}

Important:
The protocol indices use the exact time-major flattening convention of
pinn_model.read_2D_data():
    flat_index = time_index * n_spatial + spatial_index

The previous benchmark_train.py --ratio option represented mini-batch fraction,
not the fraction of supervised reference observations. In subsequent revised
training, supervised_ratio and batch_size/batch_fraction must be separate
parameters.

The validation and held-out test labels are not members of any supervised
training set. Physics collocation is an unlabeled regularization signal defined
over the reconstruction domain and must be kept identical across comparable
physics-constrained methods.
"""
    (out / "README_PROTOCOL.txt").write_text(
        readme, encoding="utf-8"
    )

    print("=" * 90)
    print("Cylinder sparse-data protocol v1.1 complete")
    print(f"Output directory: {out.resolve()}")
    print(f"Grid: nx={nx}, ny={ny}, nt={nt}, total={n_total}")
    print(
        f"Blocks: train={len(train_block_ids)}, "
        f"validation={len(val_block_ids)}, test={len(test_block_ids)}"
    )
    print(
        f"Points: train_pool={len(train_pool_indices)}, "
        f"validation={len(validation_indices)}, test={len(test_indices)}"
    )
    print(
        "Nested supervised: "
        + ", ".join(
            f"{pct}%={len(train_sets[pct])}" for pct in ratios
        )
    )
    print(f"Collocation points: {len(collocation)}")
    print(
        f"Independent physics-evaluation points: {len(physics_eval)}"
    )
    print("Nested subset checks: PASS")
    print("Partition overlap checks: PASS")
    print("=" * 90)


if __name__ == "__main__":
    main()
