
"""Exact reference-field residual audit for the Re=3900 cylinder wake.

This script evaluates the SAME simplified momentum residual used by the PINN
directly on the original CFD reference fields contained in:

    2d_cylinder_Re3900_100x100_kw_sst.mat

PINN residual audited
---------------------
fx = u_t + u*u_x + v*u_y + p_x - (1/Re)*(u_xx + u_yy)
fy = v_t + u*v_x + v*v_y + p_y - (1/Re)*(v_xx + v_yy)

The psi-p PINN does not include an explicit continuity-loss term because
u = psi_y and v = -psi_x satisfy continuity analytically. Nevertheless, this
script additionally evaluates

    rc = u_x + v_y

as a numerical reference-field diagnostic.

Important interpretation
------------------------
The MAT file contains X_star, U_star, P_star and T_star, but does not contain
k, omega, eddy viscosity, or modeled Reynolds-stress fields. Therefore this
script can evaluate the exact simplified PINN residual on the SST reference
fields, but it cannot decompose that residual into individual omitted SST
closure contributions.

Pressure-gauge note
-------------------
Only pressure gradients enter the audited residual, so an arbitrary additive
pressure constant has no effect on fx or fy.

Differentiation
---------------
The reference grid is reconstructed from X_star. Spatial and temporal
derivatives are evaluated with second-order finite differences using the actual
x, y and t coordinates. Quantitative summary statistics exclude a configurable
number of spatial boundary layers and time-end snapshots to reduce one-sided
finite-difference effects.

Outputs
-------
<results-root>/cylinder_reference_residual_audit/
    reference_residual_summary.csv
    reference_residual_time_series.csv
    reference_term_summary.csv
    reference_residual_full.npz
    sensitivity_by_margin.csv
    residual_vector_rmse_vs_time.png
    continuity_rmse_vs_time.png
    snapshot_*/...
    README_AUDIT.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import scipy.io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Exact simplified-PINN residual audit on Re=3900 SST reference fields."
    )
    p.add_argument(
        "--data-path",
        default="./2d_cylinder_Re3900_100x100_kw_sst.mat",
    )
    p.add_argument(
        "--results-root",
        default="./results_cylinder_reference_audit",
    )
    p.add_argument("--reynolds", type=float, default=3900.0)
    p.add_argument("--space-margin", type=int, default=2)
    p.add_argument("--time-margin", type=int, default=2)
    p.add_argument(
        "--snapshot-indices",
        nargs="+",
        type=int,
        default=[0, 25, 50, 75, 99],
    )
    p.add_argument(
        "--sensitivity-margins",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4],
    )
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
        raise ValueError(f"Expected X_star shape (N,2), got {X.shape}")
    if U.ndim != 3 or U.shape[1] < 2:
        raise ValueError(f"Expected U_star shape (N,>=2,T), got {U.shape}")
    if P.ndim != 2:
        raise ValueError(f"Expected P_star shape (N,T), got {P.shape}")

    x = np.unique(X[:, 0])
    y = np.unique(X[:, 1])
    nx = len(x)
    ny = len(y)
    nt = len(T)

    if nx * ny != X.shape[0]:
        raise ValueError(
            f"X_star is not a complete rectangular structured grid: "
            f"nx*ny={nx*ny}, N={X.shape[0]}"
        )

    mesh_x, mesh_y = np.meshgrid(x, y)
    if not (
        np.allclose(X[:, 0], mesh_x.reshape(-1), rtol=0.0, atol=1e-9)
        and np.allclose(X[:, 1], mesh_y.reshape(-1), rtol=0.0, atol=1e-9)
    ):
        raise ValueError(
            "X_star ordering does not match the expected y-row/x-column structured grid."
        )

    if U.shape[0] != X.shape[0] or U.shape[2] != nt:
        raise ValueError("U_star dimensions are inconsistent with X_star/T_star.")
    if P.shape != (X.shape[0], nt):
        raise ValueError("P_star dimensions are inconsistent with X_star/T_star.")

    # [time, y, x]
    u = np.stack([U[:, 0, k].reshape(ny, nx) for k in range(nt)], axis=0)
    v = np.stack([U[:, 1, k].reshape(ny, nx) for k in range(nt)], axis=0)
    p = np.stack([P[:, k].reshape(ny, nx) for k in range(nt)], axis=0)

    return {
        "x": x,
        "y": y,
        "t": T,
        "mesh_x": mesh_x,
        "mesh_y": mesh_y,
        "u": u,
        "v": v,
        "p": p,
        "mat_keys": sorted(k for k in d.keys() if not k.startswith("__")),
    }


def compute_terms(ref: Dict[str, np.ndarray], reynolds: float):
    x = ref["x"]
    y = ref["y"]
    t = ref["t"]
    u = ref["u"]
    v = ref["v"]
    p = ref["p"]

    # Second-order finite differences using actual coordinate arrays.
    u_t = np.gradient(u, t, axis=0, edge_order=2)
    v_t = np.gradient(v, t, axis=0, edge_order=2)

    u_y, u_x = np.gradient(u, y, x, axis=(1, 2), edge_order=2)
    v_y, v_x = np.gradient(v, y, x, axis=(1, 2), edge_order=2)
    p_y, p_x = np.gradient(p, y, x, axis=(1, 2), edge_order=2)

    u_xx = np.gradient(u_x, x, axis=2, edge_order=2)
    u_yy = np.gradient(u_y, y, axis=1, edge_order=2)
    v_xx = np.gradient(v_x, x, axis=2, edge_order=2)
    v_yy = np.gradient(v_y, y, axis=1, edge_order=2)

    temporal_x = u_t
    temporal_y = v_t
    convective_x = u * u_x + v * u_y
    convective_y = u * v_x + v * v_y
    pressure_x = p_x
    pressure_y = p_y
    viscous_x = (u_xx + u_yy) / float(reynolds)
    viscous_y = (v_xx + v_yy) / float(reynolds)

    fx = temporal_x + convective_x + pressure_x - viscous_x
    fy = temporal_y + convective_y + pressure_y - viscous_y
    residual_mag = np.sqrt(fx**2 + fy**2)

    continuity = u_x + v_y

    return {
        "continuity": continuity,
        "fx": fx,
        "fy": fy,
        "residual_mag": residual_mag,
        "temporal_x": temporal_x,
        "temporal_y": temporal_y,
        "convective_x": convective_x,
        "convective_y": convective_y,
        "pressure_x": pressure_x,
        "pressure_y": pressure_y,
        "viscous_x": viscous_x,
        "viscous_y": viscous_y,
    }


def build_mask(shape, space_margin: int, time_margin: int):
    nt, ny, nx = shape

    if space_margin < 0 or time_margin < 0:
        raise ValueError("Margins must be non-negative.")
    if 2 * space_margin >= min(nx, ny):
        raise ValueError("space-margin is too large.")
    if 2 * time_margin >= nt:
        raise ValueError("time-margin is too large.")

    mask = np.ones(shape, dtype=bool)

    if space_margin > 0:
        mask[:, :space_margin, :] = False
        mask[:, -space_margin:, :] = False
        mask[:, :, :space_margin] = False
        mask[:, :, -space_margin:] = False

    if time_margin > 0:
        mask[:time_margin, :, :] = False
        mask[-time_margin:, :, :] = False

    return mask


def scalar_stats(a, mask):
    z = np.asarray(a, dtype=float)[mask]
    z = z[np.isfinite(z)]
    abs_z = np.abs(z)

    return {
        "mean_abs": float(abs_z.mean()),
        "median_abs": float(np.median(abs_z)),
        "rmse": float(np.sqrt(np.mean(z**2))),
        "p95_abs": float(np.quantile(abs_z, 0.95)),
        "max_abs": float(abs_z.max()),
        "mean_signed": float(z.mean()),
        "points": int(z.size),
    }


def save_map(mesh_x, mesh_y, values, title, label, path):
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    m = ax.pcolormesh(mesh_x, mesh_y, values, shading="auto")
    cb = fig.colorbar(m, ax=ax)
    cb.set_label(label)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=250)
    plt.close(fig)


def main():
    args = parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    out = Path(args.results_root) / "cylinder_reference_residual_audit"
    out.mkdir(parents=True, exist_ok=True)

    ref = load_reference(data_path)
    terms = compute_terms(ref, args.reynolds)

    mask = build_mask(
        terms["fx"].shape,
        space_margin=args.space_margin,
        time_margin=args.time_margin,
    )

    # Main summary.
    summary_rows = []
    for name in [
        "continuity",
        "fx",
        "fy",
        "residual_mag",
    ]:
        summary_rows.append({
            "quantity": name,
            **scalar_stats(terms[name], mask),
        })

    pd.DataFrame(summary_rows).to_csv(
        out / "reference_residual_summary.csv",
        index=False,
    )

    # Individual term magnitudes.
    term_rows = []
    for name in [
        "temporal_x", "temporal_y",
        "convective_x", "convective_y",
        "pressure_x", "pressure_y",
        "viscous_x", "viscous_y",
    ]:
        term_rows.append({
            "quantity": name,
            **scalar_stats(terms[name], mask),
        })

    pd.DataFrame(term_rows).to_csv(
        out / "reference_term_summary.csv",
        index=False,
    )

    # Time-resolved metrics over strict spatial interior.
    nt = len(ref["t"])
    spatial_mask = build_mask(
        terms["fx"].shape,
        space_margin=args.space_margin,
        time_margin=0,
    )[0]

    time_rows = []
    for k in range(nt):
        valid = spatial_mask
        fx_k = terms["fx"][k][valid]
        fy_k = terms["fy"][k][valid]
        rc_k = terms["continuity"][k][valid]
        mag_k = terms["residual_mag"][k][valid]

        time_rows.append({
            "time_index": k,
            "time_value": float(ref["t"][k]),
            "continuity_rmse": float(np.sqrt(np.mean(rc_k**2))),
            "continuity_mean_abs": float(np.mean(np.abs(rc_k))),
            "fx_rmse": float(np.sqrt(np.mean(fx_k**2))),
            "fy_rmse": float(np.sqrt(np.mean(fy_k**2))),
            "physics_vector_rmse": float(np.sqrt(np.mean(fx_k**2 + fy_k**2))),
            "physics_magnitude_mean": float(np.mean(mag_k)),
            "physics_magnitude_p95": float(np.quantile(mag_k, 0.95)),
            "physics_magnitude_max": float(np.max(mag_k)),
            "included_in_global_summary": bool(
                k >= args.time_margin and k < nt - args.time_margin
            ),
        })

    time_df = pd.DataFrame(time_rows)
    time_df.to_csv(
        out / "reference_residual_time_series.csv",
        index=False,
    )

    # Sensitivity to edge exclusion.
    sensitivity_rows = []
    for m in args.sensitivity_margins:
        if 2 * m >= min(len(ref["x"]), len(ref["y"]), len(ref["t"])):
            continue
        msk = build_mask(
            terms["fx"].shape,
            space_margin=m,
            time_margin=m,
        )
        sensitivity_rows.append({
            "excluded_space_layers": int(m),
            "excluded_time_snapshots": int(m),
            "summary_points": int(msk.sum()),
            "continuity_rmse": scalar_stats(
                terms["continuity"], msk
            )["rmse"],
            "physics_vector_rmse": float(
                np.sqrt(
                    np.mean(
                        terms["fx"][msk]**2
                        + terms["fy"][msk]**2
                    )
                )
            ),
            "physics_magnitude_mean": scalar_stats(
                terms["residual_mag"], msk
            )["mean_abs"],
            "physics_magnitude_p95": scalar_stats(
                terms["residual_mag"], msk
            )["p95_abs"],
            "physics_magnitude_max": scalar_stats(
                terms["residual_mag"], msk
            )["max_abs"],
        })

    pd.DataFrame(sensitivity_rows).to_csv(
        out / "sensitivity_by_margin.csv",
        index=False,
    )

    # Full compressed traceable arrays.
    np.savez_compressed(
        out / "reference_residual_full.npz",
        x=ref["x"],
        y=ref["y"],
        t=ref["t"],
        u=ref["u"],
        v=ref["v"],
        p=ref["p"],
        continuity=terms["continuity"],
        fx=terms["fx"],
        fy=terms["fy"],
        residual_mag=terms["residual_mag"],
    )

    # Time curves.
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.plot(
        time_df["time_value"],
        time_df["physics_vector_rmse"],
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Physics-vector RMSE")
    ax.set_title("Exact simplified residual on the SST reference field")
    fig.tight_layout()
    fig.savefig(out / "residual_vector_rmse_vs_time.png", dpi=250)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.plot(
        time_df["time_value"],
        time_df["continuity_rmse"],
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Continuity RMSE")
    ax.set_title("Reference-field continuity diagnostic")
    fig.tight_layout()
    fig.savefig(out / "continuity_rmse_vs_time.png", dpi=250)
    plt.close(fig)

    # Snapshot maps.
    for idx in args.snapshot_indices:
        if idx < 0 or idx >= nt:
            continue

        snap_dir = out / f"snapshot_{idx:03d}"
        snap_dir.mkdir(parents=True, exist_ok=True)
        tval = float(ref["t"][idx])

        save_map(
            ref["mesh_x"], ref["mesh_y"],
            np.abs(terms["continuity"][idx]),
            f"Continuity residual magnitude, t={tval:.6g}",
            "|u_x + v_y|",
            snap_dir / "continuity_residual.png",
        )
        save_map(
            ref["mesh_x"], ref["mesh_y"],
            terms["residual_mag"][idx],
            f"Simplified momentum-residual magnitude, t={tval:.6g}",
            "sqrt(fx^2 + fy^2)",
            snap_dir / "momentum_residual_magnitude.png",
        )
        save_map(
            ref["mesh_x"], ref["mesh_y"],
            np.abs(terms["fx"][idx]),
            f"x-momentum residual magnitude, t={tval:.6g}",
            "|fx|",
            snap_dir / "fx_residual.png",
        )
        save_map(
            ref["mesh_x"], ref["mesh_y"],
            np.abs(terms["fy"][idx]),
            f"y-momentum residual magnitude, t={tval:.6g}",
            "|fy|",
            snap_dir / "fy_residual.png",
        )

        pd.DataFrame({
            "x": ref["mesh_x"].reshape(-1),
            "y": ref["mesh_y"].reshape(-1),
            "u": ref["u"][idx].reshape(-1),
            "v": ref["v"][idx].reshape(-1),
            "p": ref["p"][idx].reshape(-1),
            "continuity": terms["continuity"][idx].reshape(-1),
            "fx": terms["fx"][idx].reshape(-1),
            "fy": terms["fy"][idx].reshape(-1),
            "residual_magnitude": terms["residual_mag"][idx].reshape(-1),
        }).to_csv(
            snap_dir / "reference_residual_snapshot.csv",
            index=False,
        )

    residual_stats = {
        r["quantity"]: r
        for r in summary_rows
    }

    readme = f"""Cylinder Re=3900 SST reference-field residual audit

Input MAT keys:
{ref["mat_keys"]}

Grid:
nx = {len(ref["x"])}
ny = {len(ref["y"])}
nt = {len(ref["t"])}
x range = [{ref["x"].min()}, {ref["x"].max()}]
y range = [{ref["y"].min()}, {ref["y"].max()}]
t range = [{ref["t"].min()}, {ref["t"].max()}]

Re = {args.reynolds}

Audited exact simplified PINN residual:
fx = u_t + u*u_x + v*u_y + p_x - (1/Re)*(u_xx + u_yy)
fy = v_t + u*v_x + v*v_y + p_y - (1/Re)*(v_xx + v_yy)

Global summary excludes:
space layers = {args.space_margin}
time-end snapshots = {args.time_margin}

Important:
- The PINN itself does not train on an explicit continuity loss because the
  psi-p representation makes u=psi_y and v=-psi_x.
- Continuity is reported here only as a reference-field finite-difference
  diagnostic.
- The MAT file contains no k, omega, eddy viscosity, or modeled Reynolds-stress
  fields, so the exact simplified residual can be evaluated, but the omitted SST
  closure contribution cannot be decomposed from this file.
- Pressure-gauge offsets do not affect this audit because only p_x and p_y enter.

Headline values under the selected mask:
continuity RMSE = {residual_stats["continuity"]["rmse"]:.8e}
fx RMSE = {residual_stats["fx"]["rmse"]:.8e}
fy RMSE = {residual_stats["fy"]["rmse"]:.8e}
physics-vector RMSE = {float(np.sqrt(np.mean(terms["fx"][mask]**2 + terms["fy"][mask]**2))):.8e}
mean residual magnitude = {residual_stats["residual_mag"]["mean_abs"]:.8e}
P95 residual magnitude = {residual_stats["residual_mag"]["p95_abs"]:.8e}
"""
    (out / "README_AUDIT.txt").write_text(readme, encoding="utf-8")

    print("=" * 88)
    print("Cylinder Re=3900 SST reference residual audit complete")
    print(f"Output directory: {out.resolve()}")
    print(
        f"Grid: nx={len(ref['x'])}, ny={len(ref['y'])}, nt={len(ref['t'])}, "
        f"total={len(ref['x'])*len(ref['y'])*len(ref['t'])}"
    )
    print(f"MAT keys: {ref['mat_keys']}")
    print("Exact simplified pressure-inclusive PINN residual directly computable: YES")
    print("SST closure fields k/omega/eddy viscosity present in MAT: NO")
    print(
        f"Continuity RMSE: "
        f"{residual_stats['continuity']['rmse']:.6e}"
    )
    print(
        f"fx RMSE: "
        f"{residual_stats['fx']['rmse']:.6e}"
    )
    print(
        f"fy RMSE: "
        f"{residual_stats['fy']['rmse']:.6e}"
    )
    print(
        "Physics-vector RMSE: "
        f"{np.sqrt(np.mean(terms['fx'][mask]**2 + terms['fy'][mask]**2)):.6e}"
    )
    print(
        "Mean residual magnitude: "
        f"{residual_stats['residual_mag']['mean_abs']:.6e}"
    )
    print(
        "95th percentile residual magnitude: "
        f"{residual_stats['residual_mag']['p95_abs']:.6e}"
    )
    print("=" * 88)


if __name__ == "__main__":
    main()
