
"""Reference-field compatibility audit for the NASA hump 2009 LES data.

Purpose
-------
Reviewer #21 asks whether the simplified steady 2-D incompressible momentum
residual used by the PINN is compatible with the turbulent LES reference field.

Important limitation
--------------------
The public 2009 NASA hump LES mean-field file used by this project contains
velocity and Reynolds-stress quantities but no full-field pressure. Therefore
the exact pressure-inclusive PINN momentum residual cannot be evaluated directly
from the public original LES fields.

This script instead computes, directly from the published LES mean fields:

1. continuity residual:
       r_c = du/dx + dv/dy

2. convective terms:
       Cx = u du/dx + v du/dy
       Cy = u dv/dx + v dv/dy

3. molecular viscous diffusion:
       Vx = (1/Re) (d2u/dx2 + d2u/dy2)
       Vy = (1/Re) (d2v/dx2 + d2v/dy2)

4. Reynolds-stress divergence omitted by the simplified PINN residual:
       Rx = d(uu)/dx + d(uv)/dy
       Ry = d(uv)/dx + d(vv)/dy

For a statistically steady incompressible mean momentum equation written as

       C + grad(p) - V + R = 0,

the pressure gradient implied by the available mean fields is

       grad(p)_implied = -C + V - R,

and the simplified PINN residual evaluated with this RANS-consistent implied
pressure gradient is exactly

       f_simple,implied = -R.

This is NOT presented as the original LES pressure residual. It is a closure-
compatibility diagnostic that quantifies the forcing omitted when Reynolds-
stress divergence is absent from the PINN physics regularizer.

The spatial derivatives are computed on the structured curvilinear LES grid
using second-order finite differences in computational coordinates followed by
a Jacobian transformation to physical (x,y) coordinates.

Outputs
-------
<results-root>/reference_residual_audit/
    reference_residual_pointwise.csv
    reference_residual_summary.csv
    reference_residual_summary_by_quantity.csv
    continuity_residual.png
    omitted_reynolds_stress_forcing.png
    omitted_vs_convective_ratio.png
    rans_implied_pressure_gradient.png
    README_AUDIT.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hump_train import read_les_meanfield_tec


RE_HUMP_DEFAULT = 935_892.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Audit compatibility of NASA hump LES mean fields with the simplified PINN residual."
    )
    p.add_argument("--data-dir", default=".")
    p.add_argument("--results-root", default="./results_reference_audit")
    p.add_argument("--reynolds", type=float, default=RE_HUMP_DEFAULT)
    p.add_argument(
        "--exclude-boundary-layers",
        type=int,
        default=2,
        help=(
            "Number of structured-grid rows/columns excluded from quantitative "
            "summary statistics to reduce finite-difference edge effects. "
            "Pointwise CSV still contains all nodes."
        ),
    )
    return p.parse_args()


def computational_grad(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return da/dxi and da/deta on array indexed as [eta=j, xi=i]."""
    arr = np.asarray(a, dtype=float)
    da_deta, da_dxi = np.gradient(arr, edge_order=2)
    return da_dxi, da_deta


def physical_grad(
    field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Curvilinear-grid derivative transformation: return dF/dx, dF/dy."""
    f_xi, f_eta = computational_grad(field)
    x_xi, x_eta = computational_grad(x)
    y_xi, y_eta = computational_grad(y)

    jac = x_xi * y_eta - x_eta * y_xi

    scale = np.nanmedian(np.abs(jac[np.isfinite(jac)]))
    tol = max(1e-14, 1e-10 * scale)
    if np.any(np.abs(jac) < tol):
        raise FloatingPointError(
            "Near-singular curvilinear Jacobian detected. "
            "Inspect the LES grid before trusting derivatives."
        )

    df_dx = (f_xi * y_eta - f_eta * y_xi) / jac
    df_dy = (-f_xi * x_eta + f_eta * x_xi) / jac
    return df_dx, df_dy


def laplacian(
    field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return d2F/dx2, d2F/dy2, and their sum."""
    f_x, f_y = physical_grad(field, x, y)
    f_xx, _ = physical_grad(f_x, x, y)
    _, f_yy = physical_grad(f_y, x, y)
    return f_xx, f_yy, f_xx + f_yy


def norm_stats(values: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    a = np.asarray(values, dtype=float)[mask]
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {
            "mean_abs": np.nan,
            "median_abs": np.nan,
            "rmse": np.nan,
            "p95_abs": np.nan,
            "max_abs": np.nan,
            "mean_signed": np.nan,
            "points": 0,
        }
    abs_a = np.abs(a)
    return {
        "mean_abs": float(abs_a.mean()),
        "median_abs": float(np.median(abs_a)),
        "rmse": float(np.sqrt(np.mean(a**2))),
        "p95_abs": float(np.quantile(abs_a, 0.95)),
        "max_abs": float(abs_a.max()),
        "mean_signed": float(a.mean()),
        "points": int(a.size),
    }


def save_scalar_map(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    title: str,
    cbar_label: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    mesh = ax.pcolormesh(x, y, z, shading="auto")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(cbar_label)
    ax.set_xlabel("x/c")
    ax.set_ylabel("y/c")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=250)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    data_path = Path(args.data_dir) / "LES_meanfield_nasahump2009_tec.dat"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing LES mean-field file: {data_path}")

    results_dir = Path(args.results_root) / "reference_residual_audit"
    results_dir.mkdir(parents=True, exist_ok=True)

    meanfield = read_les_meanfield_tec(data_path)
    arr = meanfield["arrays"]

    x = np.asarray(arr["x"], dtype=float)
    y = np.asarray(arr["y"], dtype=float)
    u = np.asarray(arr["u"], dtype=float)
    v = np.asarray(arr["v"], dtype=float)
    uu = np.asarray(arr["uu"], dtype=float)
    vv = np.asarray(arr["vv"], dtype=float)
    uv = np.asarray(arr["uv"], dtype=float)

    # First derivatives.
    u_x, u_y = physical_grad(u, x, y)
    v_x, v_y = physical_grad(v, x, y)

    # Continuity.
    continuity = u_x + v_y

    # Convective acceleration.
    conv_x = u * u_x + v * u_y
    conv_y = u * v_x + v * v_y

    # Molecular viscous diffusion.
    u_xx, u_yy, lap_u = laplacian(u, x, y)
    v_xx, v_yy, lap_v = laplacian(v, x, y)
    visc_x = (1.0 / float(args.reynolds)) * lap_u
    visc_y = (1.0 / float(args.reynolds)) * lap_v

    # Reynolds-stress divergence omitted by the PINN residual.
    uu_x, _ = physical_grad(uu, x, y)
    _, uv_y = physical_grad(uv, x, y)
    uv_x, _ = physical_grad(uv, x, y)
    _, vv_y = physical_grad(vv, x, y)

    rsdiv_x = uu_x + uv_y
    rsdiv_y = uv_x + vv_y
    rsdiv_mag = np.sqrt(rsdiv_x**2 + rsdiv_y**2)

    conv_mag = np.sqrt(conv_x**2 + conv_y**2)
    visc_mag = np.sqrt(visc_x**2 + visc_y**2)

    # RANS-balance-implied pressure gradient.
    # C + grad(p) - V + R = 0.
    pgrad_x_implied = -conv_x + visc_x - rsdiv_x
    pgrad_y_implied = -conv_y + visc_y - rsdiv_y
    pgrad_mag_implied = np.sqrt(pgrad_x_implied**2 + pgrad_y_implied**2)

    # If this implied pressure gradient were inserted into the simplified PINN
    # residual, the mismatch would be exactly the omitted Reynolds-stress forcing.
    simple_residual_implied_x = -rsdiv_x
    simple_residual_implied_y = -rsdiv_y
    simple_residual_implied_mag = rsdiv_mag

    eps = 1e-12
    omitted_to_convective_ratio = rsdiv_mag / (conv_mag + eps)
    omitted_to_resolved_balance_ratio = rsdiv_mag / (conv_mag + visc_mag + eps)

    n_j, n_i = x.shape
    margin = int(args.exclude_boundary_layers)
    summary_mask = np.ones_like(x, dtype=bool)
    if margin > 0:
        if n_j <= 2 * margin or n_i <= 2 * margin:
            raise ValueError("exclude-boundary-layers is too large for the grid.")
        summary_mask[:] = False
        summary_mask[margin:-margin, margin:-margin] = True

    finite_mask = (
        np.isfinite(x) & np.isfinite(y)
        & np.isfinite(continuity)
        & np.isfinite(rsdiv_mag)
        & np.isfinite(conv_mag)
    )
    summary_mask &= finite_mask

    # Pointwise traceable table.
    pointwise = pd.DataFrame({
        "original_index": np.arange(x.size, dtype=int),
        "i_index": np.tile(np.arange(n_i, dtype=int), n_j),
        "j_index": np.repeat(np.arange(n_j, dtype=int), n_i),
        "x": x.reshape(-1),
        "y": y.reshape(-1),
        "u": u.reshape(-1),
        "v": v.reshape(-1),
        "uu": uu.reshape(-1),
        "vv": vv.reshape(-1),
        "uv": uv.reshape(-1),
        "u_x": u_x.reshape(-1),
        "u_y": u_y.reshape(-1),
        "v_x": v_x.reshape(-1),
        "v_y": v_y.reshape(-1),
        "continuity_residual": continuity.reshape(-1),
        "convective_x": conv_x.reshape(-1),
        "convective_y": conv_y.reshape(-1),
        "viscous_diffusion_x": visc_x.reshape(-1),
        "viscous_diffusion_y": visc_y.reshape(-1),
        "reynolds_stress_divergence_x": rsdiv_x.reshape(-1),
        "reynolds_stress_divergence_y": rsdiv_y.reshape(-1),
        "reynolds_stress_divergence_magnitude": rsdiv_mag.reshape(-1),
        "rans_implied_pressure_gradient_x": pgrad_x_implied.reshape(-1),
        "rans_implied_pressure_gradient_y": pgrad_y_implied.reshape(-1),
        "rans_implied_pressure_gradient_magnitude": pgrad_mag_implied.reshape(-1),
        "simplified_residual_implied_x": simple_residual_implied_x.reshape(-1),
        "simplified_residual_implied_y": simple_residual_implied_y.reshape(-1),
        "simplified_residual_implied_magnitude": simple_residual_implied_mag.reshape(-1),
        "omitted_to_convective_ratio": omitted_to_convective_ratio.reshape(-1),
        "omitted_to_resolved_balance_ratio": omitted_to_resolved_balance_ratio.reshape(-1),
        "included_in_summary_mask": summary_mask.reshape(-1),
    })
    pointwise.to_csv(results_dir / "reference_residual_pointwise.csv", index=False)

    quantities = {
        "continuity_residual": continuity,
        "convective_x": conv_x,
        "convective_y": conv_y,
        "viscous_diffusion_x": visc_x,
        "viscous_diffusion_y": visc_y,
        "reynolds_stress_divergence_x": rsdiv_x,
        "reynolds_stress_divergence_y": rsdiv_y,
        "reynolds_stress_divergence_magnitude": rsdiv_mag,
        "rans_implied_pressure_gradient_x": pgrad_x_implied,
        "rans_implied_pressure_gradient_y": pgrad_y_implied,
        "simplified_residual_implied_magnitude": simple_residual_implied_mag,
        "omitted_to_convective_ratio": omitted_to_convective_ratio,
        "omitted_to_resolved_balance_ratio": omitted_to_resolved_balance_ratio,
    }

    rows = []
    for name, values in quantities.items():
        row = {"quantity": name, **norm_stats(values, summary_mask)}
        rows.append(row)
    pd.DataFrame(rows).to_csv(
        results_dir / "reference_residual_summary_by_quantity.csv", index=False
    )

    summary = {
        "dataset": "NASA_hump_2009_public_LES_limited_meanfield",
        "reference_points_total": int(x.size),
        "summary_points": int(summary_mask.sum()),
        "reynolds_number": float(args.reynolds),
        "differentiation": (
            "second-order finite differences in structured computational coordinates "
            "plus curvilinear Jacobian transformation"
        ),
        "excluded_boundary_rows_and_columns": int(margin),
        "full_field_pressure_available_in_input": False,
        "exact_pressure_inclusive_PINN_residual_directly_computable": False,
        "limitation": (
            "The public 2009 LES mean-field file contains velocity and Reynolds-stress "
            "quantities but no full-field pressure. Surface Cp alone cannot determine "
            "the interior pressure-gradient field."
        ),
        "interpretation": (
            "Reynolds-stress divergence quantifies the turbulent mean-momentum forcing "
            "omitted by the simplified PINN physics regularizer. The RANS-implied "
            "pressure gradient is a derived diagnostic, not an original LES pressure field."
        ),
        "continuity_rmse": norm_stats(continuity, summary_mask)["rmse"],
        "continuity_max_abs": norm_stats(continuity, summary_mask)["max_abs"],
        "rsdiv_magnitude_mean": norm_stats(rsdiv_mag, summary_mask)["mean_abs"],
        "rsdiv_magnitude_p95": norm_stats(rsdiv_mag, summary_mask)["p95_abs"],
        "rsdiv_magnitude_max": norm_stats(rsdiv_mag, summary_mask)["max_abs"],
        "omitted_to_convective_ratio_median": norm_stats(
            omitted_to_convective_ratio, summary_mask
        )["median_abs"],
        "omitted_to_convective_ratio_p95": norm_stats(
            omitted_to_convective_ratio, summary_mask
        )["p95_abs"],
    }
    pd.DataFrame([summary]).to_csv(
        results_dir / "reference_residual_summary.csv", index=False
    )

    save_scalar_map(
        x, y, np.abs(continuity),
        "NASA hump LES continuity-residual magnitude",
        "|du/dx + dv/dy|",
        results_dir / "continuity_residual.png",
    )
    save_scalar_map(
        x, y, rsdiv_mag,
        "Omitted Reynolds-stress-divergence forcing",
        "sqrt(Rx^2 + Ry^2)",
        results_dir / "omitted_reynolds_stress_forcing.png",
    )

    # Cap only for visualization so isolated tiny convective denominators do not
    # dominate the color scale; raw uncapped values remain in the CSV.
    ratio_plot = omitted_to_convective_ratio.copy()
    finite_ratio = ratio_plot[np.isfinite(ratio_plot)]
    if finite_ratio.size:
        cap = float(np.quantile(finite_ratio, 0.98))
        ratio_plot = np.minimum(ratio_plot, cap)
    save_scalar_map(
        x, y, ratio_plot,
        "Omitted turbulent forcing relative to convective acceleration",
        "|div Reynolds stress| / (|convective| + eps)",
        results_dir / "omitted_vs_convective_ratio.png",
    )
    save_scalar_map(
        x, y, pgrad_mag_implied,
        "RANS-balance-implied pressure-gradient magnitude",
        "|grad p| (derived, not original LES pressure)",
        results_dir / "rans_implied_pressure_gradient.png",
    )

    readme = f"""NASA hump 2009 LES reference compatibility audit

This audit intentionally does NOT claim to compute the exact pressure-inclusive
PINN residual from the original LES data, because the public mean-field input
contains no full-field pressure.

Grid points: {x.size}
Reynolds number: {args.reynolds}
Summary statistics exclude {margin} rows/columns at each structured-grid edge.

Directly evaluated from the public LES:
- continuity residual
- convective acceleration
- molecular viscous diffusion
- Reynolds-stress divergence

Derived diagnostic:
- pressure gradient implied by the statistically steady RANS mean-momentum
  balance using the available velocity and Reynolds-stress fields.

Key identity for compatibility interpretation:
    full mean momentum: C + grad(p) - V + R = 0
    simplified PINN:    f = C + grad(p) - V
    therefore, with the RANS-implied pressure gradient:
        f_simple,implied = -R

Hence the magnitude of Reynolds-stress divergence directly quantifies the
closure forcing omitted from the simplified PINN regularizer, under the stated
mean-momentum balance interpretation.

Use reference_residual_summary.csv for manuscript-level headline values and
reference_residual_pointwise.csv for traceable maps/region statistics.
"""
    (results_dir / "README_AUDIT.txt").write_text(readme, encoding="utf-8")

    print("=" * 88)
    print("NASA hump 2009 LES reference compatibility audit complete")
    print(f"Output directory: {results_dir.resolve()}")
    print(f"Total grid points: {x.size}")
    print(f"Summary-mask points: {summary_mask.sum()}")
    print("Full-field pressure in public input: NO")
    print("Exact pressure-inclusive PINN residual directly computable: NO")
    print(f"Continuity RMSE: {summary['continuity_rmse']:.6e}")
    print(f"Mean omitted Reynolds-stress forcing magnitude: {summary['rsdiv_magnitude_mean']:.6e}")
    print(f"95th percentile omitted forcing magnitude: {summary['rsdiv_magnitude_p95']:.6e}")
    print(f"Median omitted/convective ratio: {summary['omitted_to_convective_ratio_median']:.6e}")
    print("=" * 88)


if __name__ == "__main__":
    main()
