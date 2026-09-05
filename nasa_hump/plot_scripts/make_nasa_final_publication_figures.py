#!/usr/bin/env python3

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.tri import Triangulation

ROOT = Path.cwd()
OUT = ROOT / "paper_ready_nasa_audit_final_v2"
OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# Utilities
# ============================================================

def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def pick_col(df, candidates):
    cmap = {norm(c): c for c in df.columns}

    for cand in candidates:
        nc = norm(cand)
        if nc in cmap:
            return cmap[nc]

    for cand in candidates:
        nc = norm(cand)
        for k, v in cmap.items():
            if nc in k or k in nc:
                return v

    return None


def xy_cols(df):
    return (
        pick_col(df, ["x", "x_coord", "coord_x", "x_coordinate"]),
        pick_col(df, ["y", "y_coord", "coord_y", "y_coordinate"]),
    )


def save(fig, stem):
    png = OUT / f"{stem}.png"
    pdf = OUT / f"{stem}.pdf"

    fig.savefig(png, dpi=400, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")

    plt.close(fig)

    print(f"[saved] {png}")
    print(f"[saved] {pdf}")


def panel(ax, label):
    ax.text(
        0.018, 0.975, label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
    )


# ============================================================
# N4
# Geometry-aware collocation audit
# ============================================================

def make_n4():

    print("\n[N4] Building sampler audit...")

    pdir = ROOT / "results_smoke_v12" / "protocol"

    legacy_path = pdir / "legacy_random_box_diagnostic_all_points.csv"
    invalid_path = pdir / "legacy_outside_local_reference_window_points.csv"
    valid_path = pdir / "valid_fluid_collocation_points.csv"

    for p in [legacy_path, invalid_path, valid_path]:
        if not p.exists():
            raise FileNotFoundError(p)

    legacy = pd.read_csv(legacy_path)
    invalid = pd.read_csv(invalid_path)
    valid = pd.read_csv(valid_path)

    lx, ly = xy_cols(legacy)
    ix, iy = xy_cols(invalid)
    vx, vy = xy_cols(valid)

    if not all([lx, ly, ix, iy, vx, vy]):
        raise RuntimeError("Could not identify x/y columns for N4.")

    print(f"  Legacy points : {len(legacy)}")
    print(f"  Invalid points: {len(invalid)}")
    print(f"  Valid points  : {len(valid)}")

    # Canonical audit safeguards
    if len(legacy) != 2000:
        print("  WARNING: expected 2000 legacy diagnostic points.")

    if len(invalid) != 276:
        print("  WARNING: expected 276 outside-window points.")

    if len(valid) != 2000:
        print("  WARNING: expected 2000 valid-fluid points.")

    fig, axes = plt.subplots(
        1, 2,
        figsize=(12.8, 5.15),
        constrained_layout=True
    )

    # --------------------------------------------------------
    # (a) Legacy
    # --------------------------------------------------------
    ax = axes[0]

    ax.scatter(
        legacy[lx],
        legacy[ly],
        s=9,
        alpha=0.24,
        label="Legacy samples (2000)",
        zorder=1,
    )

    # Explicitly show all 276 invalid points.
    # Do NOT guess the 129/147 subclass from column semantics here.
    ax.scatter(
        invalid[ix],
        invalid[iy],
        s=30,
        marker="x",
        linewidths=1.15,
        label=f"Outside local reference window ({len(invalid)})",
        zorder=5,
    )

    ax.set_title("Legacy rectangular sampling", fontsize=13)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.12)
    ax.legend(frameon=False, fontsize=9, loc="lower center")
    panel(ax, "(a)")

    # --------------------------------------------------------
    # (b) Corrected geometry-aware set
    # --------------------------------------------------------
    ax = axes[1]

    ax.scatter(
        valid[vx],
        valid[vy],
        s=10,
        alpha=0.55,
        label=f"Valid-fluid collocation ({len(valid)})",
    )

    ax.set_title("Geometry-aware sampling", fontsize=13)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.12)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    panel(ax, "(b)")

    # Same plotting limits
    xs = np.concatenate([
        legacy[lx].to_numpy(),
        invalid[ix].to_numpy(),
        valid[vx].to_numpy(),
    ])

    ys = np.concatenate([
        legacy[ly].to_numpy(),
        invalid[iy].to_numpy(),
        valid[vy].to_numpy(),
    ])

    xpad = 0.02 * max(np.ptp(xs), 1e-12)
    ypad = 0.04 * max(np.ptp(ys), 1e-12)

    for ax in axes:
        ax.set_xlim(np.nanmin(xs) - xpad, np.nanmax(xs) + xpad)
        ax.set_ylim(np.nanmin(ys) - ypad, np.nanmax(ys) + ypad)

    fig.suptitle(
        "NASA Hump Collocation Sampling Audit",
        fontsize=15,
        fontweight="bold",
    )

    fig.text(
        0.5,
        -0.012,
        "Legacy diagnostic: 276/2,000 points (13.80%) outside the local "
        "reference window; geometry-aware protocol: 2,000 valid-fluid points.",
        ha="center",
        fontsize=9,
    )

    save(fig, "Fig_N4_sampler_spatial_map_FINAL_v2")


# ============================================================
# N5
# Residual spatial diagnostics
# ============================================================

def find_residual_file(token):

    base = ROOT / "reviewer21_cross_model_residual_audit"

    if not base.exists():
        return None

    hits = []

    for p in base.rglob("*.csv"):

        name = p.name.lower()

        if token not in name:
            continue

        if "point" not in name and "residual" not in name:
            continue

        try:
            h = pd.read_csv(p, nrows=30)
        except Exception:
            continue

        x, y = xy_cols(h)

        q = pick_col(
            h,
            [
                "physics_residual_magnitude",
                "residual_magnitude",
                "physics_vector",
                "physics_residual",
                "residual_vector",
            ],
        )

        if x and y and q:
            priority = 0 if "pointwise" in name else 1
            hits.append((priority, len(str(p)), p))

    if not hits:
        return None

    hits.sort()
    return hits[0][2]


def load_residual(path):

    df = pd.read_csv(path)

    xcol, ycol = xy_cols(df)

    qcol = pick_col(
        df,
        [
            "physics_residual_magnitude",
            "residual_magnitude",
            "physics_vector",
            "physics_residual",
            "residual_vector",
        ],
    )

    if not xcol or not ycol or not qcol:
        raise RuntimeError(f"Residual columns not found in {path}")

    x = pd.to_numeric(df[xcol], errors="coerce").to_numpy()
    y = pd.to_numeric(df[ycol], errors="coerce").to_numpy()
    z = pd.to_numeric(df[qcol], errors="coerce").to_numpy()

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)

    return x[mask], y[mask], z[mask], qcol


def make_n5():

    print("\n[N5] Building residual diagnostics...")

    data_path = find_residual_file("data_only")
    pinn_path = find_residual_file("standard_pinn")

    if data_path is None or pinn_path is None:
        print("[N5 skip] Required pointwise residual files were not found.")
        return

    print(f"  Data-only : {data_path}")
    print(f"  Standard  : {pinn_path}")

    data_x, data_y, data_z, _ = load_residual(data_path)
    pinn_x, pinn_y, pinn_z, _ = load_residual(pinn_path)

    all_z = np.concatenate([data_z, pinn_z])

    # Shared robust scale.
    # Values above vmax are clipped rather than becoming blank.
    vmin = max(0.0, float(np.nanpercentile(all_z, 2)))
    vmax = float(np.nanpercentile(all_z, 98))

    if vmax <= vmin:
        vmin = float(np.nanmin(all_z))
        vmax = float(np.nanmax(all_z))

    print(f"  Shared color range: {vmin:.6g} to {vmax:.6g}")

    shared_norm = Normalize(vmin=vmin, vmax=vmax)
    levels = np.linspace(vmin, vmax, 30)

    fig, axes = plt.subplots(
        1, 2,
        figsize=(11.7, 4.8),
        constrained_layout=True,
    )

    items = [
        ("Data-only NN", data_x, data_y, data_z),
        ("Standard PINN", pinn_x, pinn_y, pinn_z),
    ]

    mappable = None

    for i, (ax, (title, x, y, z)) in enumerate(zip(axes, items)):

        # CRITICAL:
        # clip high/low values so percentile truncation does not
        # create artificial white holes.
        z_plot = np.clip(z, vmin, vmax)

        tri = Triangulation(x, y)

        mappable = ax.tricontourf(
            tri,
            z_plot,
            levels=levels,
            norm=shared_norm,
            extend="both",
        )

        ax.set_title(title, fontsize=13)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(alpha=0.08)
        panel(ax, f"({chr(97+i)})")

    cbar = fig.colorbar(
        mappable,
        ax=axes,
        pad=0.02,
        shrink=0.96,
        extend="both",
    )

    cbar.set_label("Physics-residual magnitude")

    fig.suptitle(
        "NASA Hump Physics-Residual Spatial Diagnostics",
        fontsize=15,
        fontweight="bold",
    )

    save(fig, "Fig_N5_residual_field_map_FINAL_v2")


# ============================================================
# N6
# Wall-pressure Cp profile and error audit
# ============================================================

def make_n6():

    print("\n[N6] Building wall-pressure gauge audit...")

    hdir = (
        ROOT
        / "baseline_suite"
        / "nasa_hump"
        / "bpinn_dropout"
        / "formal"
        / "archive"
        / "heldout"
    )

    point_path = hdir / "heldout_cp_pointwise_predictions.csv"

    if not point_path.exists():
        raise FileNotFoundError(point_path)

    df = pd.read_csv(point_path)

    print("  Columns:")
    for c in df.columns:
        print(f"    {c}")

    xcol = pick_col(df, ["x"])
    cp_true_col = pick_col(df, ["cp", "cp_true", "cp_reference"])

    # Explicit B-PINN columns.
    raw_col = pick_col(df, ["bpinn_dropout_cp_pred_raw"])
    aligned_col = pick_col(
        df,
        ["bpinn_dropout_cp_pred_gauge_aligned"]
    )

    if not all([xcol, cp_true_col, raw_col, aligned_col]):
        raise RuntimeError(
            "Validated B-PINN Cp columns not found.\n"
            f"Available columns: {list(df.columns)}"
        )

    x = pd.to_numeric(df[xcol], errors="coerce").to_numpy()
    cp_true = pd.to_numeric(df[cp_true_col], errors="coerce").to_numpy()
    cp_raw = pd.to_numeric(df[raw_col], errors="coerce").to_numpy()
    cp_aligned = pd.to_numeric(df[aligned_col], errors="coerce").to_numpy()

    mask = (
        np.isfinite(x)
        & np.isfinite(cp_true)
        & np.isfinite(cp_raw)
        & np.isfinite(cp_aligned)
    )

    x = x[mask]
    cp_true = cp_true[mask]
    cp_raw = cp_raw[mask]
    cp_aligned = cp_aligned[mask]

    # Sort by wall coordinate x
    order = np.argsort(x)

    x = x[order]
    cp_true = cp_true[order]
    cp_raw = cp_raw[order]
    cp_aligned = cp_aligned[order]

    raw_err = cp_raw - cp_true
    aligned_err = cp_aligned - cp_true

    raw_rmse = float(np.sqrt(np.mean(raw_err ** 2)))
    aligned_rmse = float(np.sqrt(np.mean(aligned_err ** 2)))

    shifts = cp_aligned - cp_raw
    shift = float(np.mean(shifts))
    shift_std = float(np.std(shifts))

    print(f"  Held-out Cp points   = {len(x)}")
    print(f"  Raw RMSE             = {raw_rmse:.9f}")
    print(f"  Gauge-aligned RMSE   = {aligned_rmse:.9f}")
    print(f"  Constant gauge shift = {shift:.9f}")
    print(f"  Gauge-shift std      = {shift_std:.3e}")

    # Frozen-result safety gate
    if abs(raw_rmse - 0.054260) > 5e-4:
        raise RuntimeError(
            f"Frozen RAW RMSE check failed: {raw_rmse}"
        )

    if abs(aligned_rmse - 0.051172) > 5e-4:
        raise RuntimeError(
            f"Frozen GAUGE RMSE check failed: {aligned_rmse}"
        )

    if shift_std > 1e-8:
        print(
            "WARNING: gauge correction is not numerically constant "
            f"(std={shift_std:.3e})."
        )

    fig, axes = plt.subplots(
        1, 2,
        figsize=(12.0, 4.75),
        constrained_layout=True,
    )

    # --------------------------------------------------------
    # (a) Cp wall profile
    # --------------------------------------------------------
    ax = axes[0]

    ax.plot(
        x,
        cp_true,
        marker="o",
        markersize=3.5,
        linewidth=1.4,
        label="Reference",
    )

    ax.plot(
        x,
        cp_raw,
        linewidth=1.5,
        label="B-PINN raw",
    )

    ax.plot(
        x,
        cp_aligned,
        linewidth=1.5,
        linestyle="--",
        label="B-PINN gauge-aligned",
    )

    ax.set_title(r"Wall-pressure coefficient $C_p$")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$C_p$")
    ax.grid(alpha=0.16)
    ax.legend(frameon=False, fontsize=9)
    panel(ax, "(a)")

    # --------------------------------------------------------
    # (b) Error profile
    # --------------------------------------------------------
    ax = axes[1]

    ax.axhline(
        0.0,
        linewidth=1.0,
        linestyle=":",
        label="Zero error",
    )

    ax.plot(
        x,
        raw_err,
        linewidth=1.5,
        label=rf"Raw error (RMSE={raw_rmse:.5f})",
    )

    ax.plot(
        x,
        aligned_err,
        linewidth=1.5,
        linestyle="--",
        label=rf"Gauge-aligned error (RMSE={aligned_rmse:.5f})",
    )

    # Symmetric error axis
    lim = 1.08 * np.nanmax(
        np.abs(np.concatenate([raw_err, aligned_err]))
    )

    ax.set_ylim(-lim, lim)

    ax.set_title(r"$C_p$ prediction error")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$C_p^{pred} - C_p^{ref}$")
    ax.grid(alpha=0.16)
    ax.legend(frameon=False, fontsize=9)
    panel(ax, "(b)")

    fig.suptitle(
        "NASA Hump Wall-Pressure Gauge Audit",
        fontsize=15,
        fontweight="bold",
    )

    fig.text(
        0.5,
        -0.014,
        rf"96 held-out wall-pressure points; "
        rf"raw RMSE = {raw_rmse:.5f}; "
        rf"gauge-aligned RMSE = {aligned_rmse:.5f}; "
        rf"constant gauge shift = {shift:.5f}.",
        ha="center",
        fontsize=9,
    )

    save(fig, "Fig_N6_wall_pressure_gauge_profile_FINAL_v2")


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 68)
    print("NASA HUMP FINAL PUBLICATION FIGURES — v2")
    print("Frozen evidence only.")
    print("NO TRAINING.")
    print("NO MODEL EVALUATION.")
    print("=" * 68)

    make_n4()
    make_n5()
    make_n6()

    print("\n" + "=" * 68)
    print("DONE")
    print(f"Output directory:\n{OUT}")
    print("=" * 68)


if __name__ == "__main__":
    main()

