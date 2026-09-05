from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Canonical refined-analysis sources
# ============================================================

ROOT = Path(
    "./cylinder_final_lambda_0.1/"
    "spectral_phase_analysis_refined"
)

SPECTRUM = ROOT / "refined_mean_spectrum.csv"
PHASE = ROOT / "refined_phase_amplitude_by_spatial_point.csv"

OUTDIR = Path("./evaluation/paper_ready_cylinder_diagnostics")
OUTDIR.mkdir(parents=True, exist_ok=True)

PNG = OUTDIR / "Fig_C2_refined_spectrum_phase.png"
PDF = OUTDIR / "Fig_C2_refined_spectrum_phase.pdf"
AUDIT = OUTDIR / "Fig_C2_refined_spectrum_phase_audit.csv"


# ============================================================
# Load
# ============================================================

spec = pd.read_csv(SPECTRUM)
phase = pd.read_csv(PHASE)

required_spec = {
    "frequency",
    "reference_normalized_power",
    "prediction_normalized_power",
}

required_phase = {
    "reference_amplitude_at_refined_peak",
    "amplitude_ratio_pred_over_ref",
    "absolute_phase_error_deg",
    "energetic_probe",
}

missing = required_spec - set(spec.columns)
if missing:
    raise RuntimeError(
        f"Spectrum CSV missing columns: {sorted(missing)}"
    )

missing = required_phase - set(phase.columns)
if missing:
    raise RuntimeError(
        f"Phase CSV missing columns: {sorted(missing)}"
    )


# ============================================================
# Numerical audit from saved refined-analysis data
# ============================================================

i_ref = spec["reference_normalized_power"].idxmax()
i_pred = spec["prediction_normalized_power"].idxmax()

f_ref = float(spec.loc[i_ref, "frequency"])
f_pred = float(spec.loc[i_pred, "frequency"])
delta_f = abs(f_ref - f_pred)

ref_spec = spec["reference_normalized_power"].to_numpy(dtype=float)
pred_spec = spec["prediction_normalized_power"].to_numpy(dtype=float)

spectral_relative_l2 = (
    np.linalg.norm(pred_spec - ref_spec)
    / np.linalg.norm(ref_spec)
)

energetic = phase.loc[phase["energetic_probe"].astype(bool)].copy()

if len(energetic) == 0:
    raise RuntimeError("No energetic probes found.")

# Same energy weighting used by the refined analysis:
# spectral energy is proportional to amplitude squared.
weights = (
    energetic["reference_amplitude_at_refined_peak"]
    .to_numpy(dtype=float) ** 2
)

phase_abs = energetic[
    "absolute_phase_error_deg"
].to_numpy(dtype=float)

amp_ratio = energetic[
    "amplitude_ratio_pred_over_ref"
].to_numpy(dtype=float)

weighted_phase_error = np.average(
    phase_abs,
    weights=weights
)

weighted_amplitude_ratio = np.average(
    amp_ratio,
    weights=weights
)

# Canonical finite-record resolution already established
# by refined_spectral_phase_summary.csv.
rayleigh_resolution = 0.023203


# ============================================================
# Safety checks against locked refined results
# ============================================================

assert abs(f_ref - 0.1954) < 1e-9, f_ref
assert abs(f_pred - 0.1946) < 1e-9, f_pred

assert abs(spectral_relative_l2 - 0.039427) < 5e-5
assert abs(weighted_phase_error - 3.6008) < 5e-3
assert abs(weighted_amplitude_ratio - 0.99164) < 5e-4

print("PASS: refined spectral/phase values match locked results.")


# ============================================================
# Save numerical audit
# ============================================================

audit = pd.DataFrame([
    {
        "diagnostic": "reference_dominant_frequency",
        "value": f_ref,
    },
    {
        "diagnostic": "prediction_dominant_frequency",
        "value": f_pred,
    },
    {
        "diagnostic": "absolute_frequency_difference",
        "value": delta_f,
    },
    {
        "diagnostic": "rayleigh_resolution",
        "value": rayleigh_resolution,
    },
    {
        "diagnostic": "normalized_spectral_relative_l2",
        "value": spectral_relative_l2,
    },
    {
        "diagnostic": "weighted_mean_absolute_phase_error_deg",
        "value": weighted_phase_error,
    },
    {
        "diagnostic": "weighted_amplitude_ratio",
        "value": weighted_amplitude_ratio,
    },
    {
        "diagnostic": "energetic_probe_count",
        "value": len(energetic),
    },
])

audit.to_csv(AUDIT, index=False)


# ============================================================
# Publication figure
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(11.4, 4.6),
)

# ------------------------------------------------------------
# (a) Refined spectrum
# ------------------------------------------------------------

ax = axes[0]

ax.plot(
    spec["frequency"],
    spec["reference_normalized_power"],
    linewidth=1.8,
    label="Reference",
)

ax.plot(
    spec["frequency"],
    spec["prediction_normalized_power"],
    linewidth=1.8,
    linestyle="--",
    label="Prediction",
)

ax.axvline(
    f_ref,
    linestyle=":",
    linewidth=1.2,
)

ax.axvline(
    f_pred,
    linestyle="-.",
    linewidth=1.2,
)

ax.set_xlabel("Frequency")
ax.set_ylabel("Normalized mean power")
ax.set_title("(a) Refined mean spectrum")
ax.grid(True, alpha=0.25)
ax.legend(frameon=False)

ax.text(
    0.04,
    0.95,
    (
        f"$f_{{ref}}$ = {f_ref:.4f}\n"
        f"$f_{{pred}}$ = {f_pred:.4f}"
    ),
    transform=ax.transAxes,
    va="top",
)


# ------------------------------------------------------------
# (b) Energy-weighted phase-error distribution
# ------------------------------------------------------------

ax = axes[1]

# Weighted histogram so the displayed distribution follows
# the same energetic-probe weighting used for the summary.
hist_weights = weights / weights.sum()

ax.hist(
    phase_abs,
    bins=28,
    weights=hist_weights,
    alpha=0.75,
)

ax.axvline(
    weighted_phase_error,
    linestyle="--",
    linewidth=1.5,
    label=(
        "Weighted mean "
        f"= {weighted_phase_error:.2f}°"
    ),
)

ax.set_xlabel("Absolute phase error (degrees)")
ax.set_ylabel("Energy-weighted fraction")
ax.set_title("(b) Phase-error distribution")
ax.grid(True, alpha=0.25)
ax.legend(frameon=False)

ax.text(
    0.96,
    0.95,
    (
        f"Amplitude ratio = {weighted_amplitude_ratio:.3f}\n"
        f"Spectral rel. $L_2$ = {spectral_relative_l2:.3f}"
    ),
    transform=ax.transAxes,
    ha="right",
    va="top",
)


# ============================================================
# Save
# ============================================================

fig.tight_layout()

fig.savefig(
    PNG,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    PDF,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# Terminal audit
# ============================================================

print()
print("===== FIG. C2 NUMERICAL AUDIT =====")
print(f"Reference dominant frequency : {f_ref:.6f}")
print(f"Prediction dominant frequency: {f_pred:.6f}")
print(f"Absolute difference          : {delta_f:.6f}")
print(f"Rayleigh resolution          : {rayleigh_resolution:.6f}")
print(f"Difference / resolution      : {delta_f/rayleigh_resolution:.4f}")
print(f"Spectral relative L2         : {spectral_relative_l2:.6f}")
print(f"Weighted phase error         : {weighted_phase_error:.6f} deg")
print(f"Weighted amplitude ratio     : {weighted_amplitude_ratio:.6f}")
print(f"Energetic probes             : {len(energetic)}")

print()
print("===== SAVED FIG. C2 =====")
print(PNG)
print(PDF)
print(AUDIT)

print()
print("C2 generation finished successfully.")
