
"""Spectral and phase analysis of the final cylinder held-out reconstruction.

No training and no checkpoint loading are performed.

The analysis uses the final held-out pointwise predictions. Because the held-out
spatial mask is fixed at every time snapshot, each held-out spatial location
contains a complete time series.

Main analysis
-------------
1. Build v_ref(t) and v_pred(t) at every held-out spatial location.
2. Remove each probe's temporal mean and apply a Hann window.
3. Compute the FFT at every held-out spatial location.
4. Average the power spectra over all held-out spatial locations.
5. Compare the dominant reference/predicted shedding frequencies.
6. Compute a normalized mean-spectrum L2 error.
7. At the reference dominant-frequency bin, compute phase errors over the
   energetic held-out probes.
8. Select one representative probe objectively: the held-out spatial point with
   the largest reference spectral amplitude at the dominant frequency.
9. Plot its time signal and Hilbert-phase error for interpretation.

The frequency is reported in inverse units of the supplied time coordinate. If
the paper's time coordinate is nondimensionalized by D/U_inf, this frequency is
numerically equivalent to the Strouhal number; otherwise do not label it St.

Outputs
-------
<results-root>/spectral_phase_analysis/
    spectral_phase_summary.csv
    mean_spectrum.csv
    phase_error_by_spatial_point.csv
    mean_power_spectrum.png
    representative_probe_signal.png
    phase_error_distribution.png
    representative_probe_phase_error_vs_time.png
    README_SPECTRAL_PHASE.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.signal import hilbert
except Exception:
    hilbert = None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input-csv",
        default="./cylinder_final_lambda_0.1/heldout_evaluation/"
                "heldout_pointwise_predictions.csv",
    )
    p.add_argument(
        "--results-root",
        default="./cylinder_final_lambda_0.1",
    )
    p.add_argument(
        "--variable",
        default="v",
        choices=["u", "v"],
    )
    p.add_argument(
        "--energetic-quantile",
        type=float,
        default=0.50,
        help=(
            "Only probes with reference spectral amplitude at the dominant "
            "frequency above this quantile are used for aggregate phase metrics."
        ),
    )
    p.add_argument(
        "--min-frequency",
        type=float,
        default=0.0,
    )
    p.add_argument(
        "--max-frequency",
        type=float,
        default=None,
    )
    return p.parse_args()


def wrap_phase(x):
    return np.angle(np.exp(1j * x))


def circular_mean(phases, weights=None):
    phases = np.asarray(phases, dtype=float)
    if weights is None:
        z = np.mean(np.exp(1j * phases))
    else:
        weights = np.asarray(weights, dtype=float)
        weights = weights / np.sum(weights)
        z = np.sum(weights * np.exp(1j * phases))
    return float(np.angle(z))


def main():
    args = parse_args()

    if not (0.0 <= args.energetic_quantile < 1.0):
        raise ValueError("--energetic-quantile must be in [0,1).")

    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(input_csv)

    df = pd.read_csv(input_csv)

    ref_col = f"{args.variable}_ref"
    pred_col = f"{args.variable}_pred"

    required = {
        "spatial_index",
        "time_index",
        "x",
        "y",
        "t",
        ref_col,
        pred_col,
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Verify complete time series at every held-out spatial location.
    time_table = (
        df[["time_index", "t"]]
        .drop_duplicates()
        .sort_values("time_index")
    )
    time_indices = time_table["time_index"].to_numpy(dtype=int)
    times = time_table["t"].to_numpy(dtype=float)
    nt = len(times)

    if nt < 16:
        raise RuntimeError("Too few time snapshots for a meaningful spectrum.")

    dt_all = np.diff(times)
    dt = float(np.mean(dt_all))
    if not np.allclose(dt_all, dt, rtol=1e-5, atol=1e-8):
        raise RuntimeError(
            "Time samples are not sufficiently uniform for the current FFT analysis."
        )

    spatial_counts = df.groupby("spatial_index")["time_index"].nunique()
    bad = spatial_counts[spatial_counts != nt]
    if len(bad):
        raise RuntimeError(
            f"{len(bad)} held-out spatial locations do not contain all {nt} snapshots."
        )

    spatial_info = (
        df[["spatial_index", "x", "y"]]
        .drop_duplicates("spatial_index")
        .sort_values("spatial_index")
        .reset_index(drop=True)
    )
    spatial_ids = spatial_info["spatial_index"].to_numpy(dtype=int)
    n_probe = len(spatial_ids)

    ref_pivot = df.pivot(
        index="spatial_index",
        columns="time_index",
        values=ref_col,
    ).reindex(index=spatial_ids, columns=time_indices)

    pred_pivot = df.pivot(
        index="spatial_index",
        columns="time_index",
        values=pred_col,
    ).reindex(index=spatial_ids, columns=time_indices)

    ref = ref_pivot.to_numpy(dtype=float)
    pred = pred_pivot.to_numpy(dtype=float)

    if np.isnan(ref).any() or np.isnan(pred).any():
        raise RuntimeError("NaNs found after constructing held-out time-series matrices.")

    # Remove temporal mean independently at every probe.
    ref_fluc = ref - np.mean(ref, axis=1, keepdims=True)
    pred_fluc = pred - np.mean(pred, axis=1, keepdims=True)

    window = np.hanning(nt)[None, :]
    ref_fft = np.fft.rfft(ref_fluc * window, axis=1)
    pred_fft = np.fft.rfft(pred_fluc * window, axis=1)
    freq = np.fft.rfftfreq(nt, d=dt)

    ref_power = np.abs(ref_fft) ** 2
    pred_power = np.abs(pred_fft) ** 2

    mean_ref_power = np.mean(ref_power, axis=0)
    mean_pred_power = np.mean(pred_power, axis=0)

    # Search range for dominant shedding frequency; always exclude DC.
    valid = freq > max(0.0, float(args.min_frequency))
    if args.max_frequency is not None:
        valid &= freq <= float(args.max_frequency)

    if not np.any(valid):
        raise RuntimeError("No positive FFT frequencies remain in the requested range.")

    valid_indices = np.where(valid)[0]
    ref_peak_idx = valid_indices[np.argmax(mean_ref_power[valid])]
    pred_peak_idx = valid_indices[np.argmax(mean_pred_power[valid])]

    f_ref = float(freq[ref_peak_idx])
    f_pred = float(freq[pred_peak_idx])
    f_abs_error = float(abs(f_pred - f_ref))
    f_rel_error = float(f_abs_error / abs(f_ref)) if f_ref != 0 else np.nan

    # Normalize spectra to compare shape independently of total energy.
    eps = 1e-30
    ref_norm = mean_ref_power / max(float(np.sum(mean_ref_power[valid])), eps)
    pred_norm = mean_pred_power / max(float(np.sum(mean_pred_power[valid])), eps)

    spectral_l2 = float(
        np.linalg.norm(pred_norm[valid] - ref_norm[valid])
        / max(np.linalg.norm(ref_norm[valid]), eps)
    )

    # Phase comparison at the REFERENCE dominant-frequency bin.
    ref_coeff = ref_fft[:, ref_peak_idx]
    pred_coeff = pred_fft[:, ref_peak_idx]
    ref_amp = np.abs(ref_coeff)
    pred_amp = np.abs(pred_coeff)

    threshold = float(np.quantile(ref_amp, args.energetic_quantile))
    energetic = ref_amp >= threshold

    if np.sum(energetic) < 5:
        raise RuntimeError("Too few energetic probes for robust phase statistics.")

    phase_diff = wrap_phase(np.angle(pred_coeff) - np.angle(ref_coeff))
    abs_phase = np.abs(phase_diff)

    weights = ref_amp[energetic] ** 2
    weighted_mean_abs_phase = float(
        np.sum(weights * abs_phase[energetic]) / np.sum(weights)
    )
    median_abs_phase = float(np.median(abs_phase[energetic]))
    p95_abs_phase = float(np.quantile(abs_phase[energetic], 0.95))
    circular_signed_phase = circular_mean(
        phase_diff[energetic],
        weights=weights,
    )

    # Objectively selected representative probe:
    # largest reference amplitude at the global dominant frequency.
    rep_row = int(np.argmax(ref_amp))
    rep_spatial = int(spatial_ids[rep_row])
    rep_x = float(spatial_info.iloc[rep_row]["x"])
    rep_y = float(spatial_info.iloc[rep_row]["y"])

    # Save mean spectrum.
    spectrum_df = pd.DataFrame({
        "frequency": freq,
        "reference_mean_power": mean_ref_power,
        "prediction_mean_power": mean_pred_power,
        "reference_normalized_power": ref_norm,
        "prediction_normalized_power": pred_norm,
    })

    output = Path(args.results_root) / "spectral_phase_analysis"
    output.mkdir(parents=True, exist_ok=True)

    spectrum_df.to_csv(output / "mean_spectrum.csv", index=False)

    phase_df = spatial_info.copy()
    phase_df["reference_peak_amplitude"] = ref_amp
    phase_df["prediction_amplitude_at_reference_peak"] = pred_amp
    phase_df["phase_error_rad"] = phase_diff
    phase_df["absolute_phase_error_rad"] = abs_phase
    phase_df["absolute_phase_error_deg"] = np.degrees(abs_phase)
    phase_df["energetic_probe"] = energetic
    phase_df.to_csv(
        output / "phase_error_by_spatial_point.csv",
        index=False,
    )

    summary = pd.DataFrame([{
        "variable": args.variable,
        "heldout_spatial_probes": n_probe,
        "time_snapshots": nt,
        "time_min": float(times.min()),
        "time_max": float(times.max()),
        "dt": dt,
        "frequency_resolution": float(freq[1] - freq[0]),
        "nyquist_frequency": float(freq[-1]),
        "reference_dominant_frequency": f_ref,
        "prediction_dominant_frequency": f_pred,
        "dominant_frequency_absolute_error": f_abs_error,
        "dominant_frequency_relative_error": f_rel_error,
        "normalized_mean_spectrum_relative_l2_error": spectral_l2,
        "phase_evaluation_frequency": f_ref,
        "energetic_quantile": args.energetic_quantile,
        "energetic_probe_count": int(np.sum(energetic)),
        "weighted_mean_absolute_phase_error_rad": weighted_mean_abs_phase,
        "weighted_mean_absolute_phase_error_deg": float(
            np.degrees(weighted_mean_abs_phase)
        ),
        "median_absolute_phase_error_rad": median_abs_phase,
        "median_absolute_phase_error_deg": float(np.degrees(median_abs_phase)),
        "p95_absolute_phase_error_rad": p95_abs_phase,
        "p95_absolute_phase_error_deg": float(np.degrees(p95_abs_phase)),
        "energy_weighted_circular_signed_phase_error_rad": circular_signed_phase,
        "energy_weighted_circular_signed_phase_error_deg": float(
            np.degrees(circular_signed_phase)
        ),
        "representative_spatial_index": rep_spatial,
        "representative_x": rep_x,
        "representative_y": rep_y,
        "representative_selection_rule": (
            "maximum_reference_spectral_amplitude_at_global_reference_peak"
        ),
    }])

    summary.to_csv(
        output / "spectral_phase_summary.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.plot(
        freq[valid],
        ref_norm[valid],
        label="Reference",
    )
    ax.plot(
        freq[valid],
        pred_norm[valid],
        label="Prediction",
    )
    ax.axvline(f_ref, linestyle="--", alpha=0.7, label=f"Reference peak = {f_ref:.4g}")
    ax.axvline(f_pred, linestyle=":", alpha=0.7, label=f"Prediction peak = {f_pred:.4g}")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Normalized mean power")
    ax.set_title("Held-out transverse-velocity vortex-shedding spectrum")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "mean_power_spectrum.png", dpi=250)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.plot(times, ref[rep_row], label="Reference")
    ax.plot(times, pred[rep_row], label="Prediction")
    ax.set_xlabel("Time")
    ax.set_ylabel(args.variable)
    ax.set_title(
        f"Representative held-out shedding signal at x={rep_x:.4g}, y={rep_y:.4g}"
    )
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "representative_probe_signal.png", dpi=250)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.hist(
        np.degrees(abs_phase[energetic]),
        bins=30,
    )
    ax.set_xlabel("Absolute phase error (degrees)")
    ax.set_ylabel("Energetic held-out probe count")
    ax.set_title(
        "Phase-error distribution at the reference dominant shedding frequency"
    )
    fig.tight_layout()
    fig.savefig(output / "phase_error_distribution.png", dpi=250)
    plt.close(fig)

    # Hilbert phase error for the objectively selected representative probe.
    if hilbert is not None:
        rr = ref_fluc[rep_row]
        pp = pred_fluc[rep_row]

        ref_analytic = hilbert(rr)
        pred_analytic = hilbert(pp)

        instantaneous_phase_error = wrap_phase(
            np.unwrap(np.angle(pred_analytic))
            - np.unwrap(np.angle(ref_analytic))
        )

        fig, ax = plt.subplots(figsize=(8.8, 5.0))
        ax.plot(times, np.degrees(instantaneous_phase_error))
        ax.axhline(0.0, linewidth=1.0)
        ax.set_xlabel("Time")
        ax.set_ylabel("Wrapped phase error (degrees)")
        ax.set_title(
            "Time-resolved phase difference at the representative held-out probe"
        )
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(
            output / "representative_probe_phase_error_vs_time.png",
            dpi=250,
        )
        plt.close(fig)

    readme = f"""Cylinder spectral/phase analysis

Input:
    {input_csv}

Variable:
    {args.variable}

Method:
- complete time series at every held-out spatial location
- temporal mean removed independently at each probe
- Hann window
- real FFT
- spatially averaged power spectrum
- dominant frequency from the spatially averaged spectrum
- phase error evaluated at the reference dominant-frequency bin
- aggregate phase statistics restricted to probes above the
  {args.energetic_quantile:.2f} quantile of reference peak amplitude
- representative probe selected objectively as the point with maximum
  reference spectral amplitude at the global reference peak

Important:
The frequency is expressed in inverse units of the supplied time coordinate.
Only call it a Strouhal number if the manuscript's nondimensionalization makes
that identification valid.

No training, validation, model selection, or hyperparameter tuning is performed
by this script.
"""
    (output / "README_SPECTRAL_PHASE.txt").write_text(
        readme,
        encoding="utf-8",
    )

    print("=" * 92)
    print("Cylinder spectral/phase analysis complete")
    print(f"Variable: {args.variable}")
    print(f"Held-out spatial probes: {n_probe}")
    print(f"Time snapshots: {nt}")
    print(f"dt: {dt:.8g}")
    print(f"Frequency resolution: {freq[1] - freq[0]:.8g}")
    print(f"Reference dominant frequency: {f_ref:.8g}")
    print(f"Prediction dominant frequency: {f_pred:.8g}")
    print(f"Absolute frequency error: {f_abs_error:.8g}")
    print(f"Relative frequency error: {100.0 * f_rel_error:.4f}%")
    print(f"Normalized mean-spectrum relative L2 error: {spectral_l2:.8g}")
    print(
        "Weighted mean absolute phase error: "
        f"{np.degrees(weighted_mean_abs_phase):.4f} deg"
    )
    print(
        "Median absolute phase error: "
        f"{np.degrees(median_abs_phase):.4f} deg"
    )
    print(
        "P95 absolute phase error: "
        f"{np.degrees(p95_abs_phase):.4f} deg"
    )
    print(
        "Representative held-out probe: "
        f"spatial_index={rep_spatial}, x={rep_x:.6g}, y={rep_y:.6g}"
    )
    print(f"Output: {output.resolve()}")
    print("=" * 92)


if __name__ == "__main__":
    main()
