
"""Refined spectral analysis for the final cylinder held-out reconstruction.

Why this script exists
----------------------
With only 100 time snapshots, the raw FFT Rayleigh spacing is about 1/T.
If reference and prediction peaks fall in adjacent FFT bins, the apparent
frequency error can be one full bin even when the underlying spectra almost
overlap. This script therefore performs a dense continuous-frequency projection
around the shedding band rather than reporting only the discrete FFT-bin peak.

No training and no model loading are performed.

Method
------
1. Reconstruct complete v(t) signals at all held-out spatial probes.
2. Remove each probe's temporal mean.
3. Apply a Hann window.
4. Evaluate complex Fourier projections on a dense user-defined frequency grid.
5. Spatially average power over all held-out probes.
6. Estimate reference and prediction peak frequencies from the dense grid.
7. Report:
   - raw FFT-bin frequency and Rayleigh resolution;
   - refined dense-grid dominant frequency;
   - absolute/relative refined frequency error;
   - normalized spectral-shape error;
   - phase error at the refined reference dominant frequency;
   - representative-probe amplitude ratio and phase evolution.

Important
---------
The dense-grid estimate provides a smoother peak-location estimate, but it does
not create new physical information beyond the finite observation interval.
The Rayleigh resolution 1/T is still reported and should be acknowledged when
interpreting small frequency differences.
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
    p.add_argument("--variable", default="v", choices=["u", "v"])
    p.add_argument("--fmin", type=float, default=0.12)
    p.add_argument("--fmax", type=float, default=0.28)
    p.add_argument("--df", type=float, default=0.0002)
    p.add_argument("--energetic-quantile", type=float, default=0.50)
    return p.parse_args()


def wrap_phase(x):
    return np.angle(np.exp(1j * x))


def main():
    args = parse_args()

    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(input_csv)

    df = pd.read_csv(input_csv)

    ref_col = f"{args.variable}_ref"
    pred_col = f"{args.variable}_pred"

    required = {
        "spatial_index", "time_index", "x", "y", "t",
        ref_col, pred_col
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    time_table = (
        df[["time_index", "t"]]
        .drop_duplicates()
        .sort_values("time_index")
    )
    time_indices = time_table["time_index"].to_numpy(dtype=int)
    times = time_table["t"].to_numpy(dtype=float)
    nt = len(times)

    dt_all = np.diff(times)
    dt = float(np.mean(dt_all))
    if not np.allclose(dt_all, dt, rtol=1e-5, atol=1e-8):
        raise RuntimeError("Time samples are not uniform enough.")

    duration = float(times[-1] - times[0])
    rayleigh = 1.0 / duration

    spatial_info = (
        df[["spatial_index", "x", "y"]]
        .drop_duplicates("spatial_index")
        .sort_values("spatial_index")
        .reset_index(drop=True)
    )
    spatial_ids = spatial_info["spatial_index"].to_numpy(dtype=int)

    ref = (
        df.pivot(index="spatial_index", columns="time_index", values=ref_col)
        .reindex(index=spatial_ids, columns=time_indices)
        .to_numpy(dtype=float)
    )
    pred = (
        df.pivot(index="spatial_index", columns="time_index", values=pred_col)
        .reindex(index=spatial_ids, columns=time_indices)
        .to_numpy(dtype=float)
    )

    if np.isnan(ref).any() or np.isnan(pred).any():
        raise RuntimeError("NaNs found in held-out time-series matrix.")

    ref_fluc = ref - np.mean(ref, axis=1, keepdims=True)
    pred_fluc = pred - np.mean(pred, axis=1, keepdims=True)

    window = np.hanning(nt)
    rw = ref_fluc * window[None, :]
    pw = pred_fluc * window[None, :]

    # ------------------------------------------------------------------
    # Raw FFT result for reference / comparison.
    # ------------------------------------------------------------------
    raw_freq = np.fft.rfftfreq(nt, d=dt)
    raw_ref_fft = np.fft.rfft(rw, axis=1)
    raw_pred_fft = np.fft.rfft(pw, axis=1)
    raw_ref_power = np.mean(np.abs(raw_ref_fft) ** 2, axis=0)
    raw_pred_power = np.mean(np.abs(raw_pred_fft) ** 2, axis=0)

    raw_valid = (raw_freq >= args.fmin) & (raw_freq <= args.fmax)
    raw_idx = np.where(raw_valid)[0]

    raw_ref_peak_i = raw_idx[np.argmax(raw_ref_power[raw_valid])]
    raw_pred_peak_i = raw_idx[np.argmax(raw_pred_power[raw_valid])]

    raw_f_ref = float(raw_freq[raw_ref_peak_i])
    raw_f_pred = float(raw_freq[raw_pred_peak_i])

    # ------------------------------------------------------------------
    # Dense continuous-frequency projection.
    # ------------------------------------------------------------------
    freq = np.arange(args.fmin, args.fmax + 0.5 * args.df, args.df)
    basis = np.exp(-2j * np.pi * freq[:, None] * times[None, :])

    # Shape: probes x frequencies.
    ref_coeff = rw @ basis.T
    pred_coeff = pw @ basis.T

    ref_power = np.abs(ref_coeff) ** 2
    pred_power = np.abs(pred_coeff) ** 2

    mean_ref_power = np.mean(ref_power, axis=0)
    mean_pred_power = np.mean(pred_power, axis=0)

    ref_peak_idx = int(np.argmax(mean_ref_power))
    pred_peak_idx = int(np.argmax(mean_pred_power))

    f_ref = float(freq[ref_peak_idx])
    f_pred = float(freq[pred_peak_idx])

    f_abs_error = abs(f_pred - f_ref)
    f_rel_error = f_abs_error / abs(f_ref) if f_ref != 0 else np.nan

    # Spectral shape, normalized within requested band.
    eps = 1e-30
    ref_norm = mean_ref_power / max(float(np.sum(mean_ref_power)), eps)
    pred_norm = mean_pred_power / max(float(np.sum(mean_pred_power)), eps)

    spectral_rel_l2 = float(
        np.linalg.norm(pred_norm - ref_norm)
        / max(np.linalg.norm(ref_norm), eps)
    )

    # ------------------------------------------------------------------
    # Phase at refined reference peak frequency.
    # ------------------------------------------------------------------
    ref_peak_coeff = ref_coeff[:, ref_peak_idx]
    pred_at_ref_peak_coeff = pred_coeff[:, ref_peak_idx]

    ref_amp = np.abs(ref_peak_coeff)
    pred_amp = np.abs(pred_at_ref_peak_coeff)

    threshold = float(np.quantile(ref_amp, args.energetic_quantile))
    energetic = ref_amp >= threshold

    phase_diff = wrap_phase(
        np.angle(pred_at_ref_peak_coeff) - np.angle(ref_peak_coeff)
    )
    abs_phase = np.abs(phase_diff)

    weights = ref_amp[energetic] ** 2
    weighted_mean_abs_phase = float(
        np.sum(weights * abs_phase[energetic]) / np.sum(weights)
    )
    median_abs_phase = float(np.median(abs_phase[energetic]))
    p95_abs_phase = float(np.quantile(abs_phase[energetic], 0.95))

    # Amplitude ratio at the reference peak.
    amp_ratio = pred_amp[energetic] / np.maximum(ref_amp[energetic], 1e-30)
    median_amp_ratio = float(np.median(amp_ratio))
    weighted_amp_ratio = float(
        np.sum(weights * amp_ratio) / np.sum(weights)
    )

    # Representative probe selected objectively.
    rep_row = int(np.argmax(ref_amp))
    rep_spatial = int(spatial_ids[rep_row])
    rep_x = float(spatial_info.iloc[rep_row]["x"])
    rep_y = float(spatial_info.iloc[rep_row]["y"])

    output = Path(args.results_root) / "spectral_phase_analysis_refined"
    output.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "frequency": freq,
        "reference_mean_power": mean_ref_power,
        "prediction_mean_power": mean_pred_power,
        "reference_normalized_power": ref_norm,
        "prediction_normalized_power": pred_norm,
    }).to_csv(output / "refined_mean_spectrum.csv", index=False)

    phase_df = spatial_info.copy()
    phase_df["reference_amplitude_at_refined_peak"] = ref_amp
    phase_df["prediction_amplitude_at_refined_reference_peak"] = pred_amp
    phase_df["amplitude_ratio_pred_over_ref"] = (
        pred_amp / np.maximum(ref_amp, 1e-30)
    )
    phase_df["phase_error_rad"] = phase_diff
    phase_df["absolute_phase_error_deg"] = np.degrees(abs_phase)
    phase_df["energetic_probe"] = energetic
    phase_df.to_csv(
        output / "refined_phase_amplitude_by_spatial_point.csv",
        index=False,
    )

    summary = pd.DataFrame([{
        "variable": args.variable,
        "time_snapshots": nt,
        "heldout_spatial_probes": len(spatial_ids),
        "time_duration": duration,
        "dt": dt,
        "rayleigh_frequency_resolution": rayleigh,
        "raw_fft_reference_peak": raw_f_ref,
        "raw_fft_prediction_peak": raw_f_pred,
        "raw_fft_peak_difference": abs(raw_f_pred - raw_f_ref),
        "dense_grid_df": args.df,
        "refined_reference_dominant_frequency": f_ref,
        "refined_prediction_dominant_frequency": f_pred,
        "refined_frequency_absolute_error": f_abs_error,
        "refined_frequency_relative_error": f_rel_error,
        "refined_frequency_error_over_rayleigh_resolution": (
            f_abs_error / rayleigh
        ),
        "normalized_spectral_relative_l2_error": spectral_rel_l2,
        "energetic_quantile": args.energetic_quantile,
        "energetic_probe_count": int(np.sum(energetic)),
        "weighted_mean_absolute_phase_error_deg": float(
            np.degrees(weighted_mean_abs_phase)
        ),
        "median_absolute_phase_error_deg": float(
            np.degrees(median_abs_phase)
        ),
        "p95_absolute_phase_error_deg": float(
            np.degrees(p95_abs_phase)
        ),
        "weighted_mean_amplitude_ratio_pred_over_ref": weighted_amp_ratio,
        "median_amplitude_ratio_pred_over_ref": median_amp_ratio,
        "representative_spatial_index": rep_spatial,
        "representative_x": rep_x,
        "representative_y": rep_y,
    }])
    summary.to_csv(
        output / "refined_spectral_phase_summary.csv",
        index=False,
    )

    # Refined spectrum plot.
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.plot(freq, ref_norm, label="Reference")
    ax.plot(freq, pred_norm, label="Prediction")
    ax.axvline(
        f_ref, linestyle="--", alpha=0.7,
        label=f"Refined reference peak = {f_ref:.4f}"
    )
    ax.axvline(
        f_pred, linestyle=":", alpha=0.7,
        label=f"Refined prediction peak = {f_pred:.4f}"
    )
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Normalized mean power")
    ax.set_title(
        "Refined held-out transverse-velocity vortex-shedding spectrum"
    )
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "refined_mean_power_spectrum.png", dpi=250)
    plt.close(fig)

    # Representative signal.
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

    # Phase error distribution.
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.hist(np.degrees(abs_phase[energetic]), bins=30)
    ax.set_xlabel("Absolute phase error (degrees)")
    ax.set_ylabel("Energetic held-out probe count")
    ax.set_title(
        "Phase-error distribution at the refined reference shedding frequency"
    )
    fig.tight_layout()
    fig.savefig(output / "refined_phase_error_distribution.png", dpi=250)
    plt.close(fig)

    if hilbert is not None:
        rr = ref_fluc[rep_row]
        pp = pred_fluc[rep_row]
        ref_phase = np.unwrap(np.angle(hilbert(rr)))
        pred_phase = np.unwrap(np.angle(hilbert(pp)))
        inst = wrap_phase(pred_phase - ref_phase)

        fig, ax = plt.subplots(figsize=(8.8, 5.0))
        ax.plot(times, np.degrees(inst))
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

    readme = f"""Refined cylinder spectral analysis

The previous raw FFT analysis had Rayleigh spacing:
    {rayleigh:.8g}

The raw reference/prediction peaks were:
    {raw_f_ref:.8g}
    {raw_f_pred:.8g}

Their difference was:
    {abs(raw_f_pred - raw_f_ref):.8g}

This is compared explicitly with the finite-record Rayleigh resolution.

A dense continuous-frequency projection over [{args.fmin}, {args.fmax}] with
grid spacing {args.df} is used to estimate the peak locations more smoothly.
This does not increase the physical information content of the 100-snapshot
record; the Rayleigh limit is still reported and should be acknowledged.

No model training or hyperparameter tuning is performed here.
"""
    (output / "README_REFINED_SPECTRAL.txt").write_text(
        readme, encoding="utf-8"
    )

    print("=" * 94)
    print("Refined cylinder spectral/phase analysis complete")
    print(f"Time snapshots: {nt}")
    print(f"Held-out spatial probes: {len(spatial_ids)}")
    print(f"Time duration: {duration:.8g}")
    print(f"Rayleigh frequency resolution: {rayleigh:.8g}")
    print(f"Raw FFT reference peak: {raw_f_ref:.8g}")
    print(f"Raw FFT prediction peak: {raw_f_pred:.8g}")
    print(
        "Raw FFT peak difference / Rayleigh resolution: "
        f"{abs(raw_f_pred - raw_f_ref) / rayleigh:.4f}"
    )
    print("-" * 94)
    print(f"Refined reference dominant frequency: {f_ref:.8g}")
    print(f"Refined prediction dominant frequency: {f_pred:.8g}")
    print(f"Refined absolute frequency error: {f_abs_error:.8g}")
    print(f"Refined relative frequency error: {100.0 * f_rel_error:.4f}%")
    print(
        "Refined frequency error / Rayleigh resolution: "
        f"{f_abs_error / rayleigh:.4f}"
    )
    print(f"Normalized spectral relative L2 error: {spectral_rel_l2:.8g}")
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
        "Weighted mean amplitude ratio (pred/ref): "
        f"{weighted_amp_ratio:.6f}"
    )
    print(
        "Median amplitude ratio (pred/ref): "
        f"{median_amp_ratio:.6f}"
    )
    print(f"Output: {output.resolve()}")
    print("=" * 94)


if __name__ == "__main__":
    main()
