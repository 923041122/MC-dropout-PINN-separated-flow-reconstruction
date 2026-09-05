from __future__ import annotations

"""
Compare validation-only MC50 results from the calibration-aware cylinder pilot.

Expected directory layout under --pilot-root:

    lambda_0/
      validation_mc50_calibration/final_bpinn_validation_mc50_raw_metrics.csv
    lambda_0p01/
      validation_mc50_calibration/final_bpinn_validation_mc50_raw_metrics.csv
    lambda_0p1/
      validation_mc50_calibration/final_bpinn_validation_mc50_raw_metrics.csv

The lambda=0 branch is the continued-training control.

Primary pilot gate (u/v only, because pressure has a gauge sensitivity):
    1. mean raw calibration error improves by >= 15% vs lambda=0 control
    2. mean CRPS improves by >= 5%
    3. mean MC50 RMSE degrades by no more than 5%
    4. mean 95% interval width increases by no more than 25%

This is a screening gate, not a statistical significance test. If no positive
lambda passes, the recommendation is to stop the training-level modification
and use the assessment-first fallback (validation scaling + MC convergence).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_VARIANTS = {
    "lambda_0": 0.0,
    "lambda_0p01": 0.01,
    "lambda_0p1": 0.1,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--pilot-root",
        default="./cylinder_calibration_aware_pilot",
    )
    p.add_argument("--min-calibration-improvement", type=float, default=0.15)
    p.add_argument("--min-crps-improvement", type=float, default=0.05)
    p.add_argument("--max-rmse-degradation", type=float, default=0.05)
    p.add_argument("--max-mpiw-inflation", type=float, default=0.25)
    return p.parse_args()


def safe_rel_improvement(new, base):
    if not np.isfinite(base) or abs(base) < 1e-12:
        return np.nan
    return (base - new) / abs(base)


def safe_rel_degradation(new, base):
    if not np.isfinite(base) or abs(base) < 1e-12:
        return np.nan
    return (new - base) / abs(base)


def load_variant(root: Path, name: str, lam: float) -> dict:
    path = (
        root
        / name
        / "validation_mc50_calibration"
        / "final_bpinn_validation_mc50_raw_metrics.csv"
    )
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    required = {
        "variable",
        "mc50_mean_rmse",
        "raw_mpiw_95",
        "raw_mean_absolute_calibration_error",
        "empirical_crps",
        "raw_picp_95",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{path} is missing columns: {sorted(missing)}")

    uv = df[df["variable"].isin(["u", "v"])].copy()
    if len(uv) != 2:
        raise RuntimeError(f"{path}: expected exactly u and v rows")

    p = df[df["variable"] == "p"].copy()

    row = {
        "variant": name,
        "lambda_uq": lam,
        "uv_mc50_rmse_mean": float(uv["mc50_mean_rmse"].mean()),
        "uv_raw_mace_mean": float(
            uv["raw_mean_absolute_calibration_error"].mean()
        ),
        "uv_empirical_crps_mean": float(uv["empirical_crps"].mean()),
        "uv_raw_picp95_mean": float(uv["raw_picp_95"].mean()),
        "uv_raw_mpiw95_mean": float(uv["raw_mpiw_95"].mean()),
    }

    if len(p) == 1:
        row.update({
            "p_mc50_rmse": float(p.iloc[0]["mc50_mean_rmse"]),
            "p_raw_mace": float(
                p.iloc[0]["raw_mean_absolute_calibration_error"]
            ),
            "p_empirical_crps": float(p.iloc[0]["empirical_crps"]),
            "p_raw_picp95": float(p.iloc[0]["raw_picp_95"]),
            "p_raw_mpiw95": float(p.iloc[0]["raw_mpiw_95"]),
        })

    return row


def main():
    args = parse_args()
    root = Path(args.pilot_root)

    rows = [
        load_variant(root, name, lam)
        for name, lam in DEFAULT_VARIANTS.items()
    ]
    df = pd.DataFrame(rows).sort_values("lambda_uq").reset_index(drop=True)

    control = df.loc[np.isclose(df["lambda_uq"], 0.0)].iloc[0]

    out_rows = []
    for _, r in df.iterrows():
        out = r.to_dict()

        out["uv_calibration_improvement_vs_control"] = safe_rel_improvement(
            r["uv_raw_mace_mean"],
            control["uv_raw_mace_mean"],
        )
        out["uv_crps_improvement_vs_control"] = safe_rel_improvement(
            r["uv_empirical_crps_mean"],
            control["uv_empirical_crps_mean"],
        )
        out["uv_rmse_degradation_vs_control"] = safe_rel_degradation(
            r["uv_mc50_rmse_mean"],
            control["uv_mc50_rmse_mean"],
        )
        out["uv_mpiw_inflation_vs_control"] = safe_rel_degradation(
            r["uv_raw_mpiw95_mean"],
            control["uv_raw_mpiw95_mean"],
        )

        if np.isclose(r["lambda_uq"], 0.0):
            out["pilot_pass"] = True
            out["decision_note"] = "continued-training control"
        else:
            pass_gate = (
                out["uv_calibration_improvement_vs_control"]
                >= args.min_calibration_improvement
                and out["uv_crps_improvement_vs_control"]
                >= args.min_crps_improvement
                and out["uv_rmse_degradation_vs_control"]
                <= args.max_rmse_degradation
                and out["uv_mpiw_inflation_vs_control"]
                <= args.max_mpiw_inflation
            )
            out["pilot_pass"] = bool(pass_gate)
            out["decision_note"] = (
                "candidate passes screening gate"
                if pass_gate
                else "candidate fails screening gate"
            )

        out_rows.append(out)

    result = pd.DataFrame(out_rows)

    # Among positive lambdas that pass, prefer the lowest raw calibration error,
    # then the lowest CRPS. If none pass, recommend the assessment-first fallback.
    positive_pass = result[
        (result["lambda_uq"] > 0.0) & (result["pilot_pass"])
    ].copy()

    if len(positive_pass):
        winner = positive_pass.sort_values(
            ["uv_raw_mace_mean", "uv_empirical_crps_mean"]
        ).iloc[0]
        recommendation = (
            f"PROMISING: freeze lambda_uq={winner['lambda_uq']:g} for the next "
            "formal validation step. Do not open held-out test until all settings "
            "and the validation-fitted interval scaling are frozen."
        )
    else:
        recommendation = (
            "STOP training-level novelty branch: no positive lambda passed the "
            "screening gate. Use the assessment-first fallback: retain the original "
            "MC-dropout model, validation-based interval scaling, and MC convergence/"
            "cost control. Do not keep tuning on the held-out test set."
        )

    out_path = root / "calibration_aware_pilot_comparison.csv"
    result.to_csv(out_path, index=False)

    print("\n=== Calibration-aware pilot comparison (validation only) ===")
    print(result.to_string(index=False))
    print("\n=== Recommendation ===")
    print(recommendation)
    print(f"\nSaved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
