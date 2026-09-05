"""Repeated post-warm-up timing for NASA hump checkpoints.

Place this file in the repository's ``nasa_hump`` directory and run it from the
repository root. It times every requested model on the same full LES grid,
excluding one warm-up call for each model.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from benchmark_tools import get_device
from hump_train import read_les_meanfield_tec
from hump_validation import (
    ACTIVE_HUMP_METHODS,
    HUMP_METHODS,
    load_hump_models,
    predict_on_points,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repeated post-warm-up timing on the NASA hump LES grid."
    )
    parser.add_argument("--data-dir", default="./nasa_hump")
    parser.add_argument("--models-root", default="./results_2pct_v12/models")
    parser.add_argument("--results-root", default="./results_2pct_v12")
    parser.add_argument("--methods", nargs="+", default=ACTIVE_HUMP_METHODS)
    parser.add_argument("--batch-size", type=int, default=20000)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--require-models", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    meanfield_path = Path(args.data_dir) / "LES_meanfield_nasahump2009_tec.dat"
    if not meanfield_path.exists():
        raise FileNotFoundError(f"Missing NASA hump mean-field file: {meanfield_path}")

    meanfield = read_les_meanfield_tec(meanfield_path)
    flat = meanfield["flat"][["x", "y"]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    x = flat["x"].to_numpy(dtype=float)
    y = flat["y"].to_numpy(dtype=float)

    device = get_device()
    models = load_hump_models(
        methods=args.methods,
        models_root=Path(args.models_root),
        device=device,
        require_models=args.require_models,
    )
    if not models:
        raise RuntimeError(f"No checkpoints loaded from {args.models_root}")

    rows: List[Dict[str, object]] = []

    for method, model in models.items():
        # One model-specific warm-up, excluded from the reported timing.
        predict_on_points(
            model=model,
            x=x,
            y=y,
            device=device,
            batch_size=args.batch_size,
            train_mode=False,
        )

        times: List[float] = []
        for _ in range(args.repeats):
            _, elapsed = predict_on_points(
                model=model,
                x=x,
                y=y,
                device=device,
                batch_size=args.batch_size,
                train_mode=False,
            )
            times.append(float(elapsed))

        rows.append(
            {
                "method": method,
                "label": HUMP_METHODS[method]["label"],
                "points": int(len(flat)),
                "batch_size": int(args.batch_size),
                "repeats": int(args.repeats),
                "inference_time_mean_seconds": float(np.mean(times)),
                "inference_time_median_seconds": float(np.median(times)),
                "inference_time_std_seconds": float(np.std(times, ddof=0)),
                "inference_time_min_seconds": float(np.min(times)),
                "inference_time_max_seconds": float(np.max(times)),
            }
        )

    output_dir = Path(args.results_root) / "hump_evaluation" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "hump_repeated_post_warmup_timing.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)

    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Saved: {output_path.resolve()}")


if __name__ == "__main__":
    main()
