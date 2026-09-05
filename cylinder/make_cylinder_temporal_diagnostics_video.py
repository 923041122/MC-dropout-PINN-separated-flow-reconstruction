from pathlib import Path
import shutil

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter


OUT = Path(
    "evaluation/paper_ready_cylinder_diagnostics"
)
OUT.mkdir(parents=True, exist_ok=True)

SOURCES = {
    "Standard PINN": Path(
        "cylinder_final_lambda_0.1/"
        "heldout_evaluation/"
        "heldout_time_resolved_metrics.csv"
    ),
    "Weight-decay PINN": Path(
        "baseline_suite/cylinder/weight_decay/"
        "final_heldout/heldout_evaluation/"
        "heldout_time_resolved_metrics.csv"
    ),
    "Adaptive-weight PINN": Path(
        "baseline_suite/cylinder/adaptive_weight/"
        "final_heldout/heldout_evaluation/"
        "heldout_time_resolved_metrics.csv"
    ),
}

GIF = OUT / "Video_S1_cylinder_temporal_diagnostics.gif"
MP4 = OUT / "Video_S1_cylinder_temporal_diagnostics.mp4"
POSTER = OUT / "Video_S1_cylinder_temporal_diagnostics_poster.png"
README = OUT / "README_Video_S1_FINAL.txt"


# ------------------------------------------------------------
# Load frozen time-resolved metrics
# ------------------------------------------------------------

dfs = {}

required = {
    "time_index",
    "time_value",
    "variable",
    "rmse",
}

for method, path in SOURCES.items():

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"{method}: missing columns {sorted(missing)}"
        )

    dfs[method] = df.copy()

    print(
        f"{method}: {len(df)} rows | "
        f"{df['time_index'].nunique()} snapshots"
    )


variables = [
    ("u", r"$u$"),
    ("v", r"$v$"),
    (
        "p_gauge_aligned",
        "Gauge-aligned pressure",
    ),
]


# ------------------------------------------------------------
# Build one common time axis
# ------------------------------------------------------------

base = dfs["Standard PINN"]

time_table = (
    base[["time_index", "time_value"]]
    .drop_duplicates()
    .sort_values("time_index")
)

time_indices = time_table["time_index"].to_numpy()
times = time_table["time_value"].to_numpy(dtype=float)

if len(time_indices) != 100:
    raise RuntimeError(
        f"Expected 100 snapshots, found {len(time_indices)}"
    )


# ------------------------------------------------------------
# Pre-build curves and verify all methods
# ------------------------------------------------------------

curves = {}

for method, df in dfs.items():

    curves[method] = {}

    for var, _ in variables:

        sub = (
            df[df["variable"] == var]
            .sort_values("time_index")
            .reset_index(drop=True)
        )

        if len(sub) != 100:
            raise RuntimeError(
                f"{method}/{var}: "
                f"expected 100 rows, found {len(sub)}"
            )

        if not np.array_equal(
            sub["time_index"].to_numpy(),
            time_indices,
        ):
            raise RuntimeError(
                f"{method}/{var}: time_index mismatch"
            )

        curves[method][var] = (
            sub["rmse"].to_numpy(dtype=float)
        )


# ------------------------------------------------------------
# Figure
# ------------------------------------------------------------

fig, axes = plt.subplots(
    2,
    2,
    figsize=(13.2, 8.2),
)

ax_u = axes[0, 0]
ax_v = axes[0, 1]
ax_p = axes[1, 0]
ax_txt = axes[1, 1]

plot_axes = [ax_u, ax_v, ax_p]

styles = {
    "Standard PINN": "-",
    "Weight-decay PINN": "--",
    "Adaptive-weight PINN": "-.",
}

line_objects = {}
cursor_objects = {}
marker_objects = {}

for ax, (var, label) in zip(
    plot_axes,
    variables,
):

    line_objects[var] = {}
    marker_objects[var] = {}

    all_values = []

    for method in SOURCES:

        y = curves[method][var]
        all_values.extend(y.tolist())

        line, = ax.plot(
            times,
            y,
            linestyle=styles[method],
            linewidth=1.7,
            label=method,
        )

        marker, = ax.plot(
            [times[0]],
            [y[0]],
            marker="o",
            markersize=6,
            linestyle="None",
            color=line.get_color(),
        )

        line_objects[var][method] = line
        marker_objects[var][method] = marker

    ymin = 0.0
    ymax = max(all_values) * 1.10

    ax.set_ylim(ymin, ymax)
    ax.set_xlim(times.min(), times.max())

    ax.set_xlabel("Time")
    ax.set_ylabel("RMSE")
    ax.set_title(label)

    ax.grid(alpha=0.20)

    cursor = ax.axvline(
        times[0],
        linestyle=":",
        linewidth=1.4,
        color="0.35",
    )

    cursor_objects[var] = cursor


handles, labels = ax_u.get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.925),
    ncol=3,
    frameon=False,
    fontsize=10.5,
)

ax_txt.axis("off")

title = fig.suptitle(
    "",
    fontsize=14,
)


# ------------------------------------------------------------
# Text panel
# ------------------------------------------------------------

def build_text(frame):

    ti = int(time_indices[frame])
    t = float(times[frame])

    short_names = {
        "Standard PINN": "Standard",
        "Weight-decay PINN": "Weight-decay",
        "Adaptive-weight PINN": "Adaptive",
    }

    lines = [
        "CURRENT SNAPSHOT RMSE",
        "",
        f"Snapshot {frame + 1:>3}/100    "
        f"time_index {ti:>2}    "
        f"t = {t:.3f}",
        "",
        f"{'Method':<15}"
        f"{'u':>10}"
        f"{'v':>10}"
        f"{'p':>10}",
        "-" * 45,
    ]

    for method in SOURCES:

        u = curves[method]["u"][frame]
        v = curves[method]["v"][frame]
        p = curves[method][
            "p_gauge_aligned"
        ][frame]

        lines.append(
            f"{short_names[method]:<15}"
            f"{u:>10.5f}"
            f"{v:>10.5f}"
            f"{p:>10.5f}"
        )

    lines.extend([
        "",
        "p = gauge-aligned pressure RMSE",
        "",
        "Frozen held-out results only",
        "No training / no new inference",
    ])

    return "\n".join(lines)


text_object = ax_txt.text(
    0.03,
    0.92,
    build_text(0),
    va="top",
    ha="left",
    fontsize=10.0,
    family="monospace",
    linespacing=1.35,
    transform=ax_txt.transAxes,
)


# ------------------------------------------------------------
# Animation
# ------------------------------------------------------------

def update(frame):

    t = float(times[frame])

    for var, _ in variables:

        cursor_objects[var].set_xdata([t, t])

        for method in SOURCES:

            marker_objects[var][method].set_data(
                [t],
                [curves[method][var][frame]],
            )

    text_object.set_text(
        build_text(frame)
    )

    title.set_text(
        "Cylinder frozen held-out temporal diagnostics"
        f"   |   t = {t:.3f}"
        f"   |   snapshot {frame + 1}/100"
    )

    artists = [
        title,
        text_object,
    ]

    for var, _ in variables:
        artists.append(
            cursor_objects[var]
        )
        artists.extend(
            marker_objects[var].values()
        )

    return artists


# Poster at middle snapshot
middle = len(time_indices) // 2
update(middle)

fig.tight_layout(
    rect=[0.02, 0.03, 0.98, 0.84]
)

fig.savefig(
    POSTER,
    dpi=220,
    bbox_inches="tight",
)

print("Poster:", POSTER)


animation = FuncAnimation(
    fig,
    update,
    frames=len(time_indices),
    interval=100,
    blit=False,
    repeat=True,
)


if shutil.which("ffmpeg"):

    writer = FFMpegWriter(
        fps=10,
        codec="libx264",
        bitrate=3500,
        extra_args=[
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ],
    )

    animation.save(
        MP4,
        writer=writer,
        dpi=130,
    )

    artifact = MP4

else:

    print(
        "WARNING: ffmpeg not found. "
        "Generating GIF."
    )

    animation.save(
        GIF,
        writer=PillowWriter(fps=8),
        dpi=100,
    )

    artifact = GIF


plt.close(fig)


# ------------------------------------------------------------
# README
# ------------------------------------------------------------

README.write_text(
    """Cylinder Supplementary Video S1
================================

FINAL CONTENT:
Animated temporal diagnostics on the frozen held-out
Cylinder evaluation sequence.

Panels:
- u RMSE versus time
- v RMSE versus time
- gauge-aligned pressure RMSE versus time
- current-snapshot quantitative summary

Methods:
- Standard PINN
- Weight-decay PINN
- Adaptive-weight PINN

Protocol:
- 100 temporal snapshots
- frozen held-out evaluation outputs only
- no checkpoint loading
- no new model inference
- no training
- no spatial interpolation

Scientific purpose:
The animation shows the evolution of reconstruction
error over the complete held-out temporal sequence,
including transient peaks and relative performance
among the three frozen models.

The earlier sparse held-out spatial-mask animation
should be treated as an audit visualization only
and is NOT the final supplementary video.
""",
    encoding="utf-8",
)


print()
print("=" * 72)
print("FINAL TEMPORAL VIDEO PACKAGING FINISHED")
print("=" * 72)
print("Artifact:", artifact)
print("Poster  :", POSTER)
print("README  :", README)
print()
print("NO TRAINING.")
print("NO MODEL INFERENCE.")
print("NO SPATIAL INTERPOLATION.")
print("FROZEN TIME-RESOLVED RESULTS ONLY.")
