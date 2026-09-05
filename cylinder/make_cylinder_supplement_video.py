from pathlib import Path
import shutil

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


SRC = Path(
    "./cylinder_final_lambda_0.1/"
    "heldout_evaluation/"
    "heldout_pointwise_predictions.csv"
)

OUT = Path(
    "./evaluation/paper_ready_cylinder_diagnostics"
)

OUT.mkdir(parents=True, exist_ok=True)

MP4 = OUT / "Video_S1_cylinder_time_evolution.mp4"
GIF = OUT / "Video_S1_cylinder_time_evolution.gif"
POSTER = OUT / "Video_S1_cylinder_time_evolution_poster.png"
README = OUT / "README_Video_S1.txt"


# ------------------------------------------------------------
# Load frozen predictions only
# ------------------------------------------------------------

if not SRC.exists():
    raise FileNotFoundError(SRC)

df = pd.read_csv(SRC)

required = {
    "time_index",
    "x",
    "y",
    "t",
    "u_ref",
    "u_pred",
    "v_ref",
    "v_pred",
    "p_ref",
    "p_pred_gauge_aligned",
}

missing = required - set(df.columns)

if missing:
    raise RuntimeError(
        f"Missing required columns: {sorted(missing)}"
    )

for c in required:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=list(required)).copy()

time_table = (
    df[["time_index", "t"]]
    .drop_duplicates()
    .sort_values("time_index")
)

time_indices = time_table["time_index"].to_numpy()

print("Source:", SRC)
print("Rows:", len(df))
print("Snapshots:", len(time_indices))

if len(time_indices) != 100:
    raise RuntimeError(
        f"Expected 100 snapshots, found {len(time_indices)}"
    )


# ------------------------------------------------------------
# Variables
# ------------------------------------------------------------

variables = [
    (
        "u_ref",
        "u_pred",
        r"$u$",
    ),
    (
        "v_ref",
        "v_pred",
        r"$v$",
    ),
    (
        "p_ref",
        "p_pred_gauge_aligned",
        "pressure",
    ),
]


# ------------------------------------------------------------
# Fixed spatial limits
# ------------------------------------------------------------

xmin = float(df["x"].min())
xmax = float(df["x"].max())
ymin = float(df["y"].min())
ymax = float(df["y"].max())

dx = xmax - xmin
dy = ymax - ymin

xpad = 0.01 * dx if dx > 0 else 0.01
ypad = 0.01 * dy if dy > 0 else 0.01


# ------------------------------------------------------------
# Fixed color normalization over ALL snapshots
# Reference + prediction share the same scale
# ------------------------------------------------------------

norms = []

for ref_col, pred_col, label in variables:

    values = np.concatenate([
        df[ref_col].to_numpy(dtype=float),
        df[pred_col].to_numpy(dtype=float),
    ])

    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))

    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-12

    pad = 0.02 * (vmax - vmin)

    norms.append(
        Normalize(
            vmin=vmin - pad,
            vmax=vmax + pad,
        )
    )

    print(
        f"{label}: fixed color range "
        f"[{vmin-pad:.6g}, {vmax+pad:.6g}]"
    )


# ------------------------------------------------------------
# Figure
# ------------------------------------------------------------

fig, axes = plt.subplots(
    2,
    3,
    figsize=(14.5, 7.0),
    sharex=True,
    sharey=True,
)

plt.subplots_adjust(
    left=0.07,
    right=0.94,
    bottom=0.08,
    top=0.88,
    wspace=0.20,
    hspace=0.30,
)

scatters = []

first_index = time_indices[0]
first = df[df["time_index"] == first_index]

for j, (ref_col, pred_col, label) in enumerate(variables):

    norm = norms[j]

    s_ref = axes[0, j].scatter(
        first["x"],
        first["y"],
        c=first[ref_col],
        s=7,
        norm=norm,
        linewidths=0,
    )

    s_pred = axes[1, j].scatter(
        first["x"],
        first["y"],
        c=first[pred_col],
        s=7,
        norm=norm,
        linewidths=0,
    )

    scatters.append((s_ref, s_pred))

    axes[0, j].set_title(
        f"Reference — {label}",
        fontsize=12,
    )

    pred_title = f"Standard PINN — {label}"

    axes[1, j].set_title(
        pred_title,
        fontsize=11,
    )

    sm = ScalarMappable(
        norm=norm,
        cmap=s_ref.get_cmap(),
    )

    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        ax=axes[:, j],
        fraction=0.025,
        pad=0.025,
    )

    cbar.ax.tick_params(labelsize=8)


for i, ax in enumerate(axes.flat):

    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_ylim(ymin - ypad, ymax + ypad)

    if i >= 3:
        ax.set_xlabel("x")
    else:
        ax.set_xlabel("")

    ax.set_ylabel("y")

    ax.set_aspect(
        "equal",
        adjustable="box",
    )


title = fig.suptitle(
    "",
    fontsize=15,
    y=0.97,
)


# ------------------------------------------------------------
# Frame update
# ------------------------------------------------------------

def update(frame_number):

    idx = time_indices[frame_number]

    sub = df[
        df["time_index"] == idx
    ].sort_values("spatial_index") \
        if "spatial_index" in df.columns \
        else df[df["time_index"] == idx]

    if sub.empty:
        raise RuntimeError(
            f"No points for time_index={idx}"
        )

    xy = sub[["x", "y"]].to_numpy()

    for j, (ref_col, pred_col, label) in enumerate(variables):

        s_ref, s_pred = scatters[j]

        s_ref.set_offsets(xy)
        s_pred.set_offsets(xy)

        s_ref.set_array(
            sub[ref_col].to_numpy(dtype=float)
        )

        s_pred.set_array(
            sub[pred_col].to_numpy(dtype=float)
        )

    t_value = float(sub["t"].iloc[0])

    title.set_text(
        "Cylinder wake — frozen held-out time evolution"
        f"   |   t = {t_value:.3f}"
        f"   |   snapshot {frame_number + 1}/"
        f"{len(time_indices)}"
    )

    artists = [title]

    for pair in scatters:
        artists.extend(pair)

    return artists


# ------------------------------------------------------------
# Poster frame
# ------------------------------------------------------------

middle = len(time_indices) // 2
update(middle)

fig.savefig(
    POSTER,
    dpi=220,
    bbox_inches="tight",
)

print("Poster:", POSTER)


# ------------------------------------------------------------
# Animation
# ------------------------------------------------------------

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
        bitrate=4000,
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
    artifact_type = "MP4"

else:

    print(
        "WARNING: ffmpeg not found. "
        "Generating GIF fallback."
    )

    writer = PillowWriter(
        fps=8,
    )

    animation.save(
        GIF,
        writer=writer,
        dpi=100,
    )

    artifact = GIF
    artifact_type = "GIF"


plt.close(fig)


# ------------------------------------------------------------
# Provenance
# ------------------------------------------------------------

with open(
    README,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "Cylinder Supplementary Video S1\n"
        "================================\n\n"

        "Purpose:\n"
        "Time evolution of the frozen Cylinder "
        "held-out reconstruction.\n\n"

        "Source:\n"
        f"{SRC}\n\n"

        "Model:\n"
        "Standard PINN with frozen equation-loss "
        "weight lambda_f = 0.1.\n\n"

        "Panels:\n"
        "Top row: reference u, v, pressure.\n"
        "Bottom row: Standard PINN u, v, "
        "gauge-aligned pressure prediction.\n\n"

        "Protocol:\n"
        "Existing frozen held-out pointwise "
        "predictions only.\n"
        "100 snapshots.\n"
        "No training.\n"
        "No checkpoint loading.\n"
        "No new model inference.\n"
        "No spatial interpolation.\n"
        "Fixed color normalization across time "
        "for each variable.\n"
        "Reference and prediction use identical "
        "color normalization.\n\n"

        "Pressure policy:\n"
        "Gauge-aligned pressure prediction is "
        "used for the canonical pressure panel.\n\n"

        f"Artifact type: {artifact_type}\n"
        f"Artifact: {artifact}\n"
        f"Poster: {POSTER}\n"
    )


print()
print("=" * 72)
print("SUPPLEMENTARY VIDEO PACKAGING FINISHED")
print("=" * 72)
print("Artifact:", artifact)
print("Poster  :", POSTER)
print("README  :", README)
print()
print("NO TRAINING.")
print("NO MODEL INFERENCE.")
print("FROZEN POINTWISE PREDICTIONS ONLY.")
