from pathlib import Path
import csv
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from benchmark_tools import build_model


# ============================================================
# Frozen Reviewer #21 diagnostic configuration
# ============================================================

RE_HUMP = 935892.0

LAYER_MAT_PSI = [2] + [100] * 10 + [2]

POINTS_FILE = Path(
    "physics_weight_formal_tradeoff/"
    "independent_physics_evaluation_points.csv"
)

CHECKPOINTS = {
    "data_only": Path(
        "baseline_suite/nasa_hump/data_only/formal/models/data_only_nn.pth"
    ),
    "standard_pinn": Path(
        "baseline_suite/nasa_hump/standard_pinn/formal/models/standard_pinn.pth"
    ),
    "bpinn_dropout": Path(
        "baseline_suite/nasa_hump/bpinn_dropout/formal/models/bpinn_dropout.pth"
    ),
}

OUT_DIR = Path(
    "reviewer21_cross_model_residual_audit"
)

BATCH_SIZE = 256
MC_SAMPLES = 50
SEED = 2025


# ============================================================
# Utilities
# ============================================================

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_state_dict(path, device):
    obj = torch.load(path, map_location=device)

    if isinstance(obj, nn.Module):
        return obj.state_dict()

    if not isinstance(obj, dict):
        raise TypeError(
            f"Unsupported checkpoint object in {path}: {type(obj)}"
        )

    for key in (
        "model_state_dict",
        "state_dict",
        "model",
        "net",
        "network",
    ):
        if key in obj and isinstance(obj[key], dict):
            return obj[key]

    if all(torch.is_tensor(v) for v in obj.values()):
        return obj

    raise RuntimeError(
        f"Could not identify state_dict in checkpoint: {path}"
    )


def strip_module_prefix(state):
    out = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        out[k] = v
    return out


# ============================================================
# Observation-only direct (x,y)->(u,v,p) reconstruction
#
# The archived training summary identifies this model as a
# direct observation-only MLP with 2 inputs, ten hidden
# layers of width 100, and 3 outputs.
#
# We reconstruct it from the checkpoint tensor shapes rather
# than depending on historical class naming.
# ============================================================

class DirectUVP(nn.Module):
    def __init__(self, linear_shapes):
        super().__init__()

        modules = []
        for i, (out_dim, in_dim) in enumerate(linear_shapes):
            modules.append(nn.Linear(in_dim, out_dim))
            if i != len(linear_shapes) - 1:
                modules.append(nn.Tanh())

        self.net = nn.Sequential(*modules)

    def forward(self, x, y):
        xy = torch.cat([x, y], dim=1)
        return self.net(xy)


def build_direct_uvp_from_checkpoint(path, device):
    state = strip_module_prefix(
        extract_state_dict(path, device)
    )

    # Preserve state-dict insertion order and recover linear layers
    weight_items = [
        (k, v)
        for k, v in state.items()
        if torch.is_tensor(v) and v.ndim == 2
    ]

    if not weight_items:
        raise RuntimeError(
            "No 2-D weight matrices found in Data-only checkpoint."
        )

    linear_shapes = [
        (int(v.shape[0]), int(v.shape[1]))
        for _, v in weight_items
    ]

    if linear_shapes[0][1] != 2:
        raise RuntimeError(
            f"Data-only first layer does not have 2 inputs: "
            f"{linear_shapes[0]}"
        )

    if linear_shapes[-1][0] != 3:
        raise RuntimeError(
            f"Data-only final layer does not have 3 outputs: "
            f"{linear_shapes[-1]}"
        )

    model = DirectUVP(linear_shapes).to(device)

    # Pair every 2-D weight with the matching bias prefix.
    target_linears = [
        m for m in model.net if isinstance(m, nn.Linear)
    ]

    if len(target_linears) != len(weight_items):
        raise RuntimeError("Recovered Data-only layer count mismatch.")

    for target, (weight_key, weight) in zip(
        target_linears, weight_items
    ):
        prefix = weight_key.rsplit(".", 1)[0]
        bias_key = prefix + ".bias"

        if bias_key not in state:
            raise RuntimeError(
                f"Missing matching bias for {weight_key}"
            )

        bias = state[bias_key]

        if tuple(target.weight.shape) != tuple(weight.shape):
            raise RuntimeError(
                f"Weight-shape mismatch for {weight_key}"
            )

        target.weight.data.copy_(weight.to(device))
        target.bias.data.copy_(bias.to(device))

    return model


def build_psi_model(path, dropout_rate, device):
    cfg = {
        "model_type": "psi",
        "dropout_rate": float(dropout_rate),
    }

    model = build_model(
        cfg,
        LAYER_MAT_PSI,
    ).to(device)

    state = strip_module_prefix(
        extract_state_dict(path, device)
    )

    missing, unexpected = model.load_state_dict(
        state,
        strict=False,
    )

    # Dropout has no trainable tensors, so no legitimate
    # checkpoint tensors should be lost.
    if missing:
        raise RuntimeError(
            f"Missing checkpoint tensors for {path}: {missing}"
        )

    if unexpected:
        raise RuntimeError(
            f"Unexpected checkpoint tensors for {path}: {unexpected}"
        )

    return model


# ============================================================
# Differential operators
# ============================================================

def grad_scalar(q, x, create_graph=True):
    return torch.autograd.grad(
        q.sum(),
        x,
        create_graph=create_graph,
        retain_graph=True,
    )[0]


def residual_from_uvp(u, v, p, x, y, re):
    u_x = grad_scalar(u, x)
    u_y = grad_scalar(u, y)
    v_x = grad_scalar(v, x)
    v_y = grad_scalar(v, y)

    p_x = grad_scalar(p, x)
    p_y = grad_scalar(p, y)

    u_xx = grad_scalar(u_x, x)
    u_yy = grad_scalar(u_y, y)
    v_xx = grad_scalar(v_x, x)
    v_yy = grad_scalar(v_y, y)

    fx = (
        u * u_x
        + v * u_y
        + p_x
        - (1.0 / float(re)) * (u_xx + u_yy)
    )

    fy = (
        u * v_x
        + v * v_y
        + p_y
        - (1.0 / float(re)) * (v_xx + v_yy)
    )

    continuity = u_x + v_y

    return continuity, fx, fy


def psi_fields(model, x, y):
    out = model.forward(x, y)

    psi = out[:, 0:1]
    p = out[:, 1:2]

    u = grad_scalar(psi, y)
    v = -grad_scalar(psi, x)

    return u, v, p


def psi_mc_mean_fields(model, x, y, n_mc):
    # MC predictive mean at the latent psi,p level.
    # Differentiation is then applied to the predictive mean.
    was_training = model.training
    model.train(True)

    psi_sum = None
    p_sum = None

    for _ in range(n_mc):
        out = model.forward(x, y)

        psi_i = out[:, 0:1]
        p_i = out[:, 1:2]

        psi_sum = psi_i if psi_sum is None else psi_sum + psi_i
        p_sum = p_i if p_sum is None else p_sum + p_i

    psi_mean = psi_sum / float(n_mc)
    p_mean = p_sum / float(n_mc)

    u = grad_scalar(psi_mean, y)
    v = -grad_scalar(psi_mean, x)

    model.train(was_training)

    return u, v, p_mean


# ============================================================
# Evaluation
# ============================================================

def evaluate_method(
    method,
    model,
    points,
    device,
    mode,
):
    model.eval()

    rows = []

    for start in range(0, len(points), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(points))

        xy = points[start:stop]

        x = torch.tensor(
            xy[:, 0:1],
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )

        y = torch.tensor(
            xy[:, 1:2],
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )

        with torch.enable_grad():

            if mode == "direct_uvp":
                out = model(x, y)
                u = out[:, 0:1]
                v = out[:, 1:2]
                p = out[:, 2:3]

            elif mode == "psi":
                u, v, p = psi_fields(
                    model,
                    x,
                    y,
                )

            elif mode == "psi_mc50_mean":
                u, v, p = psi_mc_mean_fields(
                    model,
                    x,
                    y,
                    MC_SAMPLES,
                )

            else:
                raise ValueError(mode)

            continuity, fx, fy = residual_from_uvp(
                u,
                v,
                p,
                x,
                y,
                RE_HUMP,
            )

            magnitude = torch.sqrt(
                fx.pow(2) + fy.pow(2)
            )

        arrays = {
            "x": x.detach().cpu().numpy().ravel(),
            "y": y.detach().cpu().numpy().ravel(),
            "u": u.detach().cpu().numpy().ravel(),
            "v": v.detach().cpu().numpy().ravel(),
            "p": p.detach().cpu().numpy().ravel(),
            "continuity": (
                continuity.detach().cpu().numpy().ravel()
            ),
            "fx": fx.detach().cpu().numpy().ravel(),
            "fy": fy.detach().cpu().numpy().ravel(),
            "residual_magnitude": (
                magnitude.detach().cpu().numpy().ravel()
            ),
        }

        n_batch = len(arrays["x"])

        for i in range(n_batch):
            rows.append({
                "method": method,
                "evaluation_mode": mode,
                **{
                    key: float(value[i])
                    for key, value in arrays.items()
                },
            })

        del x, y, u, v, p
        del continuity, fx, fy, magnitude

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return pd.DataFrame(rows)


def rmse(x):
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x ** 2)))


def summarize(df):
    c = df["continuity"].to_numpy()
    fx = df["fx"].to_numpy()
    fy = df["fy"].to_numpy()
    mag = df["residual_magnitude"].to_numpy()

    return {
        "method": df["method"].iloc[0],
        "evaluation_mode": df["evaluation_mode"].iloc[0],
        "n_points": int(len(df)),
        "reynolds_number": float(RE_HUMP),

        "continuity_mae": float(np.mean(np.abs(c))),
        "continuity_rmse": rmse(c),

        "fx_mae": float(np.mean(np.abs(fx))),
        "fx_rmse": rmse(fx),

        "fy_mae": float(np.mean(np.abs(fy))),
        "fy_rmse": rmse(fy),

        "physics_vector_rmse": float(
            np.sqrt(np.mean(fx ** 2 + fy ** 2))
        ),

        "physics_magnitude_mean": float(np.mean(mag)),
        "physics_magnitude_median": float(np.median(mag)),
        "physics_magnitude_p95": float(
            np.percentile(mag, 95.0)
        ),
        "physics_magnitude_max": float(np.max(mag)),
    }


# ============================================================
# Main
# ============================================================

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = get_device()

    print("==============================================")
    print("Reviewer #21 cross-model residual audit")
    print("==============================================")
    print("Device:", device)
    print("Re:", RE_HUMP)
    print("MC samples:", MC_SAMPLES)

    # ---------- preflight ----------
    if not POINTS_FILE.exists():
        raise FileNotFoundError(POINTS_FILE)

    for name, path in CHECKPOINTS.items():
        if not path.exists():
            raise FileNotFoundError(
                f"{name}: {path}"
            )

    points_df = pd.read_csv(POINTS_FILE)

    if not {"x", "y"}.issubset(points_df.columns):
        raise RuntimeError(
            f"{POINTS_FILE} must contain x,y columns."
        )

    points = (
        points_df[["x", "y"]]
        .to_numpy(dtype=np.float32)
    )

    print("Physics diagnostic points:", len(points))

    if len(points) != 20000:
        print(
            "WARNING: expected 20000 points, "
            f"found {len(points)}."
        )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------- models ----------
    print("\nLoading Data-only...")
    data_only = build_direct_uvp_from_checkpoint(
        CHECKPOINTS["data_only"],
        device,
    )

    print("Loading Standard PINN...")
    standard = build_psi_model(
        CHECKPOINTS["standard_pinn"],
        dropout_rate=0.0,
        device=device,
    )

    print("Loading B-PINN...")
    bpinn = build_psi_model(
        CHECKPOINTS["bpinn_dropout"],
        dropout_rate=0.002,
        device=device,
    )

    evaluations = [
        (
            "Data-only NN",
            data_only,
            "direct_uvp",
        ),
        (
            "Standard PINN",
            standard,
            "psi",
        ),
        (
            "MC-dropout B-PINN deterministic",
            bpinn,
            "psi",
        ),
        (
            "MC-dropout B-PINN MC50 predictive mean",
            bpinn,
            "psi_mc50_mean",
        ),
    ]

    summaries = []

    for method, model, mode in evaluations:
        print(
            f"\nEvaluating: {method} "
            f"[{mode}]"
        )

        df = evaluate_method(
            method,
            model,
            points,
            device,
            mode,
        )

        safe_name = (
            method.lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        pointwise_path = (
            OUT_DIR
            / f"{safe_name}_pointwise.csv"
        )

        df.to_csv(
            pointwise_path,
            index=False,
        )

        summary = summarize(df)
        summaries.append(summary)

        print(summary)
        print("Saved:", pointwise_path)

    summary_df = pd.DataFrame(summaries)

    summary_path = (
        OUT_DIR
        / "reviewer21_cross_model_residual_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    provenance = OUT_DIR / "README_AUDIT.txt"

    provenance.write_text(
        "\n".join([
            "Reviewer #21 cross-model residual diagnostic",
            "",
            "This is evaluation-only analysis.",
            "No model is trained or selected here.",
            "",
            f"Physics points: {POINTS_FILE}",
            f"Point count: {len(points)}",
            f"Reynolds number: {RE_HUMP}",
            "",
            "Residual definition matches hump_train.py:",
            "fx = u*u_x + v*u_y + p_x - Re^-1*(u_xx+u_yy)",
            "fy = u*v_x + v*v_y + p_y - Re^-1*(v_xx+v_yy)",
            "",
            "Data-only is evaluated post hoc using the same residual;",
            "it was not trained with a physics residual.",
            "",
            "B-PINN is reported both with dropout disabled",
            "and as the residual of the MC50 predictive-mean field.",
            "",
            "The 987-point held-out velocity test is not used",
            "for this diagnostic or for model selection.",
            "",
            "The NASA LES reference audit remains a separate",
            "closure-compatibility diagnostic because the released",
            "mean-field data do not provide full-field pressure.",
        ]) + "\n",
        encoding="utf-8",
    )

    print("\n==============================================")
    print("FINAL SUMMARY")
    print("==============================================")
    print(summary_df.to_string(index=False))
    print("\nSaved:", summary_path)
    print("Saved:", provenance)


if __name__ == "__main__":
    main()
