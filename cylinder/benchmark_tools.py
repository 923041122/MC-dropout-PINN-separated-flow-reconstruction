"""Utilities used by benchmark training and reviewer-response evaluation."""

import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from pinn_model import PINN_Net


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_load_state(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


class FourierFeaturePINN(nn.Module):
    """PINN with fixed random Fourier input features and psi-p outputs.

    Input:
        x, y, t

    Output:
        psi, p

    Velocity:
        u = d psi / d y
        v = - d psi / d x
    """

    def __init__(self, layer_mat, dropout_rate=0.0, fourier_features=128, sigma=3.0):
        super().__init__()

        self.dropout_rate = float(dropout_rate)
        self.fourier_features = int(fourier_features)
        self.sigma = float(sigma)

        self.register_buffer(
            "B",
            torch.randn(3, self.fourier_features) * self.sigma,
        )

        transformed_layers = list(layer_mat)
        transformed_layers[0] = self.fourier_features * 2

        self.net = PINN_Net(
            transformed_layers,
            dropout_rate=self.dropout_rate,
        )

    def forward(self, x, y, t):
        coords = torch.cat([x, y, t], dim=1)
        proj = 2.0 * math.pi * coords @ self.B
        features = torch.cat([torch.sin(proj), torch.cos(proj)], dim=1)

        return self.net.base(features)

    def predict_fields(self, x, y, t, create_graph=True):
        """Predict u, v, and p from Fourier-feature psi-p outputs."""

        if not x.requires_grad:
            x.requires_grad_(True)

        if not y.requires_grad:
            y.requires_grad_(True)

        if not t.requires_grad:
            t.requires_grad_(True)

        out = self.forward(x, y, t)

        psi = out[:, 0:1]
        p = out[:, 1:2]

        u = torch.autograd.grad(
            psi.sum(),
            y,
            create_graph=create_graph,
            retain_graph=True,
        )[0]

        v = -torch.autograd.grad(
            psi.sum(),
            x,
            create_graph=create_graph,
            retain_graph=True,
        )[0]

        return u, v, p

    def predict_fields_safe(self, x, y, t, create_graph=False, train_mode=False):
        """Safely predict u, v, p during evaluation or diagnostics.

        The psi-p formulation requires gradients even during evaluation because
        u and v are computed from derivatives of psi.
        """

        was_training = self.training

        if train_mode:
            self.train()
        else:
            self.eval()

        x_eval = x.detach().clone().requires_grad_(True)
        y_eval = y.detach().clone().requires_grad_(True)
        t_eval = t.detach().clone().requires_grad_(True)

        with torch.enable_grad():
            u_pred, v_pred, p_pred = self.predict_fields(
                x_eval,
                y_eval,
                t_eval,
                create_graph=create_graph,
            )

        if was_training:
            self.train()
        else:
            self.eval()

        return u_pred.detach(), v_pred.detach(), p_pred.detach()

    def data_mse_psi(self, x, y, t, u, v, p):
        """Supervised data loss for u, v, and p."""

        u_pred, v_pred, p_pred = self.predict_fields(
            x,
            y,
            t,
            create_graph=True,
        )

        mse = torch.nn.MSELoss()

        return (
            mse(u_pred, u)
            + mse(v_pred, v)
            + mse(p_pred, p)
        )

    def equation_mse_dimensionless_psi(self, x, y, t, Re):
        """Dimensionless incompressible Navier-Stokes residual loss."""

        u, v, p = self.predict_fields(
            x,
            y,
            t,
            create_graph=True,
        )

        u_t = torch.autograd.grad(
            u.sum(),
            t,
            create_graph=True,
            retain_graph=True,
        )[0]

        u_x = torch.autograd.grad(
            u.sum(),
            x,
            create_graph=True,
            retain_graph=True,
        )[0]

        u_y = torch.autograd.grad(
            u.sum(),
            y,
            create_graph=True,
            retain_graph=True,
        )[0]

        v_t = torch.autograd.grad(
            v.sum(),
            t,
            create_graph=True,
            retain_graph=True,
        )[0]

        v_x = torch.autograd.grad(
            v.sum(),
            x,
            create_graph=True,
            retain_graph=True,
        )[0]

        v_y = torch.autograd.grad(
            v.sum(),
            y,
            create_graph=True,
            retain_graph=True,
        )[0]

        p_x = torch.autograd.grad(
            p.sum(),
            x,
            create_graph=True,
            retain_graph=True,
        )[0]

        p_y = torch.autograd.grad(
            p.sum(),
            y,
            create_graph=True,
            retain_graph=True,
        )[0]

        u_xx = torch.autograd.grad(
            u_x.sum(),
            x,
            create_graph=True,
            retain_graph=True,
        )[0]

        u_yy = torch.autograd.grad(
            u_y.sum(),
            y,
            create_graph=True,
            retain_graph=True,
        )[0]

        v_xx = torch.autograd.grad(
            v_x.sum(),
            x,
            create_graph=True,
            retain_graph=True,
        )[0]

        v_yy = torch.autograd.grad(
            v_y.sum(),
            y,
            create_graph=True,
            retain_graph=True,
        )[0]

        fx = (
            u_t
            + u * u_x
            + v * u_y
            + p_x
            - (1.0 / Re) * (u_xx + u_yy)
        )

        fy = (
            v_t
            + u * v_x
            + v * v_y
            + p_y
            - (1.0 / Re) * (v_xx + v_yy)
        )

        zeros_x = torch.zeros_like(fx)
        zeros_y = torch.zeros_like(fy)

        mse = torch.nn.MSELoss()

        return mse(fx, zeros_x) + mse(fy, zeros_y)

    def predict_with_uncertainty(self, x, y, t, samples=30):
        """MC-dropout prediction for Fourier-feature PINN."""

        was_training = self.training

        # Keep dropout active during MC sampling.
        self.train()

        x_mc = x.detach().clone().requires_grad_(True)
        y_mc = y.detach().clone().requires_grad_(True)
        t_mc = t.detach().clone().requires_grad_(True)

        samples_u = []
        samples_v = []
        samples_p = []

        with torch.enable_grad():
            for _ in range(samples):
                u_pred, v_pred, p_pred = self.predict_fields(
                    x_mc,
                    y_mc,
                    t_mc,
                    create_graph=False,
                )

                samples_u.append(u_pred.detach().unsqueeze(0))
                samples_v.append(v_pred.detach().unsqueeze(0))
                samples_p.append(p_pred.detach().unsqueeze(0))

        if was_training:
            self.train()
        else:
            self.eval()

        u_stack = torch.cat(samples_u, dim=0)
        v_stack = torch.cat(samples_v, dim=0)
        p_stack = torch.cat(samples_p, dim=0)

        return {
            "u_mean": u_stack.mean(dim=0),
            "v_mean": v_stack.mean(dim=0),
            "p_mean": p_stack.mean(dim=0),
            "u_std": u_stack.std(dim=0, unbiased=False),
            "v_std": v_stack.std(dim=0, unbiased=False),
            "p_std": p_stack.std(dim=0, unbiased=False),
            "u_samples": u_stack,
            "v_samples": v_stack,
            "p_samples": p_stack,
        }


def build_model(method_cfg, layer_mat):
    if method_cfg.get("model_type") == "fourier_psi":
        return FourierFeaturePINN(
            layer_mat,
            dropout_rate=method_cfg.get("dropout_rate", 0.0),
            fourier_features=method_cfg.get("fourier_features", 128),
            sigma=method_cfg.get("fourier_sigma", 3.0),
        )

    return PINN_Net(
        layer_mat,
        dropout_rate=method_cfg.get("dropout_rate", 0.0),
    )


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def predict_uvp_tensor(model, x, y, t):
    """Predict u, v, p for either direct-output or psi-p models."""

    if hasattr(model, "predict_fields"):
        return model.predict_fields(
            x,
            y,
            t,
            create_graph=True,
        )

    if not x.requires_grad:
        x.requires_grad_(True)

    if not y.requires_grad:
        y.requires_grad_(True)

    if not t.requires_grad:
        t.requires_grad_(True)

    out = model.forward(x, y, t)

    if out.shape[1] >= 3:
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]

    psi = out[:, 0:1]
    p = out[:, 1:2]

    u = torch.autograd.grad(
        psi.sum(),
        y,
        create_graph=True,
        retain_graph=True,
    )[0]

    v = -torch.autograd.grad(
        psi.sum(),
        x,
        create_graph=True,
        retain_graph=True,
    )[0]

    return u, v, p


def predict_uvp_numpy(model, x_np, y_np, t_np, device, batch_size=20000, eval_mode=True):
    if eval_mode:
        model.eval()

    preds = {"u": [], "v": [], "p": []}
    start = time.perf_counter()

    for i in range(0, x_np.shape[0], batch_size):
        xb = torch.tensor(
            x_np[i:i + batch_size],
            dtype=torch.float32,
            device=device,
        ).requires_grad_(True)

        yb = torch.tensor(
            y_np[i:i + batch_size],
            dtype=torch.float32,
            device=device,
        ).requires_grad_(True)

        tb = torch.tensor(
            t_np[i:i + batch_size],
            dtype=torch.float32,
            device=device,
        ).requires_grad_(True)

        u, v, p = predict_uvp_tensor(model, xb, yb, tb)

        preds["u"].append(u.detach().cpu().numpy())
        preds["v"].append(v.detach().cpu().numpy())
        preds["p"].append(p.detach().cpu().numpy())

        del xb, yb, tb, u, v, p

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    return {
        k: np.concatenate(v, axis=0)
        for k, v in preds.items()
    }, elapsed


def load_trained_model(method_cfg, layer_mat, device):
    model = build_model(method_cfg, layer_mat).to(device)
    checkpoint = Path(method_cfg["checkpoint"])

    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

    model.load_state_dict(safe_load_state(checkpoint, device))
    model.eval()

    return model


def make_wake_region_masks(mesh_x, mesh_y):
    """Heuristic cylinder-wake regions for local error reporting."""

    x = mesh_x
    y = mesh_y

    yc = float(np.median(y))
    yr = float(np.max(np.abs(y - yc))) + 1e-12

    x_min = float(x.min())
    x_max = float(x.max())
    length = x_max - x_min + 1e-12

    x_rel = (x - x_min) / length
    y_abs = np.abs(y - yc) / yr

    masks = {
        "separation_zone": (
            (x_rel >= 0.05)
            & (x_rel < 0.25)
            & (y_abs < 0.45)
        ),
        "shear_layer": (
            (x_rel >= 0.10)
            & (x_rel < 0.60)
            & (y_abs >= 0.25)
            & (y_abs < 0.70)
        ),
        "vortex_core": (
            (x_rel >= 0.25)
            & (x_rel < 0.55)
            & (y_abs < 0.35)
        ),
        "near_wake": (
            (x_rel >= 0.20)
            & (x_rel < 0.55)
            & (y_abs < 0.60)
        ),
        "far_wake": (
            (x_rel >= 0.55)
            & (x_rel <= 1.00)
            & (y_abs < 0.70)
        ),
    }

    return masks


def metric_dict(pred, true):
    err = pred - true
    abs_err = np.abs(err)

    rel_l2 = np.linalg.norm(err.ravel()) / (
        np.linalg.norm(true.ravel()) + 1e-12
    )

    return {
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "max_abs_error": float(abs_err.max()),
        "relative_l2": float(rel_l2),
    }
