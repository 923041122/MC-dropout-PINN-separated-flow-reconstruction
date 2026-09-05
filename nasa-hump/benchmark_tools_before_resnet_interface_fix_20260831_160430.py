
"""Steady 2-D model tools for the NASA wall-mounted hump revision.

Network:
    input : x, y
    output: psi, p
    u = d psi / d y
    v = - d psi / d x

The redundant constant pseudo-time coordinate used in the earlier implementation
has been removed.
"""

from __future__ import annotations

import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn


def get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def safe_load_state(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class PINN_Net(nn.Module):
    """Steady 2-D psi-p PINN."""

    def __init__(self, layer_mat, dropout_rate: float = 0.0):
        super().__init__()
        self.layer_mat = list(layer_mat)
        self.dropout_rate = float(dropout_rate)

        if self.layer_mat[0] != 2:
            raise ValueError(
                f"NASA hump steady model requires exactly two inputs (x,y), "
                f"got {self.layer_mat[0]}."
            )
        if self.layer_mat[-1] != 2:
            raise ValueError("psi-p model requires exactly two outputs (psi,p).")

        layers = []
        for i in range(len(self.layer_mat) - 2):
            layers.append(nn.Linear(self.layer_mat[i], self.layer_mat[i + 1]))
            layers.append(nn.Tanh())
            if self.dropout_rate > 0.0:
                layers.append(nn.Dropout(p=self.dropout_rate))
        layers.append(nn.Linear(self.layer_mat[-2], self.layer_mat[-1]))

        self.base = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.base:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x, y):
        return self.base(torch.cat([x, y], dim=1))

    def predict_fields(self, x, y, create_graph: bool = True):
        if not x.requires_grad:
            x.requires_grad_(True)
        if not y.requires_grad:
            y.requires_grad_(True)

        out = self.forward(x, y)
        psi = out[:, 0:1]
        p = out[:, 1:2]

        u = torch.autograd.grad(
            psi.sum(), y, create_graph=create_graph, retain_graph=True
        )[0]
        v = -torch.autograd.grad(
            psi.sum(), x, create_graph=create_graph, retain_graph=True
        )[0]
        return u, v, p

    def predict_fields_safe(
        self,
        x,
        y,
        create_graph: bool = False,
        train_mode: bool = False,
    ):
        was_training = self.training
        self.train(mode=train_mode)

        x_eval = x.detach().clone().requires_grad_(True)
        y_eval = y.detach().clone().requires_grad_(True)

        with torch.enable_grad():
            u_pred, v_pred, p_pred = self.predict_fields(
                x_eval, y_eval, create_graph=create_graph
            )

        self.train(mode=was_training)
        return u_pred.detach(), v_pred.detach(), p_pred.detach()



class FourierFeaturePINN(PINN_Net):
    """Steady 2-D psi-p PINN with fixed random Fourier input features.

    Input:
        x, y

    Fourier mapping:
        B ~ N(0, sigma^2), fixed as a non-trainable buffer
        z = [sin(2*pi*[x,y]B), cos(2*pi*[x,y]B)]

    Output:
        psi, p

    Velocity is inherited from PINN_Net:
        u = d psi / d y
        v = - d psi / d x
    """

    def __init__(
        self,
        layer_mat,
        dropout_rate: float = 0.0,
        fourier_features: int = 128,
        sigma: float = 1.0,
    ):
        # Do not call PINN_Net.__init__ because that class requires
        # layer_mat[0] == 2, whereas the Fourier MLP receives 2*M features.
        nn.Module.__init__(self)

        self.layer_mat = list(layer_mat)
        self.dropout_rate = float(dropout_rate)
        self.fourier_features = int(fourier_features)
        self.sigma = float(sigma)

        if self.layer_mat[0] != 2:
            raise ValueError(
                "NASA Fourier model requires exactly two physical inputs (x,y)."
            )
        if self.layer_mat[-1] != 2:
            raise ValueError(
                "NASA Fourier psi-p model requires exactly two outputs (psi,p)."
            )
        if self.fourier_features <= 0:
            raise ValueError("fourier_features must be positive.")

        # Fixed random Fourier projection.
        # NASA is steady 2-D, hence B has shape (2, M).
        self.register_buffer(
            "B",
            torch.randn(2, self.fourier_features) * self.sigma,
        )

        transformed_layers = list(self.layer_mat)
        transformed_layers[0] = 2 * self.fourier_features

        layers = []
        for i in range(len(transformed_layers) - 2):
            layers.append(
                nn.Linear(transformed_layers[i], transformed_layers[i + 1])
            )
            layers.append(nn.Tanh())
            if self.dropout_rate > 0.0:
                layers.append(nn.Dropout(p=self.dropout_rate))

        layers.append(
            nn.Linear(transformed_layers[-2], transformed_layers[-1])
        )

        self.base = nn.Sequential(*layers)

        # Reuse the same Xavier initialization as the standard NASA PINN.
        self._init_weights()

    def forward(self, x, y):
        coords = torch.cat([x, y], dim=1)
        proj = 2.0 * torch.pi * (coords @ self.B)
        features = torch.cat(
            [torch.sin(proj), torch.cos(proj)],
            dim=1,
        )
        return self.base(features)




class SineLayer(nn.Module):
    """Linear layer followed by sine activation for SIREN."""

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        is_first=False,
        omega_0=30.0,
    ):
        super().__init__()

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.is_first = bool(is_first)
        self.omega_0 = float(omega_0)

        self.linear = nn.Linear(
            self.in_features,
            self.out_features,
            bias=bias,
        )

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():

            if self.is_first:
                bound = 1.0 / self.in_features

            else:
                bound = (
                    np.sqrt(6.0 / self.in_features)
                    / self.omega_0
                )

            self.linear.weight.uniform_(
                -bound,
                bound,
            )

            if self.linear.bias is not None:
                self.linear.bias.uniform_(
                    -bound,
                    bound,
                )

    def forward(self, x):
        return torch.sin(
            self.omega_0 * self.linear(x)
        )


class SirenPINN_Net(PINN_Net):
    """Steady 2-D NASA hump SIREN psi-p PINN.

    Physical input:
        x, y

    Network output:
        psi, p

    Velocity definition is inherited from PINN_Net:
        u = d psi / d y
        v = - d psi / d x

    The physical formulation is therefore unchanged.
    Only the hidden representation and initialization
    differ from the standard Tanh/Xavier PINN.
    """

    def __init__(
        self,
        layer_mat,
        first_omega_0=30.0,
        hidden_omega_0=1.0,
    ):
        # Do not call PINN_Net.__init__, because we construct
        # the SIREN hidden representation explicitly.
        nn.Module.__init__(self)

        self.layer_mat = list(layer_mat)
        self.dropout_rate = 0.0

        self.first_omega_0 = float(first_omega_0)
        self.hidden_omega_0 = float(hidden_omega_0)

        if self.layer_mat[0] != 2:
            raise ValueError(
                "NASA hump SIREN requires exactly "
                "two physical inputs (x,y)."
            )

        if self.layer_mat[-1] != 2:
            raise ValueError(
                "NASA hump SIREN psi-p model requires "
                "exactly two outputs (psi,p)."
            )

        layers = []

        # First hidden layer
        layers.append(
            SineLayer(
                self.layer_mat[0],
                self.layer_mat[1],
                is_first=True,
                omega_0=self.first_omega_0,
            )
        )

        # Remaining hidden layers
        for i in range(
            1,
            len(self.layer_mat) - 2,
        ):
            layers.append(
                SineLayer(
                    self.layer_mat[i],
                    self.layer_mat[i + 1],
                    is_first=False,
                    omega_0=self.hidden_omega_0,
                )
            )

        # Final linear layer:
        # no sine activation on psi,p output
        final_linear = nn.Linear(
            self.layer_mat[-2],
            self.layer_mat[-1],
        )

        with torch.no_grad():

            bound = (
                np.sqrt(
                    6.0 / self.layer_mat[-2]
                )
                / self.hidden_omega_0
            )

            final_linear.weight.uniform_(
                -bound,
                bound,
            )

            if final_linear.bias is not None:
                final_linear.bias.uniform_(
                    -bound,
                    bound,
                )

        layers.append(final_linear)

        self.base = nn.Sequential(*layers)

    def forward(self, x, y):
        coords = torch.cat(
            [x, y],
            dim=1,
        )
        return self.base(coords)



class ResNetPINN_Net(nn.Module):
    """Steady NASA hump residual/ResNet-type psi-p PINN."""

    def __init__(self, layer_mat):
        super().__init__()

        self.layer_mat = list(layer_mat)
        self.dropout_rate = 0.0

        hidden = self.layer_mat[1:-1]

        if self.layer_mat[0] != 2:
            raise ValueError("ResNet NASA model requires 2 inputs.")

        if self.layer_mat[-1] != 2:
            raise ValueError("ResNet NASA model requires 2 outputs.")

        if len(hidden) != 10 or len(set(hidden)) != 1:
            raise ValueError(
                "Expected locked NASA backbone: 2 -> 10x100 -> 2."
            )

        width = hidden[0]

        self.stem = nn.Linear(2, width)

        self.blocks = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(width, width),
                nn.Linear(width, width),
            ])
            for _ in range(4)
        ])

        self.tail = nn.Linear(width, width)
        self.output_layer = nn.Linear(width, 2)

        gain = nn.init.calculate_gain("tanh")

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, y):
        coords = torch.cat([x, y], dim=1)

        h = torch.tanh(self.stem(coords))

        for linear1, linear2 in self.blocks:
            residual = h
            z = torch.tanh(linear1(h))
            z = linear2(z)
            h = torch.tanh(residual + z)

        h = torch.tanh(self.tail(h))

        return self.output_layer(h)


def build_model(cfg: Dict, layer_mat):
    model_type = cfg.get("model_type", "psi")
    dropout_rate = float(cfg.get("dropout_rate", 0.0))

    if model_type == "fourier_psi":
        return FourierFeaturePINN(
            layer_mat=layer_mat,
            dropout_rate=dropout_rate,
            fourier_features=int(cfg.get("fourier_features", 128)),
            sigma=float(cfg.get("fourier_sigma", 1.0)),
        )

    if model_type == "resnet_psi":
        return ResNetPINN_Net(
            layer_mat=layer_mat,
        )

    if model_type == "siren_psi":
        return SirenPINN_Net(
            layer_mat=layer_mat,
            first_omega_0=float(
                cfg.get("first_omega_0", 30.0)
            ),
            hidden_omega_0=float(
                cfg.get("hidden_omega_0", 1.0)
            ),
        )

    if model_type != "psi":
        raise ValueError(
            f"Unsupported model_type={model_type!r}. "
            "Expected 'psi', 'fourier_psi', or 'siren_psi'."
        )

    return PINN_Net(layer_mat=layer_mat, dropout_rate=dropout_rate)


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def predict_uvp_numpy(
    model: nn.Module,
    x_np: np.ndarray,
    y_np: np.ndarray,
    device: torch.device,
    batch_size: int = 20000,
    eval_mode: bool = True,
) -> Tuple[Dict[str, np.ndarray], float]:
    """Batched prediction for the steady (x,y)->(u,v,p) model."""
    x_np = np.asarray(x_np, dtype=np.float32).reshape(-1, 1)
    y_np = np.asarray(y_np, dtype=np.float32).reshape(-1, 1)

    if x_np.shape[0] != y_np.shape[0]:
        raise ValueError("x_np and y_np must contain the same number of points.")

    n = x_np.shape[0]
    all_u, all_v, all_p = [], [], []

    was_training = model.training
    model.train(mode=not eval_mode)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()

    for start_idx in range(0, n, batch_size):
        end_idx = min(start_idx + batch_size, n)

        x = torch.tensor(
            x_np[start_idx:end_idx],
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        y = torch.tensor(
            y_np[start_idx:end_idx],
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )

        with torch.enable_grad():
            u, v, p = model.predict_fields(x, y, create_graph=False)

        all_u.append(_to_numpy(u))
        all_v.append(_to_numpy(v))
        all_p.append(_to_numpy(p))

        del x, y, u, v, p

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    model.train(mode=was_training)

    return {
        "u": np.concatenate(all_u, axis=0),
        "v": np.concatenate(all_v, axis=0),
        "p": np.concatenate(all_p, axis=0),
    }, elapsed
