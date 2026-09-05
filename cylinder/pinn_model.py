"""Core PINN model and data utilities for Re=3900 cylinder-flow experiments."""

import numpy as np
import scipy.io
import torch
import torch.nn as nn


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if device.type == "cpu":
    print("Warning: CUDA is not available, running on CPU.")


class PINN_Net(nn.Module):
    """Psi-p PINN.

    Input:
        x, y, t

    Output:
        psi, p

    Velocity is recovered by:
        u = d psi / d y
        v = - d psi / d x
    """

    def __init__(self, layer_mat, dropout_rate=0.0):
        super().__init__()

        self.layer_mat = list(layer_mat)
        self.dropout_rate = float(dropout_rate)

        layers = []

        for i in range(len(self.layer_mat) - 2):
            layers.append(nn.Linear(self.layer_mat[i], self.layer_mat[i + 1]))
            layers.append(nn.Tanh())

            if self.dropout_rate > 0.0:
                layers.append(nn.Dropout(p=self.dropout_rate))

        layers.append(nn.Linear(self.layer_mat[-2], self.layer_mat[-1]))

        self.base = nn.Sequential(*layers)

        self.init_weights()

    def init_weights(self):
        for module in self.base:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x, y, t):
        inputs = torch.cat([x, y, t], dim=1)
        return self.base(inputs)

    def predict_fields(self, x, y, t, create_graph=True):
        """Predict u, v and p from psi-p outputs.

        Important:
            x and y must require gradients because:
                u = d psi / d y
                v = - d psi / d x

        During training, the training script should set:
            x.requires_grad_(True)
            y.requires_grad_(True)
            t.requires_grad_(True)

        During evaluation, use predict_fields_safe().
        """

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
        """Safely predict u, v, p for evaluation or diagnostics.

        This function is necessary for the psi-p formulation because u and v
        are computed by autograd. Therefore, gradients must be enabled even
        during evaluation.

        Args:
            x, y, t:
                Input tensors.
            create_graph:
                Whether to create higher-order graph.
                For normal evaluation, use False.
            train_mode:
                If True, keep dropout active.
                If False, use deterministic evaluation mode.

        Returns:
            u_pred, v_pred, p_pred as detached tensors.
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
        """Supervised data loss for u, v and p."""

        u_pred, v_pred, p_pred = self.predict_fields(
            x,
            y,
            t,
            create_graph=True,
        )

        mse = nn.MSELoss()

        loss_u = mse(u_pred, u)
        loss_v = mse(v_pred, v)
        loss_p = mse(p_pred, p)

        return loss_u + loss_v + loss_p

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

        residual_x = (
            u_t
            + u * u_x
            + v * u_y
            + p_x
            - (1.0 / Re) * (u_xx + u_yy)
        )

        residual_y = (
            v_t
            + u * v_x
            + v * v_y
            + p_y
            - (1.0 / Re) * (v_xx + v_yy)
        )

        zeros_x = torch.zeros_like(residual_x)
        zeros_y = torch.zeros_like(residual_y)

        mse = nn.MSELoss()

        return mse(residual_x, zeros_x) + mse(residual_y, zeros_y)

    def predict_with_uncertainty(self, x, y, t, samples=30):
        """MC-dropout prediction for uncertainty estimation.

        Important:
            For the psi-p formulation, u and v are obtained through autograd:
                u = d psi / d y
                v = - d psi / d x

            Therefore, gradients must be enabled even during evaluation.

        This function:
            1. keeps dropout active by using model.train();
            2. clones x/y/t and sets requires_grad=True;
            3. explicitly enables torch.enable_grad();
            4. returns detached MC samples, mean, and std.
        """

        was_training = self.training

        # Activate dropout for MC sampling.
        self.train()

        x_mc = x.detach().clone().requires_grad_(True)
        y_mc = y.detach().clone().requires_grad_(True)
        t_mc = t.detach().clone().requires_grad_(True)

        u_samples = []
        v_samples = []
        p_samples = []

        with torch.enable_grad():
            for _ in range(samples):
                u_pred, v_pred, p_pred = self.predict_fields(
                    x_mc,
                    y_mc,
                    t_mc,
                    create_graph=False,
                )

                u_samples.append(u_pred.detach().unsqueeze(0))
                v_samples.append(v_pred.detach().unsqueeze(0))
                p_samples.append(p_pred.detach().unsqueeze(0))

        if was_training:
            self.train()
        else:
            self.eval()

        u_stack = torch.cat(u_samples, dim=0)
        v_stack = torch.cat(v_samples, dim=0)
        p_stack = torch.cat(p_samples, dim=0)

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


def f_equation_inverse(model, x, y, t, Re):
    """Return Navier-Stokes residuals fx and fy.

    This function is kept for compatibility with older scripts.
    """

    u, v, p = model.predict_fields(
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

    return fx, fy


def _to_column_tensor(array):
    array = np.asarray(array)
    array = array.reshape(-1, 1)
    return torch.tensor(array, dtype=torch.float32)


def _find_first_existing_key(mat_data, candidate_keys):
    for key in candidate_keys:
        if key in mat_data:
            return key

    available_keys = [key for key in mat_data.keys() if not key.startswith("__")]

    raise KeyError(
        f"None of keys {candidate_keys} found in mat file. "
        f"Available keys: {available_keys}"
    )


def _extract_uv_from_u_star(U_star, n_space, n_time):
    """Extract u and v from U_star.

    Returned:
        u_mat: shape [n_space, n_time]
        v_mat: shape [n_space, n_time]
    """

    U_star = np.asarray(U_star)

    if U_star.ndim == 3:
        if (
            U_star.shape[0] == n_space
            and U_star.shape[1] == 2
            and U_star.shape[2] == n_time
        ):
            u_mat = U_star[:, 0, :]
            v_mat = U_star[:, 1, :]
            return u_mat, v_mat

        if (
            U_star.shape[0] == n_space
            and U_star.shape[1] == n_time
            and U_star.shape[2] == 2
        ):
            u_mat = U_star[:, :, 0]
            v_mat = U_star[:, :, 1]
            return u_mat, v_mat

        if (
            U_star.shape[0] == n_time
            and U_star.shape[1] == n_space
            and U_star.shape[2] == 2
        ):
            u_mat = U_star[:, :, 0].T
            v_mat = U_star[:, :, 1].T
            return u_mat, v_mat

        if (
            U_star.shape[0] == 2
            and U_star.shape[1] == n_space
            and U_star.shape[2] == n_time
        ):
            u_mat = U_star[0, :, :]
            v_mat = U_star[1, :, :]
            return u_mat, v_mat

    if U_star.ndim == 2:
        if U_star.shape[0] == n_space * n_time and U_star.shape[1] >= 2:
            u_flat = U_star[:, 0]
            v_flat = U_star[:, 1]

            u_mat = u_flat.reshape(n_space, n_time, order="F")
            v_mat = v_flat.reshape(n_space, n_time, order="F")
            return u_mat, v_mat

        if U_star.shape[0] == n_space and U_star.shape[1] >= 2 and n_time == 1:
            u_mat = U_star[:, 0:1]
            v_mat = U_star[:, 1:2]
            return u_mat, v_mat

    raise ValueError(
        f"Unsupported U_star shape: {U_star.shape}. "
        f"Expected [N,2,Nt], [N,Nt,2], [Nt,N,2], [2,N,Nt], or [N*Nt,2]. "
        f"n_space={n_space}, n_time={n_time}"
    )


def _extract_p_from_p_star(P_star, n_space, n_time):
    """Extract p from P_star.

    Returned:
        p_mat: shape [n_space, n_time]
    """

    P_star = np.asarray(P_star)
    P_star = np.squeeze(P_star)

    if P_star.ndim == 2:
        if P_star.shape[0] == n_space and P_star.shape[1] == n_time:
            return P_star

        if P_star.shape[0] == n_time and P_star.shape[1] == n_space:
            return P_star.T

    if P_star.ndim == 1:
        if P_star.shape[0] == n_space * n_time:
            return P_star.reshape(n_space, n_time, order="F")

        if P_star.shape[0] == n_space and n_time == 1:
            return P_star.reshape(n_space, 1)

    raise ValueError(
        f"Unsupported P_star shape: {P_star.shape}. "
        f"Expected [N,Nt], [Nt,N], or [N*Nt]. "
        f"n_space={n_space}, n_time={n_time}"
    )


def _read_xstar_ustar_format(mat_data):
    """Read X_star, U_star, P_star, T_star format."""

    required_keys = ["X_star", "U_star", "P_star", "T_star"]

    for key in required_keys:
        if key not in mat_data:
            raise KeyError(f"Missing key {key} in mat file.")

    X_star = np.asarray(mat_data["X_star"])
    U_star = np.asarray(mat_data["U_star"])
    P_star = np.asarray(mat_data["P_star"])
    T_star = np.asarray(mat_data["T_star"]).reshape(-1, 1)

    if X_star.ndim != 2 or X_star.shape[1] < 2:
        raise ValueError(
            f"X_star should have shape [N, 2], but got {X_star.shape}"
        )

    x_space = X_star[:, 0:1]
    y_space = X_star[:, 1:2]

    n_space = x_space.shape[0]
    n_time = T_star.shape[0]

    u_mat, v_mat = _extract_uv_from_u_star(
        U_star=U_star,
        n_space=n_space,
        n_time=n_time,
    )

    p_mat = _extract_p_from_p_star(
        P_star=P_star,
        n_space=n_space,
        n_time=n_time,
    )

    if u_mat.shape != (n_space, n_time):
        raise ValueError(
            f"u_mat shape mismatch. Expected {(n_space, n_time)}, got {u_mat.shape}"
        )

    if v_mat.shape != (n_space, n_time):
        raise ValueError(
            f"v_mat shape mismatch. Expected {(n_space, n_time)}, got {v_mat.shape}"
        )

    if p_mat.shape != (n_space, n_time):
        raise ValueError(
            f"p_mat shape mismatch. Expected {(n_space, n_time)}, got {p_mat.shape}"
        )

    x = np.tile(x_space, (n_time, 1))
    y = np.tile(y_space, (n_time, 1))
    t = np.repeat(T_star, n_space, axis=0)

    u = u_mat.reshape(-1, 1, order="F")
    v = v_mat.reshape(-1, 1, order="F")
    p = p_mat.reshape(-1, 1, order="F")

    return x, y, t, u, v, p


def _read_direct_uvp_format(mat_data):
    """Read already flattened x, y, t, u, v, p format."""

    x_key = _find_first_existing_key(mat_data, ["x", "X", "x_star"])
    y_key = _find_first_existing_key(mat_data, ["y", "Y", "y_star"])
    t_key = _find_first_existing_key(mat_data, ["t", "T", "time", "Time"])
    u_key = _find_first_existing_key(mat_data, ["u", "U"])
    v_key = _find_first_existing_key(mat_data, ["v", "V"])
    p_key = _find_first_existing_key(mat_data, ["p", "P", "p_star"])

    x_raw = np.asarray(mat_data[x_key])
    y_raw = np.asarray(mat_data[y_key])
    t_raw = np.asarray(mat_data[t_key])
    u_raw = np.asarray(mat_data[u_key])
    v_raw = np.asarray(mat_data[v_key])
    p_raw = np.asarray(mat_data[p_key])

    x = x_raw.reshape(-1, 1)
    y = y_raw.reshape(-1, 1)
    t = t_raw.reshape(-1, 1)
    u = u_raw.reshape(-1, 1)
    v = v_raw.reshape(-1, 1)
    p = p_raw.reshape(-1, 1)

    if (
        x.shape[0] == y.shape[0]
        and x.shape[0] == t.shape[0]
        and x.shape[0] == u.shape[0]
        and x.shape[0] == v.shape[0]
        and x.shape[0] == p.shape[0]
    ):
        return x, y, t, u, v, p

    raise ValueError(
        "Direct uvp format found keys, but flattened lengths are inconsistent. "
        f"x={x.shape}, y={y.shape}, t={t.shape}, "
        f"u={u.shape}, v={v.shape}, p={p.shape}"
    )


def _read_mesh_uvp_format(mat_data):
    """Read mesh-grid X, Y, t, U, V, P format."""

    x_key = _find_first_existing_key(mat_data, ["X", "x"])
    y_key = _find_first_existing_key(mat_data, ["Y", "y"])
    t_key = _find_first_existing_key(mat_data, ["t", "T", "time", "Time"])
    u_key = _find_first_existing_key(mat_data, ["U", "u"])
    v_key = _find_first_existing_key(mat_data, ["V", "v"])
    p_key = _find_first_existing_key(mat_data, ["P", "p"])

    x_raw = np.asarray(mat_data[x_key])
    y_raw = np.asarray(mat_data[y_key])
    t_raw = np.asarray(mat_data[t_key]).reshape(-1, 1)

    u_raw = np.asarray(mat_data[u_key])
    v_raw = np.asarray(mat_data[v_key])
    p_raw = np.asarray(mat_data[p_key])

    if x_raw.shape == y_raw.shape:
        x_grid = x_raw
        y_grid = y_raw
    else:
        x_grid, y_grid = np.meshgrid(
            x_raw.reshape(-1),
            y_raw.reshape(-1),
            indexing="xy",
        )

    n_space = x_grid.size
    n_time = t_raw.shape[0]

    x = np.tile(x_grid.reshape(-1, 1), (n_time, 1))
    y = np.tile(y_grid.reshape(-1, 1), (n_time, 1))
    t = np.repeat(t_raw.reshape(-1, 1), n_space, axis=0)

    u = u_raw.reshape(-1, 1)
    v = v_raw.reshape(-1, 1)
    p = p_raw.reshape(-1, 1)

    return x, y, t, u, v, p


def read_2D_data(filename):
    """Read 2D cylinder-flow data from .mat file.

    Supported formats:
        1. X_star, U_star, P_star, T_star
        2. flattened x, y, t, u, v, p
        3. mesh-grid X, Y, t, U, V, P

    Returns:
        x, y, t, u, v, p, feature_mat
    """

    mat_data = scipy.io.loadmat(filename)

    available_keys = [key for key in mat_data.keys() if not key.startswith("__")]

    print(f"Reading data from: {filename}")
    print(f"Available mat keys: {available_keys}")

    if all(key in mat_data for key in ["X_star", "U_star", "P_star", "T_star"]):
        print("Detected format: X_star, U_star, P_star, T_star")
        x, y, t, u, v, p = _read_xstar_ustar_format(mat_data)
    else:
        try:
            print("Trying direct flattened uvp format...")
            x, y, t, u, v, p = _read_direct_uvp_format(mat_data)
        except Exception as direct_exc:
            try:
                print("Trying mesh uvp format...")
                x, y, t, u, v, p = _read_mesh_uvp_format(mat_data)
            except Exception as mesh_exc:
                raise RuntimeError(
                    f"Failed to read data file: {filename}. "
                    f"Available keys: {available_keys}. "
                    f"Direct format error: {direct_exc}. "
                    f"Mesh format error: {mesh_exc}."
                ) from mesh_exc

    min_len = min(
        x.shape[0],
        y.shape[0],
        t.shape[0],
        u.shape[0],
        v.shape[0],
        p.shape[0],
    )

    x = x[:min_len]
    y = y[:min_len]
    t = t[:min_len]
    u = u[:min_len]
    v = v[:min_len]
    p = p[:min_len]

    data_np = np.concatenate([x, y, t, u, v, p], axis=1)

    ub = np.max(data_np, axis=0, keepdims=True)
    lb = np.min(data_np, axis=0, keepdims=True)

    feature_mat = torch.tensor(
        np.concatenate([ub, lb], axis=0),
        dtype=torch.float32,
    )

    x = _to_column_tensor(x)
    y = _to_column_tensor(y)
    t = _to_column_tensor(t)
    u = _to_column_tensor(u)
    v = _to_column_tensor(v)
    p = _to_column_tensor(p)

    print("Data loaded successfully.")
    print(f"x shape: {x.shape}")
    print(f"y shape: {y.shape}")
    print(f"t shape: {t.shape}")
    print(f"u shape: {u.shape}")
    print(f"v shape: {v.shape}")
    print(f"p shape: {p.shape}")
    print(f"feature_mat shape: {feature_mat.shape}")
    print(f"x range: [{float(x.min()):.6f}, {float(x.max()):.6f}]")
    print(f"y range: [{float(y.min()):.6f}, {float(y.max()):.6f}]")
    print(f"t range: [{float(t.min()):.6f}, {float(t.max()):.6f}]")
    print(f"u range: [{float(u.min()):.6f}, {float(u.max()):.6f}]")
    print(f"v range: [{float(v.min()):.6f}, {float(v.max()):.6f}]")
    print(f"p range: [{float(p.min()):.6f}, {float(p.max()):.6f}]")

    return x, y, t, u, v, p, feature_mat


def shuffle_data(x, y, t, u, v, p):
    """Concatenate and shuffle supervised data."""

    data = torch.cat([x, y, t, u, v, p], dim=1)
    permutation = torch.randperm(data.shape[0])
    return data[permutation]


def generate_eqp_rect(lb, ub, dimension=3, points=100000):
    """Generate random collocation points in a rectangular domain."""

    lb = np.asarray(lb).reshape(1, -1)
    ub = np.asarray(ub).reshape(1, -1)

    if lb.shape[1] < dimension or ub.shape[1] < dimension:
        raise ValueError(
            f"lb and ub must have at least {dimension} dimensions."
        )

    lb = lb[:, :dimension]
    ub = ub[:, :dimension]

    random_points = np.random.rand(points, dimension)
    equation_points = lb + (ub - lb) * random_points

    return torch.tensor(equation_points, dtype=torch.float32)


def relative_l2_error(pred, ref):
    """Relative L2 error helper."""

    pred = np.asarray(pred)
    ref = np.asarray(ref)

    return np.linalg.norm(pred.reshape(-1) - ref.reshape(-1)) / (
        np.linalg.norm(ref.reshape(-1)) + 1e-12
    )


def count_parameters(model):
    """Count trainable parameters."""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def print_field_statistics(name, u_pred, v_pred, p_pred, u_ref=None, v_ref=None, p_ref=None):
    """Print min/max/mean/std statistics of predicted and reference fields.

    This function is useful for diagnosing whether B-PINN has learned the field.
    For example, if v_ref has std around 0.3 but v_pred has std around 1e-7,
    then the model has not learned the v field or the evaluation derivative is wrong.
    """

    def _stat(arr):
        if torch.is_tensor(arr):
            arr = arr.detach().cpu().numpy()

        arr = np.asarray(arr).reshape(-1)

        return (
            float(np.min(arr)),
            float(np.max(arr)),
            float(np.mean(arr)),
            float(np.std(arr)),
        )

    print(f"\n[{name}] Field statistics")

    u_min, u_max, u_mean, u_std = _stat(u_pred)
    v_min, v_max, v_mean, v_std = _stat(v_pred)
    p_min, p_max, p_mean, p_std = _stat(p_pred)

    print(
        f"u_pred: min={u_min:.6e}, max={u_max:.6e}, "
        f"mean={u_mean:.6e}, std={u_std:.6e}"
    )
    print(
        f"v_pred: min={v_min:.6e}, max={v_max:.6e}, "
        f"mean={v_mean:.6e}, std={v_std:.6e}"
    )
    print(
        f"p_pred: min={p_min:.6e}, max={p_max:.6e}, "
        f"mean={p_mean:.6e}, std={p_std:.6e}"
    )

    if u_ref is not None:
        u_min, u_max, u_mean, u_std = _stat(u_ref)
        print(
            f"u_ref : min={u_min:.6e}, max={u_max:.6e}, "
            f"mean={u_mean:.6e}, std={u_std:.6e}"
        )

    if v_ref is not None:
        v_min, v_max, v_mean, v_std = _stat(v_ref)
        print(
            f"v_ref : min={v_min:.6e}, max={v_max:.6e}, "
            f"mean={v_mean:.6e}, std={v_std:.6e}"
        )

    if p_ref is not None:
        p_min, p_max, p_mean, p_std = _stat(p_ref)
        print(
            f"p_ref : min={p_min:.6e}, max={p_max:.6e}, "
            f"mean={p_mean:.6e}, std={p_std:.6e}"
        )

