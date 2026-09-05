"""
Bayesian PINN evaluation script that visualizes predictive mean and uncertainty
for u/v/p fields at a selected time snapshot.
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

from pinn_model import PINN_Net
from read_data import read_2D_data

mpl.use("Agg")


def prepare_grid(data_stack, time_index):
    x = data_stack[:, 0].copy().reshape(-1, 1)
    y = data_stack[:, 1].copy().reshape(-1, 1)
    t = data_stack[:, 2].copy().reshape(-1, 1)
    u = data_stack[:, 3].copy().reshape(-1, 1)
    v = data_stack[:, 4].copy().reshape(-1, 1)
    p = data_stack[:, 5].copy().reshape(-1, 1)

    unique_x = np.unique(x).reshape(-1, 1)
    unique_y = np.unique(y).reshape(-1, 1)
    t_unique = np.unique(t).reshape(-1, 1)

    select_time = t_unique[time_index, 0]
    index_time = np.where(t == select_time)[0]
    mesh_x, mesh_y = np.meshgrid(unique_x, unique_y)
    x_flat = mesh_x.reshape(-1, 1)
    y_flat = mesh_y.reshape(-1, 1)
    t_flat = np.ones_like(x_flat) * select_time

    u_true = u[index_time].reshape(mesh_x.shape)
    v_true = v[index_time].reshape(mesh_x.shape)
    p_true = p[index_time].reshape(mesh_x.shape)
    return mesh_x, mesh_y, x_flat, y_flat, t_flat, select_time, (u_true, v_true, p_true)


def load_model(layer_mat, dropout_rate, model_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    layer_mat = list(layer_mat) if isinstance(layer_mat, tuple) else layer_mat
    pinn = PINN_Net(layer_mat, dropout_rate=dropout_rate)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model weights not found at {model_path}")
    # 新增 weights_only=True 解决安全警告
    pinn.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    pinn = pinn.to(device)
    pinn.eval()
    return pinn, device


def tensorize(device, *arrays):
    tensors = []
    for arr in arrays:
        tensor = torch.tensor(arr, dtype=torch.float32, device=device)
        tensor.requires_grad_(True)
        tensors.append(tensor)
    return tensors


def plot_mean_and_uncertainty(field_true, field_mean, field_std, select_time, quantity, save_dir):
    v_norm = mpl.colors.Normalize(vmin=field_true.min(), vmax=field_true.max())
    std_norm = mpl.colors.Normalize(vmin=0.0, vmax=np.percentile(field_std, 99))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(field_true, cmap="jet", norm=v_norm)
    axes[0].set_title(f"True {quantity} @ t={select_time:.2f}")
    axes[1].imshow(field_mean, cmap="jet", norm=v_norm)
    axes[1].set_title(f"Predict mean {quantity}")
    axes[2].imshow(field_std, cmap="magma", norm=std_norm)
    axes[2].set_title(f"Predict std {quantity}")
    for ax in axes:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
    fig.colorbar(axes[0].images[0], ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(axes[1].images[0], ax=axes[1], fraction=0.046, pad=0.04)
    fig.colorbar(axes[2].images[0], ax=axes[2], fraction=0.046, pad=0.04)
    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, f"{quantity}_uncertainty_t_{select_time:.2f}.png"), dpi=200)
    plt.close(fig)


def run_uncertainty_demo(
        data_path="./2d_cylinder_Re3900_100x100_kw_sst.mat",
        # 关键修改：将默认模型路径改为训练生成的 pinn_model.pth
        model_path="./training_results/pinn_model.pth",
        layer_mat=(3, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 2),
        dropout_rate=0.05,
        time_indices=None,  # 新增：支持传入多个时间索引
        samples=30,
        save_dir="./uncertainty_plots",
):
    # 读取数据
    x, y, t, u, v, p, _ = read_2D_data(data_path)
    data_stack = np.concatenate((x, y, t, u, v, p), axis=1)

    # 获取所有唯一时刻的索引（默认分析全部时刻）
    t_unique = np.unique(data_stack[:, 2]).reshape(-1, 1)
    if time_indices is None:
        time_indices = range(len(t_unique))  # 遍历所有时刻
    else:
        # 过滤超出范围的索引
        time_indices = [idx for idx in time_indices if idx < len(t_unique)]

    # 加载模型（只加载一次，避免重复加载）
    pinn, device = load_model(layer_mat, dropout_rate, model_path)

    # 遍历每个时刻生成不确定性图像
    for time_index in time_indices:
        print(f"📊 正在处理时刻索引: {time_index}")
        # 准备当前时刻的网格和真实数据
        mesh_x, mesh_y, x_flat, y_flat, t_flat, select_time, truths = prepare_grid(data_stack, time_index)

        # 转换为张量并执行贝叶斯预测（蒙特卡洛采样）
        x_t, y_t, t_t = tensorize(device, x_flat, y_flat, t_flat)
        stats = pinn.predict_with_uncertainty(x_t, y_t, t_t, samples=samples)


        # 处理预测结果（先detach分离梯度，再转CPU和NumPy）
        u_mean = stats["u_mean"].detach().cpu().numpy().reshape(mesh_x.shape)
        v_mean = stats["v_mean"].detach().cpu().numpy().reshape(mesh_x.shape)
        p_mean = stats["p_mean"].detach().cpu().numpy().reshape(mesh_x.shape)
        u_std = stats["u_std"].detach().cpu().numpy().reshape(mesh_x.shape)
        v_std = stats["v_std"].detach().cpu().numpy().reshape(mesh_x.shape)
        p_std = stats["p_std"].detach().cpu().numpy().reshape(mesh_x.shape)

        # 生成并保存当前时刻的不确定性图像（u/v/p三个物理量）
        plot_mean_and_uncertainty(truths[0], u_mean, u_std, select_time, "u", save_dir)
        plot_mean_and_uncertainty(truths[1], v_mean, v_std, select_time, "v", save_dir)
        plot_mean_and_uncertainty(truths[2], p_mean, p_std, select_time, "p", save_dir)

        # 清理内存（可选，避免GPU占用显存溢出）
        del x_t, y_t, t_t, stats
        torch.cuda.empty_cache()

    print(f"✅ 所有不确定性图像已保存至: {save_dir}")


if __name__ == "__main__":
    run_uncertainty_demo()

