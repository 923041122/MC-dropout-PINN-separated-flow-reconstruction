import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch
from pinn_model import *
import imageio
import os
from pathlib import Path
import warnings
import pandas as pd

warnings.filterwarnings('ignore')

# -------------------------- 全局配置 --------------------------
# 设置matplotlib后端（非交互式，专注文件保存）
mpl.use("Agg")

# 配置matplotlib参数，使用内置默认字体（彻底解决字体缺失问题）
mpl.rcParams.update({
    'font.size': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'figure.figsize': (12, 5),
    'font.family': 'sans-serif',  # 直接使用系统默认无衬线字体，无需指定具体字体名
    'font.sans-serif': ['DejaVu Sans', 'SimHei', 'Arial', 'Helvetica', 'Verdana', 'sans-serif'],  # 优先级 fallback
    'axes.linewidth': 0.8,
    'xtick.direction': 'in',
    'ytick.direction': 'in'
})

# -------------------------- GPU设备配置 --------------------------
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f"✅ 检测到GPU: {torch.cuda.get_device_name(0)}")
    print(f"📊 GPU数量: {torch.cuda.device_count()}")
    print(f"💾 GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    device = torch.device("cpu")
    print("⚠️  未检测到GPU，使用CPU训练（速度较慢）")

# -------------------------- 文件路径配置 --------------------------
MODEL_PATH = Path('./training_results/pinn_model.pth')
DATA_PATH = Path('./2d_cylinder_Re3900_100x100_kw_sst.mat')
OUTPUT_ROOT = Path('./prediction_comparison_results_GPU')

# 自动创建保存目录
dirs_to_create = [
    OUTPUT_ROOT / 'single_time_comparison',
    OUTPUT_ROOT / 'time_series_comparison',
    OUTPUT_ROOT / 'error_maps',
    OUTPUT_ROOT / 'gifs',
    OUTPUT_ROOT / 'csv_results'
]
for dir_path in dirs_to_create:
    dir_path.mkdir(exist_ok=True, parents=True)

# -------------------------- 物理参数与网络结构 --------------------------
U_ref = 0.1306
L_ref = 0.03
rou_ref = 998.2

# 与训练时一致的网络结构
layer_mat_1 = [3, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 2]
layer_mat_2 = [3, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 5]
layer_mat_3 = [3, 50, 50, 50, 50, 50, 5]
layer_mat_4 = [3, 20, 20, 20, 20, 20, 5]

# -------------------------- 加载模型 --------------------------
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"❌ 模型文件未找到！\n"
        f"当前配置的模型路径：{MODEL_PATH.absolute()}\n"
        f"解决方案：\n"
        f"1. 先运行 train_uv_modify.py 训练模型（会生成 pinn_model.pth）\n"
        f"2. 若已训练，检查 MODEL_PATH 是否指向训练生成的模型文件路径"
    )

pinn_net = PINN_Net(layer_mat_1)
checkpoint = torch.load(MODEL_PATH, map_location=device)
pinn_net.load_state_dict(checkpoint)
pinn_net.to(device)
pinn_net.eval()
print(f"📥 模型已加载至 {device}，路径: {MODEL_PATH.absolute()}")

# -------------------------- 加载数据 --------------------------
x, y, t, u, v, p, feature_mat = read_2D_data(str(DATA_PATH))
data_stack = np.concatenate((x, y, t, u, v, p), axis=1)
del x, y, t, u, v, p  # 释放CPU内存
print(f"📊 数据已加载完成，数据形状: {data_stack.shape}")

# -------------------------- 核心可视化函数 --------------------------
def plot_compare(q_true, q_pred, select_time, var_name, min_val, max_val, save_path):
    """真实值vs预测值+误差图（三列布局）"""
    mask = np.abs(q_true) < 1e-6
    q_error = np.zeros_like(q_true)
    q_error[~mask] = np.abs(q_pred[~mask] - q_true[~mask]) / np.abs(q_true[~mask])
    q_error[mask] = np.abs(q_pred[mask] - q_true[mask])
    error_max = min(np.percentile(q_error, 99), 0.2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'{var_name}-Component Comparison at t = {select_time:.2f} (GPU Inference)',
                 fontsize=14, fontweight='bold', y=0.95)
    norm_data = mpl.colors.Normalize(vmin=min_val, vmax=max_val)
    norm_error = mpl.colors.Normalize(vmin=0, vmax=error_max)

    # 真实值
    im1 = axes[0].imshow(q_true, cmap='jet', norm=norm_data, aspect='auto')
    axes[0].set_title('True Value', fontweight='bold')
    axes[0].set_xlabel('X Grid')
    axes[0].set_ylabel('Y Grid')
    cbar1 = plt.colorbar(im1, ax=axes[0], shrink=0.8, pad=0.02)
    cbar1.set_label(f'{var_name} Value', rotation=270, labelpad=15)

    # 预测值
    im2 = axes[1].imshow(q_pred, cmap='jet', norm=norm_data, aspect='auto')
    axes[1].set_title('Predicted Value', fontweight='bold')
    axes[1].set_xlabel('X Grid')
    axes[1].set_ylabel('Y Grid')
    cbar2 = plt.colorbar(im2, ax=axes[1], shrink=0.8, pad=0.02)
    cbar2.set_label(f'{var_name} Value', rotation=270, labelpad=15)

    # 误差图
    im3 = axes[2].imshow(q_error, cmap='Reds', norm=norm_error, aspect='auto')
    axes[2].set_title('Relative Error', fontweight='bold')
    axes[2].set_xlabel('X Grid')
    axes[2].set_ylabel('Y Grid')
    cbar3 = plt.colorbar(im3, ax=axes[2], shrink=0.8, pad=0.02)
    cbar3.set_label('Error', rotation=270, labelpad=15)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88, wspace=0.3)

    # 保存双格式
    save_path_png = save_path.with_suffix('.png')
    save_path_pdf = save_path.with_suffix('.pdf')
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path_pdf, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"📸 对比图保存: {save_path_png}")
    print(f"📄 高分辨率图保存: {save_path_pdf}")

    # 误差统计
    return {
        'time': select_time,
        'variable': var_name,
        'mean_error': np.mean(q_error),
        'max_error': np.max(q_error),
        'std_error': np.std(q_error),
        'rmse': np.sqrt(np.mean(q_error**2))
    }

def plot_compare_time_series(q_true, q_pred, select_time, var_name, min_val, max_val, save_path):
    """时间序列对比图（GIF帧）"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'{var_name}-Component | t = {select_time:.2f} (GPU)', fontsize=12, fontweight='bold')
    norm_data = mpl.colors.Normalize(vmin=min_val, vmax=max_val)

    # 真实值
    im1 = axes[0].imshow(q_true, cmap='jet', norm=norm_data, aspect='auto')
    axes[0].set_title('True', fontweight='bold')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    plt.colorbar(im1, ax=axes[0], shrink=0.7, pad=0.02)

    # 预测值
    im2 = axes[1].imshow(q_pred, cmap='jet', norm=norm_data, aspect='auto')
    axes[1].set_title('Predicted', fontweight='bold')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    plt.colorbar(im2, ax=axes[1], shrink=0.7, pad=0.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"🎞️  帧保存: {save_path}")

def plot_error_map(q_error, select_time, var_name, save_path):
    """单独误差分布图"""
    error_max = min(np.percentile(q_error, 99), 0.2)
    norm_error = mpl.colors.Normalize(vmin=0, vmax=error_max)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    im = ax.imshow(q_error, cmap='Reds', norm=norm_error, aspect='auto')
    ax.set_title(f'{var_name} Relative Error | t = {select_time:.2f} (GPU)',
                 fontweight='bold', fontsize=12)
    ax.set_xlabel('X Grid')
    ax.set_ylabel('Y Grid')
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Relative Error', rotation=270, labelpad=15)

    plt.tight_layout()
    save_path_png = save_path.with_suffix('.png')
    save_path_pdf = save_path.with_suffix('.pdf')
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path_pdf, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"❌ 误差图保存: {save_path_png}")

# -------------------------- 数据处理与对比函数 --------------------------
def compare_at_select_time(time_num, data_stack, pinn_example):
    """单时刻对比（GPU加速）"""
    x = data_stack[:, 0].copy().reshape(-1, 1)
    y = data_stack[:, 1].copy().reshape(-1, 1)
    t = data_stack[:, 2].copy().reshape(-1, 1)
    u = data_stack[:, 3].copy().reshape(-1, 1)
    v = data_stack[:, 4].copy().reshape(-1, 1)
    p = data_stack[:, 5].copy().reshape(-1, 1)

    min_data = np.min(data_stack, 0)
    max_data = np.max(data_stack, 0)
    t_unique = np.unique(t).reshape(-1, 1)
    select_time = t_unique[time_num, 0]
    index_time = np.where(t == select_time)[0]

    # 网格构建
    x_unique = np.unique(x).reshape(-1, 1)
    y_unique = np.unique(y).reshape(-1, 1)
    mesh_x, mesh_y = np.meshgrid(x_unique, y_unique)

    # 提取真实值
    u_true = u[index_time].reshape(mesh_x.shape)
    v_true = v[index_time].reshape(mesh_x.shape)
    p_true = p[index_time].reshape(mesh_x.shape)

    # 预测输入
    x_flatten = np.ndarray.flatten(mesh_x).reshape(-1, 1)
    y_flatten = np.ndarray.flatten(mesh_y).reshape(-1, 1)
    t_flatten = np.ones((x_flatten.shape[0], 1)) * select_time

    # GPU推理（启用梯度计算）
    x_tensor = torch.tensor(x_flatten, dtype=torch.float32, requires_grad=True).to(device)
    y_tensor = torch.tensor(y_flatten, dtype=torch.float32, requires_grad=True).to(device)
    t_tensor = torch.tensor(t_flatten, dtype=torch.float32, requires_grad=True).to(device)
    u_pred, v_pred, p_pred, f_x, f_y = f_equation_inverse(x_tensor, y_tensor, t_tensor, pinn_example)

    # 结果移回CPU
    u_pred = u_pred.cpu().data.numpy().reshape(mesh_x.shape)
    v_pred = v_pred.cpu().data.numpy().reshape(mesh_x.shape)
    p_pred = p_pred.cpu().data.numpy().reshape(mesh_x.shape)

    # 保存路径
    base_save_path = OUTPUT_ROOT / 'single_time_comparison' / f'time_{select_time:.2f}'
    base_save_path.mkdir(exist_ok=True)

    # 绘制对比图
    error_stats = [
        plot_compare(u_true, u_pred, select_time, 'u', min_data[3], max_data[3], base_save_path / 'u_comparison'),
        plot_compare(v_true, v_pred, select_time, 'v', min_data[4], max_data[4], base_save_path / 'v_comparison'),
        plot_compare(p_true, p_pred, select_time, 'p', min_data[5], max_data[5], base_save_path / 'p_comparison')
    ]

    # 保存误差统计
    error_df = pd.DataFrame(error_stats)
    error_df.to_csv(base_save_path / 'error_statistics.csv', index=False)
    print(f"📊 误差统计保存: {base_save_path / 'error_statistics.csv'}")

    # 保存原始数据
    np.savez(base_save_path / 'raw_data.npz',
             u_true=u_true, u_pred=u_pred, v_true=v_true, v_pred=v_pred,
             p_true=p_true, p_pred=p_pred, x=mesh_x, y=mesh_y, time=select_time)
    print(f"💾 原始数据保存: {base_save_path / 'raw_data.npz'}")

def compare_at_select_time_series(lower_time_num, upper_time_num, data_stack, pinn_example):
    """时间序列对比（GPU批量推理）"""
    x = data_stack[:, 0].copy().reshape(-1, 1)
    y = data_stack[:, 1].copy().reshape(-1, 1)
    t = data_stack[:, 2].copy().reshape(-1, 1)
    u = data_stack[:, 3].copy().reshape(-1, 1)
    v = data_stack[:, 4].copy().reshape(-1, 1)
    p = data_stack[:, 5].copy().reshape(-1, 1)

    min_data = np.min(data_stack, 0)
    max_data = np.max(data_stack, 0)
    x_unique = np.unique(x).reshape(-1, 1)
    y_unique = np.unique(y).reshape(-1, 1)
    t_unique = np.unique(t).reshape(-1, 1)
    mesh_x, mesh_y = np.meshgrid(x_unique, y_unique)

    # 时间序列索引
    time_indices = np.linspace(lower_time_num, upper_time_num, upper_time_num - lower_time_num + 1).astype(int)
    time_indices = time_indices[time_indices < len(t_unique)]
    print(f"⏰ 时间序列范围: t_num={time_indices[0]} 到 t_num={time_indices[-1]}")
    print(f"📅 对应时刻: t={t_unique[time_indices[0], 0]:.2f} 到 t={t_unique[time_indices[-1], 0]:.2f}")

    all_error_stats = []
    for idx, time_num in enumerate(time_indices):
        select_time = t_unique[time_num, 0]
        index_time = np.where(t == select_time)[0]

        # 提取真实值
        u_true = u[index_time].reshape(mesh_x.shape)
        v_true = v[index_time].reshape(mesh_x.shape)
        p_true = p[index_time].reshape(mesh_x.shape)

        # 预测输入
        x_flatten = np.ndarray.flatten(mesh_x).reshape(-1, 1)
        y_flatten = np.ndarray.flatten(mesh_y).reshape(-1, 1)
        t_flatten = np.ones((x_flatten.shape[0], 1)) * select_time

        # GPU推理（启用梯度计算）
        x_tensor = torch.tensor(x_flatten, dtype=torch.float32, requires_grad=True).to(device)
        y_tensor = torch.tensor(y_flatten, dtype=torch.float32, requires_grad=True).to(device)
        t_tensor = torch.tensor(t_flatten, dtype=torch.float32, requires_grad=True).to(device)
        u_pred, v_pred, p_pred, f_x, f_y = f_equation_inverse(x_tensor, y_tensor, t_tensor, pinn_example)

        # 结果处理
        u_pred = u_pred.cpu().data.numpy().reshape(mesh_x.shape)
        v_pred = v_pred.cpu().data.numpy().reshape(mesh_x.shape)
        p_pred = p_pred.cpu().data.numpy().reshape(mesh_x.shape)

        # 计算误差
        mask_u = np.abs(u_true) < 1e-6
        mask_v = np.abs(v_true) < 1e-6
        mask_p = np.abs(p_true) < 1e-6

        u_error = np.zeros_like(u_true)
        u_error[~mask_u] = np.abs(u_pred[~mask_u] - u_true[~mask_u]) / np.abs(u_true[~mask_u])
        u_error[mask_u] = np.abs(u_pred[mask_u] - u_true[mask_u])

        v_error = np.zeros_like(v_true)
        v_error[~mask_v] = np.abs(v_pred[~mask_v] - v_true[~mask_v]) / np.abs(v_true[~mask_v])
        v_error[mask_v] = np.abs(v_pred[mask_v] - v_true[mask_v])

        p_error = np.zeros_like(p_true)
        p_error[~mask_p] = np.abs(p_pred[~mask_p] - p_true[~mask_p]) / np.abs(p_true[~mask_p])
        p_error[mask_p] = np.abs(p_pred[mask_p] - p_true[mask_p])

        # 保存帧图和误差图
        frame_save_dir = OUTPUT_ROOT / 'time_series_comparison' / f'time_{select_time:.2f}'
        frame_save_dir.mkdir(exist_ok=True)
        plot_compare_time_series(u_true, u_pred, select_time, 'u', min_data[3], max_data[3],
                                 frame_save_dir / 'u_comparison_frame.png')
        plot_compare_time_series(v_true, v_pred, select_time, 'v', min_data[4], max_data[4],
                                 frame_save_dir / 'v_comparison_frame.png')
        plot_compare_time_series(p_true, p_pred, select_time, 'p', min_data[5], max_data[5],
                                 frame_save_dir / 'p_comparison_frame.png')

        error_save_dir = OUTPUT_ROOT / 'error_maps' / f'time_{select_time:.2f}'
        error_save_dir.mkdir(exist_ok=True)
        plot_error_map(u_error, select_time, 'u', error_save_dir / 'u_error_map')
        plot_error_map(v_error, select_time, 'v', error_save_dir / 'v_error_map')
        plot_error_map(p_error, select_time, 'p', error_save_dir / 'p_error_map')

        # 记录误差统计
        all_error_stats.extend([
            {
                'time_num': time_num, 'time_value': select_time, 'variable': 'u',
                'mean_error': np.mean(u_error), 'max_error': np.max(u_error),
                'std_error': np.std(u_error), 'rmse': np.sqrt(np.mean(u_error ** 2))
            },
            {
                'time_num': time_num, 'time_value': select_time, 'variable': 'v',
                'mean_error': np.mean(v_error), 'max_error': np.max(v_error),
                'std_error': np.std(v_error), 'rmse': np.sqrt(np.mean(v_error ** 2))
            },
            {
                'time_num': time_num, 'time_value': select_time, 'variable': 'p',
                'mean_error': np.mean(p_error), 'max_error': np.max(p_error),
                'std_error': np.std(p_error), 'rmse': np.sqrt(np.mean(p_error ** 2))
            }
        ])

        # 清理内存
        del x_tensor, y_tensor, t_tensor, u_pred, v_pred, p_pred
        del u_error, v_error, p_error
        torch.cuda.empty_cache()

        # 进度提示
        print(f"📈 时间序列进度: {idx + 1}/{len(time_indices)} (t={select_time:.2f})")

    # 保存误差统计和趋势图
    stats_df = pd.DataFrame(all_error_stats)
    stats_save_path = OUTPUT_ROOT / 'csv_results' / 'time_series_error_statistics.csv'
    stats_df.to_csv(stats_save_path, index=False)
    print(f"📊 时间序列误差统计已保存至: {stats_save_path}")
    plot_error_trend(stats_df, OUTPUT_ROOT / 'csv_results')

def plot_error_trend(error_stats_df, save_dir):
    """绘制时间序列误差趋势图"""
    variables = ['u', 'v', 'p']
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for var in variables:
        var_df = error_stats_df[error_stats_df['variable'] == var]
        time_values = var_df['time_value'].values
        mean_errors = var_df['mean_error'].values
        max_errors = var_df['max_error'].values
        rmses = var_df['rmse'].values

        # 平均误差
        axes[0].plot(time_values, mean_errors, label=f'{var}-component', linewidth=2)
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Mean Relative Error')
        axes[0].set_title('Mean Error Trend Over Time', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 最大误差
        axes[1].plot(time_values, max_errors, label=f'{var}-component', linewidth=2)
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Max Relative Error')
        axes[1].set_title('Max Error Trend Over Time', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # RMSE
        axes[2].plot(time_values, rmses, label=f'{var}-component', linewidth=2)
        axes[2].set_xlabel('Time')
        axes[2].set_ylabel('RMSE')
        axes[2].set_title('RMSE Trend Over Time', fontweight='bold')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    # 综合误差热力图
    pivot_mean = error_stats_df.pivot(index='time_value', columns='variable', values='mean_error')
    im = axes[3].imshow(pivot_mean.values, cmap='YlOrRd', aspect='auto')
    axes[3].set_xticks(range(len(pivot_mean.columns)))
    axes[3].set_xticklabels(pivot_mean.columns)
    axes[3].set_yticks(range(0, len(pivot_mean.index), max(1, len(pivot_mean.index) // 10)))
    axes[3].set_yticklabels([f'{t:.2f}' for t in pivot_mean.index[::max(1, len(pivot_mean.index) // 10)]])
    axes[3].set_xlabel('Variable')
    axes[3].set_ylabel('Time')
    axes[3].set_title('Mean Error Heatmap (Time × Variable)', fontweight='bold')
    plt.colorbar(im, ax=axes[3], shrink=0.8)

    plt.tight_layout()
    save_path_png = save_dir / 'error_trend_analysis.png'
    save_path_pdf = save_dir / 'error_trend_analysis.pdf'
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path_pdf, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"📈 误差趋势分析图已保存至: {save_path_png}")

# -------------------------- 生成GIF函数 --------------------------
def make_flow_gif(lower_time_num, upper_time_num, data_stack, interval=0.1, var_name='u', fps_num=5):
    """基于时间序列对比帧生成GIF动画"""
    t = data_stack[:, 2].copy().reshape(-1, 1)
    t_unique = np.unique(t).reshape(-1, 1)
    time_indices = np.linspace(lower_time_num, upper_time_num, upper_time_num - lower_time_num + 1).astype(int)
    time_indices = time_indices[time_indices < len(t_unique)]

    gif_images = []
    frame_dir = OUTPUT_ROOT / 'time_series_comparison'
    for time_num in time_indices:
        select_time = t_unique[time_num, 0]
        img_path = frame_dir / f'time_{select_time:.2f}' / f'{var_name}_comparison_frame.png'
        if img_path.exists():
            gif_images.append(imageio.imread(str(img_path)))
            print(f"🎞️  已添加GIF帧: {img_path}")
        else:
            print(f"⚠️  未找到帧图片: {img_path}")

    if gif_images:
        gif_save_path = OUTPUT_ROOT / 'gifs' / f'{var_name}_comparison.gif'
        imageio.mimsave(str(gif_save_path), gif_images, fps=fps_num)
        print(f"🎥 GIF动画已保存至: {gif_save_path}")
    else:
        print("❌ 未收集到任何帧图片，无法生成GIF")

# -------------------------- 主函数 --------------------------
if __name__ == "__main__":
    start_time_num = 0
    end_time_num = 99
    gif_fps = 20
    interval = 0.1

    # 1. 运行时间序列对比
    print("🚀 开始时间序列对比（GPU加速）...")
    compare_at_select_time_series(start_time_num, end_time_num, data_stack, pinn_net)

    # 2. 生成GIF动画
    print("\n🎬 开始生成GIF动画...")
    for var in ['u', 'v', 'p']:
        make_flow_gif(start_time_num, end_time_num, data_stack, interval=interval, var_name=var, fps_num=gif_fps)

    # 3. （可选）运行单时刻对比
    # compare_at_select_time(time_num=50, data_stack=data_stack, pinn_example=pinn_net)

    print("\n🎉 所有任务完成！结果已保存至: {}".format(OUTPUT_ROOT.absolute()))