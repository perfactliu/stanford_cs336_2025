import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys
import os

# 解决 Matplotlib 中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号 '-' 显示为方块的问题

# 检查 Matplotlib 后端，以确保在无显示器环境中也能保存图像
try:
    plt.get_backend()
except:
    plt.switch_backend('Agg')

# 定义数据文件路径
data_file = 'data/isoflops_curves.json'


# 1. 从JSON文件加载数据
def load_data(file_path):
    """从指定的 JSON 文件加载训练运行数据。"""
    try:
        if not os.path.exists(file_path):
            # 提供一个更友好的错误信息，并退出
            print(f"错误: 文件 '{file_path}' 未找到。请确保文件路径正确。")
            sys.exit(1)
        with open(file_path, 'r') as f:
            training_runs = json.load(f)
        print(f"成功从 '{file_path}' 加载了 {len(training_runs)} 个训练运行数据点。")
        return training_runs
    except json.JSONDecodeError:
        print(f"错误: 文件 '{file_path}' 不是有效的JSON格式。")
        sys.exit(1)


# 2. 找出每个计算预算下的最佳运行点
def get_optimal_points(data):
    """
    根据最终训练损失，找出每个计算预算下的最佳训练运行。

    Args:
        data (list): 包含所有训练运行数据的列表。

    Returns:
        tuple: 包含最佳运行的计算预算、参数量和数据集大小的 NumPy 数组。
    """
    # 按计算预算分组
    runs_by_compute = {}
    for run in data:
        cb = run.get("compute_budget")
        if cb not in runs_by_compute:
            runs_by_compute[cb] = []
        runs_by_compute[cb].append(run)

    # 找到每个计算预算下的最低损失运行
    optimal_points = []
    for cb, runs in runs_by_compute.items():
        min_loss_run = min(runs, key=lambda x: x.get("final_loss", float('inf')))
        optimal_points.append(min_loss_run)

    # 提取数据点
    compute = np.array([p.get("compute_budget") for p in optimal_points])
    parameters = np.array([p.get("parameters") for p in optimal_points])

    # 根据 C = 6 * N * D, 计算数据集大小 D = C / (6 * N)
    tokens = compute / (6 * parameters)

    return compute, parameters, tokens


# 3. 定义幂律函数进行拟合
def power_law(x, a, b):
    """用于曲线拟合的幂律函数。"""
    return a * (x ** b)


def fit_scaling_law(x_data, y_data):
    """拟合数据并返回拟合参数。"""
    try:
        popt, _ = curve_fit(power_law, x_data, y_data, p0=[1, 0.5])
        return popt
    except RuntimeError as e:
        print(f"错误: 曲线拟合失败。可能数据点太少或数据不适合幂律拟合。详细信息: {e}")
        sys.exit(1)


def plot_and_save(x_points, y_points, x_extrapolate, y_predict, title, xlabel, ylabel, filename, a, b):
    """绘制并保存缩放定律图表。"""
    plt.figure(figsize=(10, 6))
    plt.loglog(x_points, y_points, 'o', label=f'数据点: $⟨C_i, Y_{{opt}}(C_i)⟩$', markersize=8)
    plt.loglog(x_extrapolate, y_predict, '-', label=f'拟合曲线: $Y_{{opt}} = {a:.2f} C^{{{b:.2f}}}$')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig(filename)
    print(f"图表 '{filename}' 已保存。")


def main():
    """主函数，执行整个 IsoFLOPs 缩放分析。"""
    training_runs = load_data(data_file)

    # 获取最佳数据点
    C_opt, N_opt, D_opt = get_optimal_points(training_runs)

    # 拟合模型大小的缩放定律 (N_opt vs C_opt)
    popt_N = fit_scaling_law(C_opt, N_opt)
    a_N, b_N = popt_N

    # 拟合数据集大小的缩放定律 (D_opt vs C_opt)
    popt_D = fit_scaling_law(C_opt, D_opt)
    a_D, b_D = popt_D

    # 预测未来计算预算下的最佳模型大小和数据集大小
    C_extrapolate = np.logspace(np.log10(min(C_opt)), 24, 100)
    N_predict = power_law(C_extrapolate, a_N, b_N)
    D_predict = power_law(C_extrapolate, a_D, b_D)

    # 预测特定预算下的值
    N_10_23 = power_law(1e23, a_N, b_N)
    N_10_24 = power_law(1e24, a_N, b_N)
    D_10_23 = power_law(1e23, a_D, b_D)
    D_10_24 = power_law(1e24, a_D, b_D)

    print("\n--- 预测结果 ---")
    print(f"对于 10^23 FLOPs 的预算，预测的最佳模型大小为: {N_10_23:.2e}")
    print(f"对于 10^24 FLOPs 的预算，预测的最佳模型大小为: {N_10_24:.2e}")
    print(f"对于 10^23 FLOPs 的预算，预测的最佳数据集大小为: {D_10_23:.2e}")
    print(f"对于 10^24 FLOPs 的预算，预测的最佳数据集大小为: {D_10_24:.2e}")

    # 生成图表并保存
    plot_and_save(C_opt, N_opt, C_extrapolate, N_predict,
                  '模型大小与计算预算的缩放定律', '计算预算 (FLOPs)', '最佳模型大小 (参数量)',
                  'model_size_scaling_law.png', a_N, b_N)

    plot_and_save(C_opt, D_opt, C_extrapolate, D_predict,
                  '数据集大小与计算预算的缩放定律', '计算预算 (FLOPs)', '最佳数据集大小 (Tokens)',
                  'dataset_size_scaling_law.png', a_D, b_D)


if __name__ == "__main__":
    main()
