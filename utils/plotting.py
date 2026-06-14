"""
绘图模块
提供回归模型常用的可视化功能
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def set_chinese_plot_style(figure_facecolor="#f7f9fc", axes_facecolor="#ffffff"):
    """
    设置适合中文显示的绘图风格

    Args:
        figure_facecolor (str): 图像背景颜色，默认为"#f7f9fc"
        axes_facecolor (str): 坐标轴背景颜色，默认为"#ffffff"
    """
    # 配置rc字典，包含matplotlib的运行时配置
    rc = {
        "axes.unicode_minus": False,  # 解决负号显示问题
        "font.weight": "normal",  # 设置字体粗细为普通
        "axes.titleweight": "normal",  # 设置标题字体粗细为普通
        "axes.labelweight": "normal",  # 设置标签字体粗细为普通
    }
    # 如果设置了图像背景颜色，则添加到rc字典中
    if figure_facecolor is not None:
        rc["figure.facecolor"] = figure_facecolor
    # 如果设置了坐标轴背景颜色，则添加到rc字典中
    if axes_facecolor is not None:
        rc["axes.facecolor"] = axes_facecolor

    # 应用seaborn主题设置
    sns.set_theme(style="whitegrid", font="SimHei", font_scale=1.0, rc=rc)
    # 设置matplotlib的sans-serif字体，确保中文显示
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    # 设置负号显示正常
    plt.rcParams["axes.unicode_minus"] = False
    # 解决上标显示问题 - 使用DejaVu Sans作为数学文本字体
    plt.rcParams["mathtext.fontset"] = "dejavusans"
    # 设置字体粗细支持
    plt.rcParams["font.weight"] = "normal"
    plt.rcParams["axes.titleweight"] = "normal"
    plt.rcParams["axes.labelweight"] = "normal"
    # 设置支持特殊字符
    plt.rcParams["hatch.linewidth"] = 0.5


def plot_actual_vs_predicted(y_true, y_pred, title="实际值 vs 预测值", 
                            xlabel="实际值", ylabel="预测值", 
                            sample_ids=None, figsize=(8, 6)):
    """
    绘制实际值与预测值的散点图

    Args:
        y_true (array-like): 真实值
        y_pred (array-like): 预测值
        title (str): 图表标题
        xlabel (str): x轴标签
        ylabel (str): y轴标签
        sample_ids (array-like): 样本ID，用于标注点
        figsize (tuple): 图表大小

    Returns:
        matplotlib.figure.Figure: 生成的图表对象
    """
    fig, ax = plt.subplots(figsize=figsize)

    # 绘制散点图
    scatter = ax.scatter(y_true, y_pred, alpha=0.7, edgecolor='w', linewidth=0.5)

    # 绘制理想预测线（y=x）
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, label="理想预测线")

    # 添加样本ID标注
    if sample_ids is not None:
        for i, id in enumerate(sample_ids):
            ax.annotate(id, (y_true[i], y_pred[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8)

    # 设置图表属性
    ax.set_title(title, fontsize=14, fontweight='normal')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    return fig


def plot_residuals(y_true, y_pred, title="残差图", xlabel="样本", 
                  ylabel="残差", figsize=(8, 6)):
    """
    绘制残差图

    Args:
        y_true (array-like): 真实值
        y_pred (array-like): 预测值
        title (str): 图表标题
        xlabel (str): x轴标签
        ylabel (str): y轴标签
        figsize (tuple): 图表大小

    Returns:
        matplotlib.figure.Figure: 生成的图表对象
    """
    # 计算残差
    residuals = y_true - y_pred

    fig, ax = plt.subplots(figsize=figsize)

    # 绘制残差散点图
    ax.scatter(range(len(residuals)), residuals, alpha=0.7, edgecolor='w', linewidth=0.5)

    # 绘制零线
    ax.axhline(0, color='r', linestyle='--', linewidth=1.5)

    # 设置图表属性
    ax.set_title(title, fontsize=14, fontweight='normal')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    return fig


def plot_model_comparison(metrics_df, title="模型性能比较", figsize=(10, 6)):
    """
    绘制模型性能比较图

    Args:
        metrics_df (pd.DataFrame): 包含模型指标的DataFrame
        title (str): 图表标题
        figsize (tuple): 图表大小

    Returns:
        matplotlib.figure.Figure: 生成的图表对象
    """
    # 创建子图
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight='normal')

    # 绘制MAE比较
    metrics_df['MAE'].plot(kind='bar', ax=axes[0], color='skyblue')
    axes[0].set_title('MAE比较', fontweight='normal')
    axes[0].set_ylabel('MAE (万元)')
    for i, v in enumerate(metrics_df['MAE']):
        axes[0].text(i, v + 0.5, f"{v:.2f}", ha='center')

    # 绘制RMSE比较
    metrics_df['RMSE'].plot(kind='bar', ax=axes[1], color='salmon')
    axes[1].set_title('RMSE比较', fontweight='normal')
    axes[1].set_ylabel('RMSE (万元)')
    for i, v in enumerate(metrics_df['RMSE']):
        axes[1].text(i, v + 0.5, f"{v:.2f}", ha='center')

    # 绘制R2比较
    metrics_df['R2'].plot(kind='bar', ax=axes[2], color='lightgreen')
    axes[2].set_title('R2比较', fontweight='normal')
    axes[2].set_ylabel('R2 (决定系数)')
    for i, v in enumerate(metrics_df['R2']):
        axes[2].text(i, v + 0.01, f"{v:.3f}", ha='center')

    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


if __name__ == "__main__":
    # 示例用法
    np.random.seed(42)
    y_true = np.random.normal(100, 20, 50)
    y_pred = y_true + np.random.normal(0, 10, 50)

    # 设置绘图风格
    set_chinese_plot_style()

    # 绘制实际值vs预测值
    fig1 = plot_actual_vs_predicted(y_true, y_pred, sample_ids=[f"S{i+1}" for i in range(50)])
    plt.show()

    # 绘制残差图
    fig2 = plot_residuals(y_true, y_pred)
    plt.show()
