from __future__ import annotations

import os
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns


def set_chinese_plot_style(
    style: str = "whitegrid",
    font: str = "SimHei",
    font_scale: float = 1.0,
    figure_facecolor: str | None = None,
    axes_facecolor: str | None = None,
) -> None:
    """设置适合中文显示的 seaborn/matplotlib 绘图样式。"""
    rc = {"axes.unicode_minus": False}
    if figure_facecolor is not None:
        rc["figure.facecolor"] = figure_facecolor
    if axes_facecolor is not None:
        rc["axes.facecolor"] = axes_facecolor

    sns.set_theme(style=style, font=font, font_scale=font_scale, rc=rc)
    plt.rcParams["font.sans-serif"] = [font, "Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False


def save_figure(
    output_dir: str,
    filename: str,
    fig: Any | None = None,
    dpi: int = 200,
    bbox_inches: str = "tight",
) -> str:
    """保存 matplotlib/seaborn 图像，并返回保存路径。"""
    save_path = os.path.join(output_dir, filename)
    target = fig if fig is not None else plt
    target.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches)
    print(f"图片已保存: {save_path}")
    return save_path


def show_or_close(fig: Any | None = None) -> None:
    """交互环境显示图片，后台绘图环境关闭图片。"""
    if "agg" in plt.get_backend().lower():
        plt.close(fig)
    else:
        plt.show()


def close_figure(fig: Any | None = None) -> None:
    """关闭指定图像；未指定时关闭当前图像。"""
    plt.close(fig)
