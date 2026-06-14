"""
输出模块
负责创建输出目录和保存文件
"""
from pathlib import Path
from datetime import datetime
import json

import pandas as pd


# 如果当前文件位置是：项目根目录/utils/output.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 如果你的文件位置是：项目根目录/src/utils/output.py
# PROJECT_ROOT = Path(__file__).resolve().parents[2]


def create_output_dir(base_dir: str | Path | None = None) -> Path:
    """
    创建输出目录

    默认目录：
        项目根目录/output/年月日/时分秒

    Args:
        base_dir: 自定义输出目录

    Returns:
        Path: 输出目录路径
    """
    if base_dir is None:
        now = datetime.now()
        base_dir = PROJECT_ROOT / "output" / now.strftime("%Y%m%d") / now.strftime("%H%M%S")
    else:
        base_dir = Path(base_dir)

    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def save_figure(fig, filename: str, output_dir: str | Path | None = None, dpi: int = 300) -> Path:
    """
    保存图像
    """
    output_dir = create_output_dir(output_dir)
    save_path = output_dir / filename

    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return save_path


def save_dataframe(
    df: pd.DataFrame,
    filename: str,
    output_dir: str | Path | None = None,
    index: bool = False,
) -> Path:
    """
    保存 DataFrame 到 CSV 文件
    """
    output_dir = create_output_dir(output_dir)
    save_path = output_dir / filename

    df.to_csv(save_path, index=index, encoding="utf-8-sig")

    return save_path


def save_metrics(metrics_dict: dict, filename: str, output_dir: str | Path | None = None) -> Path:
    """
    保存指标到 JSON 文件
    """
    output_dir = create_output_dir(output_dir)
    save_path = output_dir / filename

    with save_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, ensure_ascii=False, indent=4)

    return save_path


def print_model_metrics(model_name: str, metrics_dict: dict) -> None:
    """
    打印模型指标
    """
    print(f"\n=== {model_name} ===")
    print(f"MAE: {metrics_dict['MAE']:.2f} 万元")
    print(f"RMSE: {metrics_dict['RMSE']:.2f} 万元")
    print(f"R2: {metrics_dict['R2']:.3f}")


if __name__ == "__main__":
    output_dir = create_output_dir()
    print(f"输出目录创建于: {output_dir}")

    data = {
        "模型": ["线性回归", "决策树", "随机森林"],
        "MAE": [5.2, 4.8, 4.5],
        "RMSE": [7.1, 6.8, 6.2],
        "R2": [0.85, 0.87, 0.89],
    }

    df = pd.DataFrame(data)

    save_path = save_dataframe(df, "模型比较.csv", output_dir)
    print(f"DataFrame 已保存至: {save_path}")

    metrics = {
        "MAE": 5.2,
        "RMSE": 7.1,
        "R2": 0.85,
    }

    metrics_path = save_metrics(metrics, "线性回归指标.json", output_dir)
    print(f"指标已保存至: {metrics_path}")

    print_model_metrics("线性回归", metrics)