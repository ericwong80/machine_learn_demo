"""
输出模块
负责创建输出目录和保存文件
"""
import os
import matplotlib.pyplot as plt
import pandas as pd


def create_output_dir(base_dir=None):
    """
    创建输出目录

    Args:
        base_dir (str or None): 基础输出目录路径，如果为None则使用项目根目录下的output/day01_XX/年月日/时分秒

    Returns:
        str: 完整的输出目录路径
    """
    # 如果没有指定输出目录，则使用项目根目录下的output目录
    if base_dir is None:
        # 获取当前文件所在目录的父目录的父目录（即项目根目录）
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        # 确保我们回到正确的项目根目录
        project_root = os.path.dirname(current_dir)
        base_dir = os.path.join(project_root, "output")

        # 添加项目目录
        project_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        base_dir = os.path.join(base_dir, project_name)

        # 添加年月日/时分秒子目录
        from datetime import datetime
        now = datetime.now()
        date_str = now.strftime("%Y%m%d")
        time_str = now.strftime("%H%M%S")
        base_dir = os.path.join(base_dir, date_str, time_str)

    # 确保基础目录存在
    os.makedirs(base_dir, exist_ok=True)

    return base_dir


def save_figure(fig, filename, output_dir=None, dpi=300):
    """
    保存图像

    Args:
        fig (matplotlib.figure.Figure): 要保存的图像对象
        filename (str): 保存的文件名
        output_dir (str): 输出目录路径，如果为None则使用默认目录
        dpi (int): 图像分辨率，默认为300

    Returns:
        str: 保存的文件路径
    """
    # 如果没有指定输出目录，则创建默认目录
    if output_dir is None:
        output_dir = create_output_dir()

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 构建完整的保存路径
    save_path = os.path.join(output_dir, filename)

    # 保存图像
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return save_path


def save_dataframe(df, filename, output_dir=None):
    """
    保存DataFrame到CSV文件

    Args:
        df (pd.DataFrame): 要保存的DataFrame
        filename (str): 保存的文件名
        output_dir (str): 输出目录路径，如果为None则使用默认目录

    Returns:
        str: 保存的文件路径
    """
    # 如果没有指定输出目录，则创建默认目录
    if output_dir is None:
        output_dir = create_output_dir()

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 构建完整的保存路径
    save_path = os.path.join(output_dir, filename)

    # 保存DataFrame
    df.to_csv(save_path, index=True, encoding='utf-8-sig')

    return save_path


def save_metrics(metrics_dict, filename, output_dir=None):
    """
    保存指标到JSON文件

    Args:
        metrics_dict (dict): 要保存的指标字典
        filename (str): 保存的文件名
        output_dir (str): 输出目录路径，如果为None则使用默认目录

    Returns:
        str: 保存的文件路径
    """
    # 如果没有指定输出目录，则创建默认目录
    if output_dir is None:
        output_dir = create_output_dir()

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 构建完整的保存路径
    save_path = os.path.join(output_dir, filename)

    # 保存指标
    import json
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_dict, f, ensure_ascii=False, indent=4)

    return save_path


def print_model_metrics(model_name, metrics_dict):
    """
    打印模型指标

    Args:
        model_name (str): 模型名称
        metrics_dict (dict): 指标字典
    """
    print(f"\n=== {model_name} ===")
    print(f"MAE: {metrics_dict['MAE']:.2f} 万元")
    print(f"RMSE: {metrics_dict['RMSE']:.2f} 万元")
    print(f"R2: {metrics_dict['R2']:.3f}")


if __name__ == "__main__":
    # 示例用法
    output_dir = create_output_dir()
    print(f"输出目录创建于: {output_dir}")

    # 示例数据
    data = {
        '模型': ['线性回归', '决策树', '随机森林'],
        'MAE': [5.2, 4.8, 4.5],
        'RMSE': [7.1, 6.8, 6.2],
        'R2': [0.85, 0.87, 0.89]
    }
    df = pd.DataFrame(data)

    # 保存DataFrame
    save_path = save_dataframe(df, "模型比较.csv", output_dir)
    print(f"DataFrame已保存至: {save_path}")

    # 保存指标
    metrics = {
        "MAE": 5.2,
        "RMSE": 7.1,
        "R2": 0.85
    }
    metrics_path = save_metrics(metrics, "线性回归指标.json", output_dir)
    print(f"指标已保存至: {metrics_path}")

    # 打印指标
    print_model_metrics("线性回归", metrics)
