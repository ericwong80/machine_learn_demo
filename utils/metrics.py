"""
评估指标计算模块
提供回归模型常用的评估指标计算功能
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(y_true, y_pred):
    """
    计算回归模型常用评估指标

    Args:
        y_true (array-like): 真实值
        y_pred (array-like): 预测值

    Returns:
        dict: 包含各项指标的字典
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


def format_metrics(metrics_dict):
    """
    格式化指标输出

    Args:
        metrics_dict (dict): 包含各项指标的字典

    Returns:
        str: 格式化后的指标字符串
    """
    return (
        f"MAE: {metrics_dict['MAE']:.2f} 万元, "
        f"RMSE: {metrics_dict['RMSE']:.2f} 万元, "
        f"R2: {metrics_dict['R2']:.3f}"
    )


def compare_models(metrics_list, model_names):
    """
    比较多个模型的指标

    Args:
        metrics_list (list): 包含多个模型指标的列表，每个元素是一个字典
        model_names (list): 模型名称列表

    Returns:
        pd.DataFrame: 模型比较表格
    """
    # 创建比较表格
    comparison_df = pd.DataFrame(metrics_list, index=model_names)

    # 按R2降序排序
    comparison_df = comparison_df.sort_values("R2", ascending=False)

    return comparison_df


if __name__ == "__main__":
    # 示例用法
    y_true = np.array([100, 120, 90, 110, 130])
    y_pred1 = np.array([105, 115, 95, 108, 125])
    y_pred2 = np.array([110, 125, 85, 115, 135])

    # 计算指标
    metrics1 = calculate_metrics(y_true, y_pred1)
    metrics2 = calculate_metrics(y_true, y_pred2)

    # 格式化输出
    print("模型1指标:", format_metrics(metrics1))
    print("模型2指标:", format_metrics(metrics2))

    # 比较模型
    metrics_list = [metrics1, metrics2]
    model_names = ["模型1", "模型2"]
    comparison = compare_models(metrics_list, model_names)
    print("\n模型比较:")
    print(comparison)
