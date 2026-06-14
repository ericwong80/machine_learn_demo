"""
数据集生成与处理模块
负责生成房价数据集，提供统一的训练集/测试集拆分功能
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class HousePriceDataGenerator:
    """房价数据生成器"""

    def __init__(self, random_state=42):
        """
        初始化数据生成器

        Args:
            random_state (int): 随机种子，确保结果可重复
        """
        np.random.seed(random_state)
        self.base_price = 3.5  # 二线城市均价3.5万/平米

    def _generate_features(self, n_samples):
        """
        生成特征数据

        Args:
            n_samples (int): 样本数量

        Returns:
            np.ndarray: 特征矩阵
        """
        area = np.clip(np.random.normal(95, 25, n_samples), 60, 180)
        floor = np.clip(np.random.normal(15, 8, n_samples), 2, 33).astype(int)
        distance = np.clip(np.random.exponential(1.2, n_samples), 0.2, 5.0)
        age = np.clip(np.random.normal(12, 8, n_samples), 1, 30).astype(int)
        return np.column_stack([area, floor, distance, age])

    def _generate_prices(self, X):
        """
        根据特征生成房价

        Args:
            X (np.ndarray): 特征矩阵

        Returns:
            np.ndarray: 房价数组
        """
        area, floor, distance, age = X[:, 0], X[:, 1], X[:, 2], X[:, 3]

        # 基础价格
        base_price = area * self.base_price

        # 位置因子：1公里内+25%，1-2公里+10%，2公里外-5%
        location_factor = np.where(distance < 1, 1.25, np.where(distance < 2, 1.1, 0.95))

        # 楼层因子：10-20层最佳，其他略低
        floor_factor = np.where((floor >= 10) & (floor <= 20), 1.05, 1.0)

        # 房龄因子：每年折价1.5%
        age_factor = 1 - 0.015 * age

        # 计算最终价格
        price = base_price * location_factor * floor_factor * age_factor
        price += np.random.normal(0, 15, len(price))  # 随机噪声
        return np.round(np.clip(price, 80, 1200), 1)

    def generate(self, n_samples):
        """
        生成完整的房价数据集

        Args:
            n_samples (int): 样本数量

        Returns:
            pd.DataFrame: 生成的房价数据集
        """
        X = self._generate_features(n_samples)
        y = self._generate_prices(X)
        df = pd.DataFrame(X, columns=["面积", "楼层", "距地铁", "房龄"])
        df["房价"] = y
        df["样本"] = [f"S{i + 1:02d}" for i in range(n_samples)]
        return df


def generate_house_price_data(n_samples=100, random_state=42):
    """
    便捷生成函数

    Args:
        n_samples (int): 样本数量，默认为100
        random_state (int): 随机种子，默认为42

    Returns:
        pd.DataFrame: 生成的房价数据集
    """
    generator = HousePriceDataGenerator(random_state)
    return generator.generate(n_samples)


def prepare_data(n_samples=100, test_size=0.2, random_state=42):
    """
    准备训练集和测试集

    Args:
        n_samples (int): 样本数量，默认为100
        test_size (float): 测试集比例，默认为0.2
        random_state (int): 随机种子，默认为42

    Returns:
        tuple: (X_train, X_test, y_train, y_test, data)
            X_train: 训练集特征
            X_test: 测试集特征
            y_train: 训练集目标值
            y_test: 测试集目标值
            data: 完整数据集
    """
    # 生成数据集
    data = generate_house_price_data(n_samples, random_state)

    # 提取特征和目标
    feature_names = ["面积", "楼层", "距地铁", "房龄"]
    X = data[feature_names].values
    y = data["房价"].values

    # 拆分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test, data, feature_names


def demo_data_generation(size: int):
    """
    演示函数，接收样本大小

    Args:
        size (int): 要生成的样本数量，必须是正整数

    Returns:
        pd.DataFrame: 生成的房价数据集

    Raises:
        ValueError: 当size不是正整数时抛出异常
    """
    if not isinstance(size, int) or size <= 0:
        raise ValueError("size参数必须是正整数")

    print(f"🏠 生成 {size} 个房价样本...")
    df = generate_house_price_data(n_samples=size, random_state=42)

    print(f"n📊 生成的数据 (前5行):")
    print(df.head().to_string())

    print(f"n📈 基本统计信息:")
    print(f"  样本数量: {len(df)}")
    print(f"  房价范围: {df['房价'].min():.1f} - {df['房价'].max():.1f} 万元")
    print(f"  平均房价: {df['房价'].mean():.1f} 万元")

    return df


if __name__ == "__main__":
    demo_data_generation(100)
