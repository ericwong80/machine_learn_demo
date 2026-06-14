"""
梯度下降线性回归演示
使用梯度下降法求解线性回归模型的参数
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from common.dataset import prepare_data
from common.metrics import calculate_metrics, format_metrics
from common.plotting import plot_actual_vs_predicted, plot_residuals, set_chinese_plot_style
from common.output import create_output_dir, save_figure, print_model_metrics


# 设置绘图风格
set_chinese_plot_style()

# 创建输出目录
output_dir = create_output_dir()

# ===== 1. 准备数据 =====
print("=== 准备数据 ===")
X_train, X_test, y_train, y_test, data, feature_names = prepare_data(n_samples=100, test_size=0.2, random_state=42)

# 为梯度下降添加偏置项（截距）
X_train_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]

print(f"训练集样本数: {X_train.shape[0]}")
print(f"测试集样本数: {X_test.shape[0]}")
print(f"特征数量: {X_train.shape[1]}")

# ===== 2. 梯度下降实现 =====
print("\n=== 梯度下降实现 ===")
class GradientDescentLinearRegression:
    """梯度下降线性回归模型"""

    def __init__(self, learning_rate=0.01, n_iterations=1000, random_state=42):
        """
        初始化模型

        Args:
            learning_rate (float): 学习率，默认为0.01
            n_iterations (int): 迭代次数，默认为1000
            random_state (int): 随机种子，默认为42
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.theta = None
        self.cost_history = []

    def fit(self, X, y):
        """
        训练模型

        Args:
            X (np.ndarray): 特征矩阵，包含偏置项
            y (np.ndarray): 目标值
        """
        # 设置随机种子
        np.random.seed(self.random_state)

        # 初始化参数
        n_samples, n_features = X.shape
        self.theta = np.random.randn(n_features)

        # 梯度下降
        for i in range(self.n_iterations):
            # 计算预测值
            predictions = X.dot(self.theta)

            # 计算成本函数（均方误差）
            cost = (1 / (2 * n_samples)) * np.sum(np.square(predictions - y))
            self.cost_history.append(cost)

            # 计算梯度
            gradient = (1 / n_samples) * X.T.dot(predictions - y)

            # 更新参数
            self.theta -= self.learning_rate * gradient

            # 打印进度
            if (i + 1) % 100 == 0:
                print(f"迭代 {i + 1}/{self.n_iterations}, 成本: {cost:.4f}")

    def predict(self, X):
        """
        预测

        Args:
            X (np.ndarray): 特征矩阵，包含偏置项

        Returns:
            np.ndarray: 预测值
        """
        return X.dot(self.theta)

    def get_params(self, include_bias=True):
        """
        获取模型参数

        Args:
            include_bias (bool): 是否包含偏置项（截距）

        Returns:
            dict: 参数字典
        """
        params = {"截距": self.theta[0]}
        for i, name in enumerate(feature_names):
            params[name] = self.theta[i + 1]
        return params

# 创建并训练模型
print("\n=== 创建和训练模型 ===")
model = GradientDescentLinearRegression(learning_rate=0.01, n_iterations=1000, random_state=42)
model.fit(X_train_b, y_train)

# ===== 3. 预测 =====
print("\n=== 预测 ===")
y_train_pred = model.predict(X_train_b)
y_test_pred = model.predict(X_test_b)

# ===== 4. 评估模型 =====
print("\n=== 评估模型 ===")
# 训练集指标
train_metrics = calculate_metrics(y_train, y_train_pred)
print("训练集指标:")
print(format_metrics(train_metrics))

# 测试集指标
test_metrics = calculate_metrics(y_test, y_test_pred)
print("\n测试集指标:")
print(format_metrics(test_metrics))

# ===== 5. 打印模型信息 =====
print("\n=== 模型信息 ===")
print("梯度下降线性回归模型公式：")
params = model.get_params()
equation_parts = [f"{params[name]:+.2f}*{name}" for name in feature_names]
print(f"房价 = {params['截距']:.2f} " + " ".join(equation_parts))

# 特征重要性（标准化系数）
x_std = X_train.std(axis=0, ddof=0)
y_std = y_train.std(ddof=0)
standardized_coef = [params[name] * x_std[i] / y_std for i, name in enumerate(feature_names)]

print("\n特征标准化影响（按重要性排序）：")
importance_df = pd.DataFrame({
    "特征": feature_names,
    "原始系数": [params[name] for name in feature_names],
    "标准化影响": standardized_coef
}).sort_values("标准化影响", key=lambda s: s.abs(), ascending=False)

for _, row in importance_df.iterrows():
    print(f"{row['特征']}: 原始系数 {row['原始系数']:+.2f}, 标准化影响 {row['标准化影响']:+.2f}")

# ===== 6. 可视化结果 =====
print("\n=== 可视化结果 ===")
# 实际值 vs 预测值
fig1 = plot_actual_vs_predicted(
    y_test, y_test_pred, 
    title="测试集：实际值 vs 预测值",
    xlabel="实际房价（万元）", 
    ylabel="预测房价（万元）",
    sample_ids=data["样本"].iloc[len(y_train):].values
)
save_figure(fig1, "gd_实际值vs预测值.png", output_dir)
plt.close(fig1)

# 残差图
fig2 = plot_residuals(
    y_test, y_test_pred,
    title="测试集：残差图",
    xlabel="样本",
    ylabel="残差（万元）"
)
save_figure(fig2, "gd_残差图.png", output_dir)
plt.close(fig2)

# 梯度下降成本曲线
fig3, ax = plt.subplots(figsize=(10, 6))
ax.plot(range(1, len(model.cost_history) + 1), model.cost_history)
ax.set_title("梯度下降成本曲线", fontsize=14, fontweight='bold')
ax.set_xlabel("迭代次数")
ax.set_ylabel("成本")
ax.grid(True, linestyle='--', alpha=0.6)
save_figure(fig3, "gd_成本曲线.png", output_dir)
plt.close(fig3)

# ===== 7. 新样本预测 =====
print("\n=== 新样本预测 ===")
# 创建一个新样本，并添加偏置项
X_new = np.array([[1, 100, 14, 0.9, 8]], dtype=float)  # 第一个1是偏置项
y_new_pred = model.predict(X_new)[0]

print(f"新样本预测：")
print(f"面积={X_new[0, 1]:.0f}㎡，楼层={X_new[0, 2]:.0f}，"
      f"距地铁={X_new[0, 3]:.1f}km，房龄={X_new[0, 4]:.0f}年 -> "
      f"{y_new_pred:.2f} 万元")

# 预测贡献分析
contributions = [params[name] * X_new[0, i + 1] for i, name in enumerate(feature_names)]
contrib_df = pd.DataFrame({
    "项目": ["截距"] + feature_names,
    "贡献": [params["截距"]] + contributions
})

print("\n预测贡献分析：")
for _, row in contrib_df.iterrows():
    print(f"{row['项目']}: {row['贡献']:+.2f} 万元")

# ===== 8. 保存结果 =====
print("\n=== 保存结果 ===")
# 保存指标
save_metrics(test_metrics, "gd_测试集指标.json", output_dir)

# 保存特征重要性
importance_df.to_csv(os.path.join(output_dir, "gd_特征重要性.csv"), index=False, encoding='utf-8-sig')

# 保存新样本预测结果
new_pred_df = pd.DataFrame({
    "样本": ["新样本"],
    "面积": [X_new[0, 1]],
    "楼层": [X_new[0, 2]],
    "距地铁": [X_new[0, 3]],
    "房龄": [X_new[0, 4]],
    "预测房价": [y_new_pred]
})
new_pred_df.to_csv(os.path.join(output_dir, "gd_新样本预测.csv"), index=False, encoding='utf-8-sig')

# 保存成本曲线
cost_df = pd.DataFrame({
    "迭代次数": range(1, len(model.cost_history) + 1),
    "成本": model.cost_history
})
cost_df.to_csv(os.path.join(output_dir, "gd_成本曲线.csv"), index=False, encoding='utf-8-sig')

print("\n=== 演示完成 ===")
