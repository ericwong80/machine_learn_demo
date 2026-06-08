"""
线性回归示例 - 身高预测体重
Author:13928
Date:2026/5/22
"""
import numpy as np  # 导入NumPy库，用于数值计算
import pandas as pd  # 导入Pandas库，用于数据处理
import matplotlib.pyplot as plt  # 导入Matplotlib库，用于绘图
import seaborn as sns  # 导入Seaborn库，用于数据可视化
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # 导入线性回归模型
from utils import create_output_dir, save_figure, set_chinese_plot_style

# 设置 seaborn 风格与中文字体
set_chinese_plot_style()  # 设置Seaborn主题样式，使用黑体，并解决负号显示问题

# 1. 准备数据
X_train = np.array([[160], [166], [172], [174], [180]])  # 训练数据：身高（厘米）
y_train = np.array([56.3, 60.6, 65.1, 68.5, 75])  # 训练数据：对应体重（千克）
X_test = np.array([[176]])  # 测试数据：身高（厘米）

# 2. 模型训练
estimator = LinearRegression()  # 创建线性回归模型实例
estimator.fit(X_train, y_train)  # 使用训练数据拟合模型

print(f"系数：{estimator.coef_}")  # 输出模型系数（斜率）
print(f"截距：{estimator.intercept_}")  # 输出模型截距

# 3. 预测
y_pred_train = estimator.predict(X_train)  # 对训练数据进行预测
y_pred_test = estimator.predict(X_test)  # 对测试数据进行预测
print(f"176cm 预测体重：{y_pred_test[0]:.1f} kg")  # 输出176cm身高的预测体重，保留一位小数

# 3.5 模型评估
# 计算模型在训练集上的评估指标
mae = mean_absolute_error(y_train, y_pred_train)  # 计算平均绝对误差(MAE)
mse = mean_squared_error(y_train, y_pred_train)   # 计算均方误差(MSE)
r2 = r2_score(y_train, y_pred_train)            # 计算决定系数(R²)


# 打印评估结果，保留指定小数位数
print(f"MAE 平均绝对误差：{mae:.2f} kg")  # 输出MAE，保留2位小数，单位为kg
print(f"MSE 均方误差：{mse:.2f}")        # 输出MSE，保留2位小数
print(f"R² 决定系数：{r2:.3f}")          # 输出R²，保留3位小数

# 4. seaborn 可视化
output_dir = create_output_dir()
df = pd.DataFrame({"身高": X_train.flatten(), "体重": y_train})  # 创建包含训练数据的DataFrame

fig, axes = plt.subplots(1, 3, figsize=(16, 5))  # 创建一个1行3列的子图，设置总大小为16x5英寸

# ── 图1：散点图 + 回归线 ──────────────────────────────────────
ax1 = axes[0]  # 选择第一个子图
sns.regplot(  # 绘制回归散点图
    data=df, x="身高", y="体重", ax=ax1,  # 使用DataFrame数据，x轴为身高，y轴为体重
    scatter_kws={"s": 80, "color": "#3498db", "edgecolor": "white", "zorder": 3},  # 设置散点样式
    line_kws={"color": "#e74c3c", "linewidth": 2},  # 设置回归线样式
)
# 标注预测点
ax1.scatter(X_test.flatten(), y_pred_test, s=120, color="#e67e22", marker="*", zorder=4, label=f"预测 176cm→{y_pred_test[0]:.1f}kg")  # 添加测试点预测值
ax1.legend(fontsize=10)  # 添加图例，设置字体大小
ax1.set_title("身高 vs 体重（回归线）", fontsize=13, fontweight="bold", pad=10)  # 设置标题
ax1.set_xlabel("身高 (cm)", fontsize=11)  # 设置x轴标签
ax1.set_ylabel("体重 (kg)", fontsize=11)  # 设置y轴标签

# ── 图2：实际值 vs 预测值 ─────────────────────────────────────
ax2 = axes[1]  # 选择第二个子图
compare_df = pd.DataFrame({  # 创建比较实际值和预测值的DataFrame
    "样本": [f"{h}cm" for h in X_train.flatten()] * 2,  # 身高标签
    "体重": np.concatenate([y_train, y_pred_train]),  # 实际体重和预测体重
    "类型": ["实际值"] * len(y_train) + ["预测值"] * len(y_train),  # 值类型
})
sns.barplot(data=compare_df, x="样本", y="体重", hue="类型", ax=ax2, width=0.6)  # 绘制柱状图
for container in ax2.containers:  # 为每个柱子添加数值标签
    ax2.bar_label(container, fmt="%.1f", fontsize=9, padding=3)
ax2.set_title("实际值 vs 预测值", fontsize=13, fontweight="bold", pad=10)  # 设置标题
ax2.set_ylabel("体重 (kg)", fontsize=11)  # 设置y轴标签
ax2.legend(fontsize=10)  # 添加图例

# ── 图3：残差图 ───────────────────────────────────────────────
ax3 = axes[2]  # 选择第三个子图
residuals = y_train - y_pred_train  # 计算残差（实际值-预测值）
residual_df = pd.DataFrame({"身高": X_train.flatten(), "残差": residuals})  # 创建残差DataFrame
sns.barplot(data=residual_df, x="身高", y="残差", hue="身高", ax=ax3,  # 绘制残差柱状图
            palette=["#2ecc71" if r >= 0 else "#e74c3c" for r in residuals],
            legend=False, width=0.5)  # 根据残差正负设置颜色
ax3.axhline(0, color="gray", linestyle="--", linewidth=0.8)  # 添加零线
for i, r in enumerate(residuals):  # 为每个残差添加数值标签
    va = "bottom" if r >= 0 else "top"  # 根据残差正负确定标签位置
    ax3.text(i, r + (0.15 if r >= 0 else -0.15), f"{r:+.2f}", ha="center", va=va, fontsize=10, fontweight="bold")
ax3.set_title("残差分布", fontsize=13, fontweight="bold", pad=10)  # 设置标题
ax3.set_ylabel("残差 (kg)", fontsize=11)  # 设置y轴标签

# ── 保存 ──────────────────────────────────────────────────────
fig.suptitle("线性回归分析 — 身高预测体重", fontsize=15, fontweight="bold", y=1.02)  # 设置总标题
plt.tight_layout()  # 自动调整子图间距
save_figure(output_dir, "fig_linear_regression.png", fig=fig, dpi=150)  # 保存图像，设置DPI和边界
plt.show()  # 显示图像
