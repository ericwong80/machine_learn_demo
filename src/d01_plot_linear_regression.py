"""
线性回归示例 - 房价预测学习仪表盘
R²决定系数表示因变量（这里指房价）的变异中可以被自变量（特征）解释的比例，取值范围：0 ≤ R² ≤ 1
demo项目未做数据集的拆分
这个 demo 不只画拟合结果，还展示：
1. 模型学到的线性公式
2. 实际值和预测值的偏差
3. 残差是否存在明显模式
4. 不同特征对预测结果的相对影响
5. 一个新样本的预测值是如何由各特征贡献叠加出来的
"""
import matplotlib.pyplot as plt  # 用于数据可视化
import numpy as np  # 用于科学计算
import pandas as pd  # 用于数据处理
import seaborn as sns  # 用于高级数据可视化
from sklearn.linear_model import LinearRegression  # 线性回归模型
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # 模型评估指标
from utils import create_output_dir, save_figure, set_chinese_plot_style, show_or_close


# ===== 输出目录与绘图风格 =====
output_dir = create_output_dir()

set_chinese_plot_style(figure_facecolor="#f7f9fc", axes_facecolor="#ffffff")


# ===== 1. 准备数据 =====
# 特征含义：
# 面积：平方米；楼层：所在楼层；距地铁：距离最近地铁站的公里数；房龄：年
feature_names = ["面积", "楼层", "距地铁", "房龄"]
# 特征数据矩阵
X = np.array(
    [
        [50, 3, 0.5, 5],
        [58, 6, 0.8, 8],
        [65, 12, 1.0, 12],
        [70, 15, 1.2, 10],
        [78, 7, 0.6, 6],
        [85, 18, 1.5, 15],
        [90, 8, 0.3, 2],
        [96, 10, 1.8, 13],
        [105, 16, 0.9, 9],
        [110, 20, 2.0, 15],
        [118, 22, 1.1, 11],
        [125, 9, 0.4, 4],
        [130, 12, 0.8, 8],
        [138, 25, 1.6, 16],
        [145, 19, 0.7, 7],
        [155, 28, 2.3, 18],
    ],
    dtype=float,
)

# 目标值：房价，单位假设为“万元”。这里保留一些噪声，让残差更接近真实建模场景。
y = np.array(
    [178, 195, 205, 218, 258, 236, 319, 255, 326, 292, 354, 393, 382, 350, 435, 395],
    dtype=float,
)

data = pd.DataFrame(X, columns=feature_names)
data["房价"] = y
data["样本"] = [f"S{i + 1:02d}" for i in range(len(data))]


# ===== 2. 训练模型与评估 =====
model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)
residuals = y - y_pred
mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

# 标准化系数用于比较“相对影响”。原始系数受单位影响，不能直接比较大小。
x_std = X.std(axis=0, ddof=0)
y_std = y.std(ddof=0)
standardized_coef = model.coef_ * x_std / y_std

coef_df = pd.DataFrame(
    {
        "特征": feature_names,
        "原始系数": model.coef_,
        "标准化影响": standardized_coef,
    }
).sort_values("标准化影响", key=lambda s: s.abs(), ascending=False)


# ===== 3. 新样本预测拆解 =====
X_new = np.array([[100, 14, 0.9, 8]], dtype=float)
y_pred_new = model.predict(X_new)[0]
contributions = model.coef_ * X_new.flatten()

contrib_df = pd.DataFrame(
    {
        "项目": ["截距"] + feature_names,
        "贡献": [model.intercept_] + contributions.tolist(),
    }
)


# ===== 4. 控制台输出 =====
equation_parts = [f"{coef:+.2f}*{name}" for name, coef in zip(feature_names, model.coef_)]
print("线性回归模型公式：")
print("房价 = " + f"{model.intercept_:.2f} " + " ".join(equation_parts))
print()
print("模型评估指标（基于训练数据）：")
print(f"MAE  平均绝对误差：{mae:.2f} 万元")
print(f"RMSE 均方根误差：{rmse:.2f} 万元")
print(f"R2   决定系数：{r2:.3f}")
print()
print("各特征原始系数与标准化影响：")
for row in coef_df.itertuples(index=False):
    print(f"{row.特征}：原始系数 {row.原始系数:+.2f}，标准化影响 {row.标准化影响:+.2f}")
print()
print(
    "新样本预测："
    f"面积={X_new[0, 0]:.0f}㎡，楼层={X_new[0, 1]:.0f}，"
    f"距地铁={X_new[0, 2]:.1f}km，房龄={X_new[0, 3]:.0f}年 -> "
    f"{y_pred_new:.2f} 万元"
)


# ===== 5. 可视化：一张图理解线性回归 =====
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle("线性回归学习仪表盘：房价预测", fontsize=18, fontweight="bold", y=0.98)

actual_pred_df = data.assign(预测值=y_pred, 残差=residuals)

# 图1：实际值 vs 预测值。越贴近对角线，拟合越好。
ax1 = axes[0, 0]
sns.scatterplot(
    data=actual_pred_df,
    x="房价",
    y="预测值",
    size="面积",
    hue="残差",
    palette="coolwarm",
    sizes=(70, 220),
    edgecolor="white",
    linewidth=1,
    ax=ax1,
)
min_price = min(y.min(), y_pred.min()) - 15
max_price = max(y.max(), y_pred.max()) + 15
ax1.plot([min_price, max_price], [min_price, max_price], "--", color="#34495e", linewidth=1.6, label="理想预测线")
for _, row in actual_pred_df.iterrows():
    ax1.text(row["房价"] + 2, row["预测值"] + 2, row["样本"], fontsize=8, color="#2c3e50")
ax1.set_xlim(min_price, max_price)
ax1.set_ylim(min_price, max_price)
ax1.set_title("实际值 vs 预测值", fontsize=14, fontweight="bold")
ax1.set_xlabel("实际房价（万元）")
ax1.set_ylabel("预测房价（万元）")
ax1.legend(loc="upper left", fontsize=9, frameon=True)

# 图2：残差分布。残差 = 实际值 - 预测值。
ax2 = axes[0, 1]
residual_palette = ["#2ecc71" if value >= 0 else "#e74c3c" for value in residuals]
sns.barplot(
    data=actual_pred_df,
    x="样本",
    y="残差",
    hue="样本",
    palette=residual_palette,
    legend=False,
    ax=ax2,
)
ax2.axhline(0, color="#34495e", linestyle="--", linewidth=1)
for i, value in enumerate(residuals):
    va = "bottom" if value >= 0 else "top"
    offset = 0.9 if value >= 0 else -0.9
    ax2.text(i, value + offset, f"{value:+.1f}", ha="center", va=va, fontsize=8)
ax2.set_title("残差图：看模型错在哪里", fontsize=14, fontweight="bold")
ax2.set_xlabel("训练样本")
ax2.set_ylabel("残差（万元）")
ax2.tick_params(axis="x", rotation=35)
ax2.text(
    0.02,
    0.95,
    f"MAE={mae:.1f}  RMSE={rmse:.1f}  R2={r2:.3f}",
    transform=ax2.transAxes,
    ha="left",
    va="top",
    fontsize=11,
    bbox={"boxstyle": "round,pad=0.35", "facecolor": "#eef3f8", "edgecolor": "#d6dee8"},
)

# 图3：标准化影响。更适合比较不同量纲的特征强弱。
ax3 = axes[1, 0]
impact_palette = ["#27ae60" if value >= 0 else "#c0392b" for value in coef_df["标准化影响"]]
sns.barplot(
    data=coef_df,
    x="标准化影响",
    y="特征",
    hue="特征",
    palette=impact_palette,
    legend=False,
    ax=ax3,
)
ax3.axvline(0, color="#34495e", linestyle="--", linewidth=1)
for i, row in enumerate(coef_df.itertuples(index=False)):
    label = f"标准化 {row.标准化影响:+.2f}\n原始 {row.原始系数:+.2f}"
    if abs(row.标准化影响) >= 0.18:
        label_x = row.标准化影响 / 2
        ha = "center"
        text_color = "white"
    else:
        label_x = row.标准化影响 + (0.015 if row.标准化影响 >= 0 else -0.015)
        ha = "left" if row.标准化影响 >= 0 else "right"
        text_color = "#2c3e50"
    ax3.text(
        label_x,
        i,
        label,
        ha=ha,
        va="center",
        fontsize=9,
        color=text_color,
        fontweight="bold" if text_color == "white" else "normal",
    )
ax3.set_xlim(-0.42, 1.08)
ax3.set_title("特征影响：标准化后再比较", fontsize=14, fontweight="bold")
ax3.set_xlabel("标准化影响强度")
ax3.set_ylabel("")

# 图4：新样本预测拆解。线性回归就是“截距 + 各特征贡献”的加法模型。
ax4 = axes[1, 1]
running_total = np.cumsum(contrib_df["贡献"])
starts = running_total - contrib_df["贡献"]
colors = ["#95a5a6"] + ["#27ae60" if value >= 0 else "#c0392b" for value in contributions]
for i, row in contrib_df.iterrows():
    ax4.bar(
        row["项目"],
        row["贡献"],
        bottom=starts[i],
        color=colors[i],
        edgecolor="white",
        linewidth=1,
    )
    label_y = starts[i] + row["贡献"] / 2
    ax4.text(i, label_y, f"{row['贡献']:+.1f}", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
ax4.axhline(y_pred_new, color="#f39c12", linestyle="--", linewidth=2, label=f"预测值 {y_pred_new:.1f} 万元")
ax4.set_title("新样本预测拆解", fontsize=14, fontweight="bold")
ax4.set_ylabel("累计预测值（万元）")
ax4.tick_params(axis="x", rotation=20)
ax4.legend(loc="upper left", fontsize=10)
ax4.text(
    0.02,
    0.04,
    "线性回归核心：预测值 = 截距 + Σ(特征值 × 系数)",
    transform=ax4.transAxes,
    ha="left",
    va="bottom",
    fontsize=11,
    bbox={"boxstyle": "round,pad=0.35", "facecolor": "#fff7e6", "edgecolor": "#f5d08a"},
)

for ax in axes.flatten():
    ax.grid(True, alpha=0.25, linestyle="--")

plt.tight_layout(rect=[0, 0, 1, 0.96])
save_figure(output_dir, "fig_linear_regression_dashboard.png", fig=fig, dpi=200)
show_or_close(fig)
