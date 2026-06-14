"""
Author:Administrator
Date:2026/6/13
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from utils import create_output_dir, save_figure, set_chinese_plot_style, show_or_close

# ===== 输出目录与绘图风格 =====
output_dir = create_output_dir()
set_chinese_plot_style(figure_facecolor="#f7f9fc", axes_facecolor="#ffffff")

# ===== 1. 准备数据 =====
feature_names = ["面积", "楼层", "距地铁", "房龄"]
from src.day01_linear_regression_models.plot_linear_regression import X, y
data = pd.DataFrame(X, columns=feature_names)
data["房价"] = y
data["样本"] = [f"S{i + 1:02d}" for i in range(len(data))]

# ===== 2. 数据分割 =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# ===== 3. 训练决策树回归模型 =====
dt_regressor = DecisionTreeRegressor(
    max_depth=3,           # 限制树深度，避免过拟合
    min_samples_split=2,   # 最小分裂样本数
    random_state=42
)
dt_regressor.fit(X_train, y_train)

# ===== 4. 训练线性回归模型（对比）=====
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# ===== 5. 预测 =====
dt_pred_train = dt_regressor.predict(X_train)
dt_pred_test = dt_regressor.predict(X_test)

lr_pred_train = lr_model.predict(X_train)
lr_pred_test = lr_model.predict(X_test)

# ===== 6. 模型评估 =====
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {
        '模型': model_name,
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2
    }

# 评估训练集和测试集
dt_train_metrics = evaluate_model(y_train, dt_pred_train, '决策树(训练集)')
dt_test_metrics = evaluate_model(y_test, dt_pred_test, '决策树(测试集)')
lr_train_metrics = evaluate_model(y_train, lr_pred_train, '线性回归(训练集)')
lr_test_metrics = evaluate_model(y_test, lr_pred_test, '线性回归(测试集)')

# 创建评估结果DataFrame
metrics_df = pd.DataFrame([
    dt_train_metrics, dt_test_metrics,
    lr_train_metrics, lr_test_metrics
])

# ===== 7. 控制台输出 =====
print("=" * 60)
print("房价预测回归模型评估结果")
print("=" * 60)
print("\n模型评估指标：")
print(metrics_df.round(2).to_string(index=False))

print("\n决策树特征重要性：")
for feature, importance in zip(feature_names, dt_regressor.feature_importances_):
    print(f"{feature}: {importance:.4f}")

# ===== 8. 创建可视化仪表盘 =====
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("决策树回归分析：房价预测", fontsize=18, fontweight="bold", y=0.98)

# 图1：决策树结构可视化
ax1 = axes[0, 0]
plot_tree(dt_regressor,
          feature_names=feature_names,
          filled=True,
          rounded=True,
          fontsize=9,
          ax=ax1,
          impurity=False,
          proportion=False)
ax1.set_title("决策树回归结构 (max_depth=3)", fontsize=14, fontweight="bold")

# 图2：预测值 vs 真实值对比
ax2 = axes[0, 1]

# 准备数据
comparison_data = pd.DataFrame({
    '样本': np.concatenate([data['样本'][X_train.shape[0]:], data['样本'][X_train.shape[0]:]]),
    '真实房价': np.concatenate([y_test, y_test]),
    '预测房价': np.concatenate([dt_pred_test, lr_pred_test]),
    '模型': ['决策树'] * len(y_test) + ['线性回归'] * len(y_test)
})

# 绘制对比
sns.scatterplot(data=comparison_data, x='真实房价', y='预测房价', hue='模型',
                style='模型', s=150, ax=ax2, alpha=0.8)

# 添加45度参考线
min_val = min(y.min(), dt_pred_test.min(), lr_pred_test.min())
max_val = max(y.max(), dt_pred_test.max(), lr_pred_test.max())
ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, linewidth=2, label='完美预测')

ax2.set_title("预测值 vs 真实值对比", fontsize=14, fontweight="bold")
ax2.set_xlabel("真实房价 (万元)")
ax2.set_ylabel("预测房价 (万元)")
ax2.legend(title='模型', loc='best')

# 图3：特征重要性
ax3 = axes[1, 0]
feature_imp_df = pd.DataFrame({
    '特征': feature_names,
    '重要性': dt_regressor.feature_importances_
}).sort_values('重要性', ascending=False)

colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(feature_imp_df)))
bars = ax3.barh(feature_imp_df['特征'], feature_imp_df['重要性'], color=colors)
ax3.set_title("决策树特征重要性", fontsize=14, fontweight="bold")
ax3.set_xlabel("重要性得分")

# 在条形上显示数值
for i, (feature, importance) in enumerate(zip(feature_imp_df['特征'], feature_imp_df['重要性'])):
    ax3.text(importance + 0.01, i, f'{importance:.3f}',
             va='center', fontweight='bold', color='#2c3e50')

# 图4：决策边界示例（使用最重要的两个特征）
ax4 = axes[1, 1]

# 选择最重要的两个特征
top_feature1 = feature_imp_df.iloc[0]['特征']
top_feature2 = feature_imp_df.iloc[1]['特征']
idx1 = feature_names.index(top_feature1)
idx2 = feature_names.index(top_feature2)

# 创建网格
x_min, x_max = X[:, idx1].min() - 5, X[:, idx1].max() + 5
y_min, y_max = X[:, idx2].min() - 1, X[:, idx2].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                     np.linspace(y_min, y_max, 50))

# 预测网格点（使用平均值填充其他特征）
grid_points = np.zeros((xx.ravel().shape[0], X.shape[1]))
grid_points[:, idx1] = xx.ravel()
grid_points[:, idx2] = yy.ravel()

# 用训练集均值填充其他特征
for i in range(X.shape[1]):
    if i != idx1 and i != idx2:
        grid_points[:, i] = X_train[:, i].mean()

# 预测
Z = dt_regressor.predict(grid_points)
Z = Z.reshape(xx.shape)

# 绘制决策边界
contour = ax4.contourf(xx, yy, Z, alpha=0.3, cmap='viridis', levels=10)
scatter = ax4.scatter(X[:, idx1], X[:, idx2], c=y, cmap='viridis',
                     s=100, edgecolors='k', alpha=0.8)

ax4.set_title(f"决策边界：{top_feature1} vs {top_feature2}", fontsize=14, fontweight="bold")
ax4.set_xlabel(top_feature1)
ax4.set_ylabel(top_feature2)
cbar = plt.colorbar(scatter, ax=ax4)
cbar.set_label('房价 (万元)')

# 添加网格线
for ax in axes.flatten():
    ax.grid(True, alpha=0.25, linestyle='--')

plt.tight_layout(rect=(0, 0, 1, 0.96))

# 保存和显示图形
save_figure(output_dir, "fig_decision_tree_regression.png", fig=fig, dpi=200)
show_or_close(fig)

# ===== 9. 额外分析：不同深度的决策树效果 =====
depths = [1, 2, 3, 4, 5, 10]
train_scores = []
test_scores = []

for depth in depths:
    dt = DecisionTreeRegressor(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)
    train_scores.append(r2_score(y_train, dt.predict(X_train)))
    test_scores.append(r2_score(y_test, dt.predict(X_test)))

# 创建深度分析图
fig2, ax = plt.subplots(figsize=(10, 6))
ax.plot(depths, train_scores, 'bo-', linewidth=2.5, markersize=10, label='训练集 R²')
ax.plot(depths, test_scores, 'ro-', linewidth=2.5, markersize=10, label='测试集 R²')

ax.set_title("决策树深度对模型性能的影响", fontsize=16, fontweight="bold")
ax.set_xlabel("树的最大深度", fontsize=12)
ax.set_ylabel("R² 分数", fontsize=12)
ax.set_xticks(depths)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12)

# 标记最佳深度
best_depth_idx = np.argmax(test_scores)
best_depth = depths[best_depth_idx]
best_score = test_scores[best_depth_idx]
ax.annotate(f'最佳深度: {best_depth}\nR²: {best_score:.3f}',
            xy=(best_depth, best_score),
            xytext=(best_depth+0.5, best_score-0.1),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f4f8", edgecolor="#2980b9"))

plt.tight_layout()
save_figure(output_dir, "fig_tree_depth_analysis.png", fig=fig2, dpi=200)
show_or_close(fig2)

print("\n" + "=" * 60)
print(f"可视化已保存至: {output_dir}/fig_decision_tree_regression.png")
print(f"深度分析图已保存至: {output_dir}/fig_tree_depth_analysis.png")
print("=" * 60)