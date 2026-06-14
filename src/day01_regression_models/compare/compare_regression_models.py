"""
回归模型横向比较
比较不同回归模型在相同数据集上的表现
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import os
from utils.dataset import prepare_data
from utils.metrics import calculate_metrics
from utils.plotting import plot_model_comparison, set_chinese_plot_style
from utils.output import create_output_dir, save_figure, save_dataframe

# 设置绘图风格
set_chinese_plot_style()

# 创建输出目录
output_dir = create_output_dir()

print(output_dir)

# ===== 1. 准备数据 =====
print("=== 准备数据 ===")
X_train, X_test, y_train, y_test, data, feature_names = prepare_data(n_samples=100, test_size=0.2, random_state=42)

print(f"训练集样本数: {X_train.shape[0]}")
print(f"测试集样本数: {X_test.shape[0]}")
print(f"特征数量: {X_train.shape[1]}")

# ===== 2. 定义和训练所有模型 =====
print("\n=== 定义和训练所有模型 ===")
models = {
    "OLS线性回归": LinearRegression(),
    "决策树回归": DecisionTreeRegressor(max_depth=4, min_samples_split=10, min_samples_leaf=5, random_state=42),
    "随机森林回归": RandomForestRegressor(n_estimators=100, max_depth=6, min_samples_split=10, 
                                           min_samples_leaf=5, random_state=42, n_jobs=-1),
    "XGBoost回归": xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, 
                                    subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
}

# 训练模型并收集结果
results = {}
for name, model in models.items():
    print(f"\n训练 {name}...")
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)

    # 计算指标
    metrics = calculate_metrics(y_test, y_pred)
    results[name] = metrics

    # 打印指标
    print(f"{name} 测试集指标:")
    print(f"MAE: {metrics['MAE']:.2f} 万元, RMSE: {metrics['RMSE']:.2f} 万元, R2: {metrics['R2']:.3f}")

# ===== 3. 模型比较 =====
print("\n=== 模型比较 ===")
# 创建比较表格
comparison_df = pd.DataFrame(results).T
print("\n模型性能比较表格:")
print(comparison_df)

# 保存比较表格
save_dataframe(comparison_df, "模型性能比较.csv", output_dir)

# ===== 4. 可视化比较结果 =====
print("\n=== 可视化比较结果 ===")
# 绘制模型比较图
fig = plot_model_comparison(comparison_df, title="回归模型性能比较", figsize=(12, 8))
save_figure(fig, "回归模型性能比较.png", output_dir)
plt.close(fig)

# 绘制实际值 vs 各模型预测值
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("各模型预测值 vs 实际值比较", fontsize=16, fontweight='normal')

for i, (name, model) in enumerate(models.items()):
    ax = axes[i//2, i%2]
    y_pred = model.predict(X_test)

    # 绘制散点图
    ax.scatter(y_test, y_pred, alpha=0.7, edgecolor='w', linewidth=0.5)

    # 绘制理想预测线（y=x）
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, label="理想预测线")

    # 设置图表属性
    ax.set_title(name, fontsize=14, fontweight='normal')
    ax.set_xlabel("实际房价（万元）", fontsize=12)
    ax.set_ylabel("预测房价（万元）", fontsize=12)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout(rect=[0, 0, 1, 0.96])
save_figure(fig, "各模型预测值vs实际值比较.png", output_dir)
plt.close(fig)

# 绘制各模型残差图
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("各模型残差图比较", fontsize=16, fontweight='normal')

for i, (name, model) in enumerate(models.items()):
    ax = axes[i//2, i%2]
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    # 绘制残差散点图
    ax.scatter(range(len(residuals)), residuals, alpha=0.7, edgecolor='w', linewidth=0.5)

    # 绘制零线
    ax.axhline(0, color='r', linestyle='--', linewidth=1.5)

    # 设置图表属性
    ax.set_title(name, fontsize=14, fontweight='normal')
    ax.set_xlabel("样本", fontsize=12)
    ax.set_ylabel("残差（万元）", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout(rect=[0, 0, 1, 0.96])
save_figure(fig, "各模型残差图比较.png", output_dir)
plt.close(fig)

# ===== 5. 特征重要性比较 =====
print("\n=== 特征重要性比较 ===")
# 收集各模型的特征重要性
feature_importance_data = {}

# 线性回归使用系数绝对值
lr_model = models["OLS线性回归"]
lr_coef = np.abs(lr_model.coef_)
lr_coef = lr_coef / lr_coef.sum()  # 归一化
feature_importance_data["OLS线性回归"] = lr_coef

# 决策树
dt_model = models["决策树回归"]
feature_importance_data["决策树回归"] = dt_model.feature_importances_

# 随机森林
rf_model = models["随机森林回归"]
feature_importance_data["随机森林回归"] = rf_model.feature_importances_

# XGBoost
xgb_model = models["XGBoost回归"]
feature_importance_data["XGBoost回归"] = xgb_model.feature_importances_

# 创建特征重要性比较表格
importance_df = pd.DataFrame(feature_importance_data, index=feature_names)
print("\n特征重要性比较表格:")
print(importance_df)

# 保存特征重要性比较表格
save_dataframe(importance_df, "特征重要性比较.csv", output_dir)

# 绘制特征重要性比较图
fig, ax = plt.subplots(figsize=(12, 8))
importance_df.plot.bar(ax=ax, width=0.7)
ax.set_title("各模型特征重要性比较", fontsize=16, fontweight='normal')
ax.set_xlabel("特征", fontsize=12)
ax.set_ylabel("重要性", fontsize=12)
ax.legend(title="模型", bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
save_figure(fig, "特征重要性比较.png", output_dir)
plt.close(fig)

# ===== 6. 总结 =====
print("\n=== 总结 ===")
# 找出最佳模型
best_model_name = comparison_df['R2'].idxmax()
print(f"\n基于R2指标，最佳模型是: {best_model_name}")
print(f"其R2值为: {comparison_df.loc[best_model_name, 'R2']:.3f}")

# 保存比较总结
summary = {
    "最佳模型": best_model_name,
    "最佳模型R2值": comparison_df.loc[best_model_name, 'R2'],
    "数据集信息": {
        "训练集样本数": X_train.shape[0],
        "测试集样本数": X_test.shape[0],
        "特征数量": X_train.shape[1],
        "特征名称": feature_names
    }
}

import json
with open(os.path.join(output_dir, "模型比较总结.json"), 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=4)

print("\n=== 比较完成 ===")
