"""
XGBoost回归演示
使用XGBoost回归模型进行建模
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 导入公共模块
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.dataset import prepare_data
from common.metrics import calculate_metrics
from common.plotting import plot_actual_vs_predicted, plot_residuals, set_chinese_plot_style
from common.output import create_output_dir, save_figure, print_model_metrics

# 设置绘图风格
set_chinese_plot_style()

# 创建输出目录
output_dir = create_output_dir()

# ===== 1. 准备数据 =====
print("=== 准备数据 ===")
X_train, X_test, y_train, y_test, data, feature_names = prepare_data(n_samples=100, test_size=0.2, random_state=42)

print(f"训练集样本数: {X_train.shape[0]}")
print(f"测试集样本数: {X_test.shape[0]}")
print(f"特征数量: {X_train.shape[1]}")

# ===== 2. 创建和训练模型 =====
print("\n=== 创建和训练模型 ===")
# 创建XGBoost回归模型
model = xgb.XGBRegressor(
    n_estimators=100,      # 树的数量
    max_depth=6,           # 树的最大深度
    learning_rate=0.1,     # 学习率
    subsample=0.8,        # 每棵树使用的样本比例
    colsample_bytree=0.8, # 每棵树使用的特征比例
    random_state=42,
    n_jobs=-1              # 使用所有可用的CPU核心
)

# 训练模型
model.fit(X_train, y_train)

# ===== 3. 预测 =====
print("\n=== 预测 ===")
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

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
print(f"XGBoost中的树数量: {model.n_estimators}")
print(f"平均树深度: {np.mean([tree.get_depth() for tree in model.get_booster().get_dump()]):.1f}")

# 特征重要性
feature_importance = model.feature_importances_
importance_df = pd.DataFrame({
    "特征": feature_names,
    "重要性": feature_importance
}).sort_values("重要性", ascending=False)

print("\n特征重要性：")
for _, row in importance_df.iterrows():
    print(f"{row['特征']}: {row['重要性']:.4f}")

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
save_figure(fig1, "xgb_实际值vs预测值.png", output_dir)
plt.close(fig1)

# 残差图
fig2 = plot_residuals(
    y_test, y_test_pred,
    title="测试集：残差图",
    xlabel="样本",
    ylabel="残差（万元）"
)
save_figure(fig2, "xgb_残差图.png", output_dir)
plt.close(fig2)

# 特征重要性条形图
fig3, ax = plt.subplots(figsize=(10, 6))
importance_df.plot.barh(x="特征", y="重要性", ax=ax, color='skyblue')
ax.set_title("特征重要性", fontsize=14, fontweight='bold')
ax.set_xlabel("重要性")
ax.set_ylabel("特征")
ax.invert_yaxis()  # 反转y轴，使最重要的特征在顶部
save_figure(fig3, "xgb_特征重要性.png", output_dir)
plt.close(fig3)

# XGBoost树结构图（第一棵树）
fig4, ax = plt.subplots(figsize=(20, 10))
xgb.plot_tree(model, num_trees=0, ax=ax)
fig4.suptitle("XGBoost中的第一棵决策树", fontsize=16, fontweight='bold')
save_figure(fig4, "xgb_第一棵决策树结构.png", output_dir)
plt.close(fig4)

# ===== 7. 新样本预测 =====
print("\n=== 新样本预测 ===")
# 创建一个新样本
X_new = np.array([[100, 14, 0.9, 8]], dtype=float)
y_new_pred = model.predict(X_new)[0]

print(f"新样本预测：")
print(f"面积={X_new[0, 0]:.0f}㎡，楼层={X_new[0, 1]:.0f}，"
      f"距地铁={X_new[0, 2]:.1f}km，房龄={X_new[0, 3]:.0f}年 -> "
      f"{y_new_pred:.2f} 万元")

# ===== 8. 保存结果 =====
print("\n=== 保存结果 ===")
# 保存指标
save_metrics(test_metrics, "xgb_测试集指标.json", output_dir)

# 保存特征重要性
importance_df.to_csv(os.path.join(output_dir, "xgb_特征重要性.csv"), index=False, encoding='utf-8-sig')

# 保存新样本预测结果
new_pred_df = pd.DataFrame({
    "样本": ["新样本"],
    "面积": [X_new[0, 0]],
    "楼层": [X_new[0, 1]],
    "距地铁": [X_new[0, 2]],
    "房龄": [X_new[0, 3]],
    "预测房价": [y_new_pred]
})
new_pred_df.to_csv(os.path.join(output_dir, "xgb_新样本预测.csv"), index=False, encoding='utf-8-sig')

# 保存XGBoost信息
xgb_info = {
    "树数量": model.n_estimators,
    "最大深度": model.max_depth,
    "学习率": model.learning_rate,
    "特征数量": model.n_features_in_,
    "特征名称": feature_names
}
import json
with open(os.path.join(output_dir, "xgb_XGBoost信息.json"), 'w', encoding='utf-8') as f:
    json.dump(xgb_info, f, ensure_ascii=False, indent=4)

print("\n=== 演示完成 ===")
