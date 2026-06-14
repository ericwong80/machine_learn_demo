"""
OLS普通线性回归演示
使用普通最小二乘法(OLS)进行线性回归建模
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

from utils.dataset import prepare_data
from utils.metrics import calculate_metrics, format_metrics
from utils.plotting import plot_actual_vs_predicted, plot_residuals, set_chinese_plot_style
from utils.output import create_output_dir, save_figure, print_model_metrics

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
model = LinearRegression()
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
print("线性回归模型公式：")
equation_parts = [f"{coef:+.2f}*{name}" for name, coef in zip(feature_names, model.coef_)]
print(f"房价 = {model.intercept_:.2f} " + " ".join(equation_parts))

# 特征重要性（标准化系数）
x_std = X_train.std(axis=0, ddof=0)
y_std = y_train.std(ddof=0)
standardized_coef = model.coef_ * x_std / y_std

print("\n特征标准化影响（按重要性排序）：")
importance_df = pd.DataFrame({
    "特征": feature_names,
    "原始系数": model.coef_,
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
save_figure(fig1, "ols_实际值vs预测值.png", output_dir)
plt.close(fig1)

# 残差图
fig2 = plot_residuals(
    y_test, y_test_pred,
    title="测试集：残差图",
    xlabel="样本",
    ylabel="残差（万元）"
)
save_figure(fig2, "ols_残差图.png", output_dir)
plt.close(fig2)

# ===== 7. 新样本预测 =====
print("\n=== 新样本预测 ===")
# 创建一个新样本
X_new = np.array([[100, 14, 0.9, 8]], dtype=float)
y_new_pred = model.predict(X_new)[0]

print(f"新样本预测：")
print(f"面积={X_new[0, 0]:.0f}㎡，楼层={X_new[0, 1]:.0f}，"
      f"距地铁={X_new[0, 2]:.1f}km，房龄={X_new[0, 3]:.0f}年 -> "
      f"{y_new_pred:.2f} 万元")

# 预测贡献分析
contributions = model.coef_ * X_new.flatten()
contrib_df = pd.DataFrame({
    "项目": ["截距"] + feature_names,
    "贡献": [model.intercept_] + contributions.tolist(),
})

print("\n预测贡献分析：")
for _, row in contrib_df.iterrows():
    print(f"{row['项目']}: {row['贡献']:+.2f} 万元")

# ===== 8. 保存结果 =====
print("\n=== 保存结果 ===")
# 保存指标
save_metrics(test_metrics, "ols_测试集指标.json", output_dir)

# 保存特征重要性
importance_df.to_csv(os.path.join(output_dir, "ols_特征重要性.csv"), index=False, encoding='utf-8-sig')

# 保存新样本预测结果
new_pred_df = pd.DataFrame({
    "样本": ["新样本"],
    "面积": [X_new[0, 0]],
    "楼层": [X_new[0, 1]],
    "距地铁": [X_new[0, 2]],
    "房龄": [X_new[0, 3]],
    "预测房价": [y_new_pred]
})
new_pred_df.to_csv(os.path.join(output_dir, "ols_新样本预测.csv"), index=False, encoding='utf-8-sig')

print("\n=== 演示完成 ===")
