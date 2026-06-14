"""
Author: Administrator
Date: 2026/5/17
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import create_output_dir, save_figure, set_chinese_plot_style

set_chinese_plot_style()

np.random.seed(42)

# ============================================================
# 1. 生成更适合学习梯度下降的数据
# ============================================================

n_samples = 50

# 为了保持最终输出不变，仍然使用“房屋面积”这个业务含义
areas_m2 = np.linspace(90, 150, n_samples)

# 真实规律仍然不变
true_w = 250
true_b = 30

# 噪声适当减小，方便初学者观察拟合效果
noise = np.random.normal(loc=0, scale=80, size=n_samples)

prices = true_w * areas_m2 + true_b + noise

print(f"房屋数量：{len(areas_m2)}套")
print(f"面积范围：{areas_m2.min():.1f} - {areas_m2.max():.1f} 平方米")
print(f"价格范围：{prices.min():.1f} - {prices.max():.1f} 元")
print(f"真实规律：房价 = 250 * 面积(平方米) + 30")


# ============================================================
# 2. 特征标准化
# ============================================================
"""
为什么要标准化？

原始面积是 90~150，数值比较大。
如果直接做梯度下降，由于损失函数的梯度与特征值成正比，会导致梯度非常大，容易越过最优解（发散）。

标准化之后：
面积会变成均值为 0，标准差为 1 的数据。

这样学习率可以设大一些，训练过程更稳定，也更适合教学。
"""

area_mean = areas_m2.mean()
area_std = areas_m2.std()

areas_scaled = (areas_m2 - area_mean) / area_std

print("\n标准化信息：")
print(f"面积均值：{area_mean:.2f}")
print(f"面积标准差：{area_std:.2f}")
print(f"标准化后范围：{areas_scaled.min():.2f} - {areas_scaled.max():.2f}")


# ============================================================
# 3. 梯度下降
# ============================================================

def gradient_descent(X, y, lr=0.05, epochs=300):
    """
    梯度下降算法实现线性回归

    注意：
    这里传入的 X 是标准化后的面积，不是原始面积。

    模型：
        y_pred = w_scaled * X_scaled + b_scaled

    损失：
        MSE = mean((y_pred - y) ** 2)

    返回：
        w_scaled, b_scaled, history
    """
    w, b = 0.0, 0.0
    n = len(X)

    initial_loss = np.mean((w * X + b - y) ** 2)

    history = {
        'w': [w],
        'b': [b],
        'loss': [initial_loss]
    }

    print("\n开始训练...")

    for epoch in range(epochs):
        y_pred = w * X + b

        error = y_pred - y

        dw = (2 / n) * np.sum(error * X)
        db = (2 / n) * np.sum(error)

        w = w - lr * dw
        b = b - lr * db

        current_loss = np.mean((w * X + b - y) ** 2)

        history['w'].append(w)
        history['b'].append(b)
        history['loss'].append(current_loss)

        if epoch % 30 == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch:3d}/{epochs}, "
                f"w_scaled={w:10.4f}, "
                f"b_scaled={b:10.4f}, "
                f"Loss={current_loss:12.4f}"
            )

    return w, b, history


# ============================================================
# 4. 使用标准化数据训练
# ============================================================

w_scaled, b_scaled, history_scaled = gradient_descent(
    areas_scaled,
    prices,
    lr=0.05,
    epochs=300
)


# ============================================================
# 5. 把标准化参数还原为原始面积尺度
# ============================================================
"""
训练时模型是：

    price = w_scaled * area_scaled + b_scaled

而：

    area_scaled = (area - area_mean) / area_std

代入可得：

    price = w_scaled * ((area - area_mean) / area_std) + b_scaled

展开：

    price = (w_scaled / area_std) * area + (b_scaled - w_scaled * area_mean / area_std)

所以：

    w = w_scaled / area_std
    b = b_scaled - w_scaled * area_mean / area_std

这样最终得到的 w、b 仍然是原始面积尺度下的参数。
"""

w = w_scaled / area_std
b = b_scaled - w_scaled * area_mean / area_std

# 为了兼容你后面的可视化代码，把 history 也还原成原始尺度
history = {
    'w': [],
    'b': [],
    'loss': history_scaled['loss']
}

for ws, bs in zip(history_scaled['w'], history_scaled['b']):
    w_original = ws / area_std
    b_original = bs - ws * area_mean / area_std

    history['w'].append(w_original)
    history['b'].append(b_original)


print(f"\n拟合结果：w = {w:.2f}, b = {b:.2f}")
print(f"真实参数：w = 250.00, b = 30.00")
print(f"误差：w误差 = {abs(w-250):.2f}, b误差 = {abs(b-30):.2f}")


# 可视化结果
plt.figure(figsize=(15, 10))

# 1. 数据点和拟合直线
plt.subplot(2, 2, 1)
plt.scatter(areas_m2, prices, alpha=0.6, label='真实数据', color='blue')
x_line = np.linspace(areas_m2.min(), areas_m2.max(), 100)
y_line = w * x_line + b
plt.plot(x_line, y_line, 'r-', linewidth=2, label=f'拟合直线: y = {w:.2f}x + {b:.2f}')
plt.xlabel('房屋面积 (平方米)', fontsize=12)
plt.ylabel('房价 (元)', fontsize=12)
plt.title('房屋面积与房价关系', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3, linestyle='--')

# 2. 损失函数变化
plt.subplot(2, 2, 2)
valid_losses = np.array(history['loss'])
valid_losses = valid_losses[np.isfinite(valid_losses)]
plt.plot(valid_losses, 'b-', linewidth=2)
plt.xlabel('迭代次数', fontsize=12)
plt.ylabel('损失值 (MSE)', fontsize=12)
plt.title('训练过程损失变化', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle='--')
plt.yscale('log')

# 3. w参数变化
plt.subplot(2, 2, 3)
valid_w = np.array(history['w'])[np.isfinite(history['w'])]
plt.plot(valid_w, 'g-', linewidth=2, label='训练值')
plt.axhline(y=250, color='r', linestyle='--', linewidth=2, label='真实值 (250)')
plt.xlabel('迭代次数', fontsize=12)
plt.ylabel('权重 w', fontsize=12)
plt.title('权重w训练过程变化', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3, linestyle='--')

# 4. b参数变化
plt.subplot(2, 2, 4)
valid_b = np.array(history['b'])[np.isfinite(history['b'])]
plt.plot(valid_b, 'm-', linewidth=2, label='训练值')
plt.axhline(y=30, color='r', linestyle='--', linewidth=2, label='真实值 (30)')
plt.xlabel('迭代次数', fontsize=12)
plt.ylabel('偏置 b', fontsize=12)
plt.title('偏置b训练过程变化', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout(pad=3.0)
output_dir = create_output_dir()
save_figure(output_dir, 'linear_regression_results_safe.png', dpi=300)
plt.show()

# 预测新数据
new_area = 120
predicted_price = w * new_area + b
true_price = 250 * new_area + 30
print(f"\n预测结果：")
print(f"面积 = {new_area} 平方米")
print(f"预测房价 = {predicted_price:.2f} 元")
print(f"真实房价 = {true_price:.2f} 元")
print(f"预测误差 = {abs(predicted_price - true_price):.2f} 元")
