"""
Author:Administrator
Date:2026/5/18
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import create_output_dir, save_figure, set_chinese_plot_style

set_chinese_plot_style()

url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)

from sklearn.model_selection import train_test_split

X = df.drop('Survived',axis=1)
y = df['Survived']

X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

# 缺失值统计
missing = X_train.isnull().sum()
missing_pct = (missing/len(X_train)*100).round(1)

# 只保留“缺失数”这一列的值大于0的行。其效果相当于 df[df["缺失数"] > 0]，但写法更简洁。
print(pd.DataFrame({"缺失数":missing, "缺失率":missing_pct}).query("缺失数 > 0"))
print(X_train.describe().round(2))

# ===== EDA 可视化 =====
fig, axes = plt.subplots(3, 2, figsize=(14, 15))
fig.suptitle('Titanic 数据探索性分析', fontsize=18, fontweight='bold', y=1.01)

# ----- 1. 存活率分布 -----
ax1 = axes[0, 0]
survive_counts = df['Survived'].value_counts()
survive_labels = ['未存活', '存活']
sns.countplot(data=df, x='Survived', hue='Survived', ax=ax1,
              palette=['#e74c3c', '#2ecc71'], legend=False, width=0.5)
ax1.set_xticks([0, 1])
ax1.set_xticklabels(survive_labels, fontsize=12)
ax1.set_title('存活率分布', fontsize=14, fontweight='bold', pad=10)
ax1.set_ylabel('人数', fontsize=12)
ax1.set_xlabel('')
# 标注百分比
total = len(df)
for p in ax1.patches:
    height = p.get_height()
    pct = height / total * 100
    ax1.text(p.get_x() + p.get_width() / 2, height + 5,
             f'{height}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=11, fontweight='bold')

# ----- 2. 年龄分布 -----
ax2 = axes[0, 1]
sns.histplot(data=df, x='Age', bins=30, kde=True, ax=ax2,
             color='#3498db', edgecolor='white', linewidth=0.5)
ax2.axvline(df['Age'].mean(), color='#e74c3c', linestyle='--', linewidth=1.5, label=f'均值: {df["Age"].mean():.1f}')
ax2.axvline(df['Age'].median(), color='#f39c12', linestyle='--', linewidth=1.5, label=f'中位数: {df["Age"].median():.1f}')
ax2.set_title('年龄分布', fontsize=14, fontweight='bold', pad=10)
ax2.set_ylabel('人数', fontsize=12)
ax2.set_xlabel('年龄', fontsize=12)
ax2.legend(fontsize=10)

# ----- 3. 票价分布 -----
ax3 = axes[1, 0]
sns.histplot(data=df, x='Fare', bins=40, kde=True, ax=ax3,
             color='#9b59b6', edgecolor='white', linewidth=0.5)
ax3.axvline(df['Fare'].mean(), color='#e74c3c', linestyle='--', linewidth=1.5, label=f'均值: {df["Fare"].mean():.1f}')
ax3.axvline(df['Fare'].median(), color='#f39c12', linestyle='--', linewidth=1.5, label=f'中位数: {df["Fare"].median():.1f}')
ax3.set_title('票价分布', fontsize=14, fontweight='bold', pad=10)
ax3.set_ylabel('人数', fontsize=12)
ax3.set_xlabel('票价', fontsize=12)
ax3.legend(fontsize=10)

# ----- 4. 性别 VS 存活 -----
ax4 = axes[1, 1]
sns.countplot(data=df, x='Sex', hue='Survived', ax=ax4,
              palette=['#e74c3c', '#2ecc71'], width=0.5)
ax4.set_xticks([0, 1])
ax4.set_xticklabels(['男性', '女性'], fontsize=12)
ax4.set_title('性别 VS 存活', fontsize=14, fontweight='bold', pad=10)
ax4.set_ylabel('人数', fontsize=12)
ax4.set_xlabel('')
ax4.legend(['未存活', '存活'], fontsize=11)
# 标注数值
for p in ax4.patches:
    height = p.get_height()
    if height > 0:
        ax4.text(p.get_x() + p.get_width() / 2, height + 3,
                 f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# ----- 5. 船票等级 VS 存活 -----
ax5 = axes[2, 0]
pclass_labels = ['头等舱', '二等舱', '三等舱']
sns.countplot(data=df, x='Pclass', hue='Survived', ax=ax5,
              palette=['#e74c3c', '#2ecc71'], width=0.5)
ax5.set_xticks([1, 2, 3])
ax5.set_xticklabels(pclass_labels, fontsize=12)
ax5.set_title('船票等级 VS 存活', fontsize=14, fontweight='bold', pad=10)
ax5.set_ylabel('人数', fontsize=12)
ax5.set_xlabel('')
ax5.legend(['未存活', '存活'], fontsize=11)
# 标注数值
for p in ax5.patches:
    height = p.get_height()
    if height > 0:
        ax5.text(p.get_x() + p.get_width() / 2, height + 3,
                 f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# ----- 6. 相关性热力图 -----
ax6 = axes[2, 1]
numeric_cols = df.select_dtypes(include='number')
corr = numeric_cols.corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            ax=ax6, linewidths=0.5, vmin=-1, vmax=1,
            annot_kws={'fontsize': 9})
ax6.set_title('特征相关性热力图', fontsize=14, fontweight='bold', pad=10)

plt.tight_layout()
output_dir = create_output_dir()
save_figure(output_dir, "fig_titanic_eda.png", fig=fig, dpi=200)
plt.show()
