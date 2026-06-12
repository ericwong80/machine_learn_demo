"""
Author:Administrator
Date:2026/5/20
鸢尾花数据集 EDA（探索性数据分析）
"""
from __future__ import annotations

from sklearn.datasets import load_iris
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import warnings
from utils import create_output_dir, save_figure, set_chinese_plot_style

set_chinese_plot_style(font_scale=1.1)
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# 类别名称映射
TARGET_NAMES = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
FEATURE_NAMES_CN = {
    'sepal length (cm)': '花萼长度',
    'sepal width (cm)': '花萼宽度',
    'petal length (cm)': '花瓣长度',
    'petal width (cm)': '花瓣宽度',
}


def _load_iris():
    """加载鸢尾花数据集并返回DataFrame和原始数据"""
    iris_data = load_iris()
    df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
    df['target'] = iris_data.target
    df['species'] = df['target'].map(TARGET_NAMES)
    print(f"数据描述：{iris_data.DESCR}")
    return df, iris_data

def eda_basic_info(df):
    """1. 数据基本信息"""
    print("=" * 60)
    print("1. 数据基本信息")
    print("=" * 60)
    print(f"数据集形状: {df.shape[0]} 行 x {df.shape[1]} 列")
    print(f"特征数量: {df.shape[1] - 2} (排除target和species)")
    print(f"类别数量: {df['target'].nunique()}")
    print(f"类别名称: {list(TARGET_NAMES.values())}")
    print(f"\n数据类型:\n{df.dtypes}")
    print(f"\n缺失值统计:\n{df.isnull().sum()}")
    print(f"\n重复行数量: {df.duplicated().sum()}")


def eda_descriptive_stats(df):
    """2. 描述性统计"""
    feature_cols = [c for c in df.columns if c not in ('target', 'species')]

    print("\n" + "=" * 60)
    print("2. 描述性统计（整体）")
    print("=" * 60)
    print(df[feature_cols].describe().round(2))

    print("\n" + "=" * 60)
    print("3. 描述性统计（按类别）")
    print("=" * 60)
    for name, group in df.groupby('species'):
        print(f"\n--- {name} ---")
        print(group[feature_cols].describe().round(2))


def eda_class_distribution(df, output_dir: str):
    """4. 类别分布"""
    print("\n" + "=" * 60)
    print("4. 类别分布")
    print("=" * 60)
    counts = df['species'].value_counts()
    print(counts)
    print(f"\n各类别占比:")
    for name, cnt in counts.items():
        print(f"  {name}: {cnt / len(df) * 100:.1f}%")

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # 柱状图
    counts.plot.bar(ax=axes[0], color=['#4C72B0', '#DD8452', '#55A868'], rot=0)
    axes[0].set_title('Class Count')
    axes[0].set_xlabel('Species')
    axes[0].set_ylabel('Count')

    # 饼图
    axes[1].pie(counts, labels=counts.index, autopct='%1.1f%%',
                colors=['#4C72B0', '#DD8452', '#55A868'], startangle=90)
    axes[1].set_title('Class Proportion')

    plt.tight_layout()
    save_figure(output_dir, "01_class_distribution.png", fig=fig, dpi=200)
    plt.show()


def eda_distribution(df, output_dir: str):
    """5. 各特征分布（KDE核密度估计）"""
    feature_cols = [c for c in df.columns if c not in ('target', 'species')]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, col in enumerate(feature_cols):
        for species in TARGET_NAMES.values():
            subset = df[df['species'] == species][col]
            subset.plot.kde(ax=axes[i], label=species)
        cn = FEATURE_NAMES_CN.get(col, col)
        axes[i].set_title(f'{cn} Distribution (KDE)')
        axes[i].set_xlabel(col)
        axes[i].legend()

    plt.suptitle('Feature KDE by Class', fontsize=14, y=1.02)
    plt.tight_layout()
    save_figure(output_dir, "02_feature_distribution.png", dpi=200)
    plt.show()


def eda_boxplot(df, output_dir: str):
    """6. 箱线图"""
    feature_cols = [c for c in df.columns if c not in ('target', 'species')]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, col in enumerate(feature_cols):
        sns.boxplot(data=df, x='species', y=col, ax=axes[i],
                    palette='Set2', hue='species', legend=False)
        cn = FEATURE_NAMES_CN.get(col, col)
        axes[i].set_title(f'{cn} Boxplot')
        axes[i].set_xlabel('Species')
        axes[i].set_ylabel(col)

    plt.suptitle('Feature Boxplot by Class', fontsize=14, y=1.02)
    plt.tight_layout()
    save_figure(output_dir, "03_feature_boxplot.png", dpi=200)
    plt.show()


def eda_violin(df, output_dir: str):
    """7. 小提琴图"""
    feature_cols = [c for c in df.columns if c not in ('target', 'species')]

    # 将宽表转长表
    df_melted = df.melt(id_vars=['target', 'species'],
                        value_vars=feature_cols,
                        var_name='feature',
                        value_name='value')

    plt.figure(figsize=(14, 6))
    sns.violinplot(data=df_melted, x='feature', y='value', hue='species',
                   palette='Set2', split=False)
    plt.title('Feature Violin Plot by Class', fontsize=14)
    plt.xlabel('Feature')
    plt.ylabel('Value')
    plt.xticks(ticks=range(len(feature_cols)),
               labels=[FEATURE_NAMES_CN.get(c, c) for c in feature_cols])
    plt.tight_layout()
    save_figure(output_dir, "04_feature_violin.png", dpi=200)
    plt.show()


def eda_correlation(df, output_dir: str):
    """8. 相关性分析"""
    feature_cols = [c for c in df.columns if c not in ('target', 'species')]

    print("\n" + "=" * 60)
    print("5. 相关性分析")
    print("=" * 60)
    corr = df[feature_cols + ['target']].corr(numeric_only=True)
    print(corr.round(3))

    print("\nCorrelation with target (desc):")
    print(corr['target'].drop('target').sort_values(ascending=False).round(3))

    # 热力图
    plt.figure(figsize=(8, 6))
    labels = [FEATURE_NAMES_CN.get(c, c) if c != 'target' else 'Target' for c in corr.columns]
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                square=True, linewidths=0.5,
                xticklabels=labels, yticklabels=labels)
    plt.title('Correlation Heatmap', fontsize=14)
    plt.tight_layout()
    save_figure(output_dir, "05_correlation_heatmap.png", dpi=200)
    plt.show()


def eda_pairplot(df, output_dir: str):
    """9. 配对散点图"""
    feature_cols = [c for c in df.columns if c not in ('target', 'species')]

    # 使用简短列名方便显示
    df_plot = df[feature_cols + ['species']].copy()
    df_plot.columns = [FEATURE_NAMES_CN.get(c, c) for c in df_plot.columns[:-1]] + ['species']

    g = sns.pairplot(df_plot, hue='species', palette='Set2',
                     diag_kind='kde', height=2.2,
                     plot_kws={'alpha': 0.6, 's': 30, 'edgecolor': 'k', 'linewidth': 0.3})
    g.figure.suptitle('Pairplot', fontsize=14, y=1.02)
    plt.tight_layout()
    save_figure(output_dir, "06_pairplot.png", fig=g, dpi=200)
    plt.show()


def eda_scatter_key(df, output_dir: str):
    """10. 关键特征散点图（花萼 vs 花瓣）"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 花萼
    sns.scatterplot(data=df, x='sepal length (cm)', y='sepal width (cm)',
                    hue='species', palette='Set2', s=60, ax=axes[0],
                    edgecolor='k', linewidth=0.3)
    axes[0].set_title('Sepal Length vs Sepal Width')
    axes[0].set_xlabel('Sepal Length (cm)')
    axes[0].set_ylabel('Sepal Width (cm)')

    # 花瓣
    sns.scatterplot(data=df, x='petal length (cm)', y='petal width (cm)',
                    hue='species', palette='Set2', s=60, ax=axes[1],
                    edgecolor='k', linewidth=0.3)
    axes[1].set_title('Petal Length vs Petal Width')
    axes[1].set_xlabel('Petal Length (cm)')
    axes[1].set_ylabel('Petal Width (cm)')

    plt.suptitle('Key Feature Scatter', fontsize=14, y=1.02)
    plt.tight_layout()
    save_figure(output_dir, "07_key_feature_scatter.png", dpi=200)
    plt.show()


def eda_anderson(df):
    """11. 各类别各特征的正态性检验（Anderson-Darling）"""
    feature_cols = [c for c in df.columns if c not in ('target', 'species')]

    print("\n" + "=" * 60)
    print("6. Normality Test (Anderson-Darling, 5% significance)")
    print("=" * 60)
    header = f"{'Class':<12} {'Feature':<20} {'Statistic':>10} {'Crit(5%)':>10} {'Normal':>8}"
    print(header)
    print("-" * 62)

    for species in TARGET_NAMES.values():
        for col in feature_cols:
            data = df[df['species'] == species][col]
            result = stats.anderson(data, dist='norm')
            stat = result.statistic
            crit_5 = result.critical_values[2]  # 5% 显著性水平
            is_normal = "Yes" if stat < crit_5 else "No"
            cn = FEATURE_NAMES_CN.get(col, col)
            print(f"{species:<12} {cn:<20} {stat:>10.4f} {crit_5:>10.4f} {is_normal:>8}")


def run_eda():
    """运行完整EDA"""
    df, iris_data = _load_iris()
    output_dir = create_output_dir()
    print(f"输出目录: {output_dir}")

    # 文本分析
    eda_basic_info(df)
    eda_descriptive_stats(df)
    eda_class_distribution(df, output_dir)
    eda_correlation(df, output_dir)
    eda_anderson(df)

    # 可视化
    eda_distribution(df, output_dir)
    eda_boxplot(df, output_dir)
    eda_violin(df, output_dir)
    eda_pairplot(df, output_dir)
    eda_scatter_key(df, output_dir)

    print("\n" + "=" * 60)
    print("EDA Complete!")
    print("=" * 60)
    print("""
Summary:
1. 150 samples, 3 classes x 50 each, balanced, no missing values.
2. Petal features (length/width) are most discriminative, corr with target > 0.95.
3. Sepal features are less discriminative with more class overlap.
4. Setosa is linearly separable from the other two classes on petal features.
5. Versicolor and Virginica have slight overlap on petal features.
6. Petal length and petal width are highly correlated (r ~ 0.96), multicollinearity risk.
7. Most features are approximately normal within each class.
""")


if __name__ == '__main__':
    run_eda()


