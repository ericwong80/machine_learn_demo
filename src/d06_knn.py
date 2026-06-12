"""
KNN 鸢尾花分类学习版
Author: 13928
说明：
    这个脚本不是只追求最短代码，而是为了学习 KNN 的完整建模流程。

学习目标：
    1. 了解鸢尾花数据集的结构
    2. 学会训练集 / 测试集拆分
    3. 明白为什么 KNN 一般要做标准化
    4. 学会用 Pipeline 串联 标准化 + 模型
    5. 学会比较不同 K 值的效果
    6. 学会查看准确率、分类报告、混淆矩阵
    7. 输出几张辅助理解的图片

机器学习流程：
    加载数据
    -> 转换为 DataFrame 方便观察
    -> 数据集拆分
    -> 标准化 + KNN 建模
    -> 模型评估
    -> 新样本预测
    -> 图片输出
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from utils import close_figure, create_output_dir, save_figure, set_chinese_plot_style


# ============================================================
# 0. 全局配置
# ============================================================

RANDOM_STATE = 23
TEST_SIZE = 0.2
DEFAULT_K = 3



# ============================================================
# 1. 加载数据
# ============================================================

def load_iris_as_dataframe() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str], np.ndarray]:
    """
    加载鸢尾花数据集，并转换为 DataFrame。

    返回：
        iris_df: 带特征、标签、类别名称的 DataFrame
        X: 特征矩阵，形状为 (150, 4)
        y: 标签数组，形状为 (150,)
        feature_names: 特征名称
        target_names: 类别名称
    """
    iris = load_iris()

    feature_names = iris.feature_names
    target_names = iris.target_names

    X = iris.data
    y = iris.target

    iris_df = pd.DataFrame(X, columns=feature_names)
    iris_df["label"] = y
    iris_df["species"] = iris_df["label"].map(lambda idx: target_names[idx])

    return iris_df, X, y, feature_names, target_names


def print_dataset_info(iris_df: pd.DataFrame, feature_names: list[str], target_names: np.ndarray) -> None:
    """
    打印数据集基础信息。
    """
    print("=" * 70)
    print("1. 数据集基本信息")
    print("=" * 70)

    print(f"样本数量: {len(iris_df)}")
    print(f"特征数量: {len(feature_names)}")
    print(f"特征名称: {feature_names}")
    print(f"类别名称: {target_names.tolist()}")

    print("\n前 5 行数据:")
    print(iris_df.head())

    print("\n每个类别的样本数量:")
    print(iris_df["species"].value_counts())

    print("\n特征统计信息:")
    print(iris_df[feature_names].describe())


# ============================================================
# 2. 可视化：认识数据
# ============================================================

def plot_pairplot(iris_df: pd.DataFrame, output_dir: str) -> None:
    """
    输出特征两两关系图。

    作用：
        帮助理解不同类别的鸢尾花在特征空间里是否容易分开。
    """
    pair_grid = sns.pairplot(
        iris_df,
        vars=[
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)",
        ],
        hue="species",
        diag_kind="hist",
        plot_kws={"alpha": 0.75, "s": 45, "edgecolor": "white", "linewidth": 0.5},
    )
    pair_grid.fig.suptitle("鸢尾花数据集：特征两两关系", y=1.02, fontsize=16, fontweight="bold")

    save_figure(output_dir, "01_iris_pairplot.png", fig=pair_grid, dpi=200)
    close_figure(pair_grid.fig)



def plot_feature_distribution(iris_df: pd.DataFrame, feature_names: list[str], output_dir: str) -> None:
    """
    输出每个特征在不同类别下的分布图。

    作用：
        看哪些特征对分类更有区分度。
    """
    long_df = iris_df.melt(
        id_vars=["species"],
        value_vars=feature_names,
        var_name="feature",
        value_name="value",
    )

    plt.figure(figsize=(13, 7))
    sns.boxplot(
        data=long_df,
        x="feature",
        y="value",
        hue="species",
    )

    plt.title("不同类别下的特征分布", fontsize=16, fontweight="bold")
    plt.xlabel("特征")
    plt.ylabel("特征值")
    plt.xticks(rotation=15)
    plt.tight_layout()

    save_figure(output_dir, "02_feature_distribution.png", dpi=200)
    close_figure()



# ============================================================
# 3. 数据集拆分
# ============================================================

def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    拆分训练集和测试集。

    stratify=y 的作用：
        保证训练集和测试集中，各个类别的比例尽量一致。
        分类任务中比较推荐这样写。
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\n" + "=" * 70)
    print("2. 数据集拆分")
    print("=" * 70)
    print(f"训练集特征形状: {X_train.shape}")
    print(f"测试集特征形状: {X_test.shape}")
    print(f"训练集标签形状: {y_train.shape}")
    print(f"测试集标签形状: {y_test.shape}")

    return X_train, X_test, y_train, y_test


def plot_split_distribution(
    y_train: np.ndarray,
    y_test: np.ndarray,
    target_names: np.ndarray,
    output_dir: str,
) -> None:
    """
    输出训练集和测试集的类别分布图。

    作用：
        检查拆分后是否类别失衡。
    """
    train_df = pd.DataFrame({"dataset": "train", "label": y_train})
    test_df = pd.DataFrame({"dataset": "test", "label": y_test})
    split_df = pd.concat([train_df, test_df], ignore_index=True)
    split_df["species"] = split_df["label"].map(lambda idx: target_names[idx])

    plt.figure(figsize=(9, 6))
    sns.countplot(
        data=split_df,
        x="species",
        hue="dataset",
    )

    plt.title("训练集 / 测试集类别分布", fontsize=16, fontweight="bold")
    plt.xlabel("类别")
    plt.ylabel("样本数量")
    plt.tight_layout()

    save_figure(output_dir, "03_train_test_distribution.png", dpi=200)
    close_figure()


# ============================================================
# 4. 建模：标准化 + KNN
# ============================================================

def build_knn_pipeline(k: int = DEFAULT_K) -> Pipeline:
    """
    创建 KNN 分类 Pipeline。

    为什么用 Pipeline？
        KNN 依赖距离，特征尺度会影响距离计算，所以一般要先标准化。
        Pipeline 可以把 标准化 和 模型 绑定在一起，减少忘记 transform 的风险。

    Pipeline 流程：
        原始特征
        -> StandardScaler 标准化
        -> KNeighborsClassifier 训练/预测
    """
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=k)),
        ]
    )

    return model


def train_and_evaluate_model(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    target_names: np.ndarray,
    output_dir: str,
    k: int = DEFAULT_K,
) -> Pipeline:
    """
    训练并评估 KNN 模型。
    """
    print("\n" + "=" * 70)
    print("3. 模型训练与评估")
    print("=" * 70)

    model = build_knn_pipeline(k=k)

    # 训练模型：只用训练集
    model.fit(X_train, y_train)

    # 预测训练集和测试集
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # 计算准确率
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    print(f"K 值: {k}")
    print(f"训练集准确率: {train_acc:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")

    print("\n测试集分类报告:")
    print(
        classification_report(
            y_test,
            y_test_pred,
            target_names=target_names,
            digits=4,
        )
    )

    plot_confusion_matrix(y_test, y_test_pred, target_names, output_dir)

    return model


def plot_confusion_matrix(
    y_test: np.ndarray,
    y_test_pred: np.ndarray,
    target_names: np.ndarray,
    output_dir: str,
) -> None:
    """
    输出混淆矩阵。

    作用：
        看模型到底把哪些类别分错了。
    """
    cm = confusion_matrix(y_test, y_test_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=target_names,
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")

    ax.set_title("测试集混淆矩阵", fontsize=16, fontweight="bold")
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    plt.tight_layout()

    save_figure(output_dir, "04_confusion_matrix.png", fig=fig, dpi=200)
    close_figure(fig)


# ============================================================
# 5. 比较不同 K 值
# ============================================================

def compare_different_k_values(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    output_dir: str,
) -> pd.DataFrame:
    """
    比较不同 K 值下，训练集和测试集准确率的变化。

    学习重点：
        K 太小：模型容易受个别样本影响，可能过拟合。
        K 太大：模型过于平滑，可能欠拟合。
    """
    print("\n" + "=" * 70)
    print("4. 比较不同 K 值")
    print("=" * 70)

    results = []

    for k in range(1, 21):
        model = build_knn_pipeline(k=k)
        model.fit(X_train, y_train)

        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)

        results.append(
            {
                "K值": k,
                "训练集准确率": train_acc,
                "测试集准确率": test_acc,
            }
        )

    k_df = pd.DataFrame(results)
    print(k_df.to_string(index=False))

    best_row = k_df.loc[k_df["测试集准确率"].idxmax()]
    print(
        f"\n测试集上表现最好的 K: {int(best_row['K值'])}, "
        f"测试集准确率: {best_row['测试集准确率']:.4f}"
    )

    plot_k_accuracy(k_df, output_dir)

    return k_df


def plot_k_accuracy(k_df: pd.DataFrame, output_dir: str) -> None:
    """
    输出 K 值与准确率关系图。
    """
    plot_df = k_df.melt(
        id_vars="K值",
        value_vars=["训练集准确率", "测试集准确率"],
        var_name="数据集",
        value_name="准确率",
    )

    plt.figure(figsize=(11, 6))
    sns.lineplot(
        data=plot_df,
        x="K值",
        y="准确率",
        hue="数据集",
        marker="o",
    )

    plt.title("不同 K 值下的准确率变化", fontsize=16, fontweight="bold")
    plt.xlabel("K 值，即 n_neighbors")
    plt.ylabel("准确率")
    plt.xticks(k_df["K值"])
    plt.ylim(0.85, 1.02)
    plt.tight_layout()

    save_figure(output_dir, "05_k_value_accuracy.png", dpi=200)
    close_figure()


# ============================================================
# 6. 可视化：用两个特征画 KNN 分类边界
# ============================================================

def plot_knn_decision_boundary(
    iris_df: pd.DataFrame,
    target_names: np.ndarray,
    output_dir: str,
    k: int = DEFAULT_K,
) -> None:
    """
    使用两个最有区分度的特征画分类边界。

    注意：
        真实模型使用的是 4 个特征。
        这里为了画二维图，只使用 petal length 和 petal width 两个特征。
    """
    feature_x = "petal length (cm)"
    feature_y = "petal width (cm)"

    X_2d = iris_df[[feature_x, feature_y]].values
    y = iris_df["label"].values

    model_2d = build_knn_pipeline(k=k)
    model_2d.fit(X_2d, y)

    x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
    y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )

    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model_2d.predict(grid_points)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 7))
    plt.contourf(xx, yy, Z, alpha=0.25, levels=np.arange(len(target_names) + 1) - 0.5)

    sns.scatterplot(
        data=iris_df,
        x=feature_x,
        y=feature_y,
        hue="species",
        edgecolor="white",
        linewidth=0.8,
        s=70,
    )

    plt.title(f"KNN 二维分类边界示意图：K={k}", fontsize=16, fontweight="bold")
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.tight_layout()

    save_figure(output_dir, "06_knn_decision_boundary_2d.png", dpi=200)
    close_figure()


# ============================================================
# 7. 新样本预测
# ============================================================

def predict_new_sample(
    model: Pipeline,
    target_names: np.ndarray,
) -> None:
    """
    对新样本进行预测。
    """
    print("\n" + "=" * 70)
    print("5. 新样本预测")
    print("=" * 70)

    # 新样本格式：
    # [sepal length, sepal width, petal length, petal width]
    new_sample = np.array([[7.8, 2.1, 3.9, 1.6]])

    pred_label = model.predict(new_sample)[0]
    pred_name = target_names[pred_label]
    pred_proba = model.predict_proba(new_sample)[0]

    print("新样本特征:")
    print("sepal length=7.8, sepal width=2.1, petal length=3.9, petal width=1.6")
    print(f"预测类别编号: {pred_label}")
    print(f"预测类别名称: {pred_name}")

    print("\n每个类别的预测概率:")
    for name, proba in zip(target_names, pred_proba):
        print(f"{name}: {proba:.4f}")


# ============================================================
# 8. 主函数
# ============================================================

def main() -> None:
    """
    主流程入口。
    """
    output_dir = create_output_dir()
    set_chinese_plot_style(figure_facecolor="#f7f9fc", axes_facecolor="#ffffff")

    print(f"输出目录: {output_dir}")

    # 1. 加载数据
    iris_df, X, y, feature_names, target_names = load_iris_as_dataframe()
    print_dataset_info(iris_df, feature_names, target_names)

    # 2. 先画图认识数据
    plot_pairplot(iris_df, output_dir)
    plot_feature_distribution(iris_df, feature_names, output_dir)

    # 3. 拆分训练集和测试集
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    plot_split_distribution(y_train, y_test, target_names, output_dir)

    # 4. 训练一个默认 K=3 的模型
    model = train_and_evaluate_model(
        X_train,
        X_test,
        y_train,
        y_test,
        target_names,
        output_dir,
        k=DEFAULT_K,
    )

    # 5. 比较不同 K 值
    compare_different_k_values(
        X_train,
        X_test,
        y_train,
        y_test,
        output_dir,
    )

    # 6. 画二维分类边界，帮助理解 KNN 的“邻居投票”思想
    plot_knn_decision_boundary(
        iris_df,
        target_names,
        output_dir,
        k=DEFAULT_K,
    )

    # 7. 新样本预测
    predict_new_sample(model, target_names)

    print("\n" + "=" * 70)
    print("运行完成")
    print("=" * 70)
    print(f"所有图片已输出到: {output_dir}")


if __name__ == "__main__":
    main()
