"""
Author:Administrator
Date:2026/6/13
决策树是一种常用的机器学习算法，主要用于分类和回归任务。它的核心思想是通过树状结构对数据进行决策，每个内部节点代表一个特征判断，每个叶节点代表一个类别或预测值。

核心特点：
直观易懂：树形结构可视化强，决策过程可解释
无需复杂预处理：对缺失值和异常值不敏感，不需要特征缩放
自动特征选择：通过信息增益、基尼系数等指标自动选择重要特征

工作原理：
节点分裂：从根节点开始，选择最优特征进行分裂
递归构建：对每个子集重复分裂过程，直到满足停止条件
剪枝优化：防止过拟合，通过预剪枝或后剪枝简化树结构

常用算法：
ID3（信息增益）
C4.5（信息增益率）
CART（基尼系数，支持分类和回归）

优缺点：
✅ 优点：解释性强、计算效率高、可处理数值和类别数据
❌ 缺点：容易过拟合、对数据变化敏感、可能产生复杂树结构

决策树是随机森林、梯度提升树等集成算法的基础，在金融风控、医疗诊断、客户分群等领域有广泛应用。
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

from utils import create_output_dir, save_figure, set_chinese_plot_style
import seaborn as sns  # 用于高级数据可视化

iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier(
    criterion='gini',  # 划分特征选择标准，gini不纯度，entropy信息增益
    max_depth=3,  # 树的最大深度
    random_state=42
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(f"决策树模型的预测准确度{accuracy_score(y_test, y_pred) * 100:.2f}%\n")


