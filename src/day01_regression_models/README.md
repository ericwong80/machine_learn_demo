# 回归模型入门专题

## 项目概述

本项目旨在通过同一个房价预测问题，展示不同回归模型的建模过程和效果差异。我们将从基础线性模型开始，逐步探索更复杂的非线性模型和集成模型。

## 目录结构

```
day01_regression_models/
├── README.md                 # 项目说明文档
├── common/                   # 公共工具模块
│   ├── dataset.py           # 数据生成与处理
│   ├── metrics.py           # 评估指标计算
│   ├── plotting.py          # 通用绘图功能
│   └── output.py            # 输出与保存功能
├── demos/                   # 各模型独立演示
│   ├── demo_01_ols_linear_regression.py
│   ├── demo_02_gradient_descent_regression.py
│   ├── demo_03_decision_tree_regression.py
│   ├── demo_04_random_forest_regression.py
│   └── demo_05_xgboost_regression.py
├── compare/                 # 模型横向比较
│   └── compare_regression_models.py
└── output/                  # 输出结果目录
```

## 学习顺序

建议按照以下顺序学习：

1. OLS普通线性回归 (`demo_01_ols_linear_regression.py`)
2. 梯度下降线性回归 (`demo_02_gradient_descent_regression.py`)
3. 决策树回归 (`demo_03_decision_tree_regression.py`)
4. 随机森林回归 (`demo_04_random_forest_regression.py`)
5. XGBoost回归 (`demo_05_xgboost_regression.py`)
6. 模型横向对比 (`compare_regression_models.py`)

## 核心目标

理解不同回归模型对同一问题的建模方式、效果差异及适用场景。
