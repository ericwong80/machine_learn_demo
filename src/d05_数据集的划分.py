"""
Author:13928
Date:2026/5/20
"""
from sklearn.model_selection import train_test_split
from d04_鸢尾花 import _load_iris as load_data


def _dataset_split():
    dataset = load_data()
    X_train,X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2, random_state=42)
    # print(f"数据总量：{len(dataset.data)}")
    # print(f"训练集数量：{len(X_train)}")
    # print(f"测试集数量：{len(X_test)}")




if __name__ == '__main__':
    _dataset_split()