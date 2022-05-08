import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris, load_breast_cancer, load_digits

from rf_importance import cal_weight_feature_importance

warnings.filterwarnings('ignore')


class FeatureSelection(object):
    def __init__(self, feature, target, feature_names):
        self.SEED = 9
        self.TEST_SIZE = 0.2

        self.feature_names = feature_names
        feature = pd.DataFrame(feature, columns=self.feature_names)
        target = pd.Series(target)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(feature, target,
                                                                                test_size=self.TEST_SIZE,
                                                                                random_state=self.SEED,
                                                                                )

    def cal_importance(self, rf_params=None):
        if rf_params is None:
            rf_params = dict()
        rf_params['random_state'] = self.SEED

        model = RandomForestClassifier(**rf_params)
        model.fit(self.x_train, self.y_train)

        raw_importance = list(zip(self.feature_names, model.feature_importances_))
        change_importance = list(cal_weight_feature_importance(model, self.feature_names).items())
        raw_importance.sort(key=lambda x: -x[1])
        change_importance.sort(key=lambda x: -x[1])
        self.raw_importance = raw_importance
        self.change_importance = change_importance

    def select_feature(self, kind='count', value=30):
        # kind : count fraction threshold
        self.need_feature = []
        if kind == 'count':
            self.need_feature.append([x[0] for x in self.raw_importance[:value]])
            self.need_feature.append([x[0] for x in self.change_importance[:value]])
        elif kind == 'fraction':
            num = round(len(self.raw_importance) * value)
            self.need_feature.append([x[0] for x in self.raw_importance[:num]])
            self.need_feature.append([x[0] for x in self.change_importance[:num]])
        elif kind == 'threshold':
            num = 0
            for x in self.raw_importance:
                if x[1] >= value:
                    num += 1
            self.need_feature.append([x[0] for x in self.raw_importance[:num]])
            self.need_feature.append([x[0] for x in self.change_importance[:num]])
        else:
            raise ValueError(f"invalid kind type:{kind},only support count fraction threshold")

    def train_model(self, model_func, params=None):
        if not params:
            params = dict()
        params['random_state'] = self.SEED
        self.raw_res = []
        self.change_res = []
        num = len(self.need_feature[0])
        for i in range(1, num + 1):
            raw_model = model_func(**params)
            change_model = model_func(**params)
            raw_model.fit(self.x_train[self.need_feature[0][:i]], self.y_train)
            change_model.fit(self.x_train[self.need_feature[1][:i]], self.y_train)
            self.raw_res.append(accuracy_score(self.y_test, raw_model.predict(self.x_test[self.need_feature[0][:i]])))
            self.change_res.append(
                accuracy_score(self.y_test, change_model.predict(self.x_test[self.need_feature[1][:i]])))

    def plot_res(self):
        plt.plot(self.raw_res, label='raw_feature')
        plt.plot(self.change_res, label='change_feature')
        plt.legend()
        plt.title('accuracy on test data')
        plt.show()


if __name__ == '__main__':
    # 示例1
    data = load_digits()
    # 示例2
    # data = load_breast_cancer()

    feature = data['data']
    target = data['target']
    feature_names = data['feature_names'] if 'feature_names' in data else [f'f_{i}' for i in range(feature.shape[1])]

    # 传全部数据集特征、全部数据集标签、对应特征列的名称
    fs = FeatureSelection(feature, target, feature_names)

    # rf_params传随机森林的参数
    rf_params = {}
    fs.cal_importance(rf_params=rf_params)

    # 传筛选方法kind和筛选阈值value
    fs.select_feature()

    # 传分类模型和模型参数
    params = {}
    fs.train_model(LogisticRegression, params=params)

    fs.plot_res()
