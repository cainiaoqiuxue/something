# -*- coding:utf-8 -*-
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class XgbModel:
    def __init__(self, df, feature_col=None, label_col=None):
        self.df = df
        self.feature_col = feature_col or ['微博数', '点赞', '转发', '评论', '粉丝', '关注', '地域', '情感倾向']
        self.label_col = label_col or '热度'
        self.model = None

    def train(self, params=None, epochs=10):
        if params is None:
            params = {
                # 'objective': 'reg:squarederror',
                'objective': 'reg:squaredlogerror',
                'learning_rate': 0.3,
                'seed': 42,
                'verbosity': 1
            }
        dtrain = xgb.DMatrix(data=self.df[self.feature_col],
                             label=self.df[self.label_col],
                             feature_names=self.feature_col)
        model = xgb.train(params, dtrain, num_boost_round=epochs)
        self.model = model

    @staticmethod
    def save_model(obj, path):
        joblib.dump(obj, path)

    @staticmethod
    def load_model(path):
        return joblib.load(path)

    def plot_importance(self):
        fig, ax = plt.subplots(figsize=(10, 8))
        xgb.plot_importance(self.model, ax=ax)

    def predict(self, data):
        data = data[self.feature_col]
        data = xgb.DMatrix(data=data)
        return self.model.predict(data)


class Eval:
    def __init__(self, target, prediction):
        self.target = target
        self.prediction = prediction

    def cal_mae(self):
        return mean_absolute_error(self.target, self.prediction)

    def cal_mse(self):
        return mean_squared_error(self.target, self.prediction)

    def plot(self):
        plt.figure(figsize=(10, 8))
        idx = range(len(self.target))
        plt.scatter(idx, self.target, label='target')
        plt.scatter(idx, self.prediction, label='prediction')
        plt.legend()
        plt.show()