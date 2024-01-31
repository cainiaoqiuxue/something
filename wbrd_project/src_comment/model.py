# -*- coding:utf-8 -*-

import joblib  # 导入 joblib 库用于模型的保存和加载
import xgboost as xgb  # 导入 xgboost 库并简称为 xgb
import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 用于绘图，并简称为 plt
from sklearn.metrics import mean_squared_error, mean_absolute_error  # 导入 sklearn 库中的两种误差计算函数

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置 matplotlib 图表中的字体为 SimHei（黑体），用于显示中文
plt.rcParams['axes.unicode_minus'] = False  # 设置 matplotlib 在坐标轴上正确显示负号

class XgbModel:
    # 定义 XgbModel 类用于建立和训练 XGBoost 模型

    def __init__(self, df, feature_col=None, label_col=None):
        # 类的初始化方法
        self.df = df  # 保存数据集至对象变量 df
        # 指定特征列，如果调用时未指定，则默认使用给定的中文特征名称列表
        self.feature_col = feature_col or ['微博数', '点赞', '转发', '评论', '粉丝', '关注', '地域', '情感倾向']
        self.label_col = label_col or '热度'  # 指定标签列，如果调用时未指定，则默认标签列为 '热度'
        self.model = None  # 初始化一个变量 model，用于存储训练后的模型，初始值为 None

    def train(self, params=None, epochs=10):
        # 训练模型的方法
        if params is None:
            # 如果调用时没有指定 params 参数，则使用以下默认参数
            params = {
                'objective': 'reg:squaredlogerror',  # 指定目标函数类型为平方对数误差回归
                'learning_rate': 0.3,  # 学习率为 0.3
                'seed': 42,  # 随机种子设为 42
                'verbosity': 1  # 输出训练过程中的信息
            }
        # 创建一个 DMatrix 对象：xgboost 中的数据结构，用于高效地存储和处理数据
        dtrain = xgb.DMatrix(data=self.df[self.feature_col],
                             label=self.df[self.label_col],
                             feature_names=self.feature_col)
        # 使用 xgboost 的 train 方法训练模型
        model = xgb.train(params, dtrain, num_boost_round=epochs)
        self.model = model  # 保存训练好的模型

    @staticmethod
    def save_model(obj, path):
        # 静态方法，保存模型至指定路径
        joblib.dump(obj, path)

    @staticmethod
    def load_model(path):
        # 静态方法，从指定路径加载模型
        return joblib.load(path)

    def plot_importance(self):
        # 方法：绘制特征重要性
        fig, ax = plt.subplots(figsize=(10, 8))  # 创建画布和坐标轴，设定大小为 10x8
        xgb.plot_importance(self.model, ax=ax)  # 调用 xgboost 提供的 plot_importance 方法显示特征重要性

    def predict(self, data):
        # 预测方法
        data = data[self.feature_col]  # 获取用于预测的特征数据
        data = xgb.DMatrix(data=data)  # 通过 XGBoost 的 DMatrix 对象处理数据
        return self.model.predict(data)  # 返回模型的预测结果

class Eval:
    # 定义 Eval 类用于评估模型性能

    def __init__(self, target, prediction):
        # 类的初始化方法
        self.target = target  # 真实值
        self.prediction = prediction  # 预测值

    def cal_mae(self):
        # 计算并返回平均绝对误差 (Mean Absolute Error, MAE)
        return mean_absolute_error(self.target, self.prediction)

    def cal_mse(self):
        # 计算并返回均方误差 (Mean Squared Error, MSE)
        return mean_squared_error(self.target, self.prediction)

    def plot(self):
        # 绘制目标和预测值的散点图
        plt.figure(figsize=(10, 8))  # 设置画布大小
        idx = range(len(self.target))  # 创建索引
        plt.scatter(idx, self.target, label='target')  # 绘制真实值的散点图
        plt.scatter(idx, self.prediction, label='prediction')  # 绘制预测值的散点图
        plt.legend()  # 显示图例
        plt.show()  # 显示图表