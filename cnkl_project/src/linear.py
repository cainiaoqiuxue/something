# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


class LinearModel:
    def __init__(self, topic_value):
        self.df = topic_value
        self.index = self.df.index
        self.columns = self.df.columns
        self.idx = None
        self.nx = None
        self.ny = None
        self.model = None

    def fit(self, idx):
        self.idx = idx
        value = self.df.iloc[:, idx].values
        self.model = LinearRegression()
        self.nx = np.arange(len(value)).reshape(-1, 1)
        self.ny = value.reshape(-1, 1)
        self.model.fit(self.nx, self.ny)

    def plot_model_fit(self):
        pred = self.model.predict(self.nx)
        plt.scatter(self.nx, self.ny)
        plt.plot(self.nx, pred, color='r', linestyle='--')
        plt.xticks(self.nx.reshape(-1), labels=self.index)
        plt.title('Trending of Topic_{}'.format(self.idx + 1))
        plt.show()

    def get_slope(self):
        pred = self.model.predict(self.nx).reshape(-1)
        slope = (pred[-1] - pred[0]) / (len(pred) - 1)
        return slope

    def select_topic(self, topn=5):
        slopes = []
        for i in range(len(self.columns)):
            self.fit(i)
            slopes.append(self.get_slope())
        means = self.df.mean().values
        means = tuple(zip(range(means.shape[0]), means))
        means = sorted(means, key=lambda x: -x[1])
        topics = []
        for m in means:
            if slopes[m[0]] > 0:
                topics.append(m[0])
        topics = topics[:topn]
        return topics, [self.columns[i] for i in topics]

