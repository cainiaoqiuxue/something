# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


class ArimaModel:
    def __init__(self, data: pd.DataFrame, **kwargs):
        self.data = data
        self.config = kwargs
        self.p = self.config.get('p', 1)
        self.q = self.config.get('q', 1)
        self.d = self.config.get('d', 1)
        self.model = None

    def fit(self):
        self.model = ARIMA(self.data.values, order=(self.p, self.d, self.q)).fit()
        print(self.model.summary())

    def predict(self, step):
        start = len(self.data)
        end = start + step
        result = self.model.predict(start=start, end=end)
        return result

    def show_data(self):
        self.data.plot()
        plt.title('Time Series')
        plt.show()

        self.data.hist()
        plt.title('Data Distribution')
        plt.show()

    def adf_check(self):
        # 如果p值不小于显著性水平（通常是0.05），则序列可能是非平稳的，需要进行差分处理。
        result = adfuller(self.data.values.flatten())
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])

    def acf_plot(self):
        # 绘制ACF图
        plot_acf(self.data)
        plt.title('ACF')
        plt.show()

        # 绘制PACF图
        plot_pacf(self.data)
        plt.title('PACF')
        plt.show()

    def set_params(self, **kwargs):
        self.config = kwargs
        self.p = self.config.get('p', self.p)
        self.q = self.config.get('q', self.q)
        self.d = self.config.get('d', self.d)
        print('new params: p: {} d: {} q : {}'.format(self.p, self.d, self.q))

    def plot_predict(self, pred=None, step=None):
        if step is not None:
            pred = self.predict(step)
        plt.figure(figsize=(10, 6))
        plt.plot(np.concatenate([self.data.values, pred]), label='prediction')
        plt.plot(self.data.values, label='true')
        idxs = list(self.data.index)
        idx = int(idxs[-1])
        idxs += [str(idx + i + 1) for i in range(len(pred))]
        plt.xticks(range(len(idxs)), labels=idxs, rotation=30)
        plt.legend()
        plt.show()
