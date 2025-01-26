import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.special import gamma
from .model import Config, CNNModel, sigmoid, sigmoid_derivative
from .model_matirx import CNNModelMatrix


class CNNModelRegressor(CNNModelMatrix):
    def __init__(self, cfg, feature, target):
        self.cfg = cfg
        np.random.seed(self.cfg.seed)
        self.kernel_size = cfg.kernel_size
        self.padding_size = cfg.padding_size
        self.kw1 = np.random.randn(self.cfg.hidden_size, self.cfg.kernel_size)
        self.w2 = np.random.randn(self.cfg.output_size, self.cfg.hidden_size)
        self.x = np.concatenate([feature, np.zeros((feature.shape[0], self.padding_size))], axis=1)
        self.y = target
        self.slide = self.x.shape[1] // self.kernel_size
        self.w1 = np.concatenate([self.kw1 for _ in range(self.slide)], axis=1)
        # 训练集 测试集 划分
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.x, self.y, test_size=0.2, random_state=42)
    
    def caputo_i2h_matrix(self):
        alpha = self.cfg.alpha
        cmin = min(self.kw1.min(), self.w2.min())
        weight = 1 / ((1 - alpha) * gamma(1 - alpha))
        f_j = np.matmul(self.w2, sigmoid(np.matmul(self.kw1, self.x.T)))
        f_dj = -1
        theta_j = (self.y.T - f_j) * f_dj
        g_j = sigmoid(np.matmul(self.kw1, self.x.T))
        res = (weight * theta_j * g_j).mean(axis=1) * (self.w2 - cmin) ** (1 - alpha)
        return res
    
    def caputo_h2o_matrix(self):
        alpha = self.cfg.alpha
        cmin = min(self.kw1.min(), self.w2.min())
        weight = 1 / ((1 - alpha) * gamma(1 - alpha))
        f_j = np.matmul(self.w2, sigmoid(np.matmul(self.kw1, self.x.T)))
        f_dj = -1
        theta_j = (self.y.T - f_j) * f_dj
        g_dj = sigmoid_derivative(np.matmul(self.kw1, self.x.T))
        # res = weight * theta_j * self.w2 * g_dj * self.x.T * (self.w1 - cmin) ** (1 - alpha)
        res = weight * theta_j * g_dj 
        res = np.einsum('ij,kj->ikj', res, self.x.T).mean(axis=-1)
        res = res * self.w2.T * (self.kw1 - cmin) ** (1 - alpha)
        return res
    
    
    def anti_caputo_i2h_matrix(self):
        alpha = self.cfg.alpha
        cmax = max(self.kw1.max(), self.w2.max())
        weight = 1 / ((1 - alpha) * gamma(1 - alpha))
        f_j = np.matmul(self.w2, sigmoid(np.matmul(self.kw1, self.x.T)))
        f_dj = -1
        theta_j = (self.y.T - f_j) * f_dj
        g_j = sigmoid(np.matmul(self.kw1, self.x.T))
        res = (weight * theta_j * g_j).mean(axis=1) * (cmax - self.w2) ** (1 - alpha)
        return res
        
    def anti_caputo_h2o_matrix(self):
        alpha = self.cfg.alpha
        cmax = max(self.kw1.max(), self.w2.max())
        weight = 1 / ((1 - alpha) * gamma(1 - alpha))
        f_j = np.matmul(self.w2, sigmoid(np.matmul(self.kw1, self.x.T)))
        f_dj = -1
        theta_j = (self.y.T - f_j) * f_dj
        g_dj = sigmoid_derivative(np.matmul(self.kw1, self.x.T))
        res = weight * theta_j * g_dj 
        res = np.einsum('ij,kj->ikj', res, self.x.T).mean(axis=-1)
        res = res * self.w2.T * (cmax - self.kw1) ** (1 - alpha)
        return res
    
    
    def non_caputo_i2h_matrix(self):
        weight = 1 / (2 * np.sin(self.cfg.alpha * np.pi / 2))
        return weight * (self.caputo_i2h_matrix() - self.anti_caputo_i2h_matrix())
    
    def non_caputo_h2o_matrix(self):
        weight = 1 / (2 * np.sin(self.cfg.alpha * np.pi / 2))
        return weight * (self.caputo_h2o_matrix() - self.anti_caputo_h2o_matrix())

    def update(self, type='caputo'):
        # lr = self.cfg.learning_rate
        # n, m = self.w1.shape
        # func_dic = {
        #     'caputo': (self.caputo_i2h, self.caputo_h2o),
        #     'anti_caputo': (self.anti_caputo_i2h, self.anti_caputo_h2o),
        #     'non_caputo': (self.non_caputo_i2h, self.non_caputo_h2o),
        # }

        # i2h, h2o = func_dic.get(type)
        # for i in range(n):
        #     self.w2[:, i] = self.w2[:, i] - lr * i2h(i)
        # for i in range(n):
        #     for r in range(m):
        #         self.kw1[i][r % self.kernel_size] = self.w1[i][r] - lr * h2o(i,r) / self.slide
        lr = self.cfg.learning_rate
        func_dic = {
            'caputo': (self.caputo_i2h_matrix, self.caputo_h2o_matrix),
            'anti_caputo': (self.anti_caputo_i2h_matrix, self.anti_caputo_h2o_matrix),
            'non_caputo': (self.non_caputo_i2h_matrix, self.non_caputo_h2o_matrix),
        }

        i2h, h2o = func_dic.get(type)
        w2_update = i2h()
        w1_update = h2o()
        self.w2 = self.w2 - lr * w2_update
        self.kw1 = self.kw1 - lr * w1_update
        self.w1 = np.concatenate([self.kw1 for _ in range(self.slide)], axis=1)

    def predict(self, x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.shape[1] < self.w1.shape[1]:
            x = np.concatenate([x, np.zeros((x.shape[0], self.padding_size))], axis=1)
        return np.matmul(self.w2, sigmoid(np.matmul(self.w1, x.T))).reshape(-1)

    def train(self, epochs=100, train_type='caputo'):
        print('-' * 10, ' begin task ', '-' * 10)
        print(self.cfg)
        for epoch in range(epochs):
            self.update(train_type)
            print('update: {} / {}'.format(epoch + 1, epochs))
            print('r2: ', r2_score(self.y, self.predict(self.x)))

