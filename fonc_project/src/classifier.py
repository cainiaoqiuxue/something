import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from .model import Config, CNNModel, sigmoid, sigmoid_derivative
from .model_matirx import CNNModelMatrix
from scipy.special import gamma


class CNNModelClassifier(CNNModelMatrix):
    def __init__(self, cfg, feature, target):
        self.cfg = cfg
        np.random.seed(self.cfg.seed)
        self.kernel_size = cfg.kernel_size
        self.padding_size = cfg.padding_size
        # self.kw1 = np.random.randn(self.cfg.hidden_size, self.cfg.kernel_size)
        # self.w2 = np.random.randn(self.cfg.output_size, self.cfg.hidden_size)
        self.kw1 = np.random.normal(0.0, self.cfg.hidden_size ** -0.5, (self.cfg.hidden_size, self.cfg.kernel_size))
        self.w2 = np.random.normal(0.0, self.cfg.output_size ** -0.5, (self.cfg.output_size, self.cfg.hidden_size))
        self.x = np.concatenate([feature, np.zeros((feature.shape[0], self.padding_size))], axis=1)
        self.y = target
        self.slide = self.x.shape[1] // self.kernel_size
        self.w1 = np.concatenate([self.kw1 for _ in range(self.slide)], axis=1)
        # 训练集 测试集 划分
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.x, self.y, test_size=0.2, random_state=42)

    def caputo_i2h(self, i):
        alpha = self.cfg.alpha
        cmin = min(self.w1.min(), self.w2.min())
        weight = 1 / ((1 - alpha) * gamma(1 - alpha))
        result = []
        for j in range(len(self.x)):
            x_j = self.x[j]
            f_j = sigmoid(np.matmul(self.w2, sigmoid(np.matmul(self.w1, x_j))))
            f_dj = -sigmoid_derivative(np.matmul(self.w2, sigmoid(np.matmul(self.w1, x_j))))
            theta_j = (self.y[j] - f_j) * f_dj
            g_j = sigmoid(np.matmul(self.w1[i, :], x_j))
            res = weight * theta_j * g_j * (self.w2[:, i] - cmin) ** (1 - alpha)
            result.append(res)
        result = np.array(result).mean(axis=0)
        return result
    
    def caputo_i2h_matrix(self):
        alpha = self.cfg.alpha
        cmin = min(self.kw1.min(), self.w2.min())
        weight = 1 / ((1 - alpha) * gamma(1 - alpha))
        # f_j = sigmoid(np.matmul(self.w2, sigmoid(np.matmul(self.kw1, self.x.T))))
        # f_dj = -sigmoid_derivative(np.matmul(self.w2, sigmoid(np.matmul(self.kw1, self.x.T))))
        f_j = np.matmul(self.w2, sigmoid(np.matmul(self.kw1, self.x.T)))
        f_dj = -1
        theta_j = (self.y.T - f_j) * f_dj
        g_j = sigmoid(np.matmul(self.kw1, self.x.T))
        res = weight * np.matmul(theta_j, g_j.T) / theta_j.shape[1] * (self.w2 - cmin) ** (1 - alpha)
        return res
        
    def caputo_h2o(self, i, r):
        alpha = self.cfg.alpha
        cmin = min(self.w1.min(), self.w2.min())
        weight = 1 / ((1 - alpha) * gamma(1 - alpha))
        result = []
        for j in range(len(self.x)):
            x_j = self.x[j]
            f_j = sigmoid(np.matmul(self.w2, sigmoid(np.matmul(self.w1, x_j))))
            f_dj = -sigmoid_derivative(np.matmul(self.w2, sigmoid(np.matmul(self.w1, x_j))))
            theta_j = (self.y[j] - f_j) * f_dj
            g_dj = sigmoid_derivative(np.matmul(self.w1[i, :], x_j))
            res = weight * theta_j * self.w2[:, i] * g_dj * x_j[r] * (self.w1[i, r] - cmin) ** (1 - alpha)
            result.append(res.mean())
        result = np.array(result).mean(axis=0)
        return result

    def caputo_h2o_matrix(self):
        alpha = self.cfg.alpha
        cmin = min(self.kw1.min(), self.w2.min())
        weight = 1 / ((1 - alpha) * gamma(1 - alpha))
        # f_j = sigmoid(np.matmul(self.w2, sigmoid(np.matmul(self.kw1, self.x.T))))
        # f_dj = -sigmoid_derivative(np.matmul(self.w2, sigmoid(np.matmul(self.kw1, self.x.T))))
        f_j = np.matmul(self.w2, sigmoid(np.matmul(self.kw1, self.x.T)))
        f_dj = -1
        theta_j = (self.y.T - f_j) * f_dj
        g_dj = sigmoid_derivative(np.matmul(self.kw1, self.x.T))
        res = weight * np.matmul(self.w2.T, theta_j) * g_dj 
        res = np.einsum('ij,kj->ikj', res, self.x.T).mean(axis=-1)
        res = res * (self.kw1 - cmin) ** (1 - alpha)
        return res
    
    def anti_caputo_i2h_matrix(self):
        alpha = self.cfg.alpha
        cmax = max(self.kw1.max(), self.w2.max())
        weight = 1 / ((1 - alpha) * gamma(1 - alpha))
        # f_j = sigmoid(np.matmul(self.w2, sigmoid(np.matmul(self.kw1, self.x.T))))
        # f_dj = -sigmoid_derivative(np.matmul(self.w2, sigmoid(np.matmul(self.kw1, self.x.T))))
        f_j = np.matmul(self.w2, sigmoid(np.matmul(self.kw1, self.x.T)))
        f_dj = -1
        theta_j = (self.y.T - f_j) * f_dj
        g_j = sigmoid(np.matmul(self.kw1, self.x.T))
        res = weight * np.matmul(theta_j, g_j.T) / theta_j.shape[1] * (cmax - self.w2) ** (1 - alpha)
        return res
    
    def caputo_h2o_matrix(self):
        alpha = self.cfg.alpha
        cmax = max(self.kw1.max(), self.w2.max())
        weight = 1 / ((1 - alpha) * gamma(1 - alpha))
        # f_j = sigmoid(np.matmul(self.w2, sigmoid(np.matmul(self.kw1, self.x.T))))
        # f_dj = -sigmoid_derivative(np.matmul(self.w2, sigmoid(np.matmul(self.kw1, self.x.T))))
        f_j = np.matmul(self.w2, sigmoid(np.matmul(self.kw1, self.x.T)))
        f_dj = -1
        theta_j = (self.y.T - f_j) * f_dj
        g_dj = sigmoid_derivative(np.matmul(self.kw1, self.x.T))
        res = weight * np.matmul(self.w2.T, theta_j) * g_dj 
        res = np.einsum('ij,kj->ikj', res, self.x.T).mean(axis=-1)
        res = res * (cmax - self.kw1) ** (1 - alpha)
        return res
    
    def non_caputo_i2h_matrix(self):
        weight = 1 / (2 * np.sin(self.cfg.alpha * np.pi / 2))
        return weight * (self.caputo_i2h_matrix() - self.anti_caputo_i2h_matrix())
    
    def non_caputo_h2o_matrix(self):
        weight = 1 / (2 * np.sin(self.cfg.alpha * np.pi / 2))
        return weight * (self.caputo_h2o_matrix() - self.anti_caputo_h2o_matrix())
    
    def predict(self, x):
        # res = sigmoid(np.matmul(self.w2, sigmoid(np.matmul(self.w1, x.T))))
        res = np.matmul(self.w2, sigmoid(np.matmul(self.w1, x.T)))
        return res.T
    
    def update(self, type='caputo'):
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

if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import OneHotEncoder

    oe = OneHotEncoder()
    df = pd.read_csv('mnist_test.csv')
    label = oe.fit_transform(df['label'].values.reshape(-1, 1))
    label = label.toarray()
    cfg = Config(alpha=0.5, input_size=768, output_size=10, learning_rate=0.08, hidden_size=100, kernel_size=784,
                 padding_size=0)
    x = df.iloc[:, 1:].values / 255
    y = label
    cnn = CNNModelClassifier(cfg, x, y)
    
    print(cnn.caputo_i2h_matrix())
    print(cnn.caputo_h2o_matrix())