import numpy as np
from scipy.special import gamma, expit
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class Config:
    input_size: int = field(default=13)
    hidden_size: int = field(default=20)
    output_size: int = field(default=1)
    kernel_size: int = field(default=3)
    padding_size: int = field(default=1)
    learning_rate: float = field(default=5e-5)
    alpha: float = field(default=0.5)
    seed: int = field(default=42)
    bound: float = field(default=0.99)

def sigmoid(x):
    # return 1 / (1 + np.exp(-x))
    return expit(x)

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# class CNNModel:
#     def __init__(self, cfg, feature, target):
#         self.cfg = cfg
#         np.random.seed(self.cfg.seed)
#         self.w1 = np.random.randn(self.cfg.input_size, self.cfg.hidden_size)
#         self.w2 = np.random.randn(self.cfg.hidden_size, self.cfg.output_size)
#         self.x = feature
#         self.y = target
        
#     def caputo_input_to_hidden(self):
#         alpha = self.cfg.alpha
#         cmin = min(self.w1.min(), self.w2.min())
#         weight = 1 / ((1 - alpha) * gamma(1 - alpha))
#         v1 = -sigmoid_derivative(np.matmul(sigmoid(np.matmul(self.x, self.w1)), self.w2))
#         v2 = np.sum(sigmoid(np.matmul(self.x, self.w1)), axis=1, keepdims=True)
#         result = weight * np.sum(v1 * v2) * (self.w2 - cmin) ** (1 - alpha)
#         return result
    
#     def caputo_hidden_to_output(self):
#         alpha = self.cfg.alpha
#         cmin = min(self.w1.min(), self.w2.min())
#         weight = 1 / ((1 - alpha) * gamma(1 - alpha))
        
class CNNModel:
    def __init__(self, cfg, feature, target):
        self.cfg = cfg
        np.random.seed(self.cfg.seed)
        self.w1 = np.random.randn(self.cfg.hidden_size, self.cfg.input_size)
        self.w2 = np.random.randn(self.cfg.output_size, self.cfg.hidden_size)
        self.x = feature
        self.y = target

        
    def caputo_i2h(self, i):
        alpha = self.cfg.alpha
        cmin = min(self.w1.min(), self.w2.min())
        weight = 1 / ((1 - alpha) * gamma(1 - alpha))
        result = []
        for j in range(len(self.x)):
            x_j = self.x[j]
            # f_j = sigmoid(np.matmul(self.w2, sigmoid(np.matmul(self.w1, x_j))))
            # f_dj = -sigmoid_derivative(np.matmul(self.w2, sigmoid(np.matmul(self.w1, x_j))))
            f_j = np.matmul(self.w2, sigmoid(np.matmul(self.w1, x_j)))
            f_dj = -1
            theta_j = (self.y[j] - f_j) * f_dj
            g_j = sigmoid(np.matmul(self.w1[i, :], x_j))
            res = weight * theta_j * g_j * (self.w2[:, i] - cmin) ** (1 - alpha)
            result.append(res)
        result = np.array(result).mean(axis=0)
        return result
        
    def caputo_h2o(self, i, r):
        alpha = self.cfg.alpha
        cmin = min(self.w1.min(), self.w2.min())
        weight = 1 / ((1 - alpha) * gamma(1 - alpha))
        result = []
        for j in range(len(self.x)):
            x_j = self.x[j]
            # f_j = sigmoid(np.matmul(self.w2, sigmoid(np.matmul(self.w1, x_j))))
            # f_dj = -sigmoid_derivative(np.matmul(self.w2, sigmoid(np.matmul(self.w1, x_j))))
            f_j = np.matmul(self.w2, sigmoid(np.matmul(self.w1, x_j)))
            f_dj = -1
            theta_j = (self.y[j] - f_j) * f_dj
            g_dj = sigmoid_derivative(np.matmul(self.w1[i, :], x_j))
            res = weight * theta_j * self.w2[:, i] * g_dj * x_j[r] * (self.w1[i, r] - cmin) ** (1 - alpha)
            result.append(res)
        result = np.array(result).mean(axis=0)
        return result
    
    def anti_caputo_i2h(self, i):
        alpha = self.cfg.alpha
        cmax = max(self.w1.max(), self.w2.max())
        weight = 1 / ((1 - alpha) * gamma(1 - alpha))
        result = []
        for j in range(len(self.x)):
            x_j = self.x[j]
            f_j = np.matmul(self.w2, sigmoid(np.matmul(self.w1, x_j)))
            f_dj = -1
            theta_j = (self.y[j] - f_j) * f_dj
            g_j = sigmoid(np.matmul(self.w1[i, :], x_j))
            res = weight * theta_j * g_j * (cmax - self.w2[:, i]) ** (1 - alpha)
            result.append(res)
        result = np.array(result).mean(axis=0)
        return result
        
    def anti_caputo_h2o(self, i, r):
        alpha = self.cfg.alpha
        cmax = max(self.w1.max(), self.w2.max())
        weight = 1 / ((1 - alpha) * gamma(1 - alpha))
        result = []
        for j in range(len(self.x)):
            x_j = self.x[j]
            f_j = np.matmul(self.w2, sigmoid(np.matmul(self.w1, x_j)))
            f_dj = -1
            theta_j = (self.y[j] - f_j) * f_dj
            g_dj = sigmoid_derivative(np.matmul(self.w1[i, :], x_j))
            res = weight * theta_j * self.w2[:, i] * g_dj * x_j[r] * (cmax - self.w1[i, r]) ** (1 - alpha)
            result.append(res)
        result = np.array(result).mean(axis=0)
        return result
    
    def non_caputo_i2h(self, i):
        weight = 1 / (2 * np.sin(self.cfg.alpha * np.pi / 2))
        return weight * (self.caputo_i2h(i) - self.anti_caputo_i2h(i))

    def non_caputo_h2o(self, i, r):
        weight = 1 / (2 * np.sin(self.cfg.alpha * np.pi / 2))
        return weight * (self.caputo_h2o(i, r) - self.anti_caputo_h2o(i, r))
    

    def update(self, type='caputo'):
        lr = self.cfg.learning_rate
        n, m = self.w1.shape
        func_dic = {
            'caputo': (self.caputo_i2h, self.caputo_h2o),
            'anti_caputo': (self.anti_caputo_i2h, self.anti_caputo_h2o),
            'non_caputo': (self.non_caputo_i2h, self.non_caputo_h2o),
        }

        i2h, h2o = func_dic.get(type)
        for i in range(n):
            self.w2[:, i] = self.w2[:, i] - lr * i2h(i)
        for i in range(n):
            for r in range(m):
                self.w1[i][r] = self.w1[i][r] - lr * h2o(i,r)
        
    def predict(self, x):
        return np.matmul(self.w2, sigmoid(np.matmul(self.w1, x.T))).reshape(-1)
    
    def save_model(self, dir):
        dir = Path(dir)
        np.save(dir / 'w1.npy', self.w1)
        np.save(dir / 'w2.npy', self.w2)

    def load(self, dir):
        dir = Path(dir)
        self.w1 = np.load(dir / 'w1.npy')
        self.w2 = np.load(dir / 'w2.npy')

    def train(self, epoch, train_type, save_dir=None):
        for i in range(epoch):
            print('train epoch: {} / {}'.format(i + 1, epoch))
            self.update(train_type)

        if save_dir:
            self.save_model(save_dir)
    
    
if __name__ == '__main__':
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    root_dir = Path(__file__).parent
    df = pd.read_csv(root_dir / '../data/housing.csv', header=None, sep='\s+') 
    df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    x = StandardScaler().fit_transform(x)
    # y = np.random.randint(0, 1, (len(x), 3))
    cfg = Config(alpha=0.5, input_size=x.shape[1], output_size=1, learning_rate=0.05, hidden_size=20)
    cnn = CNNModel(cfg, x, y)
    for epoch in range(100):
        cnn.update(type='non_caputo')
        print('update: {}'.format(epoch + 1))
        # print(cnn.w2[0])
    
    cnn.save_model(root_dir / '../data')
    print(cnn.predict(x[0]))
    
    

