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

        
class CNNModelMatrix:
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
            f_j = np.matmul(self.w2, sigmoid(np.matmul(self.w1, x_j)))
            f_dj = -1
            theta_j = (self.y[j] - f_j) * f_dj
            g_j = sigmoid(np.matmul(self.w1[i, :], x_j))
            res = weight * theta_j * g_j * (self.w2[:, i] - cmin) ** (1 - alpha)
            result.append(res)
        result = np.array(result).mean(axis=0)
        return result
    
    def caputo_i2h_batch(self, i):
        alpha = self.cfg.alpha
        cmin = min(self.w1.min(), self.w2.min())
        weight = 1 / ((1 - alpha) * gamma(1 - alpha))
        f_j = np.matmul(self.w2, sigmoid(np.matmul(self.w1, self.x.T)))
        f_dj = -1
        theta_j = (self.y.T - f_j) * f_dj
        g_j = sigmoid(np.matmul(self.w1[i, :], self.x.T))
        res = weight * theta_j * g_j * (self.w2[:, i] - cmin) ** (1 - alpha)
        return res.mean(axis=1)
    
    def caputo_i2h_matrix(self):
        alpha = self.cfg.alpha
        cmin = min(self.w1.min(), self.w2.min())
        weight = 1 / ((1 - alpha) * gamma(1 - alpha))
        f_j = np.matmul(self.w2, sigmoid(np.matmul(self.w1, self.x.T)))
        f_dj = -1
        theta_j = (self.y.T - f_j) * f_dj
        g_j = sigmoid(np.matmul(self.w1, self.x.T))
        res = (weight * theta_j * g_j).mean(axis=1) * (self.w2 - cmin) ** (1 - alpha)
        return res
        
    def caputo_h2o(self, i, r):
        alpha = self.cfg.alpha
        cmin = min(self.w1.min(), self.w2.min())
        weight = 1 / ((1 - alpha) * gamma(1 - alpha))
        result = []
        for j in range(len(self.x)):
            x_j = self.x[j]
            f_j = np.matmul(self.w2, sigmoid(np.matmul(self.w1, x_j)))
            f_dj = -1
            theta_j = (self.y[j] - f_j) * f_dj
            g_dj = sigmoid_derivative(np.matmul(self.w1[i, :], x_j))
            res = weight * theta_j * self.w2[:, i] * g_dj * x_j[r] * (self.w1[i, r] - cmin) ** (1 - alpha)
            result.append(res)
        result = np.array(result).mean(axis=0)
        return result
    
    def caputo_h2o_batch(self, i, r):
        alpha = self.cfg.alpha
        cmin = min(self.w1.min(), self.w2.min())
        weight = 1 / ((1 - alpha) * gamma(1 - alpha))
        f_j = np.matmul(self.w2, sigmoid(np.matmul(self.w1, self.x.T)))
        f_dj = -1
        theta_j = (self.y.T - f_j) * f_dj
        g_dj = sigmoid_derivative(np.matmul(self.w1[i, :], self.x.T))
        res = weight * theta_j * self.w2[:, i] * g_dj * self.x.T[r, :] * (self.w1[i, r] - cmin) ** (1 - alpha)
        return res.mean(axis=1)
    
    def caputo_h2o_matrix(self):
        alpha = self.cfg.alpha
        cmin = min(self.w1.min(), self.w2.min())
        weight = 1 / ((1 - alpha) * gamma(1 - alpha))
        f_j = np.matmul(self.w2, sigmoid(np.matmul(self.w1, self.x.T)))
        f_dj = -1
        theta_j = (self.y.T - f_j) * f_dj
        g_dj = sigmoid_derivative(np.matmul(self.w1, self.x.T))
        # res = weight * theta_j * self.w2 * g_dj * self.x.T * (self.w1 - cmin) ** (1 - alpha)
        res = weight * theta_j * g_dj 
        res = np.einsum('ij,kj->ikj', res, self.x.T).mean(axis=-1)
        res = res * self.w2.T * (self.w1 - cmin) ** (1 - alpha)
        return res
    
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
    
    def anti_caputo_i2h_matrix(self):
        alpha = self.cfg.alpha
        cmax = max(self.w1.max(), self.w2.max())
        weight = 1 / ((1 - alpha) * gamma(1 - alpha))
        f_j = np.matmul(self.w2, sigmoid(np.matmul(self.w1, self.x.T)))
        f_dj = -1
        theta_j = (self.y.T - f_j) * f_dj
        g_j = sigmoid(np.matmul(self.w1, self.x.T))
        res = (weight * theta_j * g_j).mean(axis=1) * (cmax - self.w2) ** (1 - alpha)
        return res
        
    def anti_caputo_h2o_matrix(self):
        alpha = self.cfg.alpha
        cmax = max(self.w1.max(), self.w2.max())
        weight = 1 / ((1 - alpha) * gamma(1 - alpha))
        f_j = np.matmul(self.w2, sigmoid(np.matmul(self.w1, self.x.T)))
        f_dj = -1
        theta_j = (self.y.T - f_j) * f_dj
        g_dj = sigmoid_derivative(np.matmul(self.w1, self.x.T))
        res = weight * theta_j * g_dj 
        res = np.einsum('ij,kj->ikj', res, self.x.T).mean(axis=-1)
        res = res * self.w2.T * (cmax - self.w1) ** (1 - alpha)
        return res
    
    def non_caputo_i2h(self, i):
        weight = 1 / (2 * np.sin(self.cfg.alpha * np.pi / 2))
        return weight * (self.caputo_i2h(i) - self.anti_caputo_i2h(i))
    
    def non_caputo_i2h_matrix(self):
        weight = 1 / (2 * np.sin(self.cfg.alpha * np.pi / 2))
        return weight * (self.caputo_i2h_matrix() - self.anti_caputo_i2h_matrix())

    def non_caputo_h2o(self, i, r):
        weight = 1 / (2 * np.sin(self.cfg.alpha * np.pi / 2))
        return weight * (self.caputo_h2o(i, r) - self.anti_caputo_h2o(i, r))
    
    def non_caputo_h2o_matrix(self):
        weight = 1 / (2 * np.sin(self.cfg.alpha * np.pi / 2))
        return weight * (self.caputo_h2o_matrix() - self.anti_caputo_h2o_matrix())

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
        self.w1 = self.w1 - lr * w1_update
        
    def predict(self, x):
        return np.matmul(self.w2, sigmoid(np.matmul(self.w1, x.T))).reshape(-1)
    
    def save_model(self, dir):
        dir = Path(dir)
        np.save(dir / 'w1.npy', self.w1)
        np.save(dir / 'w2.npy', self.w2)

    def load_model(self, dir):
        dir = Path(dir)
        self.w1 = np.load(dir / 'w1.npy')
        self.w2 = np.load(dir / 'w2.npy')
        
    def save(self, path):
        np.savez(path, w1=self.w1, w2=self.w2)

    def load(self, path):
        w = np.load(path)
        self.w1 = w['w1']
        self.w2 = w['w2']
        
    def train(self, epoch, train_type, save_dir=None):
        for i in range(epoch):
            print('train epoch: {} / {}'.format(i + 1, epoch))
            self.update(train_type)

        if save_dir:
            self.save_model(save_dir)
            
    def custom_i2h(self, i):
        pass
    
    def custom_h2o(self, i, r):
        pass
    
    def custom_update(self):
        lr = self.cfg.learning_rate
        n, m = self.w1.shape
        w2_param = np.zeros_like(self.w2)
        w1_param = np.zeros_like(self.w1)
        for i in range(n):
            w2_param[:, i] = self.custom_i2h(i)
        for i in range(n):
            for r in range(m):
                w1_param[i][r] = self.custom_h2o(i, r)
        
        self.w1 = self.w1 - lr * w1_param
        self.w2 = self.w2 - lr * w2_param
    
    
if __name__ == '__main__':
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score
    root_dir = Path(__file__).parent
    df = pd.read_csv(root_dir / '../housing.csv', header=None, sep='\s+') 
    df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    x = StandardScaler().fit_transform(x)
    
    cfg = Config(alpha=0.5, input_size=x.shape[1], output_size=1, learning_rate=0.05, hidden_size=20)
    cnn = CNNModelMatrix(cfg, x, y)
    cnn.caputo_h2o_matrix()
    for epoch in range(100):
        cnn.update(type='caputo')
        print('update: {}'.format(epoch + 1))
        print('r2: ', r2_score(cnn.y, cnn.predict(x)))
    
    # cnn.save_model(root_dir / '../data')
    # print(cnn.predict(x[0]))
    
    

