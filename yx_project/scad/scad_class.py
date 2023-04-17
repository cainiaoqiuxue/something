import numpy as np


class Scad(object):
    def __init__(self, x, y, weight=None):
        self.x = np.array(x)
        self.y = np.array(y).reshape(-1)
        self.a = 3.7
        self.lambda_ = 0.025
        if weight is not None:
            self.weight = np.array(weight)
        else:
            self.weight = np.zeros(self.x.shape[1])

    @staticmethod
    def log_likelihood(x, y, b):
        linear_comb = np.matmul(x, b)
        res = np.sum(y * linear_comb) + np.sum(np.log(1 / (1 + np.exp(linear_comb))))
        return res

    def gauss_seidel(self, x, y, b0=None, diff=1, eps=1e-5, max_iter=10000):
        if b0 is None:
            b0 = np.zeros(x.shape[1])
        epoch = 0
        res = [self.log_likelihood(x, y, b0)]
        b_best = np.zeros_like(b0)
        while (diff > eps and epoch < max_iter):
            for j in range(len(b0)):
                linear_comb = np.matmul(x, b0)
                nominator = np.sum(y * x[:, j] - x[:, j] * np.exp(linear_comb) / (1 + np.exp(linear_comb)))
                denominator = -np.sum(x[:, j] ** 2 * np.exp(linear_comb) / (1 + np.exp(linear_comb)) ** 2)
                b0[j] = b0[j] - nominator / denominator
                res.append(self.log_likelihood(x, y, b0))
                if res[-1] > res[-2]:
                    b_best[j] = b0[j]
                diff = np.abs((res[-1] - res[-2]) / res[-2])
                epoch += 1
                if diff < eps:
                    break
        return b_best

    def get_p_lambda(self, theta):
        a = self.a
        lambda_ = self.lambda_
        theta_abs = np.abs(theta)
        if theta_abs > lambda_:
            if a * lambda_ > theta_abs:
                return (theta_abs ** 2 - 2 * a * lambda_ * theta_abs + lambda_ ** 2) / (2 - 2 * a)
            else:
                return (a + 1) * lambda_ ** 2 / 2
        else:
            return lambda_ * theta_abs

    def get_p_lambda_d(self, theta):
        a = self.a
        lambda_ = self.lambda_
        theta_abs = np.abs(theta)
        if theta_abs > lambda_:
            if a * lambda_ > theta_abs:
                return (a * lambda_ - theta) / (a - 1)
            else:
                return 0
        else:
            return lambda_

    def log_likelihood_scad(self, x, y, b):
        linear_comb = np.matmul(x, b)
        p_lambda = np.zeros_like(b)
        for i in range(len(b)):
            p_lambda[i] = self.get_p_lambda(b[i])
        res = np.sum(y * linear_comb) + np.sum(np.log(1 / (1 + np.exp(linear_comb)))) + x.shape[0] * np.sum(p_lambda)
        return res

    def scad_iter(self, x, y, b0=None, diff=1, eps=1e-5, max_iter=10000):
        if b0 is None:
            b0 = np.zeros(x.shape[1])
        epoch = 0
        res = [self.log_likelihood_scad(x, y, b0)]
        b_best = np.zeros_like(b0)
        while diff > eps and epoch < max_iter:
            for j in range(len(b0)):
                if np.abs(b0[j]) < 1e-6:
                    continue
                else:
                    linear_comb = np.matmul(x, b0)
                    nominator = np.sum(y * x[:, j] - x[:, j] * np.exp(linear_comb) / (1 + np.exp(linear_comb))) \
                                + x.shape[0] * b0[j] * self.get_p_lambda_d(b0[j]) / np.abs(b0[j])
                    denominator = -np.sum(x[:, j] ** 2 * np.exp(linear_comb) / (1 + np.exp(linear_comb)) ** 2) \
                                  + x.shape[0] * self.get_p_lambda_d(b0[j]) / np.abs(b0[j])
                    b0[j] = b0[j] - nominator / denominator
                    if np.abs(b0[j]) < 1e-6:
                        b0[j] = 0
            res.append(self.log_likelihood_scad(x, y, b0))
            if res[-1] > res[-2]:
                b_best = b0
            epoch += 1
        return b_best

    def cal_weight_with_scad(self,x=None, y=None):
        if x is None:
            x = self.x
            y = self.y
        b0 = self.gauss_seidel(x, y)
        return self.scad_iter(x, y, b0)



if __name__ == '__main__':
    import pandas as pd

    x = pd.read_csv('x_test.csv')
    y = pd.read_csv('y_test.csv')
    x = x.drop(columns=x.columns[0])
    y = y.drop(columns=y.columns[0])
    x = x.values
    y = y.values.reshape(-1)
    print(x.shape)
    print(y.shape)
    scad = Scad(x, y)
    print(scad.gauss_seidel(x, y))
    print(scad.cal_weight_with_scad(x, y))

