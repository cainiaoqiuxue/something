# -*- coding:utf-8 -*-
import numpy as np
from scipy.special import gammaln
from scipy.optimize import minimize
from scipy.special import hyp2f1
import warnings
warnings.filterwarnings("ignore")


class BgNbd:
    # https://benalexkeen.com/bg-nbd-model-for-customer-base-analysis-in-python/
    def __init__(self, frequency, recency, period, init_params=None):
        self.scale = 1 / period.max()
        self.x = frequency
        self.t_x = recency
        self.T = period
        self.scale_tx = recency * self.scale
        self.scale_T = period * self.scale
        # params = [r, alpha, a, b]
        self.params = np.array(init_params) if init_params else np.ones(4) * 0.1

    def negative_log_likelihood(self, params):
        if np.any(np.asarray(params)) <= 0:
            return np.inf
        r, alpha, a, b = params
        ln_A_1 = gammaln(r + self.x) - gammaln(r) + r * np.log(alpha)
        ln_A_2 = gammaln(a + b) + gammaln(b + self.x) - gammaln(b) - gammaln(a + b + self.x)
        ln_A_3 = -(r + self.x) * np.log(alpha + self.scale_T)
        ln_A_4 = self.x.copy()
        ln_A_4[ln_A_4 > 0] = (np.log(a) - np.log(b + ln_A_4[ln_A_4 > 0] - 1) - (r + ln_A_4[ln_A_4 > 0]) * np.log(alpha + self.scale_tx))
        delta = np.where(self.x > 0, 1, 0)
        log_likelihood = ln_A_1 + ln_A_2 + np.log(np.exp(ln_A_3) + delta * np.exp(ln_A_4))
        return -log_likelihood.sum()

    @staticmethod
    def _func_caller(params, func_args, function):
        return function(params, *func_args)

    def negative_log_likelihood_v2(self, log_params):
        params = np.exp(log_params)
        r, alpha, a, b = params

        A_1 = gammaln(r + self.x) - gammaln(r) + r * np.log(alpha)
        A_2 = gammaln(a + b) + gammaln(b + self.x) - gammaln(b) - gammaln(a + b + self.x)
        A_3 = -(r + self.x) * np.log(alpha + self.scale_T)
        A_4 = np.log(a) - np.log(b + np.maximum(self.x, 1) - 1) - (r + self.x) * np.log(self.scale_tx + alpha)

        max_A_3_A_4 = np.maximum(A_3, A_4)

        penalizer_term = sum(params ** 2)
        weights = np.ones_like(self.x, dtype=int)
        ll =  weights * (A_1 + A_2 + np.log(np.exp(A_3 - max_A_3_A_4) + np.exp(A_4 - max_A_3_A_4) * (self.x > 0)) + max_A_3_A_4)

        return -ll.sum() / weights.sum() + penalizer_term
    
    def fit(self, **kwargs):
        method = kwargs.get('method', 'Nelder-Mead')
        tol = kwargs.get('tol', 1e-7)
        output = minimize(self.negative_log_likelihood,
                          method=method,
                          tol=tol,
                          x0=self.params,
                          options={'maxiter': 2000}
                          )
        self.params = np.array([output.x[i] for i in range(4)])
        self.params[1] /= self.scale

    def expected_sales_to_time_t(self, t):
        r, alpha, a, b = self.params
        hyp2f1_a = r
        hyp2f1_b = b
        hyp2f1_c = a + b - 1
        hyp2f1_z = t / (alpha + t)
        hyp_term = hyp2f1(hyp2f1_a, hyp2f1_b, hyp2f1_c, hyp2f1_z)
        return ((a + b - 1) / (a - 1)) * (1 - (((alpha / (alpha + t)) ** r) * hyp_term))

    def calculate_conditional_expectation(self, t, x, t_x, period):
        r, alpha, a, b = self.params
        first_term = (a + b + x - 1) / (a - 1)
        hyp2f1_a = r + x
        hyp2f1_b = b + x
        hyp2f1_c = a + b + x - 1
        hyp2f1_z = t / (alpha + period + t)
        hyp_term = hyp2f1(hyp2f1_a, hyp2f1_b, hyp2f1_c, hyp2f1_z)
        second_term = (1 - ((alpha + period) / (alpha + period + t)) ** (r + x) * hyp_term)
        delta = 1 if x > 0 else 0
        denominator = 1 + delta * (a / (b + x - 1)) * ((alpha + period) / (alpha + t_x)) ** (r + x)
        return first_term * second_term / denominator


if __name__ == '__main__':
    import pandas as pd
    df = pd.read_csv('../../Python_Project/lifetimes-master/lifetimes/datasets/cdnow_customers_summary.csv')
    model = BgNbd(df['frequency'], df['recency'], df['T'])
    model.fit()
    print(model.params)
    print(model.expected_sales_to_time_t(52))
    print(model.calculate_conditional_expectation(39, 2, 30.42, 38.86))
