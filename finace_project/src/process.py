import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind, f, t
from scipy.optimize import minimize


@dataclass
class Config:
    data_path: str = field(default='./data/数据集/f-rx(1973.01-2022.12)_clear.xlsx')
    extra_feature_data_path: str = field(default='./data/数据集/宏观数据1973.01-2022.12.csv')
    nber_data_path: str = field(default='./data/数据集/NBER based US_Recession Indicators.xls')
    start_year: int = field(default=2005)
    end_year: int = field(default=2016)
    interval: int = field(default=6)
    gamma: int = field(default=5)
    epsilon: int = field(default=50)
    

class Process:
    def __init__(self, cfg):
        self.cfg = cfg
        self.df1 = pd.read_csv(Path(cfg.extra_feature_data_path), encoding='gbk', skiprows=4).drop(0).reset_index(drop=True)
        self.df2 = pd.read_excel(Path(cfg.data_path))
        self.df3 = pd.read_excel(Path(cfg.nber_data_path), skiprows=10)
        

        self.df2['year'] = self.df2['month'].apply(lambda x: x // 100)
        self.feature_cols = ['y_1', 'f_2', 'f_3', 'f_4', 'f_5', 'f_6', 'f_7', 'f_8', 'f_9', 'f_10']
        self.label_cols = ['rx_2', 'rx_3', 'rx_4', 'rx_5', 'rx_7', 'rx_10', 'rx_ew1']
        self.front_flag = False
        self.back_flag = False

    
    @staticmethod
    def cal_p_value(y_true, y_pred):
        return ttest_ind(y_true, y_pred)[1].item()
    
    def norm_df2(self):
        ss = StandardScaler()
        self.df2[self.feature_cols] = ss.fit_transform(self.df2[self.feature_cols])

    
    def cal_f_p_value(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_mean = np.mean(y_true)
        SST = np.sum((y_true - y_mean) ** 2)
        SSR = np.sum((y_pred - y_mean) ** 2)
        SSE = np.sum((y_true - y_pred) ** 2)

        k = len(self.feature_cols)
        n = len(y_true)
        F_statistic = (SSR / k) / (SSE / (n - k - 1))
        return F_statistic
    
    def split_train_test_data(self, start_year, interval):
        train = self.df2[self.df2['year'].isin([start_year + i for i in range(interval)])].copy()
        test = self.df2[self.df2['year'].isin([start_year + interval])].copy()
        return train, test
    
    
    def split_feature_label_data(self, df, feature_col=None, label_col=None):
        if not feature_col:
            feature_col = self.feature_cols
        if not label_col:
            label_col = self.label_cols[0]
            
        feature = df[feature_col]
        label = df[label_col]
        return feature, label
    
    
    def evaluate_period(self, model, data):
        train_x, train_y, test_x, test_y = data
        model.reset()
        model.fit(train_x, train_y)
        y_pred = model.predict(test_x)
        return (test_y.values, y_pred), r2_score(test_y, y_pred), self.cal_p_value(test_y, y_pred)
    
    
    def evaluate(self, model, feature_col, label_col):
        periods = range(self.cfg.start_year, self.cfg.end_year + 1)
        result = []
        for i in periods:
            train, test = self.split_train_test_data(i, self.cfg.interval)
            train_x, train_y = self.split_feature_label_data(train, feature_col, label_col)
            test_x, test_y = self.split_feature_label_data(test, feature_col, label_col)
            result.append(self.evaluate_period(model, (train_x, train_y, test_x, test_y)))
        return result
    
    def res_analysis_v1(self, result, weight=True):
        n = len(result)
        r = [i[1] for i in result]
        p = [i[2] for i in result]
        if weight:
            weights = [0] * len(r)
            for i in range(len(r)):
                if r[i] > 0:
                    weights[i] = r[i] * self.cfg.epsilon
                else:
                    weights[i] = abs(r[i] / self.cfg.epsilon)
            weights = [i / sum(weights) for i in weights]
            r = [r[i] * weights[i] for i in range(len(r))]
            return sum(r), sum(p) / n
            
        return sum(r) / n, sum(p) / n
    
    def res_analysis_v2(self, result, weight=False):
        y_true = []
        y_pred = []
        for r in result:
            y_true.extend(r[0][0].tolist())
            y_pred.extend(r[0][1].tolist())
        return r2_score(y_true, y_pred), self.cal_p_value(y_true, y_pred)
    
    def res_analysis_v3(self, result):
        valid_count = self.cfg.end_year - self.cfg.start_year + 1
        assert valid_count == len(result)
        nber = []
        for i in range(self.cfg.start_year, self.cfg.end_year + 1):
            valid_year = i + self.cfg.interval
            nber.append(int(self.df3[self.df3['year3'] == valid_year]['USREC'].sum() > 0))
        y_true_0 = []
        y_true_1 = []
        y_pred_0 = []
        y_pred_1 = []
        for i, r in enumerate(result):
            if nber[i] == 0:
                y_true_0.extend(r[0][0].tolist())
                y_pred_0.extend(r[0][1].tolist())
            else:
                y_true_1.extend(r[0][0].tolist())
                y_pred_1.extend(r[0][1].tolist())
        return r2_score(y_true_0, y_pred_0), self.cal_p_value(y_true_0, y_pred_0), r2_score(y_true_1, y_pred_1), self.cal_p_value(y_true_1, y_pred_1)

    
    def res_analysis(self, result, weight=False, kernel_type='v2'):
        if kernel_type == 'v1':
            return self.res_analysis_v1(result, weight)
        elif kernel_type == 'v2':
            return self.res_analysis_v2(result, weight)
        elif kernel_type == 'v3':
            return self.res_analysis_v3(result)
        else:
            raise TypeError("not valid kernel type")
        
    def show_bond_risk(self, model):
        for i in range(len(self.label_cols)):
            print(self.label_cols[i])
            result = self.evaluate(model, self.feature_cols, self.label_cols[i])
            r2, p = self.res_analysis(result)
            if r2 > 0:
                print('r2: {} p-value: {}'.format(r2, p))
            else:
                print('r2: {} p : -'.format(r2, p))
            print('-' * 30)

    
    def concat_cp_factor(self, drop=False):
        self.df1['Month'] = self.df1['Month'].astype('int')
        self.df1.fillna(0, inplace=True)
        self.df2 = pd.merge(left=self.df2, right=self.df1, left_on='month', right_on='Month')

        extra_features = self.df1.columns[1:].tolist()
        ss = StandardScaler()
        self.df2[extra_features] = ss.fit_transform(self.df2[extra_features])

        self.feature_cols.extend(extra_features)
        if drop:
            self.feature_cols = self.feature_cols[10:]
        self.cal_p_value = self.cal_f_p_value

    def group_factor(self):
        self.df3['month'] = self.df3['observation_date'].astype(str).apply(lambda x: int(x.replace('-', '')[:6]))
        self.df3['year3'] = self.df3['month'].apply(lambda x: x // 100)
        self.df2 = pd.merge(left=self.df2, right=self.df3, on='month')

    def show_group_bond_risk_front(self, model):
        if not self.front_flag:
            self.group_factor()
            self.front_flag = True
        self.label_cols = ['rx_2', 'rx_5', 'rx_10']
        print('Fwd rates:')
        for i in range(len(self.label_cols)):
            print(self.label_cols[i])
            result = self.evaluate(model, self.feature_cols, self.label_cols[i])
            r2_0, p_0, r2_1, p_1 = self.res_analysis(result, kernel_type='v3')
            print('Exp r2: {} Rec r2: {}'.format(r2_0, r2_1))

    def show_group_bond_risk_back(self, model):
        if not self.back_flag:
            self.concat_cp_factor()
            self.back_flag = True
        print('Fwd rates + Macro')
        for i in range(len(self.label_cols)):
            print(self.label_cols[i])
            result = self.evaluate(model, self.feature_cols, self.label_cols[i])
            r2_0, p_0, r2_1, p_1 = self.res_analysis(result, kernel_type='v3')
            print('Exp r2: {} Rec r2: {}'.format(r2_0, r2_1))

    def get_feature_importance(self, model):
            i = -1
            result = self.evaluate(model, self.feature_cols, self.label_cols[i])
            importance = model.feature_importance
            return importance
    
    def mean_variance_utility(self, returns, gamma=5):
        expected_return = np.mean(returns)
        variance = np.var(returns)
        utility = expected_return - 0.5 * gamma * variance
        return utility

    def power_utility(self, returns, gamma=2):
        expected_power_return = np.mean(returns ** (1 - gamma))
        utility = (expected_power_return) / (1 - gamma)
        return utility
    
    def cal_cer(self, predictions):
        def objective(weights):
            portfolio_return = np.dot(predictions, weights)
            portfolio_variance = np.dot(weights.T, np.dot(np.cov(predictions, rowvar=False), weights))
            return -np.mean(portfolio_return) + 0.5 * 5 * portfolio_variance  

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  
        bounds = tuple((0, 1) for _ in range(len(predictions)))
        initial_guess = np.ones(len(predictions)) / len(predictions)

        result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        cer = -result.fun
        return cer
    
    def cal_power(self, predictions):
        def power_utility(weights, gamma, returns):
            portfolio_return = np.dot(weights, returns)
            return (portfolio_return ** (1 - gamma)) / (1 - gamma)
        risk_averse_coefficient = 2
        def objective(weights):
            expected_utility = np.mean(power_utility(weights, risk_averse_coefficient, predictions))
            return -expected_utility
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(len(predictions)))
        initial_guess = np.ones(len(predictions)) / len(predictions)
        
        result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        cer = -result.fun
        cer = np.clip(cer, -10, 10)
        return cer
        

    def show_utility_bond_risk(self, model, model_cfgs):
        self.label_cols = ['rx_2', 'rx_10', 'rx_ew1']
        res= []
        for label in self.label_cols:
            print(label)
            for model_config in model_cfgs:
                task_name = model_config['name']
                model_name = model_config['config']['model']
                params = model_config['config']['params']

                model.set_params(params)
                model.set_model(model_name)
                print('#' * 20)
                print(task_name)
                print('#' * 20)
                result = self.evaluate(model, self.feature_cols, label)
                y_true = []
                y_pred = []
                for r in result:
                    y_true.extend(r[0][0].tolist())
                    y_pred.extend(r[0][1].tolist())
                y_true = np.array(y_true)
                y_pred = np.array(y_pred)
                mvu_pred = self.cal_cer(y_pred)
                mvu_true = self.cal_cer(y_true)
                pu_pred = self.cal_power(y_pred)
                pu_true = self.cal_power(y_true)
                res.append({
                    'label': label,
                    'task_name': task_name,
                    'y_true': y_true,
                    'y_pred': y_pred,
                    'mvu_pred': mvu_pred,
                    'mvu_true': mvu_true,
                    'pu_pred': pu_pred,
                    'pu_true': pu_true
                })
        return res
                
                # print('mean variance utility: {} p-value: {}'.format(mvu_pred[0] - mvu_true[0], self.cal_p_value(y_true[1], y_pred[1])))
                # print('power utility: {} p-value: {}'.format(pu_pred[0] - pu_true[0], self.cal_p_value(y_true[1], y_pred[1])))
        

            