# -*- coding:utf-8 -*-

"""
 FileName     : clv.py
 Type         : pyspark/pysql/python
 Arguments    : None
 Author       : xingyuanfan@tencent.com
 Date         : 2023-09-11
 Description  : 
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from src.process import RFMData
from src.models import BgNbd, NewBgNbd
from lifetimes import GammaGammaFitter, BetaGeoFitter


class CLV:
    def __init__(self, cig_data, wine_data, other_data):
        self.cig_data = cig_data
        self.wine_data = wine_data
        self.other_data = other_data

    def gather_clv(self):
        df = pd.concat([self.cig_data, self.wine_data, self.other_data])
        rfm = RFMData(df)
        data = rfm.get_features(rfm.train_data)
        bg_model = BgNbd(data['frequency'], data['recency'], data['T'])
        bg_model.fit()
        gg_model = GammaGammaFitter()
        gg_model.fit(data['frequency'], data['money'])
        return {'rfm': rfm, 'bg_model': bg_model, 'gg_model': gg_model}

    def partial_clv(self, type=0):
        rfm_cig = RFMData(self.cig_data)
        data = rfm_cig.get_features(rfm_cig.train_data)
        if type == 0:
            bg_cig = BgNbd(data['frequency'], data['recency'], data['T'], init_params=np.ones(4))
        elif type == 1:
            bg_cig = NewBgNbd(data['frequency'], data['recency'], data['T'], data['preference'], 0, np.ones(5))
        else:
            bg_cig = NewBgNbd(data['frequency'], data['recency'], data['T'], data['preference'], 1, np.ones(6))
        bg_cig.fit()
        gg_cig = GammaGammaFitter()
        gg_cig.fit(data['frequency'], data['money'])
        
        rfm_wine = RFMData(self.wine_data)
        data = rfm_wine.get_features(rfm_wine.train_data)
        # bg_wine = BgNbd(data['frequency'] - 1, data['recency'], data['T'])
        if type == 0:
            bg_wine = BgNbd(data['frequency'] -1, data['recency'], data['T'], init_params=np.ones(4))
        elif type == 1:
            bg_wine = NewBgNbd(data['frequency'] - 1, data['recency'], data['T'], data['preference'], 0, np.ones(5))
        else:
            bg_wine = NewBgNbd(data['frequency'] - 1, data['recency'], data['T'], data['preference'], 1, np.ones(6))
        bg_wine.fit()
        gg_wine = GammaGammaFitter()
        gg_wine.fit(data['frequency'], data['money'])
        
        rfm_other = RFMData(self.other_data)
        data = rfm_other.get_features(rfm_other.train_data)
        # bg_other = BgNbd(data['frequency'], data['recency'], data['T'])
        if type == 0:
            bg_other = BgNbd(data['frequency'], data['recency'], data['T'], init_params=np.ones(4))
        elif type == 1:
            bg_other = NewBgNbd(data['frequency'], data['recency'], data['T'], data['preference'], 0, np.ones(5))
        else:
            bg_other = NewBgNbd(data['frequency'], data['recency'], data['T'], data['preference'], 1, np.ones(6))
        bg_other.fit()
        gg_other = GammaGammaFitter()
        gg_other.fit(data['frequency'], data['money'])
        return {
            'rfm_cig': rfm_cig,
            'bg_cig': bg_cig,
            'gg_cig': gg_cig,
            'rfm_wine': rfm_wine,
            'bg_wine': bg_wine,
            'gg_wine': gg_wine,
            'rfm_other': rfm_other,
            'bg_other': bg_other,
            'gg_other': gg_other,
        }


class Metric:
    def __init__(self, y_true, y_pred, name='clv'):
        self.name = name
        self.y_true = y_true
        self.y_pred = y_pred

    def cal_mae(self):
        return np.abs(self.y_true - self.y_pred).mean()

    def cal_mdae(self):
        return np.abs(self.y_true - self.y_pred).median()

    def cal_rmse(self):
        value = (self.y_true - self.y_pred) ** 2
        return np.sqrt(value.mean())

    def cal_pearson(self):
        return pd.DataFrame({'true': self.y_true, 'pred': self.y_pred}).corr().iloc[0, 1]

    def cal_spearmanr(self):
        return pd.DataFrame({'true': self.y_true, 'pred': self.y_pred}).corr('spearman').iloc[0, 1]

    def summary(self):
        return pd.DataFrame({
            'mae': self.cal_mae(),
            'mdae': self.cal_mdae(),
            'rmse': self.cal_rmse(),
            'pearson': self.cal_pearson(),
            'spearmanr': self.cal_spearmanr(),
            'loss': str(round(((np.abs(self.y_true - self.y_pred) / self.y_true).mean() * 100), 2)) + '%'
        }, index=[self.name])