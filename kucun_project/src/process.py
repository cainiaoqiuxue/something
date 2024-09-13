import pandas as pd
import numpy as np
from pathlib import Path


class Process:
    def __init__(self):
        self.data_root = Path(__file__) / '../../data'
        self.df = pd.read_csv(self.data_root / '2023.csv', dtype=str)
        # self.remove = pd.read_csv(self.data_root / 'Fixed Location.csv')
        # self.volume = pd.read_csv(self.data_root / 'Volume.csv')
        self.id = 'Short Item No'
        self.sid = '2nd Item Number'
        self.tid = 'Task Number'
        self.date_col = 'Date Updated'
        self.cub = 'Used Cubic Dim'
        self.weight = 'Used Weight'
        self.key = 'Task Number'

    def remove_fix(self):
        remove_df = pd.read_csv(self.data_root / 'Fixed Location.csv', dtype=str)
        self.df = self.df[~self.df[self.id].isin(remove_df[self.id])].reset_index(drop=True)

    
    def clean_data(self):
        self.remove_fix()
        self.df[self.cub] = self.df[self.cub].str.replace(',', '').astype(float)
        self.df[self.weight] = self.df[self.weight].str.replace(',', '').astype(float)
        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])

    @staticmethod
    def exponential_weighted_function(x, alpha=0.5):
        if isinstance(x, pd.Series):
            x = x.values
        x0 = 0
        for i in x:
            x0 = alpha * x0 + (1 - alpha) * i
        return x0

    def cal_proior(self):
        data_gather = sorted(self.df[self.date_col].unique())
        x = np.arange(len(data_gather)) * 0.1
        weight = np.exp(x) / np.sum(np.exp(x))
        weight_map = dict(zip(data_gather, weight))
        self.df['weight_score'] = self.df[self.date_col].map(weight_map)
        weight_freq = self.df.groupby(self.id)['weight_score'].sum()
        weight_freq = weight_freq.sort_values(ascending=False)

        high = int(len(weight_freq) * 0.2)
        medium = int(len(weight_freq) * 0.5) - high
        low = len(weight_freq) - high - medium

        labels = ['High'] * high + ['Medium'] * medium + ['low'] * low
        label_dict = dict(zip(weight_freq.index, labels))
        self.df['label'] = self.df[self.id].map(label_dict)

        return label_dict


if __name__ == '__main__':
    p = Process()
    p.clean_data()
    wd = p.cal_proior()
    p.df[[p.id, p.sid, 'label']].groupby(p.id).first().reset_index().to_csv('result1.csv', index=False)