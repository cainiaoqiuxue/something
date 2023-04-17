import os
import datetime
import pandas as pd

from utils.weather_util import get_all_day_weather
from utils.patient_util import read_patient_xls
from utils.log_util import get_logger

logger = get_logger()


class DataProcessor(object):
    def __init__(self):
        self.patient_df = read_patient_xls()
        self.weather_df = get_all_day_weather()
        self.feature_col = ['Sex', 'Age', 'Occupation', 'date', 'Aver RH', 'Aver pres', 'Aver temp',
                            'High pres', 'High temp', 'Low pres', 'Low temp', 'Min RH']
        self.weather_col = ['Aver RH', 'Aver pres', 'Aver temp', 'High pres', 'High temp', 'Low pres', 'Low temp',
                            'Min RH']
        self.df = pd.merge(left=self.patient_df, right=self.weather_df, on=['year', 'month', 'day'], how='left')
        self.df = self.df.dropna(how='any', subset=['Aver RH'])
        self.df = self.df.reset_index(drop=True)
        self.add_feature()

    def get_label(self):
        year_count = self.df.groupby('year').count()['Sex'].to_dict()
        day_count = self.df.groupby('date').count()['Sex'].to_dict()
        res_count = {}
        for key in day_count:
            res_count[key] = day_count[key] / year_count[key.year]
        label_df = pd.DataFrame({'date': list(res_count.keys()), 'label': list(res_count.values())})
        df = pd.merge(left=label_df, right=self.df, on='date', how='left')
        return df

    def get_train_test_data(self):
        df = self.get_label()
        df = df.drop_duplicates(subset=['date']).reset_index(drop=True)
        return df

    def add_feature(self):
        self.df['Diff temp'] = self.df['High temp'] - self.df['Low temp']
        self.df['Diff pres'] = self.df['High pres'] - self.df['Low pres']


