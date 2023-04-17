import os
import datetime
import pandas as pd

from utils.weather_util import read_all_weather, weather_data_preprocess
from utils.patient_util import read_patient_data, patient_data_preprocess
from utils.log_util import get_logger
from config import Config as config

logger = get_logger()

class DataProcessor(object):
    patient_df = None
    weather_df = None
    df = None

    @classmethod
    def initialize(cls):
        logger.info('DataProcessor initialize...')
        p_df = read_patient_data(os.path.join(config.data_root_path, 'patient.xlsx'))
        p_df = patient_data_preprocess(p_df)
        w_df = read_all_weather()
        w_df = weather_data_preprocess(w_df)
        w_df['日气压日较差(hpa)'] = w_df['日最高气压(hpa)'] - w_df['日最低气压(hpa)']
        p_df = p_df[['地址', '日期']].drop_duplicates()
        df = pd.merge(left=p_df, right=w_df, left_on=['地址', '日期'], right_on=['area', 'date'], how='left')
        df = df.dropna(subset=config.weather_columns_float, how='all')
        logger.info('DataProcessor Done')

        cls.patient_df = p_df
        cls.weather_df = w_df
        cls.df = df

    @classmethod
    def get_label(cls):
        # pos_sample
        df = cls.df.sort_values('date').reset_index(drop=True)
        # neg sample
        dates = df['date'].unique()
        neg_df = pd.DataFrame()
        for date in dates:
            now_df = df[df['date'] == date]
            pos_area = now_df['area'].values.tolist()
            neg_area = [area for area in config.areas if area not in pos_area]
            now_df = cls.weather_df[(cls.weather_df['area'].isin(neg_area)) & (cls.weather_df['date'] == date)]
            neg_df = pd.concat([neg_df, now_df], ignore_index=True)
        # get label
        df['label'] = 1
        neg_df['label'] = 0
        df = pd.concat([df, neg_df], axis=0, join='inner', ignore_index=True)
        return df

    @classmethod
    def filter_na_feature(cls, feature_df, threshold=1):
        logger.info('----- na fraction of feature list -----')
        s = feature_df.isna().mean(axis=0)
        res = []
        for f in feature_df.columns:
            v = s[f]
            logger.info('{f}: {v:.2f}'.format(f=f, v=v))
            if v >= threshold:
                res.append(f)
        logger.info('----- end -----')
        logger.info('threshold is {t}, drop columns: {r} because of nan'.format(t=threshold, r=res))
        feature_names = [f for f in config.weather_columns_float if f not in res]
        return feature_names

    @classmethod
    def slide_feature(cls, feature_df, window=3):
        logger.info('slide window is {w}'.format(w=window))
        start_date = feature_df['date'] - datetime.timedelta(days=window - 1)
        areas = feature_df['area']
        tmp_df = pd.DataFrame({'date': start_date, 'area': areas})
        dfs = []
        for i in range(window - 1):
            day_df = pd.merge(tmp_df, cls.weather_df, on=['date', 'area'], how='left')
            day_df = day_df[config.weather_columns_float]
            day_df = day_df.rename(columns={k: f'{k}_{window - i -1}day_before' for k in day_df.columns})
            dfs.append(day_df)
            tmp_df['date'] = tmp_df['date'] + datetime.timedelta(days=1)
        feature_df = feature_df[['date', 'area'] + config.weather_columns_float]
        for tdf in dfs:
            feature_df = pd.concat([feature_df, tdf], axis=1)
        return feature_df