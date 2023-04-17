import pandas as pd

from utils.weather_util import read_all_weather, weather_data_preprocess
from utils.patient_util import read_patient_data, patient_data_preprocess
from utils.log_util import get_logger
from config import Config as config

logger = get_logger()


def get_data():
    p_df = read_patient_data('./data/patient.xlsx')
    p_df = patient_data_preprocess(p_df)
    w_df = read_all_weather()
    w_df = weather_data_preprocess(w_df)
    df = pd.merge(left=p_df, right=w_df, left_on=['地址', '日期'], right_on=['area', 'date'], how='left')
    df = df.sort_values('date').reset_index(drop=True)
    df = df.dropna(subset=config.weather_columns_float, how='all')
    return df


def get_label(pos_df, n=5):
    pos_df['label'] =1
    neg_df = pd.DataFrame()
    w_df = read_all_weather()
    w_df = weather_data_preprocess(w_df)
    for i in range(len(pos_df)):
        pos_date = pos_df.loc[i, 'date']
        neg_samples = pos_df.iloc[i+1:i+n-1, :].copy()
        neg_samples['date'] = pos_date
        neg_samples = neg_samples[['date', 'area']]
        neg_samples = pd.merge(left=neg_samples, right=w_df, on=['date', 'area'], how='left')
        neg_df = pd.concat([neg_df, neg_samples], ignore_index=True)
    neg_df['label'] = 0
    df = pd.concat([pos_df, neg_df], ignore_index=True)
    return df


def get_data_v2():
    p_df = read_patient_data('./data/patient.xlsx')
    p_df = patient_data_preprocess(p_df)
    w_df = read_all_weather()
    w_df = weather_data_preprocess(w_df)
    p_df = p_df[['地址', '日期']].drop_duplicates()
    df = pd.merge(left=p_df, right=w_df, left_on=['地址', '日期'], right_on=['area', 'date'], how='left')
    df = df.dropna(subset=config.weather_columns_float, how='all')
    #pos_sample
    df = df.sort_values('date').reset_index(drop=True)
    # neg sample
    dates = df['date'].unique()
    neg_df = pd.DataFrame()
    for date in dates:
        now_df = df[df['date'] == date]
        pos_area = now_df['area'].values.tolist()
        neg_area = [area for area in config.areas if area not in pos_area]
        now_df = w_df[(w_df['area'].isin(neg_area)) & (w_df['date'] == date)]
        neg_df = pd.concat([neg_df, now_df], ignore_index=True)
    # get label
    df['label'] = 1
    neg_df['label'] = 0
    df = pd.concat([df, neg_df], axis=0, join='inner', ignore_index=True)
    return df


def filter_na(df, threshold=1):
    logger.info('--na fraction of feature list--')
    s = df.isna().mean(axis=0)
    res = []
    for f in config.weather_columns_float:
        v = s[f]
        logger.info('{f}: {v:.2f}'.format(f=f, v=v))
        if v >= threshold:
            res.append(f)
    logger.info('--end--')
    logger.info('threshold is {t}, drop columns: {r} because of nan'.format(t=threshold, r=res))
    feature_names = [f for f in config.weather_columns_float if f not in res]
    return feature_names

if __name__ == '__main__':
    df = get_data_v2()
    print(df.head())