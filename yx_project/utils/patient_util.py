import os
import re
import numpy as np
import pandas as pd
from config import Config as config

def read_patient_data(file_path):
    """excel to DataFrame

    :param file_path: excel path
    :return: DataFrame
    """
    return pd.read_excel(file_path)


def patient_data_preprocess(df):
    """clean patient data
    convert address to area
    convert date to datetime

    :param df:raw DataFrame
    :return:cleaned DataFrame
    """
    df['地址'] = df['地址1'].map(config.add2area)

    extra_df = df[~df['备注日期'].isna()].copy()
    extra_df['日期'] = extra_df['备注日期']
    df = pd.concat([df, extra_df], axis=0, ignore_index=True)
    df['日期'] = pd.to_datetime(df['日期'])
    df = df[['地址', '日期', '病因', '诊断类型']]
    return df

def encode_time(x):
    if type(x) == str:
        s = re.findall('\d+', x)
        if len(s) >= 3:
            s = s[:3]
            if len(s[2]) == 4:
                return s[2] + '/' + s[1] + '/' + s[0]
            else:
                return s[0] + '/' + s[1] + '/' +s [2]
        elif len(s) == 1:
            if len(s[0]) == 8:
                s = s[0]
                return s[:4] + '/' + s[4:6] + '/' + s[6:8]
        else:
            return ''
    else:
        return x


def convert_float(x):
    try:
        res = float(x)
    except:
        res = np.NaN
    return res


def read_patient_xls(path_dir=None):
    if path_dir is None:
        path_dir = os.path.join(config.data_root_path, 'Appendix-C1')
    files = os.listdir(path_dir)
    df = pd.DataFrame()
    for file in files:
        tmp_df = pd.read_excel(os.path.join(path_dir, file))
        tmp_df['Time of incidence'] = tmp_df['Time of incidence'].astype(str)
        tmp_df['Time of incidence'] = tmp_df['Time of incidence'].apply(encode_time)
        tmp_df['date'] = pd.to_datetime(tmp_df['Time of incidence'], errors='coerce')
        df = pd.concat([df, tmp_df], axis=0 ,ignore_index=True)
    df = df.dropna(how='any', subset=['date']).reset_index(drop=True)
    df['year'] = df['date'].apply(lambda x: x.year)
    df['month'] = df['date'].apply(lambda x: x.month)
    df['day'] = df['date'].apply(lambda x: x.day)
    df['Age'] = df['Age'].apply(convert_float)
    df['Occupation'] = df['Occupation'].apply(convert_float)
    return df