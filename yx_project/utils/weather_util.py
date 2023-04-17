import os
import re
import numpy as np
import pandas as pd
from utils.log_util import get_logger
from config import Config as config

logger = get_logger()


def read_weather_txt(file_path):
    """txt to DataFrame

    :param file_path: txt file
    :return: DataFrame
    """
    logger.debug(f"read {file_path}")
    with open(file_path, mode='r', encoding='gbk') as f:
        res = f.readlines()
    columns = res[0].split()
    data = [s.split() for s in res[1:]]
    df = pd.DataFrame(data, columns=columns)
    return df


def read_weather_dir(dir_path):
    """data concat

    :param dir_path: txt dirs
    :return: concat DataFrame
    """
    logger.info(f"read dir: {dir_path}")
    files = os.listdir(dir_path)
    df = pd.DataFrame()
    for file in files:
        df = pd.concat([df, read_weather_txt(os.path.join(dir_path, file))], ignore_index=True)
    return df


def read_all_weather():
    """concat all weather data

    :return: DataFrame
    """
    df = pd.DataFrame()
    for file_dir in config.data_dirs:
        file_dir = os.path.join(config.data_root_path, file_dir)
        df = pd.concat([df, read_weather_dir(file_dir)], ignore_index=True)
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def get_weather_date(df):
    """get weather_date

    :param df: DataFrame
    :return: Series
    """
    def convert(x): return str(int(x))
    series = df.apply(lambda x: '-'.join(map(convert, [x['年份'], x['月份'], x['日期']])), axis=1)
    series = pd.to_datetime(series)
    return series


def get_code2area():
    """code to area

    :return: dict
    """
    pattern = re.compile('(\d+) ?-?(.*)?')
    code2area_dict = {}
    for content in config.data_dirs:
        g = re.search(pattern, content)
        if g is None:
            raise ValueError("area not match")
        code2area_dict[int(g.group(1))] = g.group(2)
    return code2area_dict


def weather_data_preprocess(df, default_value=None):
    """clean weather data
    replace nan

    :param df: raw DataFrame
    :return: cleaned DataFrame
    """
    df = df.replace('-', np.nan)
    df = df.replace('*', np.nan)
    df = df.replace('/', np.nan)
    df[config.weather_columns_int] = df[config.weather_columns_int].astype('int')
    df[config.weather_columns_float] = df[config.weather_columns_float].astype('float')
    if default_value is not None:
        df = df.fillna(default_value)
    df['date'] = get_weather_date(df)
    code2area = get_code2area()
    df['area'] = df['区站号'].map(code2area)
    return df


def read_weather_xls(path, sheet_name):
    df = pd.read_excel(path, sheet_name=sheet_name, skiprows=2, index_col='Date')
    months = [i+1 for i in range(12)]
    weather_col = df.columns[:8]
    cols = [['{w}_{i}'.format(w=w, i=i) for w in weather_col] for i in months]
    cols = [c for col in cols for c in col]
    df.columns = cols
    return df

def get_one_day_weather(year, month, day):
    year = str(year)[2:]
    path = os.path.join(config.data_root_path, 'Appendix-C2', 'data5.xls')
    df = read_weather_xls(path, sheet_name=year)
    cols = df.columns[df.columns.str.endswith('_{m}'.format(m=month))]
    value = df.loc[day, cols]
    value.index = [i.split('_')[0] for i in value.index]
    return value

def get_all_day_weather():
    file_path = os.path.join(config.data_root_path, 'Appendix-C2', 'weather_data.xlsx')
    if os.path.exists(file_path):
        return pd.read_excel(file_path)
    logger.info("get all day weather")
    years = ['2007', '2008', '2009', '2010']
    months = [i for i in range(1, 13)]
    days = [i for i in range(1, 32)]
    res = []
    for year in years:
        logger.info("begin to year {y}".format(y=year))
        for month in months:
            logger.info("begin to month {m}".format(m=month))
            for day in days:
                tmp_dict = {'year': year, 'month': month, 'day': day}
                value = get_one_day_weather(year, month, day)
                tmp_dict.update(value.to_dict())
                res.append(tmp_dict)
    logger.info("get weather done")
    df = pd.DataFrame(res)
    df['year'] = df['year'].astype(int)
    df.to_excel(file_path, index=False)
    return df






if __name__ == '__main__':
    dir_path = '../data/57707毕节'
    df = read_weather_dir(dir_path)
    df = weather_data_preprocess(df)
    print(df.head())