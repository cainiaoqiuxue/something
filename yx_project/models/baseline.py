import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
import lightgbm as lgb

from v1_code.data_process import DataProcessor as DataProcessor_v1
from data_process import DataProcessor as DataProcessor_v2


def concat_data():
    DataProcessor_v1.initialize()
    d = DataProcessor_v2()
    df_v1 = DataProcessor_v1.df.copy()
    df_v2 = d.df.copy()

    new_col = ['Aver RH', 'Aver pres', 'Aver temp', 'High pres', 'High temp', 'Low pres', 'Low temp', 'Min RH',
               'Diff temp', 'Diff pres', 'date']
    old_col = ['日平均相对湿度(%)', '日平均气压(hpa)', '日平均气温(℃)', '日最高气压(hpa)', '日最高气温(℃)', '日最低气压(hpa)', '日最低气温(℃)', '日最小相对湿度(%)',
               '日气温日较差(℃)', '日气压日较差(hpa)', 'date']
    map_dict = dict(zip(old_col, new_col))
    df_v1 = df_v1.rename(columns=map_dict)

    df = pd.concat([df_v1[new_col], df_v2[new_col]], axis=0, ignore_index=True)
    return df


def get_label(df):
    df['year'] = df['date'].apply(lambda x: x.year)
    year_count = df.groupby('year').count()['Aver RH'].to_dict()
    day_count = df.groupby('date').count()['Aver RH'].to_dict()
    res_count = {}
    for key in day_count:
        res_count[key] = day_count[key] / year_count[key.year]
    label_df = pd.DataFrame({'date': list(res_count.keys()), 'label': list(res_count.values())})
    df = pd.merge(left=label_df, right=df, on='date', how='left')
    df = df.drop_duplicates(subset=['date']).reset_index(drop=True)
    df = df[df['label'] < 0.01].reset_index(drop=True)
    return df


def split_train_test_data(df, year=2010):
    train_df = df[df['year'] < year].reset_index(drop=True)
    test_df = df[df['year'] >= year].reset_index(drop=True)
    return train_df, test_df


def multi_linear_model(x_train, y_train, kind='linear'):
    model_dict = {'linear': LinearRegression, 'ridge': Ridge, 'svm': SVR}
    model = model_dict[kind]()
    model.fit(x_train, y_train)
    return model


def lgb_model(x_train, y_train):
    params = {
        'objective': 'regression',
        'seed': 42,
        'verbosity': -1,
    }
    data = lgb.Dataset(x_train, label=y_train)
    model = lgb.train(params, data)
    return model