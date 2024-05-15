import os
import numpy as np
import pandas as pd
import datetime

from code.config import Config


def decorate(func):
    def wrap(*args, **kwargs):
        print('start ', func.__name__, '...')
        res = func(*args, **kwargs)
        print('done')
        return res

    return wrap


@decorate
def read_raw_data():
    return pd.read_csv(Config.data_path)


@decorate
def drop_column(df, cols):
    return df.drop(columns=cols)


@decorate
def get_label(df, label_col):
    y = df.pop(label_col)
    return df, y


@decorate
def make_dummies(df, cols):
    if isinstance(cols, (list, set, tuple)):
        new_df = pd.get_dummies(df, columns=cols)
    elif isinstance(cols, str):
        new_df = pd.get_dummies(df, columns=[cols])
    else:
        raise TypeError("wrong cols type")
    return new_df


def diff_time(target, now, format):
    target_time = datetime.datetime.strptime(target, format)
    diff = now - target_time
    return diff.days


@decorate
def convert_time_col(df, col, now=datetime.datetime.now(), format='%Y-%m-%d', inplace=True):
    if inplace:
        df[col] = df[col].apply(diff_time, args=(now, format))
    else:
        df[col + '_convert'] = df[col].apply(diff_time, args=(now, format))
    return df


@decorate
def convert_employmentLength(df):
    df['employmentLength'] = df['employmentLength'].map({
        np.NaN: -1,
        '< 1 year': 0,
        '1 year': 1,
        '2 years': 2,
        '3 years': 3,
        '4 years': 4,
        '5 years': 5,
        '6 years': 6,
        '7 years': 7,
        '8 years': 8,
        '9 years': 9,
        '10+ years': 10,
    })
    return df


@decorate
def convert_earliesCreditLine(df, now=datetime.datetime.now()):
    months = {
        'Jan': 1,
        'Feb': 2,
        'Mar': 3,
        'Apr': 4,
        'May': 5,
        'Jun': 6,
        'Jul': 7,
        'Aug': 8,
        'Sep': 9,
        'Oct': 10,
        'Nov': 11,
        'Dec': 12
    }

    def transfer_time(target):
        m = target[:3]
        new_target = str(months[m]) + target[3:]
        return datetime.datetime.strptime(new_target, '%m-%Y')

    df['earliesCreditLine'] = df['earliesCreditLine'].apply(lambda x: (now - transfer_time(x)).days)
    return df


def data_preprocess_v1(label_split=True):
    '''
    data clean version1

    drop
        id --no sense
        employmentTitle --too many kinds
        postCode --too many kinds
        title --too many kinds
        policyCode --only one value
    convert time:
        employmentLength
        issueDate
        earliesCreditLine
    dummy(one-hot):
        term
        grade
        subGrade
        homeOwnership
        verificationStatus
        purpose
        regionCode
        initialListStatus
        applicationType

    :return: cleaned data
    '''
    load_path = os.path.join(Config.data_file, 'data_preprocess_v1.csv')
    if os.path.exists(load_path):
        df = pd.read_csv(load_path)
        print('data has been pre-loaded')
    else:
        df = read_raw_data()
        need_drop = ['id', 'employmentTitle', 'postCode', 'title', 'policyCode']
        df = drop_column(df, need_drop)
        df = convert_time_col(df, 'issueDate')
        df = convert_employmentLength(df)
        df = convert_earliesCreditLine(df)
        need_one_hot = ['term', 'grade', 'subGrade', 'homeOwnership', 'verificationStatus', 'purpose', 'regionCode',
                        'initialListStatus', 'applicationType']
        df = pd.get_dummies(df, columns=need_one_hot, dummy_na=True)
        df.to_csv(load_path, index=False)
    if label_split:
        feature, label = get_label(df, 'isDefault')
        return feature, label
    else:
        return df


def data_preprocessing_with_missing(n2v):
    feature, label = data_preprocess_v1()
    feature = feature.fillna(n2v)
    return feature, label


def balance_data(pos_scale=2):
    df = data_preprocess_v1(label_split=False)
    n0 = df[df['isDefault'] == 1].shape[0]
    sampler = df[df['isDefault'] == 0].sample(n0 * pos_scale, random_state=Config.seed)
    new_df = pd.concat([df[df['isDefault'] == 1], sampler])
    feature, label = get_label(new_df, 'isDefault')
    return feature, label


def data_preprocessing_v2(label_split=True):
    df = read_raw_data()
    need_drop = ['id', 'employmentTitle', 'title', 'policyCode']
    df = drop_column(df, need_drop)
    df['postCode'] = df['postCode'].apply(lambda x: x if x > 10 else np.NaN)
    df = convert_time_col(df, 'issueDate')
    df = convert_employmentLength(df)
    df = convert_earliesCreditLine(df)
    need_one_hot = ['term', 'grade', 'subGrade', 'homeOwnership', 'verificationStatus', 'purpose', 'regionCode',
                    'initialListStatus', 'applicationType','postCode']
    df = pd.get_dummies(df, columns=need_one_hot, dummy_na=True)
    if label_split:
        feature, label = get_label(df, 'isDefault')
        return feature, label
    else:
        return df
