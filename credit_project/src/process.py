# !usr/bin/env python
# -*- coding:utf-8 -*-

import os
import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from imblearn.combine import SMOTETomek


def log_func(func):
    def wrap(*args, **kwargs):
        print('start: {} ...'.format(func.__name__), end='')
        res = func(*args, **kwargs)
        print('done')
        return res
    return wrap


class Processor:
    def __init__(self, data_path):
        self.df = self.read_data(data_path)

    @log_func
    def read_data(self, data_path):
        if isinstance(data_path, str):
            df = pd.read_csv(data_path)
        elif isinstance(data_path, pd.DataFrame):
            df = data_path
        else:
            raise RuntimeError("invalid data type")
        return df

    @log_func
    def drop_column(self, cols):
        self.df = self.df.drop(columns=cols)

    @log_func
    def get_label(self, label_col):
        columns = self.df.columns
        feature_col = [i for i in columns if i != label_col]
        # y = self.df.pop(label_col)
        return self.df[feature_col], self.df[label_col]

    @log_func
    def make_dummies(self, cols, dummy_na=False):
        if isinstance(cols, (list, set, tuple)):
            new_df = pd.get_dummies(self.df, columns=cols, dummy_na=dummy_na, dtype=int)
        elif isinstance(cols, str):
            new_df = pd.get_dummies(self.df, columns=[cols], dummy_na=dummy_na, dtype=int)
        else:
            raise TypeError("wrong cols type")
        return new_df

    @staticmethod
    def diff_time(target, now, formats):
        target_time = datetime.datetime.strptime(target, formats)
        diff = now - target_time
        return diff.days

    @log_func
    def convert_time_col(self, col, now=datetime.datetime.now(), formats='%Y-%m-%d', inplace=True):
        if inplace:
            self.df[col] = self.df[col].apply(self.diff_time, args=(now, formats))
        else:
            self.df[col + '_convert'] = self.df[col].apply(self.diff_time, args=(now, formats))

    @log_func
    def convert_employment_length(self):
        self.df['employmentLength'] = self.df['employmentLength'].map({
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

    @log_func
    def convert_earlies_credit_line(self, now=datetime.datetime.now()):
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

        self.df['earliesCreditLine'] = self.df['earliesCreditLine'].apply(lambda x: (now - transfer_time(x)).days)

    def data_preprocess_v1(self, label_split=True, n2v=-1):
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
        need_drop = ['id', 'employmentTitle', 'postCode', 'title', 'policyCode']
        self.drop_column(need_drop)
        self.convert_time_col('issueDate')
        self.convert_employment_length()
        self.convert_earlies_credit_line()
        need_one_hot = ['term', 'grade', 'subGrade', 'homeOwnership', 'verificationStatus', 'purpose', 'regionCode',
                        'initialListStatus', 'applicationType']
        self.df = self.make_dummies(need_one_hot, dummy_na=True)
        # df = pd.get_dummies(df, columns=need_one_hot, dummy_na=True)
        if label_split:
            feature, label = self.get_label('isDefault')
            return feature.fillna(n2v), label
        else:
            return self.df.fillna(n2v)

    @staticmethod
    def imbalance_sample(feature, label):
        smt = SMOTETomek(random_state=42)
        feature, label = smt.fit_resample(feature, label)
        return feature, label

    def sample(self, frac=None, pos_frac=None, neg_frac=None):
        label_col = 'isDefault'
        pos = self.df[self.df[label_col] == 1]
        neg = self.df[self.df[label_col] == 0]
        pos_frac = pos_frac or frac
        neg_frac = neg_frac or frac
        pos = pos.sample(frac=pos_frac, random_state=42)
        neg = neg.sample(frac=neg_frac, random_state=42)
        self.df = pd.concat([pos, neg])
        self.df = self.df.sample(n=len(self.df)).reset_index(drop=True)


class TranTestSplit:
    def __init__(self, feature, label):
        self.feature = feature
        self.label = label

    def split(self, test_size=0.1):
        train_x, test_x, train_y, test_y = train_test_split(self.feature,
                                                            self.label,
                                                            test_size=test_size,
                                                            random_state=42)
        return train_x, train_y, test_x, test_y

    def kfold(self, k=5):
        skf = StratifiedKFold(n_splits=k, shuffle=False)
        groups = skf.split(self.feature, self.label)
        train_idxs = []
        test_idxs = []
        for train, test in groups:
            train_idxs.append(train)
            test_idxs.append(test)
        return train_idxs, test_idxs

