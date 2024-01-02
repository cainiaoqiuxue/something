# -*- coding:utf-8 -*-
# 数据处理：归一化，窗口化，数据连接，缺失值补充等
import os
import sys
import torch
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from utils import get_logger, save_pkl, load_pkl


class StockProcess:
    def __init__(self, config, add_text=False):
        self.config = config
        self.logger = get_logger('stock_process')
        self.add_text = add_text
        if self.add_text:
            path = os.path.abspath(os.path.join(os.path.dirname(__file__), self.config['data']['sentiment_feature']))
            self.text_feature = load_pkl(path)

    def read_data(self, path):
        df = pd.read_csv(path, sep='\t', header=None)
        df.columns = self.config['process']['all_col']
        if self.add_text:
            name = path.split('/')[-1].split('.')[0]
            res = self.text_feature.get(name, {})
            new_feature = df.apply(
                lambda x: res.get(x[self.config['process']['date_col']], torch.tensor([0., 0., 0.])).tolist(),
                axis=1,
                result_type='expand'
            )
            df = pd.concat([df, new_feature], axis=1)
            return df
        else:
            return df

    def split_train_test_data(self, df):
        year = self.config['process']['split_date']
        train_df = df[df['Date'] < year]
        test_df = df[df['Date'] > year]
        return train_df, test_df

    def make_model_data(self, df):
        config = self.config['process']
        df[config['date_col']] = pd.to_datetime(df[config['date_col']])
        df = df.sort_values(config['date_col'])
        if self.add_text:
            feature_col = config['feature_col'] + [0, 1, 2]
        else:
            feature_col = config['feature_col']
        feature = df[feature_col].values
        label = df[config['label_col']].values
        data = []
        target = []
        window = config['window']
        for i in range(window, len(feature) + 1):
            data.append(feature[i - window: i])
            if label[i - 1] > label[i - 2]:
                target.append(1)
            else:
                target.append(0)
        data = np.array(data)
        target = np.array(target)
        return data, target

    def make_single_file(self, path):
        df = self.read_data(path)
        train_df, test_df = self.split_train_test_data(df)
        train_data = self.make_model_data(train_df)
        test_data = self.make_model_data(test_df)
        return train_data, test_data

    def make_multi_file(self, dir_path, is_save=False):
        files = os.listdir(dir_path)
        train_feature = []
        train_label = []
        test_feature = []
        test_label = []
        for file in files:
            path = os.path.join(dir_path, file)
            train, test = self.make_single_file(path)
            train_feature.append(train[0])
            train_label.append(train[1])
            test_feature.append(test[0])
            test_label.append(test[1])
        train_data = (np.concatenate(train_feature), np.concatenate(train_label))
        test_data = (np.concatenate(test_feature), np.concatenate(test_label))
        if is_save:
            root_path = os.path.dirname(__file__)
            if self.add_text:
                train_path = os.path.abspath(os.path.join(root_path, self.config['data']['train']['add_feature_path']))
                test_path = os.path.abspath(os.path.join(root_path, self.config['data']['test']['add_feature_path']))
            else:
                train_path = os.path.abspath(os.path.join(root_path, self.config['data']['train']['path']))
                test_path = os.path.abspath(os.path.join(root_path, self.config['data']['test']['path']))
            save_pkl(train_data, train_path)
            save_pkl(test_data, test_path)
        return train_data, test_data
