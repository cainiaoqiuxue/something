# -*- coding:utf-8 -*-
import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from collections import Counter


class Process:
    def __init__(self, raw_data_dir, data_name):
        self.raw_data_dir = raw_data_dir
        self.data_name = data_name
        self.flag = False
        self.df = None
        self.clean_data()

    @staticmethod
    def read_file(path):
        file_type = path.split('.')[-1]
        if file_type == 'csv':
            return pd.read_csv(path)
        elif file_type == 'xlsx':
            return pd.read_excel(path)
        else:
            raise RuntimeError("unsupported file type: {}".format(file_type))

    def read_data(self):
        data_path = os.path.join(self.raw_data_dir, self.data_name)
        if os.path.exists(data_path):
            self.flag = True
            return self.read_file(data_path)
        else:
            dfs = []
            for file in os.listdir(self.raw_data_dir):
                dfs.append(self.read_file(os.path.join(self.raw_data_dir, file)))
            data = pd.concat(dfs, ignore_index=False)
            # data.to_csv(data_path, index=False, encoding='utf_8_sig')
            return data

    def clean_data(self):
        if self.df is None:
            self.df = self.read_data()

        # 去除性别异常
        self.df = self.df[self.df['性别'].isin(['男', '女'])]

        # 去除年龄异常
        self.df = self.df[self.df['年龄'] != '-']
        # self.df = self.df[self.df['年龄'].astype(int) < 100]

        # 姓名 + 电话 构成唯一id
        self.df['id'] = self.df.apply(lambda x: str(x['会员姓名']) + str(x['电话']), axis=1)

        #  条转盒
        self.df.loc[self.df['单位'] == '条', '商品数量'] = self.df.loc[self.df['单位'] == '条', '商品数量'] * 10
        self.df.loc[self.df['单位'] == '条', '商品名称'] = self.df.loc[self.df['单位'] == '条', '商品名称'].str[:-3]

        # 去除消费时间为空，商品名称为空的数据
        self.df = self.df.dropna(subset=['消费时间', '商品名称'])

        # 选取购买香烟的记录
        # self.df = self.df[self.df['商品名称'].isin(self.get_cig_name())]

    @staticmethod
    def get_cig_name(path='data/第二阶段数据及结果/卷烟.xlsx'):
        cig_enum = pd.read_excel(path)
        cig_name = cig_enum['商品名称'].unique().tolist()
        return cig_name

    @staticmethod
    def get_wine_name(path='data/第二阶段数据及结果/酒.xlsx'):
        df = pd.read_excel(path)
        return df['商品名称'].unique().tolist()

    def get_kind_data(self, kind):
        if kind == 0:
            return self.df[self.df['商品名称'].isin(self.get_cig_name())]
        elif kind == 1:
            return self.df[self.df['商品名称'].isin(self.get_wine_name())]
        else:
            return self.df[~self.df['商品名称'].isin(self.get_cig_name() + self.get_wine_name())]


class Scaler:
    def __init__(self, df, select_cols):
        self.df = df
        self.select_cols = select_cols
        self.params = {}
        self.train_data = None
        self. test_data = None
        self._init()

    def split_data(self):
        data = self.df[self.select_cols]
        self.train_data = data[data['消费时间'].apply(lambda x: x.month).isin([9, 10])]
        self.test_data = data[data['消费时间'].apply(lambda x: x.month).isin([11])]

    @staticmethod
    def get_date_diff(gdf):
        gdf['消费时间'] = gdf['消费时间'].dt.floor('D')
        gdf = gdf.sort_values('消费时间')
        gdf = gdf.drop_duplicates(subset='消费时间')
        return gdf['消费时间'].diff().apply(lambda x: x.days).mean()

    @staticmethod
    def count_prefer(gdf):
        goods = gdf['商品名称'].tolist()
        cgoods = Counter(goods)
        name, count = cgoods.most_common()[0]
        return name, count / sum(cgoods.values())

    @staticmethod
    def get_cig_info(path='data/第二阶段数据及结果/卷烟.xlsx'):
        cig_df = pd.read_excel(path)
        return dict(zip(cig_df['商品名称'], cig_df['建议零售价']))

    def fit_transform(self):
        data1 = self.train_data.groupby('id')['实收金额'].agg(**{'amount': 'sum', 'amount_top': 'max', 'frequency': 'count'}).reset_index()
        data2 = self.train_data.groupby('id')['商品数量'].agg(**{'number': 'sum'}).reset_index()
        data3 = pd.DataFrame({'interval': self.train_data.groupby('id').apply(self.get_date_diff)}).reset_index()
        data4 = (pd.to_datetime('2022-10-31') - self.train_data.groupby('id')['消费时间'].max().dt.floor('D')).apply(lambda x: x.days)
        data4 = pd.DataFrame({'last_time': 60 - data4})
        data4 = data4.fillna(60)
        data5 = pd.DataFrame({'level': self.train_data.groupby('id').apply(self.count_prefer)}).reset_index()
        data5['price'] = data5['level'].apply(lambda x: x[0])
        data5['level'] = data5['level'].apply(lambda x: x[1])
        data5['price'] = data5['price'].map(self.get_cig_info())
        data6 = self.train_data.groupby('id')[['性别', '年龄']].first().reset_index()
        data6.columns = ['id', 'gender', 'age']
        data6['gender'] = data6['gender'].map({'男': 1, '女': 0})

        data = data1.merge(data2, on='id').merge(data3, on='id').merge(data4, on='id').merge(data5, on='id').merge(data6, on='id')
        self.train_data = data

    def drop_outer(self, columns, threshold=0.99):
        for col in columns:
            value = self.train_data[col].quantile(threshold)
            self.train_data = self.train_data[self.train_data[col] <= value]
        self.train_data = self.train_data.reset_index(drop=True)

    def max_min_scaler(self, columns):
        for col in columns:
            v_max = self.train_data[col].max()
            v_min = self.train_data[col].min()
            self.params['{}_max'.format(col)] = v_max
            self.params['{}_min'.format(col)] = v_min
            self.train_data[col] = (self.train_data[col] - v_min) / (v_max - v_min)

    def get_label(self):
        label_id = self.test_data['id'].unique().tolist()
        self.train_data['label'] = self.train_data['id'].apply(lambda x: 1 if x in label_id else 0)

    def _init(self):
        self.split_data()
        self.fit_transform()
        self.drop_outer(['amount', 'amount_top', 'frequency', 'number', 'interval', 'last_time', 'price', 'age'])
        self.max_min_scaler(['amount', 'amount_top', 'frequency', 'number', 'interval', 'last_time', 'price', 'age'])
        self.get_label()


class RFMData:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.train_data = None
        self.test_data = None
        self._init()

    def split_data(self):
        self.df = self.df[(self.df['实收金额'] > 0) & (self.df['实收金额'] < 3000)]

        self.df['消费时间'] = pd.to_datetime(self.df['消费时间']).dt.floor('D')
        self.df = self.df.drop_duplicates(subset=['id', '消费时间'])

        self.train_data = self.df[(self.df['消费时间'] <= '2022-10-30') & (self.df['消费时间'] >= '2022-09-01')]
        self.test_data = self.df[(self.df['消费时间'] <= '2022-11-30') & (self.df['消费时间'] >= '2022-11-01')]

    def get_features(self, data: pd.DataFrame):
        data1 = data.groupby('id')['消费时间'].nunique().reset_index()
        data1.columns = ['id', 'frequency']
        data2 = data.groupby('id')['消费时间'].max() - data.groupby('id')['消费时间'].min()
        data2 = data2.reset_index()
        data2.columns = ['id', 'recency']
        data2['recency'] = data2['recency'].dt.days
        data3 = data['消费时间'].max() - data.groupby('id')['消费时间'].min()
        data3 = data3.reset_index()
        data3.columns = ['id', 'T']
        data3['T'] = data3['T'].dt.days
        data4 = data.groupby('id').apply(self.count_prefer).reset_index()
        data4.columns = ['id', 'preference']
        data5 = data.groupby('id')['实收金额'].mean().reset_index()
        data5.columns = ['id', 'money']
        data = data1.merge(data2, on='id').merge(data3, on='id').merge(data4, on='id').merge(data5, on='id')
        return data

    @staticmethod
    def count_prefer(gdf):
        goods = gdf['商品名称'].tolist()
        cgoods = Counter(goods)
        name, count = cgoods.most_common()[0]
        return count / sum(cgoods.values())

    def _init(self):
        self.split_data()
        # self.train_data = self.get_features(self.train_data)
        # self.test_data = self.get_features(self.test_data)
