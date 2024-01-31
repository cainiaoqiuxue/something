# -*- coding:utf-8 -*-
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class DataProcess:
    def __init__(self, data_path, data_dir):
        self.df = pd.read_excel(data_path)
        self.data_dir = data_dir
        self.label_col = '热度'

    def get_data_id(self, tid):
        try:
            df = pd.read_excel(os.path.join(self.data_dir, '{}.xlsx'.format(tid)))
        except:
            df = None
        return df

    def get_stats_value(self, row):
        tid = row['序号']
        df = self.get_data_id(tid)
        if df is None:
            return None, None, None, None, None, None, None
        num = df.shape[0]
        dz = df['点赞'].mean()
        zf = df['转发'].mean()
        pl = df['评论'].mean()
        fs = df['粉丝数'].mean()
        gz = df['关注数'].mean()
        dy = df['地域'].nunique()
        return num, dz, zf, pl, fs, gz, dy

    def cal_numerical_feature(self):
        tmp = self.df.apply(self.get_stats_value, axis=1, result_type='expand')
        self.df['微博数 点赞 转发 评论 粉丝 关注 地域'.split(' ')] = tmp

    def split_data(self, train_size=0.8):
        train_df, test_df = train_test_split(self.df, train_size=train_size, random_state=42)
        return train_df, test_df

    def get_sentiment_data(self, sentiment):
        sentiment_map = {
            '正向情感': 0,
            '中立情感': 1,
            '负向情感': 2
        }
        return self.df[self.df['情感倾向'] == sentiment_map[sentiment]].copy()

    def plot_sentiment(self):
        data = self.df['情感倾向'].value_counts().to_dict()
        sentiment_map = {
            0: '正向情感',
            1: '中立情感',
            2: '负向情感',
        }
        values = data.values()
        labels = [sentiment_map[i] for i in data.keys()]
        plt.figure(figsize=(10, 8))
        plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=180)
        plt.axis('equal')
        plt.title('情感分析占比')
        plt.legend()
        plt.show()