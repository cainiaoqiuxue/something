# -*- coding:utf-8 -*-
import os  # 引入操作系统模块，用于文件路径操作和环境变量用途等
import pandas as pd  # 引入Pandas库，常用于数据分析和操作
import matplotlib.pyplot as plt  # 引入matplotlib的pyplot，用于数据可视化绘图
from sklearn.model_selection import train_test_split  # 引入train_test_split函数用于数据划分

# 设置matplotlib绘图时的字符显示，指定用SimHei字体来显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
# 解决matplotlib绘图时的负号显示问题
plt.rcParams['axes.unicode_minus'] = False

# 定义一个数据处理的类
class DataProcess:
    def __init__(self, data_path, data_dir):
        self.df = pd.read_excel(data_path)  # 读取指定路径Excel文件，加载到DataFrame中
        self.data_dir = data_dir  # 存储数据的目录路径
        self.label_col = '热度'  # 指定"热度"为标签列名

    def get_data_id(self, tid):
        try:
            # 尝试根据tid(假设是一个ID)，拼接文件名，读取相应的Excel文件
            df = pd.read_excel(os.path.join(self.data_dir, '{}.xlsx'.format(tid)))
        except:
            # 如果在尝试读取文件时发生异常，则返回None
            df = None
        return df

    def get_stats_value(self, row):
        tid = row['序号']  # 获取当前行的'序号'列的值作为tid
        df = self.get_data_id(tid)  # 获取tid对应的数据
        if df is None:
            # 如果没有获取到数据，则返回七个None
            return None, None, None, None, None, None, None
        # 如果成功获取数据，计算以下统计值
        num = df.shape[0]  # 计算微博数（数据框的行数）
        dz = df['点赞'].mean()  # 计算点赞的平均数
        zf = df['转发'].mean()  # 计算转发的平均数
        pl = df['评论'].mean()  # 计算评论的平均数
        fs = df['粉丝数'].mean()  # 计算粉丝数的平均数
        gz = df['关注数'].mean()  # 计算关注数的平均数
        dy = df['地域'].nunique()  # 计算不同地域的数量
        # 返回统计值
        return num, dz, zf, pl, fs, gz, dy

    def cal_numerical_feature(self):
        # 使用apply方法对DataFrame的每行执行get_stats_value函数，
        # result_type='expand'表示将多列结果扩展成DataFrame
        tmp = self.df.apply(self.get_stats_value, axis=1, result_type='expand')
        # 将计算结果合并到类对象的df属性，新增对应的列名
        self.df['微博数', '点赞', '转发', '评论', '粉丝', '关注', '地域'] = tmp

    def split_data(self, train_size=0.8):
        # 使用train_test_split分割数据集，train_size表示训练集比例，random_state确保可复现
        train_df, test_df = train_test_split(self.df, train_size=train_size, random_state=42)
        return train_df, test_df  # 返回分割后的训练集和测试集

    def get_sentiment_data(self, sentiment):
        # 定义情感映射字典
        sentiment_map = {
            '正向情感': 0,
            '中立情感': 1,
            '负向情感': 2
        }
        # 通过情感映射字典获取对应情感的DataFrame
        return self.df[self.df['情感倾向'] == sentiment_map[sentiment]].copy()

    def plot_sentiment(self):
        # 统计情感倾向列的每个值的数量，并转化为字典
        data = self.df['情感倾向'].value_counts().to_dict()
        # 定义情感倾向的映射关系
        sentiment_map = {
            0: '正向情感',
            1: '中立情感',
            2: '负向情感',
        }
        # 提取字典中的值（饼图的大小）
        values = list(data.values())
        # 转换键（情感倾向的数值）为相应的标签
        labels = [sentiment_map[i] for i in data.keys()]
        # 创建一个饼图的绘图对象，大小为10x8英寸
        plt.figure(figsize=(10, 8))
        # 绘制饼图，显示每部分的百分比，从180度开始
        plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=180)
        # 确保饼图为正圆形
        plt.axis('equal')
        # 设置饼图的标题
        plt.title('情感分析占比')
        # 显示图例
        plt.legend()
        # 展示绘制好的图
        plt.show()