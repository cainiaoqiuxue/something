import os
import re
import yaml
import warnings
import jieba
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import spearmanr
import pyecharts.options as opts
from pyecharts.charts import ThemeRiver
from wordcloud import WordCloud


warnings.filterwarnings('ignore')
jieba.initialize()
plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['simsun']})
plt.rcParams['figure.dpi'] = 150
plt.style.use('bmh')


class Process:
    def __init__(self, config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        self.data_dir = os.path.abspath(os.path.join(os.path.dirname(config_file), self.config['data_dir']))
        self.df = None
        self.area = None

    def set_area(self, area):
        self.area = area
        self.read_data()

    def read_data(self):
        files = os.listdir(self.data_dir)
        file_prefix = tuple(self.config[self.area]['file_prefix'])
        df = pd.DataFrame()
        for file in files:
            if file.startswith(file_prefix):
                df = pd.concat([df, pd.read_json(os.path.join(self.data_dir, file))], ignore_index=True)
        self.df = df
        self.clean_nick()
        self.clean_date()

    def clean_nick(self):
        patterns = [
            re.compile('//@.*?:'),
            re.compile('\\\\u[\da-zA-Z]{4}'),
            re.compile('(收起d)|(展开c)')
        ]
        for pattern in patterns:
            self.df['text'] = self.df['text'].apply(lambda x: re.sub(pattern, '', x))

    def clean_date(self):
        self.df = self.df[~self.df['date'].str.contains('今天')].copy()
        self.df['date'] = self.df['date'].apply(lambda x: x[:-5].replace('月', '-').replace('年', '-').replace('日', ''))
        self.df['date'] = pd.to_datetime(self.df['date'])

    def date_count_plot_v0(self):
        self.df['date'].value_counts().plot()
        plt.xlabel('日期')
        plt.ylabel('每日发微博数')
        plt.show()

    def date_count_plot(self):
        num = self.df['date'].value_counts().sort_index().tolist()
        offsets = self.config[self.area]['fix']
        for offset in offsets:
            num[offset[0]] -= offset[1]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(num)
        vlines = self.config[self.area]['period']
        for i in vlines:
            ax.axvline(i, color='gray', linestyle='--')
        ax.set_xticks(vlines)

        # ax.text(0.5, 600, '征兆期')
        # ax.text(4, 600, '爆发期')

        ax.set_xlabel('天数')
        ax.set_ylabel('每日发微博数')

    def action_count_plot(self):
        action_data = self.df.groupby('date').sum()
        forward = action_data['forward'].sort_index().tolist()
        comment = action_data['comment'].sort_index().tolist()
        like = action_data['like'].sort_index().tolist()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(forward, label='转发数')
        ax.plot(comment, label='评论数')
        ax.plot(like, label='点赞数')
        vlines = self.config[self.area]['period']
        for i in vlines:
            ax.axvline(i, color='gray', linestyle='--')
        ax.set_xticks(vlines)

        plt.legend()

    def cal_spearman(self):
        name = ['转发', '评论', '点赞']
        matrix, value = spearmanr(self.df[['forward', 'comment', 'like']])
        matrix = pd.DataFrame(matrix, index=name, columns=name)
        value = pd.DataFrame(value, index=name, columns=name)
        print('Correlation Matrix:')
        print(matrix)
        print('Sig (two-tailed) values:')
        print(value)

    @staticmethod
    def stream_topic_plot(df, save_path):
        idx = df.index
        cols = df.columns
        x_data = cols
        y_data = []
        for col in cols:
            y_data += [*zip(idx, df[col], [col for _ in range(df.shape[0])])]

        ThemeRiver().add(
            series_name=x_data,
            data=y_data,
            singleaxis_opts=opts.SingleAxisOpts(
                pos_top="50", pos_bottom="50", type_="time"
            ),
        ).set_global_opts(
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="line")
        ).render(save_path)


def stream_topic(topic_data, raw_data):
    df = pd.read_csv(topic_data, encoding='gbk')
    df = df[df['话题'].notnull()][['Topic', '话题']]
    topic_map = dict(zip(df['Topic'].values, df['话题'].values))
    df = pd.read_csv(raw_data)
    df['topic'] = df['topic'].map(topic_map)
    df = df.dropna(subset=['topic'])
    tmp = df.groupby(['date', 'topic']).count()['nick'].reset_index()
    df = df.groupby(['date', 'topic']).count()['nick']
    x_data = tmp['topic'].unique().tolist()
    date = tmp['date'].unique().tolist()
    y_data = []
    for idx, x in tmp.iterrows():
        y_data.append([x['date'], x['nick'], x['topic']])

    return ThemeRiver().add(
        series_name=x_data,
        data=y_data,
        singleaxis_opts=opts.SingleAxisOpts(
            pos_top="50", pos_bottom="50", type_="time"
        ),
    ).set_global_opts(
        tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="line")
    ).render_notebook()


def topic_corr(topic_data, raw_data):
    sig_map = {'微博官方认证': '官方', '微博个人认证': '大V用户', '未认证': '普通用户'}
    df = pd.read_csv(topic_data, encoding='gbk')
    df = df[df['话题'].notnull()]
    topic_map = dict(zip(df['Topic'].values, df['话题'].values))
    df = pd.read_csv(raw_data)
    df['topic'] = df['topic'].map(topic_map)
    df = df.dropna(subset=['topic'])
    sig = df['sig'].unique()
    for s in sig:
        print(sig_map[s], '关注top5:')
        res = df[df['sig'] == s].groupby('topic').count()['forward'].sort_values(ascending=False)[:5].index.tolist()
        for r in res:
            print(r)
        print('-' * 50)


def topic_word_cloud(topic_data, raw_data):
    df = pd.read_csv(topic_data, encoding='gbk').astype(str)
    df = df[df['话题'].notnull()][['Topic', '话题']]
    topic_map = dict(zip(df['Topic'].values, df['话题'].values))
    df = pd.read_csv(raw_data).astype(str)
    df['topic'] = df['topic'].map(topic_map)
    df = df.dropna(subset=['topic'])
    stop_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../config/stop.txt'))
    with open(stop_path, 'r', encoding='utf-8') as f:
        stopwords = f.readlines()

    stopwords = set([i.strip() for i in stopwords])
    stopwords.add('\u200b')
    stopwords.add('##')

    docs = [jieba.lcut(i) for i in df['text'].tolist()]
    words = []
    for d in docs:
        for w in d:
            if w not in stopwords:
                words.append(w)

    word_dic = Counter(words)
    wd = WordCloud(font_path='C:\Windows\Fonts\simsun.ttc', background_color='white', scale=60)
    wd.generate_from_frequencies(word_dic)
    plt.axis('off')
    plt.imshow(wd)
    name = topic_data.split('/')[-1].split('_')[0]
    path = os.path.dirname(os.path.abspath(topic_data))
    plt.savefig(os.path.join(path, '{}_word_cloud.png'.format(name)), dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    process = Process('../config/area.yaml')
    process.set_area('qinghai')
    process.date_count_plot()
    plt.show()
