# -*- coding:utf-8 -*-

"""
 FileName     : tw_bash.py
 Type         : pyspark/pysql/python
 Arguments    : None
 Author       : xingyuanfan@tencent.com
 Date         : 2023-07-25
 Description  : 
"""
import os
import sys
import warnings
from collections import Counter

import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from read_data import read_csv, clean_data, read_stopwords_v2, split_text, read_csv_dir

warnings.filterwarnings('ignore')


class TWPipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.prefix_path = cfg.output_name
        self.data_path = cfg.output_name + '.csv'
        self.root_dir = os.path.join(cfg.project_root, self.prefix_path)
        self.input_dir = os.path.join(cfg.project_root, cfg.raw_data_dir)

        if not os.path.exists(self.root_dir):
            os.mkdir(self.root_dir)
            print('创建文件夹 {}'.format(self.prefix_path))
        else:
            print('文件夹已存在，如果要覆盖先删除重新创建，不覆盖原文件，断点重启可忽略')

    @staticmethod
    def save_file(df, file_name):
        file_type = file_name.split('.')[-1]
        if file_type == 'csv':
            df.to_csv(file_name, index=False, encoding='utf_8_sig')
        elif file_type == 'xlsx':
            df.to_excel(file_name, index=False, encoding='utf_8_sig', engine='xlsxwriter')
        else:
            raise RuntimeError("invalid file type in: {}".format(file_name))

    def data_gather(self):
        df = read_csv_dir(self.input_dir)
        if self.cfg.drop_duplicates:
            df = df.drop_duplicates()
        self.save_file(df, os.path.join(self.root_dir, self.data_path))
        print('data_gather done')

    def data_clean(self):
        df = read_csv(os.path.join(self.root_dir, self.data_path))
        df['clean_data'] = df['Embedded_text'].apply(clean_data)
        df['tokens'] = df['clean_data'].apply(split_text, args=(read_stopwords_v2(),))
        self.save_file(df, os.path.join(self.root_dir, self.prefix_path + '_res.csv'))
        print('data_clean done')

    def sentiment_analysis(self):
        df = pd.read_csv(os.path.join(self.root_dir, self.prefix_path + '_res.csv'))
        tokenizer = AutoTokenizer.from_pretrained(self.cfg.sentiment_model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(self.cfg.sentiment_model_dir)

        def predict(text):
            token = tokenizer(text, return_tensors='pt')
            with torch.no_grad():
                output = model(**token).logits
            return output.argmax().item() - 1

        contents = df['clean_data'].tolist()
        labels = []
        for i in tqdm(contents):
            labels.append(predict(str(i)))
        df['sentiment'] = labels
        self.save_file(df, os.path.join(self.root_dir, self.prefix_path + '_res_sentiment.csv'))
        print('sentiment_analysis done')

    def lda_analysis(self):
        df = pd.read_csv(os.path.join(self.root_dir, self.prefix_path + '_res_sentiment.csv'))

        neg_df = df[df['sentiment'] == -1]
        neu_df = df[df['sentiment'] == 0]
        pos_df = df[df['sentiment'] == 1]

        def topic_analysis(tdf, n_components, top_n):
            contents = [' '.join(eval(i)) for i in tdf['tokens'].to_list()]
            tfidf = TfidfVectorizer(ngram_range=(1, self.cfg.n_gram))
            x = tfidf.fit_transform(contents)
            model = LatentDirichletAllocation(n_components=n_components, random_state=42)
            model.fit(x)
            if hasattr(tfidf, 'get_feature_names_out'):
                feature_names = tfidf.get_feature_names_out()
            else:
                feature_names = tfidf.get_feature_names()
            rows = []
            for topic in model.components_:
                topwords = [feature_names[i] for i in topic.argsort()[: -top_n - 1:-1]]
                rows.append(topwords)
            return rows

        def write_lda(rows, sentiment_type):
            with open(os.path.join(self.root_dir, 'lda.txt'), 'a') as f:
                f.write(sentiment_type)
                f.write('\n')
                for idx, row in enumerate(rows):
                    f.write('topic: {}\n'.format(idx + 1))
                    f.write(str(row))
                    f.write('\n')
                f.write('\n')
                f.write('=' * 100)
                f.write('\n')

        res_row = topic_analysis(df, self.cfg.n_components, self.cfg.top_n)
        write_lda(res_row, 'total')
        res_row = topic_analysis(pos_df, self.cfg.n_components, self.cfg.top_n)
        write_lda(res_row, 'positive')
        res_row = topic_analysis(neu_df, self.cfg.n_components, self.cfg.top_n)
        write_lda(res_row, 'neutrality')
        res_row = topic_analysis(neg_df, self.cfg.n_components, self.cfg.top_n)
        write_lda(res_row, 'negative')

        self.save_file(neg_df, os.path.join(self.root_dir, self.prefix_path + '_res_sentiment_neg.xlsx'))
        self.save_file(pos_df, os.path.join(self.root_dir, self.prefix_path + '_res_sentiment_pos.xlsx'))
        self.save_file(neu_df, os.path.join(self.root_dir, self.prefix_path + '_res_sentiment_neu.xlsx'))
        print('lda_analysis done')

    def word_count(self):
        df = pd.read_csv(os.path.join(self.root_dir, self.prefix_path + '_res_sentiment.csv'))
        sentiment_map = {'pos': 1, 'neu': 0, 'neg': -1}

        def sentiment_word_count(tdf, seg_name=None):
            if seg_name is not None:
                tdf = tdf[tdf['sentiment'] == sentiment_map[seg_name]]
            words = []
            for i in tdf['tokens']:
                words.extend(eval(i))
            words_dic = Counter(words)
            words = pd.DataFrame(words_dic.items(), columns=['word', 'count'])
            self.save_file(words, os.path.join(self.root_dir, self.prefix_path + '_word_count_{}.xlsx'.format(seg_name)))

        sentiment_word_count(df)
        sentiment_word_count(df, 'pos')
        sentiment_word_count(df, 'neg')
        sentiment_word_count(df, 'neu')
        print('word_count done')

    def word_cloud(self):
        df = pd.read_csv(os.path.join(self.root_dir, self.prefix_path + '_res_sentiment.csv'))
        words = []
        for i in df['tokens']:
            words.extend(eval(i))

        res_dic = Counter(words)
        wd = WordCloud(font_path=r'C:\Windows\Fonts\STXINGKA.TTF', background_color='white', scale=60)
        wd.generate_from_frequencies(res_dic)
        plt.imshow(wd)
        plt.axis('off')
        plt.savefig(os.path.join(self.root_dir, self.prefix_path + '_word_cloud.png'), dpi=self.cfg.dpi,
                    bbox_inches='tight')
        print('word_cloud done')

    def run(self):
        print('开始运行')
        self.data_gather()
        self.data_clean()
        self.sentiment_analysis()
        self.lda_analysis()
        self.word_count()
        self.word_cloud()
