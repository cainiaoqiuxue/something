# -*- coding:utf-8 -*-
import os
import re
import jieba
import pandas as pd
import matplotlib.pyplot as plt


class DataProcess:
    def __init__(self, data_path, stopwords=None):
        jieba.initialize()
        self.ori_data = self.read_data(data_path)
        self.data = self.extract(self.ori_data)
        if stopwords:
            self.stopwords = self.read_txt(stopwords)
            self.stopwords = set(self.stopwords.split('\n'))
        else:
            self.stopwords = set()

    @staticmethod
    def read_txt(path):
        with open(path, 'r', encoding='utf-8') as f:
            res = f.read()
        return res

    def read_data(self, path):
        if os.path.isdir(path):
            files = os.listdir(path)
            res = ''
            for file in files:
                if file.endswith('txt'):
                    res += self.read_txt(os.path.join(path, file)) + '\n'
            return res
        else:
            return self.read_txt(path)

    # @staticmethod
    # def _extract(data):
    #     data = data.strip().split('\n\n')
    #     patterns = {
    #         'database': re.compile('SrcDatabase-来源库: (.*?)$'),
    #         'title': re.compile('Title-题名: (.*?)$'),
    #         'organ': re.compile('Organ-单位: (.*?)$'),
    #         'source': re.compile('Source-文献来源: (.*?)$'),
    #         'keyword': re.compile('Keyword-关键词: (.*?)$'),
    #         'summary': re.compile('Summary-摘要: (.*?)$'),
    #         'pubtime': re.compile('PubTime-发表时间: (.*?)$'),
    #     }
    #     res = []
    #     for d in data:
    #         item = dict()
    #         for pattern in patterns:
    #             search_res = re.search(patterns[pattern], d)
    #             if search_res:
    #                 item[pattern] = search_res.group(1)
    #         res.append(item)
    #     return res

    @staticmethod
    def extract(data):
        data = data.strip().split('SrcDatabase')
        patterns = {
            'database': re.compile('SrcDatabase-来源库: (.*?)\n'),
            'title': re.compile('Title-题名: (.*?)\n'),
            'organ': re.compile('Organ-单位: (.*?)\n'),
            'source': re.compile('Source-文献来源: (.*?)\n'),
            'keyword': re.compile('Keyword-关键词: (.*?)\n'),
            'summary': re.compile('Summary-摘要: (.*?)\n'),
            'pubtime': re.compile('PubTime-发表时间: (.*?)$'),
        }
        res = []
        for d in data:
            if len(d) == 0:
                continue
            d = 'SrcDatabase' + d.replace('\n\n', '\n')
            item = dict()
            for pattern in patterns:
                search_res = re.search(patterns[pattern], d)
                if search_res:
                    item[pattern] = search_res.group(1)
            res.append(item)
        return res

    def get_tokens(self, col):
        res = []
        for data in self.data:
            tokens = jieba.lcut(data.get(col, ''))
            token_list = []
            for token in tokens:
                if token not in self.stopwords and len(token) > 1:
                    token_list.append(token)
            res.append(token_list)
        return res

    def get_date(self, year_only=True):
        res = []
        for d in self.data:
            if 'pubtime' in d:
                res.append(d['pubtime'][:10])
            else:
                res.append(res[-1])
        if year_only:
            res = [i[:4] for i in res]
        return res

    def split_keywords(self, gather=True):
        res = []
        for data in self.data:
            if 'keyword' in data:
                keywords = data['keyword'].strip().split(';')
            else:
                keywords = []
            res.append(keywords)
        if gather:
            res = list(set([i for j in res for i in j if len(i) > 0]))
        return res

    def year_count(self):
        pd.DataFrame(self.get_date(), columns=['date']).groupby('date').size().plot.bar()
        plt.xticks(rotation=0)
        plt.show()

    @staticmethod
    def add_words(words):
        for word in words:
            jieba.add_word(word)
