# -*- coding:utf-8 -*-
import os
import numpy as np
import pandas as pd
import jieba
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Process:
    def __init__(self, cfg):
        self.cfg = cfg['data']
        data_path = os.path.join(self.cfg['root_dir'], self.cfg['name'])
        stp_path = os.path.join(self.cfg['root_dir'], self.cfg['stopwords'])
        self.df = pd.read_excel(data_path)
        with open(stp_path, 'r', encoding='utf-8') as f:
            self.stp = f.read().split('\n')
        self.stp = set(self.stp)
        jieba.initialize()

    def tokenize(self):
        columns = self.df.columns[1:]
        for col in columns:
            self.df[col + '_token'] = self.df[col].apply(lambda x: jieba.lcut(self.remove_letter(x)))
            self.df[col + '_token'] = self.df[col + '_token'].apply(lambda x: [w for w in x if w not in self.stp])
        self.df = self.df[self.df.iloc[:, -1].str.len() != 0].reset_index(drop=True)

    @staticmethod
    def remove_letter(content):
        letters = ', . ， 。 \n \t'.split(' ')
        for letter in letters:
            content = content.replace(letter, ' ')
        return content

    def get_tokens(self):
        self.tokenize()
        tokens = self.df[self.df.columns[-2]].tolist() + self.df[self.df.columns[-1]].tolist()
        return tokens


class CBowModel:
    def __init__(self, cfg):
        self.cfg = cfg['cbow']
        self.model = None
        self.isTrain = False

    def fit(self, sentences):
        self.model = Word2Vec(sentences,
                              vector_size=self.cfg['vector_size'],
                              window=self.cfg['window'],
                              min_count=self.cfg['min_count'],
                              sg=self.cfg['sg'],
                              epochs=self.cfg['epochs'],
                              seed=42
                              )
        self.isTrain = True

    def get_vector(self, word):
        if self.isTrain:
            return self.model.wv[word]


class TFIDFModel:
    def __init__(self, cfg):
        self.cfg = cfg['tfidf']
        self.model = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        self.isTrain = False
        self.wv = None
        self.idx2word = None
        self.word2idx = None

    def fit(self, sentences):
        sentences = [" ".join(s) for s in sentences]
        self.wv = self.model.fit_transform(sentences)
        self.isTrain = True
        self.word2idx = self.model.vocabulary_
        self.idx2word = {i: word for word, i in self.word2idx.items()}

    def cal_weights(self, contents):
        res = []
        for i, words in enumerate(contents):
            weights = []
            for word in words:
                idx = self.word2idx.get(word)
                if idx:
                    weights.append(self.wv[i, idx])
                else:
                    weights.append(0)
            weights = np.array(weights)
            weights_sum = np.sum(weights)
            if len(weights) == 0:
                res.append(None)
            elif weights_sum == 0:
                res.append(np.where(weights == 0, 1 / len(weights), weights))
            else:
                res.append(weights / weights_sum)
        return res


class IDEAData:
    def __init__(self, cfg):
        self.process = Process(cfg)
        self.cbow = CBowModel(cfg)
        sentences = self.process.get_tokens()
        self.cbow.fit(sentences)
        self.tfidf = TFIDFModel(cfg)
        self.tfidf.fit(sentences[: len(sentences) // 2])

    @property
    def title(self):
        return self.process.df[self.process.df.columns[-2]]

    @property
    def content(self):
        return self.process.df[self.process.df.columns[-1]]

    def get_vector(self, word):
        return self.cbow.get_vector(word)

    def get_title_weights(self):
        return self.tfidf.cal_weights(self.title)

    def cal_title_vector(self):
        weights = self.get_title_weights()
        titles = self.title
        res = []
        for i in range(len(weights)):
            if weights[i] is None:
                res.append(np.zeros(self.cbow.cfg['vector_size']))
            else:
                title_vector = np.array([self.get_vector(t) for t in titles[i]])
                title_vector = weights[i].reshape(-1, 1) * title_vector
                res.append(np.sum(title_vector, axis=0))
        return np.array(res)

    def cal_content_vector(self):
        title_vectors = self.cal_title_vector()
        contents = self.content
        content_vectors = []
        for content in contents:
            content_vectors.append([self.get_vector(w) for w in content])

        content_weights = []
        for i in range(len(contents)):
            title_vector = title_vectors[i].reshape(1, -1)
            content_vector = np.array(content_vectors[i])
            similarity = cosine_similarity(content_vector, title_vector).reshape(-1)
            content_weights.append(self.softmax(similarity))
        res = []
        for i in range(len(contents)):
            content_vector = np.array(content_vectors[i])
            content_weight = content_weights[i].reshape(-1, 1)
            res.append(np.sum(content_vector * content_weight, axis=0))

        return np.array(res)

    @staticmethod
    def softmax(x):
        return np.exp(x) / np.sum(np.exp(x))

    @property
    def title_vector(self):
        return self.cal_title_vector()

    @property
    def content_vector(self):
        return self.cal_content_vector()
